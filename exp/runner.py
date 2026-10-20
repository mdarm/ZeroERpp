"""Label-free within-panel configuration selection and full-data replay."""
from hashlib import sha256
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import sqlite3
from time import perf_counter
import zlib

import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState
from sqlalchemy.engine import make_url

from exp.config import BENCH_DIR, MATCHING_STORAGE, STORAGE
from utils.decisions import decision_rows, threshold_curve
from models.matcher import ZeroERMatcher
from data.features.panels import PanelTransform, load_cell, record_disjoint_splits


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return sha256(_json(value).encode()).hexdigest()


def _code_hash():
    root = Path(__file__).resolve().parents[1]
    return sha256(b"".join((root / name).read_bytes() for name in (
        "exp/runner.py", "data/features/panels.py", "data/features/embeddings.py",
        "data/features/bank.py", "utils/decisions.py",
        "models/zeroer.py", "models/matcher.py"))).hexdigest()


def _database(storage):
    url = make_url(storage)
    if url.drivername != "sqlite" or not url.database or url.database == ":memory:":
        raise ValueError("Use a separate file-backed SQLite database for these experiments")
    path = Path(url.database).resolve()
    if path in {Path(make_url(old).database).resolve() for old in (STORAGE, MATCHING_STORAGE)}:
        raise ValueError("Unsupervised experiments require a new database; legacy results are protected")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=8)
def _cached_storage(storage, process_id):
    with sqlite3.connect(_database(storage), timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    return optuna.storages.RDBStorage(storage, engine_kwargs={"connect_args": {"timeout": 60}})


def _storage(storage):
    return _cached_storage(storage, os.getpid())


def _configurations(configs):
    configs = dict(configs) if isinstance(configs, dict) else {
        "config-{}".format(i): config for i, config in enumerate(configs)}
    if not configs:
        raise ValueError("At least one configuration is required")
    configs = {str(name): dict(config) for name, config in configs.items()}
    policies = {config.get("missing", "zero") for config in configs.values()}
    if len(policies) != 1:
        raise ValueError("Compare NLL within one missingness policy; use separate studies")
    if any(config.get("transitivity", False) for config in configs.values()):
        raise ValueError("Use decision-stage greedy decoding; mixture selection requires transitivity=False")
    return configs


def _distributions(configs):
    keys = {key for config in configs.values() for key in config}
    if any(set(config) != keys for config in configs.values()):
        raise ValueError("Configurations must explicitly specify the same parameter keys")
    return {key: CategoricalDistribution(list(dict.fromkeys(config[key] for config in configs.values())))
            for key in sorted(keys)}


def _prepare(cell, panel, train, validation):
    start = perf_counter()
    attributes = [a for a in (panel.get("attributes") or cell["attributes"]) if a != "record"]
    transformer = PanelTransform(panel["models"], attributes,
                                 fusion=panel.get("fusion", "separate"),
                                 metrics=panel.get("metrics", ["cos"]))
    transformer.fit(cell["bank"].iloc[train], cell["present"].iloc[train])
    x_train = transformer.transform(cell["bank"].iloc[train], cell["present"].iloc[train])
    x_valid = transformer.transform(cell["bank"].iloc[validation], cell["present"].iloc[validation])
    return x_train, x_valid, perf_counter() - start


def _observed(present, columns):
    blocks = list(dict.fromkeys(name.split("_")[0] for name in columns))
    return present[blocks].to_numpy(dtype=bool), blocks


def _diagnostics(matcher, scores, present, columns):
    observed, blocks = _observed(present, columns)
    predicted = np.asarray(scores["p_match"]) >= .5
    strata, missingness = [], []
    nll = -np.asarray(scores["log_density"])
    for count in np.unique(observed.sum(axis=1)):
        keep = observed.sum(axis=1) == count
        strata.append(dict(n_blocks=int(count), n=int(keep.sum()),
                           nll_sum=float(nll[keep].sum()), nll=float(nll[keep].mean())))
    for i, block in enumerate(blocks):
        mask = observed[:, i]
        a, b = int((predicted & mask).sum()), int((predicted & ~mask).sum())
        c, d = int((~predicted & mask).sum()), int((~predicted & ~mask).sum())
        denom = float((a + b) * (c + d) * (a + c) * (b + d)) ** .5
        missingness.append(dict(attribute=block, present=int(mask.sum()), n=len(mask),
                                predicted_present=a, predicted_absent=b,
                                phi=(a * d - b * c) / denom if denom else None))
    model = matcher.model
    informed = observed.any(axis=1)
    order = {block: float(np.mean(np.asarray(model.Mu_M)[[j for j, c in enumerate(columns)
                        if c.split("_")[0] == block]] - np.asarray(model.Mu_U)[[j for j, c in enumerate(columns)
                        if c.split("_")[0] == block]])) for block in blocks}
    return dict(pi=float(scores["pi"]), init_fraction=float(np.mean(model.y)),
                converged=bool(model.converged), n_iter=int(model.n_iter),
                kappa_capped=float(np.mean(np.asarray(model.kappas) >= 1)),
                n_features=len(columns), feature_names=list(columns),
                n_no_features=int((observed.sum(axis=1) == 0).sum()),
                n_informative=int(informed.sum()),
                informative_nll=float(nll[informed].mean()) if informed.any() else None,
                component_mean_difference=order, presence_strata=strata, missingness=missingness)


def _fit(config, pairs, x_train, present_train, x_valid, present_valid):
    start = perf_counter()
    matcher = ZeroERMatcher(**config)
    matcher.match_pairs(pairs, x_train, present=present_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    scores = matcher.model.score_components(x_valid.to_numpy(), present=present_valid)
    if not np.isfinite(scores["log_density"]).all():
        raise ValueError("Non-finite validation log density")
    if not 0 < scores["pi"] < 1:
        raise ValueError("Mixture collapsed to a single component")
    score_seconds = perf_counter() - start
    attrs = _diagnostics(matcher, scores, present_valid, x_train.columns)
    attrs.update(fit_seconds=fit_seconds, score_seconds=score_seconds)
    return matcher, scores, attrs


def run_search(dataset, side, k, panel, configs, split_spec, storage,
               experiment="unsupervised-v1", retry_failed=False):
    """Minimise held-out mixture NLL inside a fixed panel; never read gold."""
    _database(storage)
    configs = _configurations(configs)
    distributions = _distributions(configs)
    panel = dict(panel, models=sorted(panel["models"]), metrics=list(panel.get("metrics", ["cos"])))
    cell = load_cell(dataset, side, k, panel["models"], metrics=panel["metrics"],
                     include_record=panel.get("include_record", True))
    panel.setdefault("attributes", list(cell["attributes"]))
    panel.setdefault("fusion", "separate")
    panel_name = panel.get("name", "panel")
    spec = dict(dataset=dataset, side=side, k=int(k), panel=panel, configs=configs,
                split_spec=dict(split_spec), experiment=experiment,
                input_hash=cell["input_hash"], code_hash=_code_hash())
    protocol_hash = _hash(spec)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", panel_name)
    name = "{}_{}_{}_k{}_{}_{}".format(dataset, experiment, side, k, label, _hash(panel)[:8])
    study = optuna.create_study(study_name=name, storage=_storage(storage), direction="minimize", load_if_exists=True)
    old_hash = study.user_attrs.get("protocol_hash")
    if old_hash is not None and old_hash != protocol_hash:
        raise ValueError("Study protocol, inputs or code changed; use a new experiment name")
    if old_hash is None:
        if study.trials:
            raise ValueError("Existing study has trials without protocol metadata")
        for key, value in dict(spec=spec, protocol_hash=protocol_hash, phase="search", dataset=dataset,
                               side=side, k=int(k), panel_name=panel_name, experiment=experiment,
                               objective="validation_nll", load_timings=cell.get("timings", {})).items():
            study.set_user_attr(key, value)
    finished = {t.user_attrs.get("config_hash") for t in study.trials
                if t.state == TrialState.COMPLETE or (t.state == TrialState.FAIL and not retry_failed)}
    todo = [(name, config) for name, config in configs.items() if _hash(config) not in finished]
    if not todo:
        return study
    splits = record_disjoint_splits(cell["ids"], **split_spec)
    prepared = []
    for split in splits:
        try:
            if not len(split["train"]) or not len(split["validation"]):
                raise ValueError("Empty training or validation fold")
            prepared.append(_prepare(cell, panel, split["train"], split["validation"]))
        except ValueError as error:
            prepared.append(error)
    study.set_user_attr("splits", [split["metadata"] for split in splits])
    for config_name, config in todo:
        rows, start = [], perf_counter()
        for fold, (split, inputs) in enumerate(zip(splits, prepared)):
            train, validation = split["train"], split["validation"]
            row = dict(split["metadata"], fold=fold,
                       n_train=len(train), n_validation=len(validation), n_excluded=len(split["excluded"]))
            try:
                if isinstance(inputs, Exception):
                    raise inputs
                x_train, x_valid, row["preprocessing_seconds"] = inputs
                _, scores, attrs = _fit(config, cell["pairs"][train], x_train,
                    cell["present"].iloc[train], x_valid, cell["present"].iloc[validation])
                nll_sum = float(-np.asarray(scores["log_density"]).sum())
                row.update(attrs, nll_sum=nll_sum, nll=nll_sum / len(validation))
            except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                row["error"] = repr(error)
            rows.append(row)
        errors = sum("error" in row for row in rows)
        n = sum(row["n_validation"] for row in rows)
        value = sum(row["nll_sum"] for row in rows) / n if not errors and n else None
        attrs = dict(config_name=config_name, config_hash=_hash(config), folds=rows,
                     n_errors=errors, n_validation=n, seconds=perf_counter() - start)
        if value is not None:
            attrs.update(validation_nll=value, nll_sum=sum(row["nll_sum"] for row in rows))
        study.add_trial(optuna.trial.create_trial(params=config, distributions=distributions,
            value=value, state=TrialState.COMPLETE if value is not None else TrialState.FAIL,
            user_attrs=attrs))
    return study


def _load_gold(dataset, ids):
    gold = pd.read_csv(Path(BENCH_DIR) / dataset / "matches.csv", dtype=str)
    pairs = set(map(tuple, gold.iloc[:, :2].to_numpy()))
    return np.array([pair in pairs for pair in map(tuple, ids.to_numpy())]), len(pairs)


def _baselines(cell, panel):
    bank, present = cell["bank"], cell["present"]
    models, attributes = panel["models"], [a for a in panel["attributes"] if a != "record"]
    record = cell.get("record_scores", pd.DataFrame(index=bank.index))
    for model in record:
        yield "record_cos:" + model, record[model].to_numpy(), present["record"].to_numpy(dtype=bool)
    if len(record.columns):
        values = record.mean(axis=1).to_numpy()
        yield "record_cos:mean", values, present["record"].to_numpy(dtype=bool)
    for metrics, name in [(["cos"], "attribute_cos:mean"), (panel["metrics"], "attribute_mixed:mean")]:
        if name.startswith("attribute_mixed") and len(metrics) < 2:
            continue
        columns = ["{}_emb_{}__{}".format(a, metric, model)
                   for a in attributes for metric in metrics for model in models]
        observed = np.column_stack([present[a].to_numpy(dtype=bool)
                                   for a in attributes for metric in metrics for model in models])
        count = observed.sum(axis=1)
        values = np.divide(np.where(observed, bank[columns].to_numpy(), 0).sum(axis=1), count,
                           out=np.zeros(len(bank)), where=count > 0)
        yield name, values, count > 0


def _store_decisions(storage, study, number, rows, curves):
    """Queryable result rows plus complete exact threshold curves, in the new DB."""
    with sqlite3.connect(_database(storage), timeout=120) as connection:
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS unsupervised_decisions (
            study_name TEXT, trial_number INTEGER, row_number INTEGER, score_name TEXT,
            decoder TEXT, rule TEXT, threshold_json TEXT, uses_gold INTEGER,
            tp INTEGER, predicted INTEGER, n_gold INTEGER, precision REAL, recall REAL,
            f1 REAL, decode_seconds REAL, PRIMARY KEY(study_name, trial_number, row_number));
        CREATE TABLE IF NOT EXISTS unsupervised_thresholds (
            study_name TEXT, trial_number INTEGER, score_name TEXT, decoder TEXT,
            n_points INTEGER, encoding TEXT, curve BLOB,
            PRIMARY KEY(study_name, trial_number, score_name, decoder));
        """)
        connection.executemany("INSERT OR REPLACE INTO unsupervised_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (study.study_name, number, i, row.get("score_name", row.get("score")), row["decoder"], row["rule"],
                 _json(row["threshold"]), int(row["uses_gold"]), row["tp"], row["predicted"], row["n_gold"],
                 row["precision"], row["recall"], row["f1"], row.get("decode_seconds"))
                for i, row in enumerate(rows)])
        connection.executemany("INSERT OR REPLACE INTO unsupervised_thresholds VALUES (?,?,?,?,?,?,?)", [
            (study.study_name, number, score_name, decoder, len(curve), "json+zlib",
             zlib.compress(_json(curve).encode())) for score_name, decoder, curve in curves])


def load_threshold_curve(storage, study_name, trial_number, score_name, decoder):
    with sqlite3.connect(_database(storage)) as connection:
        row = connection.execute("""SELECT curve FROM unsupervised_thresholds
            WHERE study_name=? AND trial_number=? AND score_name=? AND decoder=?""",
            (study_name, trial_number, score_name, decoder)).fetchone()
    if row is None:
        raise KeyError((study_name, trial_number, score_name, decoder))
    return pd.DataFrame(json.loads(zlib.decompress(row[0])))


def run_refit(study, storage, thresholds, output_dir, config_override=None, selection="cv_nll", retry_failed=False):
    """Refit the selected label-free configuration, then attach gold diagnostics."""
    _database(storage)
    if isinstance(study, str):
        study = optuna.load_study(study_name=study, storage=_storage(storage))
    candidates = [t for t in study.trials if t.state == TrialState.COMPLETE
                  and t.value is not None and np.isfinite(t.value)]
    if not candidates:
        raise ValueError("No complete finite-NLL configuration is available")
    selected = min(candidates, key=lambda t: (t.value, t.number))
    spec = study.user_attrs["spec"]
    config = dict(selected.params, **(config_override or {}))
    policy = dict(source_protocol=study.user_attrs["protocol_hash"], source_trial=selected.number,
                  thresholds=thresholds, decoder_policy="threshold-and-greedy-no-evidence-abstain-v1",
                  code_hash=_code_hash(), config_override=config_override or {}, selection=selection)
    refit_hash = _hash(policy)
    suffix = "_full" if not config_override and selection == "cv_nll" else "_{}_full".format(
        re.sub(r"[^A-Za-z0-9_.-]+", "-", selection))
    full = optuna.create_study(study_name=study.study_name + suffix, storage=_storage(storage),
                               direction="minimize", load_if_exists=True)
    if full.user_attrs.get("refit_hash", refit_hash) != refit_hash:
        raise ValueError("Refit selection or threshold policy changed; use a new experiment name")
    if any(t.state == TrialState.COMPLETE or (t.state == TrialState.FAIL and not retry_failed)
           for t in full.trials):
        return full
    for key, value in dict(spec=spec, policy=policy, refit_hash=refit_hash, phase="full",
                           dataset=spec["dataset"], side=spec["side"], k=spec["k"],
                           panel_name=study.user_attrs["panel_name"], experiment=spec["experiment"],
                           role="ablation" if config_override else "selected", selection=selection).items():
        full.set_user_attr(key, value)
    start = perf_counter()
    panel = spec["panel"]
    cell = load_cell(spec["dataset"], spec["side"], spec["k"], panel["models"],
                     metrics=panel["metrics"], include_record=panel.get("include_record", True))
    if cell["input_hash"] != spec["input_hash"] or _code_hash() != spec["code_hash"]:
        raise ValueError("Inputs or code changed since selection; rerun under a new experiment name")
    all_rows = np.arange(len(cell["ids"]))
    try:
        features, _, preprocessing_seconds = _prepare(cell, panel, all_rows, all_rows)
        _, scores, attrs = _fit(config, cell["pairs"], features, cell["present"],
                               features, cell["present"])
    except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        full.add_trial(optuna.trial.create_trial(params=config, distributions=_distributions({selection: config}),
            state=TrialState.FAIL, user_attrs=dict(error=repr(error), selection=selection,
                source_study=study.study_name, source_trial=selected.number, config_hash=_hash(config),
                seconds=perf_counter() - start)))
        return full
    observed, _ = _observed(cell["present"], features.columns)
    eligible = observed.any(axis=1)
    # Gold is first opened here, after model/configuration selection and refitting.
    is_gold, n_gold = _load_gold(spec["dataset"], cell["ids"])
    p_match = np.asarray(scores["p_match"])
    clipped = np.clip(p_match, 1e-15, 1 - 1e-15)
    attrs.update(brier=float(np.mean((p_match - is_gold) ** 2)),
                 label_log_loss=float(-np.mean(is_gold * np.log(clipped) + (~is_gold) * np.log1p(-clipped))),
                 calibration_population="candidate pairs", n_gold=n_gold,
                 tp_blocking=int(is_gold.sum()), candset_size=len(is_gold))
    score_specs = [("llr", scores["llr"], eligible, thresholds["llr"]),
                   ("posterior", p_match, eligible, thresholds["posterior"])]
    score_specs += [(name, values, valid, thresholds["cosine"]) for name, values, valid in _baselines(cell, panel)]
    decisions, curves = [], []
    saved = cell["ids"].copy()
    for name in ("log_match", "log_nonmatch", "llr", "p_match", "log_density", "n_blocks"):
        saved[name] = scores[name]
    saved["eligible"] = eligible
    for name, values, valid, cuts in score_specs:
        decisions.extend(decision_rows(cell["pairs"], values, is_gold, n_gold, name, cuts, eligible=valid))
        for decoder in ("threshold", "greedy"):
            curves.append((name, decoder, threshold_curve(cell["pairs"], values, is_gold, n_gold,
                                                          decoder=decoder, eligible=valid)))
        saved[name if name != "posterior" else "p_match"] = values
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (full.study_name + "__" + refit_hash[:12] + ".parquet")
    saved.to_parquet(path, index=False)
    attrs.update(selection=selection, source_study=study.study_name, source_trial=selected.number,
                 config_hash=_hash(config), validation_nll=selected.value,
                 scores_path=str(path.resolve()), decisions=decisions,
                 preprocessing_seconds=preprocessing_seconds, load_timings=cell.get("timings", {}),
                 seconds=perf_counter() - start)
    _json(attrs)  # Reject invalid numeric values before persisting a completed result.
    number = len(full.trials)
    _store_decisions(storage, full, number, decisions, curves)
    full.add_trial(optuna.trial.create_trial(params=config, distributions=_distributions({selection: config}),
                                            value=selected.value, user_attrs=attrs))
    return full

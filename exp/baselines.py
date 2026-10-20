"""Original ZeroER baselines on locally prepared experiment candidates.

The primary prediction is the original last-E-step posterior >= .5. No final
parameter refresh, missing-aware treatment, embedding feature or re-blocking is
introduced. Gold is opened only after fitting, for external reporting.
"""
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from exp.config import FEATURES_DIR, LADDER, cell_name
from utils.decisions import decision_rows, threshold_curve
from data.benchmarks import DROP_GROUPS
from models.matcher import ZeroERMatcher
from exp.runner import (_code_hash as _shared_code_hash, _database, _distributions,
                          _hash, _json, _load_gold, _storage, _store_decisions)
from data.features.panels import _candidate_input


BASELINES = {
    "B-paper": dict(c_bay=.01, init_threshold=.5, tol=1e-5, max_iter=200,
                    init_statistic="sum", missing="zero", hard=False,
                    transitivity=False, pi_M=None),
    "B-code": dict(c_bay=.1, init_threshold=.8, tol=None, max_iter=40,
                   init_statistic="sum", missing="zero", hard=False,
                   transitivity=False, pi_M=None),
}


def _code_hash():
    return sha256(_shared_code_hash().encode() + Path(__file__).read_bytes()).hexdigest()


def _load_cell(dataset, side, k):
    """Generated string similarities, in the embedding experiment pair order."""
    started = perf_counter()
    k_max = max(kk for ss, kk in LADDER[dataset] if ss == side)
    exact_source = Path(FEATURES_DIR) / (cell_name(dataset, side, k) + ".csv.gz")
    if exact_source.exists():
        k_max = k
    if not 1 <= k <= k_max:
        raise ValueError("k must lie within the cached candidate range")
    source = Path(FEATURES_DIR) / (cell_name(dataset, side, k_max) + ".csv.gz")
    if not source.exists():
        raise FileNotFoundError(
            f"Missing generated string features: {source}. "
            "Run `python main_zeroer.py prepare --scope all` before the baselines.")
    stored = pd.read_csv(source, dtype={"ltable_id": str, "rtable_id": str})
    if not {"ltable_id", "rtable_id"} <= set(stored.columns):
        raise ValueError("String feature cache must contain both candidate ID columns")
    if len(stored) % k_max:
        raise ValueError("stored feature rows do not form complete candidate lists")
    stored = stored[np.arange(len(stored)) % k_max < k].reset_index(drop=True)
    ids, pairs, _, _ = _candidate_input(dataset, side, k)
    if not ids.equals(stored[["ltable_id", "rtable_id"]]):
        raise ValueError("Magellan and embedding experiments must use identical candidate rows")
    features = stored.drop(columns=["ltable_id", "rtable_id"])
    if not len(features.columns) or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("String feature cache must contain finite numeric similarity features")
    blocks = features.columns.str.split("_").str[0]
    features = features.loc[:, ~blocks.isin(DROP_GROUPS.get(dataset, []))]
    raw_features = features.copy()
    features = features.loc[:, features.nunique() > 1]
    digest = sha256()
    for frame in (ids, features):
        digest.update(_json(list(frame.columns)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    from data.prepare import validate_feature_cache
    provenance = validate_feature_cache(source, dataset, side, k_max)
    return dict(ids=ids, pairs=pairs, features=features, input_hash=digest.hexdigest(),
                raw_features=raw_features, source_k=k_max, provenance=provenance,
                source=str(source.resolve()), loading_seconds=perf_counter() - started)


def run_baselines(dataset, side, k, storage, output_dir,
                  baselines=("B-paper", "B-code"), thresholds=(.5,),
                  experiment="unsupervised-v1", retry_failed=False):
    """Fit fixed original baselines and persist decisions in the new database.

    Study values are the constant zero: there is no baseline selection or
    gold objective. Gold F1, declared cuts and exact oracle curves are reporting
    attributes. The original posterior cut .5 is always included. Feature
    extraction runs in ``prepare``, so only load/fit/reporting time is timed here.
    """
    _database(storage)
    names = tuple(baselines)
    if not names or len(set(names)) != len(names) or set(names) - set(BASELINES):
        raise ValueError("baselines must name B-paper and/or B-code without duplicates")
    cuts = list(dict.fromkeys([.5] + [float(value) for value in thresholds]))
    if not np.isfinite(cuts).all():
        raise ValueError("declared baseline thresholds must be finite")
    configs = {name: dict(BASELINES[name]) for name in names}
    cell = _load_cell(dataset, side, k)
    spec = dict(dataset=dataset, side=side, k=int(k), configs=configs, thresholds=cuts,
                experiment=experiment, input_hash=cell["input_hash"], code_hash=_code_hash(),
                feature_source=cell["source"], score_state="legacy_last_e_step",
                feature_policy="magellan_style_exclude_known_leakage_drop_constants",
                feature_provenance=cell["provenance"])
    protocol_hash = _hash(spec)
    name = "{}_{}_{}_k{}_baseline".format(dataset, experiment, side, k)
    study = optuna.create_study(study_name=name, storage=_storage(storage),
                                direction="minimize", load_if_exists=True)
    previous = study.user_attrs.get("protocol_hash")
    if previous is not None and previous != protocol_hash:
        raise ValueError("Baseline inputs, settings or code changed; use a new experiment name")
    if previous is None:
        if study.trials:
            raise ValueError("Existing baseline trials lack protocol metadata")
        for key, value in dict(spec=spec, protocol_hash=protocol_hash, phase="baseline",
                               dataset=dataset, side=side, k=int(k), panel_name="handcrafted",
                               experiment=experiment, role="baseline",
                               objective="fixed_configuration_no_selection").items():
            study.set_user_attr(key, value)
    finished = {trial.user_attrs.get("baseline") for trial in study.trials
                if trial.state == TrialState.COMPLETE
                or (trial.state == TrialState.FAIL and not retry_failed)}
    distributions = _distributions(configs)
    for baseline, config in configs.items():
        if baseline in finished:
            continue
        started = perf_counter()
        attrs = dict(baseline=baseline, config_name=baseline, config_hash=_hash(config),
                     code_hash=spec["code_hash"], input_hash=cell["input_hash"],
                     score_state="legacy_last_e_step", feature_source=cell["source"],
                     feature_names=cell["features"].columns.tolist(),
                     n_features=cell["features"].shape[1], candset_size=len(cell["ids"]),
                     loading_seconds=cell["loading_seconds"],
                     feature_generation_seconds=None, feature_generation="outside_fit_runtime",
                     feature_provenance=cell["provenance"])
        try:
            matcher = ZeroERMatcher(**config)
            matcher.match_pairs(cell["pairs"], cell["features"])
            posterior = np.asarray(matcher.posteriors, dtype=float).copy()
            if posterior.shape != (len(cell["ids"]),) or not np.isfinite(posterior).all():
                raise ValueError("baseline posteriors must be finite and candidate-aligned")
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            attrs.update(error=repr(error), seconds=perf_counter() - started)
            study.add_trial(optuna.trial.create_trial(params=config, distributions=distributions,
                                                      state=TrialState.FAIL, user_attrs=attrs))
            continue
        attrs.update(fit_seconds=perf_counter() - started,
                     n_iter=int(matcher.model.n_iter), converged=bool(matcher.model.converged),
                     init_fraction=float(np.mean(matcher.model.y)),
                     final_parameter_pi=float(matcher.model.pi_M))
        # Labels cannot influence fitting or any declared decision rule.
        is_gold, n_gold = _load_gold(dataset, cell["ids"])
        score_name = "legacy_posterior"
        rows = decision_rows(cell["pairs"], posterior, is_gold, n_gold, score_name, cuts)
        curves = [(score_name, decoder, threshold_curve(cell["pairs"], posterior,
                   is_gold, n_gold, decoder=decoder)) for decoder in ("threshold", "greedy")]
        primary = next(row for row in rows if row["decoder"] == "threshold"
                       and row["rule"] == "fixed" and row["threshold"] == .5)
        path = Path(output_dir) / (study.study_name + "__" + baseline + "__"
                                   + protocol_hash[:12] + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        cell["ids"].assign(p_match=posterior, original_match=posterior >= .5).to_parquet(path, index=False)
        attrs.update(scores_path=str(path.resolve()), decisions=rows, primary_decision=primary,
                     tp_blocking=int(is_gold.sum()), n_gold=int(n_gold),
                     **{key: primary[key] for key in ("tp", "predicted", "precision", "recall", "f1")},
                     seconds=perf_counter() - started)
        _json(attrs)
        _store_decisions(storage, study, len(study.trials), rows, curves)
        study.add_trial(optuna.trial.create_trial(params=config, distributions=distributions,
                                                  value=0., user_attrs=attrs))
    return study

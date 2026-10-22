"""Reporting-only manuscript diagnostics from saved scores and raw record IDs.

No model is fitted or rescored here. Score rows are joined by both external IDs;
greedy ties use full-table integer row positions, as in the experiment runner.
Gold is opened only for descriptive evaluation, never for score construction.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from data.benchmarks import DATASETS
from data.features.bank import load_presence
from exp import config
from exp.protocol import MODELS, PRIMARY_CELLS, CELLS
from utils.decisions import decode

IDS = ["ltable_id", "rtable_id"]
CELL = ["dataset", "side", "k"]
TIE_TOLERANCE = .0005
REFERENCE_KEYS = {
    "benchmark_datasets": ["dataset"], "fz_class_diagnostic": ["dataset"],
    "primary_raw_mean": CELL, "primary_posterior_saturation": CELL,
    "primary_presence_diagnostics": CELL + ["method", "indicator"],
    "primary_same_model_comparison": CELL + ["model"], "kalm_union_comparison": CELL + ["model"],
}


def _ids(frame):
    if frame[IDS].isna().any().any() or frame[IDS].astype(str).duplicated().any():
        raise ValueError("candidate IDs must be nonmissing and unique")
    return pd.MultiIndex.from_frame(frame[IDS].astype(str))


def align_presence(scores, presence):
    """Require identical pair populations, returning masks in score-row order."""
    score_ids, presence_ids = _ids(scores), _ids(presence)
    positions = presence_ids.get_indexer(score_ids)
    if len(score_ids) != len(presence_ids) or (positions < 0).any():
        raise ValueError("saved scores and presence masks have different candidate IDs")
    return presence.iloc[positions].reset_index(drop=True)


def raw_feature_mean(attribute_mean, n_present, record_cosine):
    """Unweighted observed attribute cosines plus one whole-record cosine."""
    attribute_mean, n_present, record_cosine = [np.asarray(v, dtype=float)
                                              for v in (attribute_mean, n_present, record_cosine)]
    if (attribute_mean.ndim != 1 or attribute_mean.shape != n_present.shape
            or attribute_mean.shape != record_cosine.shape):
        raise ValueError("raw-mean inputs must be row-aligned vectors")
    if (not all(np.isfinite(v).all() for v in (attribute_mean, n_present, record_cosine))
            or (n_present < 0).any() or (n_present != np.floor(n_present)).any()):
        raise ValueError("raw-mean inputs require finite scores and nonnegative feature counts")
    return (attribute_mean * n_present + record_cosine) / (n_present + 1)


def binary_association(predicted, indicator):
    """Cross-tab counts and phi; constant margins yield an undefined phi."""
    predicted, indicator = np.asarray(predicted), np.asarray(indicator)
    if (predicted.ndim != 1 or predicted.shape != indicator.shape
            or not np.isin(predicted, [False, True]).all()
            or not np.isin(indicator, [False, True]).all()):
        raise ValueError("association inputs must be row-aligned boolean vectors")
    predicted, indicator = predicted.astype(bool), indicator.astype(bool)
    n11 = int((predicted & indicator).sum())
    n10 = int((predicted & ~indicator).sum())
    n01 = int((~predicted & indicator).sum())
    n00 = int((~predicted & ~indicator).sum())
    denominator = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return dict(n_pairs=len(predicted), predicted=int(predicted.sum()),
                indicator_present=int(indicator.sum()), n11=n11, n10=n10, n01=n01, n00=n00,
                phi=(n11 * n00 - n10 * n01) / denominator if denominator else np.nan)


def _scores(database, trial):
    stored = trial.get("scores_path")
    if not isinstance(stored, str) or not stored:
        raise FileNotFoundError("selected trial has no saved scores_path")
    # Fresh outputs can move from /app (Docker) to another checkout or machine.
    local = Path(database).parent / "unsupervised_scores" / Path(stored).name
    path = local if local.is_file() else Path(stored)
    if not path.is_file():
        raise FileNotFoundError("saved scores missing: " + str(local))
    scores = pd.read_parquet(path)
    _ids(scores)
    posterior = scores["p_match"].to_numpy(dtype=float)
    if not np.isfinite(posterior).all() or ((posterior < 0) | (posterior > 1)).any():
        raise ValueError("stored posterior must be finite and within [0, 1]")
    return scores


def _trial(trials, dataset, side, k, baseline=None):
    if trials.empty:
        raise FileNotFoundError("selected completed trial unavailable")
    mask = trials.dataset.eq(dataset) & trials.side.eq(side) & trials.k.eq(k) & trials.state.eq("COMPLETE")
    if baseline:
        mask &= trials.phase.eq("baseline") & trials.baseline.eq(baseline)
    else:
        mask &= trials.phase.eq("full") & trials.role.eq("selected") & trials.panel_name.eq("single:" + config.MODEL)
    chosen = trials.loc[mask].sort_values("trial_number").drop_duplicates("study_name", keep="last")
    if len(chosen) != 1:
        raise FileNotFoundError("selected completed trial unavailable or ambiguous")
    return chosen.iloc[0]


def _population(dataset, scores, data_dir):
    """Return positional decoder IDs and full-gold membership for saved pairs."""
    positions = []
    offset = 0
    for side, column in zip(("left", "right"), IDS):
        ids = pd.read_csv(data_dir / dataset / (side + ".csv"), usecols=["id"], dtype=str)["id"]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError("raw table IDs must be nonmissing and unique")
        rows = pd.Index(ids).get_indexer(scores[column].astype(str))
        if (rows < 0).any():
            raise ValueError("saved score IDs absent from raw tables")
        positions.append(rows + offset)
        offset += len(ids)
    gold = pd.read_csv(data_dir / dataset / "matches.csv", dtype=str).iloc[:, :2]
    gold_pairs = set(map(tuple, gold.to_numpy()))
    is_gold = np.array([pair in gold_pairs for pair in map(tuple, scores[IDS].astype(str).to_numpy())])
    return np.column_stack(positions), is_gold, len(gold_pairs)


def _metrics(keep, is_gold, n_gold):
    tp, predicted = int((keep & is_gold).sum()), int(keep.sum())
    return dict(tp=tp, predicted=predicted, n_gold=n_gold,
                precision=tp / predicted if predicted else 0.,
                recall=tp / n_gold if n_gold else 0.,
                f1=2 * tp / (predicted + n_gold) if predicted + n_gold else 0.)


def same_model_comparisons(decisions, cells, models):
    """Pair each selected single-model fit with its own no-cut cosine control."""
    rows = []
    for dataset, side, k in cells:
        for model in models:
            row = dict(dataset=dataset, side=side, k=k, model=model, available=False,
                       tie_tolerance=TIE_TOLERANCE, metrics_use_gold=1)
            selected = decisions
            if not decisions.empty:
                selected = decisions.loc[decisions.dataset.eq(dataset) & decisions.side.eq(side)
                    & decisions.k.eq(k) & decisions.role.eq("selected")
                    & decisions.panel_name.eq("single:" + model) & decisions.decoder.eq("greedy")
                    & decisions.rule.eq("pure_transitivity")]
                if not selected.empty:
                    selected = selected.loc[selected.trial_number.eq(selected.trial_number.max())]
            fitted = selected.loc[selected.score_name.eq("llr")] if not selected.empty else selected
            cosine = selected.loc[selected.score_name.eq("record_cos:" + model)] if not selected.empty else selected
            if len(fitted) == len(cosine) == 1 and fitted.iloc[0].study_name == cosine.iloc[0].study_name:
                fit, control = fitted.iloc[0], cosine.iloc[0]
                if fit.n_gold != control.n_gold:
                    raise ValueError("paired controls have different full-gold denominators")
                delta = float(fit.f1 - control.f1)
                row.update(available=True, llr_f1=float(fit.f1), record_cosine_f1=float(control.f1),
                           f1_difference=delta, outcome="win" if delta > TIE_TOLERANCE else
                           "loss" if delta < -TIE_TOLERANCE else "tie")
            row["status"] = "available" if row["available"] else "unavailable: paired decisions missing"
            rows.append(row)
    return pd.DataFrame(rows)


def _comparison_summary(frame, by_model=False):
    groups = [("all", frame)] + (list(frame.groupby("model", sort=False)) if by_model else [])
    rows = []
    for model, group in groups:
        available = group.loc[group.available]
        outcomes = available.outcome.value_counts() if len(available) else {}
        delta = available.f1_difference if len(available) else pd.Series(dtype=float)
        rows.append(dict(model=model, expected_cells=len(group), available_cells=len(available),
            complete=len(available) == len(group), wins=outcomes.get("win", 0),
            ties=outcomes.get("tie", 0), losses=outcomes.get("loss", 0),
            mean_f1_difference=delta.mean(), min_f1_difference=delta.min(), max_f1_difference=delta.max(),
            max_absolute_f1_difference=delta.abs().max(), tie_tolerance=TIE_TOLERANCE,
            metrics_use_gold=1))
    return pd.DataFrame(rows)


def _dataset_reports(data_dir, record):
    rows, leak = [], []
    for dataset, spec in DATASETS.items():
        try:
            left, right = [pd.read_csv(data_dir / dataset / (side + ".csv"), dtype=str)
                           for side in ("left", "right")]
            gold = pd.read_csv(data_dir / dataset / "matches.csv", dtype=str).iloc[:, :2].drop_duplicates()
            rows.append(dict(dataset=dataset, n_left=len(left), n_right=len(right), n_gold=len(gold),
                maximum_left_gold_degree=gold.iloc[:, 0].value_counts().max(),
                maximum_right_gold_degree=gold.iloc[:, 1].value_counts().max(),
                attributes=", ".join(spec["attributes"]), metrics_use_gold=1))
            record("benchmark_datasets", (dataset, None, None))
            if dataset == "fodors_zagats":
                pairs = left[["id", "class"]].dropna().merge(right[["id", "class"]].dropna(), on="class")
                predicted = set(map(tuple, pairs[["id_x", "id_y"]].to_numpy()))
                truth = set(map(tuple, gold.to_numpy()))
                leak.append(dict(dataset=dataset, predicted=len(predicted), n_gold=len(truth),
                    tp=len(predicted & truth), false_positive=len(predicted - truth),
                    false_negative=len(truth - predicted), exact_gold_reconstruction=predicted == truth,
                    metrics_use_gold=1))
                record("fz_class_diagnostic", (dataset, None, None))
        except (FileNotFoundError, KeyError, ValueError) as error:
            if not rows or rows[-1]["dataset"] != dataset:
                record("benchmark_datasets", (dataset, None, None), error)
            if dataset == "fodors_zagats":
                record("fz_class_diagnostic", (dataset, None, None), error)
    return dict(benchmark_datasets=pd.DataFrame(rows), fz_class_diagnostic=pd.DataFrame(leak))


def publication_diagnostics(database, trials, decisions, *, data_dir=None):
    """Build descriptive exports; unavailable inputs stay explicit, never refit.

    ``trials`` and ``decisions`` follow ``utils.report_tables.read_sources``.
    Prepared pair-presence files and relocated scores are read beside the DB.
    Raw tables and gold come from ``data_dir`` (the CLI's configured data root).
    """
    data_dir = Path(data_dir or config.BENCH_DIR)
    bank_dir = Path(database).parent / "cosine_bank"
    availability, raw_mean, saturation, associations = [], [], [], []

    def record(diagnostic, cell, error=None, method="KaLM"):
        availability.append(dict(zip(CELL, cell), diagnostic=diagnostic, method=method,
            available=error is None, status="available" if error is None else "unavailable: " + str(error)))

    reports = _dataset_reports(data_dir, record)
    for cell in PRIMARY_CELLS:
        dataset, side, k = cell
        try:
            trial = _trial(trials, *cell)
            scores = _scores(database, trial)
            pairs, is_gold, n_gold = _population(dataset, scores, data_dir)
            if not scores["eligible"].isin([False, True]).all():
                raise ValueError("saved eligibility must be boolean and nonmissing")
            eligible = scores["eligible"].to_numpy(dtype=bool)
            posterior = scores["p_match"].to_numpy(dtype=float)
            llr_keep = decode(pairs, scores["llr"], "greedy", None, eligible)
            posterior_keep = decode(pairs, posterior, "greedy", None, eligible)
            changed = llr_keep ^ posterior_keep
            saturation.append(dict(zip(CELL, cell), n_pairs=len(scores),
                posterior_zero=int((posterior == 0).sum()), posterior_one=int((posterior == 1).sum()),
                accepted_mask_xor=int(changed.sum()), xor_at_zero=int((changed & (posterior == 0)).sum()),
                xor_at_one=int((changed & (posterior == 1)).sum()),
                llr_predicted=int(llr_keep.sum()), posterior_predicted=int(posterior_keep.sum()),
                llr_f1=_metrics(llr_keep, is_gold, n_gold)["f1"],
                posterior_f1=_metrics(posterior_keep, is_gold, n_gold)["f1"], metrics_use_gold=1))
            record("primary_posterior_saturation", cell)
        except (FileNotFoundError, KeyError, ValueError) as error:
            record("primary_posterior_saturation", cell, error)
            record("primary_raw_mean", cell, error)
        else:
            try:
                presence = align_presence(scores, load_presence(*cell, bank_dir=bank_dir))
                columns = ["present_" + attr for attr in DATASETS[dataset]["attributes"]]
                masks = presence[columns]
                if not masks.isin([False, True]).all().all():
                    raise ValueError("presence flags must be boolean and nonmissing")
                values = raw_feature_mean(scores["attribute_cos:mean"], masks.sum(axis=1),
                                          scores["record_cos:" + config.MODEL])
                keep = decode(pairs, values, "greedy", None)
                metrics = _metrics(keep, is_gold, n_gold)
                raw_mean.append(dict(zip(CELL, cell), **metrics, llr_f1=_metrics(llr_keep, is_gold, n_gold)["f1"],
                    fit_minus_raw_mean_f1=_metrics(llr_keep, is_gold, n_gold)["f1"] - metrics["f1"],
                    decoder="greedy", rule="pure_transitivity", post_hoc=1, metrics_use_gold=1))
                record("primary_raw_mean", cell)
            except (FileNotFoundError, KeyError, ValueError) as error:
                record("primary_raw_mean", cell, error)
        if dataset not in ("abt_buy", "amazon_googleproducts"):
            continue
        attribute = "price" if dataset == "abt_buy" else "manufacturer"
        for method in ("B-paper", "KaLM"):
            try:
                trial = _trial(trials, *cell, baseline="B-paper" if method == "B-paper" else None)
                scores = _scores(database, trial)
                presence = align_presence(scores, load_presence(*cell, bank_dir=bank_dir))
                _, is_gold, _ = _population(dataset, scores, data_dir)
                # Legacy last-E-step posteriors are preserved exactly as stored.
                predicted = scores["p_match"].to_numpy(dtype=float) >= .5
                for indicator, values in ((attribute + "_present", presence["present_" + attribute]),
                                          ("gold", is_gold)):
                    associations.append(dict(zip(CELL, cell), method=method, indicator=indicator,
                        threshold=.5, **binary_association(predicted, values), metrics_use_gold=1))
                record("primary_presence_diagnostics", cell, method=method)
            except (FileNotFoundError, KeyError, ValueError) as error:
                record("primary_presence_diagnostics", cell, error, method)
    primary = same_model_comparisons(decisions, PRIMARY_CELLS, MODELS)
    union = same_model_comparisons(decisions, CELLS, [config.MODEL])
    for name, frame in (("primary_same_model_comparison", primary), ("kalm_union_comparison", union)):
        for row in frame.itertuples():
            record(name, (row.dataset, row.side, row.k),
                   None if row.available else "paired decisions missing", row.model)
    reports.update(primary_raw_mean=pd.DataFrame(raw_mean), primary_presence_diagnostics=pd.DataFrame(associations),
        primary_posterior_saturation=pd.DataFrame(saturation), primary_same_model_comparison=primary,
        primary_same_model_summary=_comparison_summary(primary, by_model=True),
        kalm_union_comparison=union, kalm_union_summary=_comparison_summary(union),
        publication_diagnostic_availability=pd.DataFrame(availability))
    return reports


def compare_publication_diagnostics(reports, reference_path=None):
    """Require complete diagnostic evidence and compare with the original run."""
    path = Path(reference_path or Path(__file__).resolve().parents[1]
                / "data" / "publication-diagnostics-reference.json")
    reference = json.loads(path.read_text())
    if reference.get("schema") != "zeroerpp-publication-diagnostics-v1":
        raise AssertionError("Unknown publication diagnostic reference schema")
    availability = reports["publication_diagnostic_availability"]
    if availability.empty or not availability.available.all():
        missing = availability.loc[~availability.available, "diagnostic"].unique().tolist()
        raise AssertionError("Publication diagnostic inputs unavailable: " + ", ".join(missing))
    checked = {}
    for name, keys in REFERENCE_KEYS.items():
        expected = pd.DataFrame(reference["reports"][name])
        actual = reports[name]
        if set(actual) != set(expected) or actual.duplicated(keys).any():
            raise AssertionError(name + ": diagnostic columns or unique keys differ")
        actual = actual[expected.columns].sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                      check_exact=False, rtol=0, atol=1e-12, obj=name)
        checked[name] = len(actual)
    return dict(passed=True, checked=checked, numeric_atol=1e-12,
                reference_database_sha256=reference["provenance"]["database_sha256"])

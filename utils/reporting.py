"""Recreate publication tables from raw experiment records, without refitting.

Gold ranks the reported models and thresholds. Only configuration selection
inside each panel is label-free. Every metric uses the full dataset gold count.
"""
from pathlib import Path
import json
import sqlite3
import zlib

import numpy as np
import pandas as pd

from exp.config import LADDER, MODEL
from utils.report_tables import (EXPERIMENT, REFERENCE_EXPERIMENT, PRIMARY_CELLS,
    SENSITIVITY_CELLS, REPORT_CELLS, EXPECTED_PANELS, read_sources, make_reports,
    terminal_outcomes, completion_status)

DATASET_CODES = {"fodors_zagats": "FZ", "dblp_acm": "DA", "dblp_scholar": "DS",
                 "abt_buy": "AB", "amazon_googleproducts": "AG"}
CELL = ["dataset", "side", "k"]
DECISION_KEY = CELL + ["panel_name", "role", "selection", "baseline", "score_name",
                       "decoder", "rule", "threshold_json"]
ILLUSTRATION_MODEL = "codefuse-ai_F2LLM-v2-4B"


def _read_connection(database):
    return sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True, timeout=120)


def _scope(frame, cells):
    if frame.empty:
        return frame.copy()
    return frame.merge(pd.DataFrame(sorted(cells), columns=CELL), on=CELL,
                       how="inner", validate="many_to_one")


def audit_decisions(frame):
    """Reject inconsistent count-derived metrics before creating any summary."""
    if frame.empty:
        return {"rows": 0, "max_metric_residual": 0.0}
    if (frame[["tp", "predicted", "n_gold"]] < 0).any().any():
        raise ValueError("Negative evaluation count")
    if (frame.tp > frame.predicted).any() or (frame.tp > frame.n_gold).any():
        raise ValueError("True positives exceed prediction or full-gold counts")
    expected = {
        "precision": np.divide(frame.tp, frame.predicted, out=np.zeros(len(frame)), where=frame.predicted.ne(0)),
        "recall": np.divide(frame.tp, frame.n_gold, out=np.zeros(len(frame)), where=frame.n_gold.ne(0)),
        "f1": np.divide(2 * frame.tp, frame.predicted + frame.n_gold,
                        out=np.zeros(len(frame)), where=(frame.predicted + frame.n_gold).ne(0)),
    }
    residual = 0.0
    for name, values in expected.items():
        if not np.isfinite(frame[name]).all():
            raise ValueError("Nonfinite " + name)
        delta = float(np.max(np.abs(frame[name].to_numpy() - values)))
        residual = max(residual, delta)
        if delta > 1e-12:
            raise ValueError("Stored {} disagrees with full-gold counts".format(name))
    if frame.groupby("dataset").n_gold.nunique().gt(1).any():
        raise ValueError("Full-gold denominator differs within a dataset")
    return {"rows": len(frame), "max_metric_residual": residual}


def candidate_diagnostics(database, trials):
    """Use stored diagnostic counts or calculate them from regenerated pairs."""
    with _read_connection(database) as connection:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='unsupervised_candidate_diagnostics'").fetchone():
            return pd.read_sql_query("SELECT * FROM unsupervised_candidate_diagnostics ORDER BY dataset,side,k", connection)
    if trials.empty:
        return pd.DataFrame(columns=CELL)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from exp.config import BENCH_DIR
    from data.prepare import load_pairs
    rows = []
    cells = trials.loc[trials.state.eq("COMPLETE") & trials.phase.ne("search"), CELL].drop_duplicates()
    for dataset, side, k in cells.itertuples(index=False, name=None):
        try:
            pairs = load_pairs(dataset, side, int(k)).iloc[:, :2].astype(str)
            gold = set(map(tuple, pd.read_csv(Path(BENCH_DIR) / dataset / "matches.csv", dtype=str).iloc[:, :2].to_numpy()))
        except FileNotFoundError:
            # A read-only database report remains useful without raw inputs.
            # Structural bounds remain missing, and comparison fails explicitly.
            continue
        kept = np.array([tuple(pair) in gold for pair in pairs.to_numpy()])
        left, left_ids = pd.factorize(pairs.iloc[:, 0])
        right, right_ids = pd.factorize(pairs.iloc[:, 1])
        graph = coo_matrix((np.ones(int(kept.sum())), (left[kept], right[kept])),
                           shape=(len(left_ids), len(right_ids))).tocsr()
        cap = int(np.sum(maximum_bipartite_matching(graph, perm_type="column") >= 0))
        n_gold, retained = len(gold), int(kept.sum())
        rows.append(dict(dataset=dataset, side=side, k=int(k), n_pairs=len(pairs),
            n_left=len(left_ids), n_right=len(right_ids), n_gold=n_gold,
            gold_in_candidates=retained, maximum_gold_in_one_to_one=cap,
            blocking_recall=retained / n_gold, one_to_one_recall_ceiling=cap / n_gold,
            one_to_one_f1_ceiling=2 * cap / (cap + n_gold), uses_gold=1, role="reporting_only"))
    return pd.DataFrame(rows, columns=CELL + ["n_pairs", "n_left", "n_right", "n_gold",
        "gold_in_candidates", "maximum_gold_in_one_to_one", "blocking_recall",
        "one_to_one_recall_ceiling", "one_to_one_f1_ceiling", "uses_gold", "role"])


def _family_specs(frame):
    selected = frame.role.eq("selected")
    main = frame.role.isin(["selected", "baseline"])
    fitted = frame.score_name.isin(["llr", "posterior"])
    single = selected & fitted & frame.panel_name.str.startswith("single:")
    combinations = selected & fitted & ~frame.panel_name.str.startswith("single:")
    record = selected & frame.score_name.str.startswith("record_cos:")
    attribute = selected & frame.score_name.str.startswith("attribute_")
    fixed = frame.rule.eq("fixed") & frame.uses_gold.eq(0)
    no_cut = frame.rule.eq("pure_transitivity") & frame.uses_gold.eq(0)
    cut = pd.to_numeric(frame.threshold_json, errors="coerce")
    specs = []
    for baseline in ("B-paper", "B-code"):
        arm = frame.role.eq("baseline") & frame.baseline.eq(baseline)
        specs += [("reference", baseline + " · posterior ≥ .5", arm & fixed & cut.eq(.5) & frame.decoder.eq("threshold")),
                  ("reference", baseline + " · posterior ≥ .5 + greedy", arm & fixed & cut.eq(.5) & frame.decoder.eq("greedy")),
                  ("reference", baseline + " · no-cut greedy", arm & no_cut)]
    for name, arm in (("Single embedding", single), ("Embedding combinations", combinations)):
        for score, threshold, label in (("llr", 0., "LLR ≥ 0"), ("posterior", .5, "posterior ≥ .5")):
            exact = arm & frame.score_name.eq(score)
            specs += [("reference", name + " · " + label, exact & fixed & cut.eq(threshold) & frame.decoder.eq("threshold")),
                      ("reference", name + " · " + label + " + greedy", exact & fixed & cut.eq(threshold) & frame.decoder.eq("greedy")),
                      ("reference", name + " · " + score + " no-cut greedy", exact & no_cut)]
    specs += [("reference", "KaLM record cosine · no-cut greedy", record & no_cut & frame.score_name.eq("record_cos:" + MODEL)),
              ("reference", "Record cosine/models or means · no-cut greedy", record & no_cut),
              ("reference", "Attribute similarity means · no-cut greedy", attribute & no_cut)]
    for name, arm in (("B-paper", frame.role.eq("baseline") & frame.baseline.eq("B-paper")),
                      ("B-code", frame.role.eq("baseline") & frame.baseline.eq("B-code")),
                      ("Single embedding", single), ("Embedding combinations", combinations),
                      ("Record cosine/models or means", record), ("Attribute similarity means", attribute)):
        specs.append(("grid", name + " · best declared cut/decoder", arm & fixed))
    for name, selection in (("Hard EM diagnostic (MiniLM6)", "hard_em"), ("Zero-fill diagnostic (MiniLM6)", "zero_fill")):
        arm = frame.role.eq("ablation") & frame.selection.eq(selection) & fitted
        specs += [("diagnostic", name + " · posterior ≥ .5", arm & fixed & frame.score_name.eq("posterior") & cut.eq(.5) & frame.decoder.eq("threshold")),
                  ("diagnostic", name + " · best declared cut/decoder", arm & fixed)]
    specs.append(("oracle", "Exact score-threshold oracle · best scored method", main & frame.uses_gold.eq(1)))
    return specs


def _dataset_winners(frame, sensitivity=False):
    """Average each complete variant before any retrospective winner selection."""
    if frame.empty:
        return frame.copy()
    if sensitivity:
        keys = ["dataset", "panel_name", "role", "selection", "baseline", "score_name", "decoder", "rule"]
        # Oracle threshold varies per cell; fixed-grid thresholds must be shared.
        frame = frame.copy()
        frame["threshold_json"] = frame.threshold_json.where(frame.uses_gold.eq(0), np.nan)
        keys.append("threshold_json")
        frame = frame.groupby(keys, dropna=False).agg(f1=("f1", "mean"),
            cells=("study_name", "nunique")).reset_index()
        frame = frame.loc[frame.cells.eq(frame.dataset.map({d: len(c) for d, c in LADDER.items()}))]
    else:
        frame = frame.copy()
        frame["cells"] = 1
    return frame.sort_values(["f1", "panel_name", "score_name", "decoder", "threshold_json"],
        ascending=[False, True, True, True, True], kind="stable").drop_duplicates("dataset")


def publication_tables(decisions, diagnostics):
    """Build the 70-row method-family table and 27-panel primary table."""
    table, winners = [], []
    if decisions.empty:
        return {name: pd.DataFrame() for name in ("experiment_table", "experiment_winners", "panel_table", "panel_winners")}
    for scope, cells in (("primary", set(PRIMARY_CELLS)), ("sensitivity", SENSITIVITY_CELLS)):
        frame = _scope(decisions, cells)
        for section, label, mask in _family_specs(frame):
            chosen = _dataset_winners(frame.loc[mask], sensitivity=scope == "sensitivity")
            row = dict(scope=scope, section=section, experiment_label=label)
            row.update({code: float(chosen.loc[chosen.dataset.eq(dataset), "f1"].iloc[0])
                        if chosen.dataset.eq(dataset).any() else np.nan for dataset, code in DATASET_CODES.items()})
            table.append(row)
            chosen = chosen.copy()
            chosen["scope"], chosen["section"], chosen["experiment_label"] = scope, section, label
            chosen["summary_selection_uses_gold"] = 1
            winners.append(chosen)
        candidate = _scope(diagnostics, cells)
        for label, numerator, denominator in (
                ("Accept every candidate", "gold_in_candidates", "n_pairs"),
                ("Perfect matcher on retained candidates", "gold_in_candidates", "gold_in_candidates"),
                ("Perfect one-to-one matcher on retained candidates", "maximum_gold_in_one_to_one", "maximum_gold_in_one_to_one")):
            row = dict(scope=scope, section="bound", experiment_label=label)
            values = pd.Series(dtype=float)
            if not candidate.empty:
                score = 2 * candidate[numerator] / (candidate[denominator] + candidate.n_gold)
                grouped = candidate.assign(f1=score).groupby("dataset").agg(f1=("f1", "mean"), cells=("k", "size"))
                expected = {d: len(c) for d, c in LADDER.items()} if scope == "sensitivity" else {d: 1 for d in DATASET_CODES}
                values = grouped.f1.where(grouped.cells.eq(pd.Series(expected)))
            row.update({code: values.get(dataset, np.nan) for dataset, code in DATASET_CODES.items()})
            table.append(row)
    primary = _scope(decisions, set(PRIMARY_CELLS))
    eligible = primary.loc[primary.role.eq("selected") & primary.score_name.isin(["llr", "posterior"])
                           & primary.rule.eq("fixed") & primary.uses_gold.eq(0)]
    panel_winners = eligible.sort_values(["f1", "panel_name", "score_name", "decoder", "threshold_json"],
        ascending=[False, True, True, True, True], kind="stable").drop_duplicates(["panel_name", "dataset"])
    panels = panel_winners.pivot(index="panel_name", columns="dataset", values="f1").rename(columns=DATASET_CODES)
    panels = panels.reindex(columns=list(DATASET_CODES.values())).reset_index()
    panels.columns.name = None
    return dict(experiment_table=pd.DataFrame(table), experiment_winners=pd.concat(winners, ignore_index=True),
                panel_table=panels, panel_winners=panel_winners.sort_values(["panel_name", "dataset"]))


def primary_reports(reports):
    output = {}
    for name in ("decisions", "practical", "oracles", "calibration", "timings", "cosine_timings", "baseline_timings"):
        output["primary_" + name] = _scope(reports[name], set(PRIMARY_CELLS))
    studies, trials = reports["studies"], reports["trials"]
    rows = []
    for dataset, side, k in PRIMARY_CELLS:
        study = _scope(studies, {(dataset, side, k)})
        trial = _scope(trials, {(dataset, side, k)})
        counts = dict(dataset=dataset, side=side, k=k, expected_panels=EXPECTED_PANELS,
                      expected_baseline_trials=2, search_studies=0, finished_search_studies=0,
                      search_studies_without_valid_configuration=0,
                      selected_refits=0, selected_refits_terminal=0, selected_refits_failed=0,
                      baseline_trials=0, baseline_trials_terminal=0, baseline_trials_failed=0)
        if not study.empty:
            counts.update(search_studies=int(study.phase.eq("search").sum()),
                          finished_search_studies=int((study.phase.eq("search") & study.finished).sum()))
            selections = _scope(reports["selection"], {(dataset, side, k)})
            if not selections.empty:
                counts["search_studies_without_valid_configuration"] = int((selections.finished & selections.trial_id.isna()).sum())
        for prefix, mask, keys in (("selected_refits", trial.phase.eq("full") & trial.role.eq("selected"), ["study_name"]),
                                   ("baseline_trials", trial.phase.eq("baseline"), ["study_name", "config_hash"])) if not trial.empty else ():
            outcomes = terminal_outcomes(trial.loc[mask], keys)
            counts[prefix] = int(outcomes.complete.sum())
            counts[prefix + "_terminal"] = int(outcomes.terminal.sum())
            counts[prefix + "_failed"] = int(outcomes.failed.sum())
        counts["status"] = completion_status(EXPECTED_PANELS, counts["finished_search_studies"],
            counts["search_studies_without_valid_configuration"],
            counts["selected_refits"], counts["selected_refits_terminal"], counts["baseline_trials"], counts["baseline_trials_terminal"], 2)
        rows.append(counts)
    output["primary_coverage"] = pd.DataFrame(rows)
    practical = output["primary_practical"]
    for name, roles in (("primary_development_summary", ["selected", "baseline"]),
                         ("primary_ablation_summary", ["ablation"])):
        if practical.empty:
            output[name] = pd.DataFrame()
            continue
        subset = practical.loc[practical.role.isin(roles)].copy()
        subset["method"] = subset.baseline.where(subset.phase.eq("baseline"), subset.selection.fillna("cv_nll"))
        grouped = subset.groupby(["panel_name", "role", "method", "decision_label"], dropna=False).agg(
            dataset_equal_f1=("f1", "mean"), dataset_equal_precision=("precision", "mean"),
            dataset_equal_recall=("recall", "mean"), datasets=("dataset", "nunique")).reset_index()
        grouped["expected_datasets"], grouped["ranking_uses_gold"] = 5, 1
        grouped["complete_coverage"] = grouped.datasets.eq(5)
        output[name] = grouped
    fixed = output["primary_decisions"]
    if fixed.empty:
        output["primary_fixed_summary"] = pd.DataFrame()
        return output
    fixed = fixed.loc[fixed.role.isin(["selected", "baseline"]) & fixed.uses_gold.eq(0)
                      & fixed.rule.isin(["fixed", "pure_transitivity"])].copy()
    fixed["method"] = fixed.baseline.where(fixed.phase.eq("baseline"), fixed.selection.fillna("cv_nll"))
    fixed["decision_label"] = np.where(fixed.rule.eq("pure_transitivity"), fixed.score_name + ":greedy:no-cut",
        fixed.score_name + ":" + fixed.decoder + ":" + fixed.threshold_json)
    summary = fixed.groupby(["panel_name", "role", "method", "decision_label"], dropna=False).agg(
        dataset_equal_f1=("f1", "mean"), dataset_equal_precision=("precision", "mean"),
        dataset_equal_recall=("recall", "mean"), datasets=("dataset", "nunique"),
        cells=("study_name", "nunique"), observations=("f1", "size")).reset_index()
    summary["expected_datasets"], summary["expected_cells"] = 5, 5
    summary["complete_coverage"] = summary.datasets.eq(5) & summary.cells.eq(5) & summary.observations.eq(5)
    summary["ranking_uses_gold"] = 1
    output["primary_fixed_summary"] = summary.sort_values(
        ["complete_coverage", "dataset_equal_f1", "panel_name", "role", "method", "decision_label"],
        ascending=[False, False, True, True, True, True], kind="stable").reset_index(drop=True)
    return output


def primary_timing_comparison(reports):
    """Same-model LLR versus direct cosine; every row shares fit costs across rules."""
    rows = []
    for label, model in (("KaLM", MODEL), ("F2LLM-4B", ILLUSTRATION_MODEL),
                         ("F2LLM-8B", "codefuse-ai_F2LLM-v2-8B"), ("E5-large-v2", "e5_large_v2")):
        for pipeline, source, score, threshold in (
                ("llr_greedy_no_cut", "primary_timings", "llr", "null"),
                ("llr_greedy_minus5", "primary_timings", "llr", "-5.0"),
                ("record_cosine_greedy_no_cut", "primary_cosine_timings", "record_cos:" + model, "null")):
            frame = reports[source]
            if frame.empty:
                continue
            keep = frame.score_name.eq(score) & frame.decoder.eq("greedy") & frame.threshold_json.eq(threshold)
            if source == "primary_timings":
                keep &= frame.panel_name.eq("single:" + model) & frame.role.eq("selected")
            for item in frame.loc[keep].to_dict("records"):
                row = {key: item[key] for key in CELL + ["decoder", "threshold_json", "repeats"]}
                row.update(model=label, model_id=model, pipeline=pipeline)
                for stage in ("loading", "similarity", "preprocessing", "fit", "score", "aggregation", "decode", "matching", "total"):
                    row[stage + "_seconds"] = item.get(stage + "_seconds_median", 0.)
                row["matching_seconds_repeat_min"] = item["matching_seconds_min"]
                row["matching_seconds_repeat_max"] = item["matching_seconds_max"]
                rows.append(row)
    return pd.DataFrame(rows)


def _compact(frame, columns):
    return frame[[column for column in columns if column in frame]].copy()


def comparison_frames(reports):
    """Stable natural keys, counts and selected configurations; no local paths."""
    trials = reports["trials"]
    decisions = reports["decisions"]
    selected = trials.loc[trials.phase.ne("search")] if not trials.empty else trials
    return {
        "fit_configs": _compact(selected, CELL + ["panel_name", "phase", "role", "selection", "baseline", "state",
            "params_json", "source_validation_nll", "converged", "n_iter", "candset_size", "tp_blocking", "n_gold"]),
        "decision_counts": _compact(decisions, DECISION_KEY + ["uses_gold", "tp", "predicted", "n_gold", "precision", "recall", "f1"]),
        "selection_configs": _compact(reports["selection"], CELL + ["panel_name", "choice_status", "params_json", "cv_validation_nll"]),
    }


def timing_coverage(reports):
    """Verify where fresh timings exist without equating machine runtimes."""
    rows = []
    for kind, name, extra in (("fitted", "timing_repeats", ["panel_name"]),
                              ("cosine", "cosine_timing_repeats", []),
                              ("strings", "baseline_timing_repeats", ["baseline"])):
        frame = reports[name]
        if frame.empty:
            continue
        required = CELL + extra + ["repeat", "similarity_seconds", "matching_seconds", "total_seconds"]
        if not set(required) <= set(frame):
            raise ValueError(kind + " timing schema is incomplete")
        if not np.isfinite(frame[["similarity_seconds", "matching_seconds", "total_seconds"]]).all().all():
            raise ValueError(kind + " timing contains nonfinite durations")
        grouped = frame.groupby(CELL + extra, dropna=False).agg(repeats=("repeat", "nunique"), rows=("repeat", "size")).reset_index()
        for item in grouped.to_dict("records"):
            item["kind"] = kind
            rows.append(item)
    return pd.DataFrame(rows, columns=["kind"] + CELL + ["panel_name", "baseline", "repeats", "rows"])


def _write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression, float_format="%.17g")


def compare_reports(reports, reference_dir):
    """Strict metric/count/configuration comparison; wall times are not targets."""
    reference_dir = Path(reference_dir)
    frames = dict(comparison_frames(reports), experiment_table=reports["experiment_table"],
                  panel_table=reports["panel_table"])
    specifications = {
        "fit_configs": (CELL + ["panel_name", "phase", "role", "selection", "baseline"],
                        ["state", "params_json", "candset_size", "tp_blocking", "n_gold"]),
        "selection_configs": (CELL + ["panel_name"], ["choice_status", "params_json"]),
        "decision_counts": (DECISION_KEY, ["uses_gold", "tp", "predicted", "n_gold", "precision", "recall", "f1"]),
        "experiment_table": (["scope", "section", "experiment_label"], list(DATASET_CODES.values())),
        "panel_table": (["panel_name"], list(DATASET_CODES.values())),
    }
    checked = []
    for name, (keys, values) in specifications.items():
        reference = pd.read_csv(reference_dir / (name + ".csv.gz" if name.endswith("configs") or name == "decision_counts" else name + ".csv"), keep_default_na=False)
        actual = frames[name].copy().fillna("")
        if len(actual) != len(reference):
            raise AssertionError("{}: {} rows, expected {}; partial runs cannot establish full replication".format(name, len(actual), len(reference)))
        for frame in (reference, actual):
            if "params_json" in frame:
                frame["params_json"] = frame.params_json.map(lambda v: json.dumps(json.loads(v), sort_keys=True) if v else "")
            for key in keys:
                if key == "k":
                    frame[key] = pd.to_numeric(frame[key]).astype(int).astype(str)
                else:
                    frame[key] = frame[key].astype(str)
        # Oracle cut positions are floating-point scores, not fixed policy
        # constants. Compare their values separately while retaining exact
        # decision-count requirements. Fixed cuts remain exact natural keys.
        has_oracles = False
        if name == "decision_counts":
            has_oracles = True
            for frame in (reference, actual):
                frame["oracle_cut"] = pd.to_numeric(frame.threshold_json.where(frame.uses_gold.eq(1)), errors="coerce")
                frame.loc[frame.uses_gold.eq(1), "threshold_json"] = "oracle"
        if actual.duplicated(keys).any() or reference.duplicated(keys).any():
            raise AssertionError(name + ": duplicate natural keys")
        actual = actual.sort_values(keys).reset_index(drop=True)
        reference = reference.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[keys], reference[keys], check_dtype=False)
        if has_oracles:
            np.testing.assert_allclose(actual.oracle_cut, reference.oracle_cut,
                rtol=1e-8, atol=1e-6, equal_nan=True, err_msg="Oracle score cut drift")
        for column in values:
            if column in ("params_json", "state", "choice_status"):
                pd.testing.assert_series_equal(actual[column], reference[column], check_dtype=False)
            else:
                a = pd.to_numeric(actual[column], errors="coerce").to_numpy()
                b = pd.to_numeric(reference[column], errors="coerce").to_numpy()
                # Integer counts must agree exactly; tolerance only covers CSV float roundoff.
                atol = 1e-12 if column in ("precision", "recall", "f1", *DATASET_CODES.values()) else 0
                np.testing.assert_allclose(a, b, rtol=0, atol=atol, equal_nan=True,
                                           err_msg=name + ": " + column)
        checked.append(dict(table=name, rows=len(actual)))
    timing = timing_coverage(reports).fillna("")
    expected_timing = pd.read_csv(reference_dir / "timing_coverage.csv", keep_default_na=False)
    timing_keys = ["kind"] + CELL + ["panel_name", "baseline"]
    completed_timing = 0
    if not timing.empty:
        compared = timing.merge(expected_timing, on=timing_keys, how="left", suffixes=("", "_expected"),
                                validate="one_to_one", indicator=True)
        if not compared._merge.eq("both").all():
            raise AssertionError("Timing coverage contains an unexpected benchmark group")
        if not (compared.rows * compared.repeats_expected).eq(compared.rows_expected * compared.repeats).all():
            raise AssertionError("Timing decision-rule coverage differs from the reference")
        completed_timing = int(compared.repeats.ge(compared.repeats_expected).sum())
    timing_status = ("complete" if completed_timing == len(expected_timing)
                     else "partial" if len(timing) else "not_run")
    return {"passed": True, "checked": checked, "metric_atol": 1e-12,
            "oracle_score_cut_atol": 1e-6, "oracle_score_cut_rtol": 1e-8,
            "timing_coverage": {"status": timing_status, "complete_groups": completed_timing,
                                "expected_groups": len(expected_timing)},
            "counts_and_selected_configurations": "exact", "wall_time_comparison": "not compared"}


def report(database, output_dir, *, blocking_database=None, compare_reference=False, reference_dir=None):
    """Write deterministic CSV reports and a curve figure; never change a DB."""
    database, output_dir = Path(database), Path(output_dir)
    if not database.is_file():
        raise FileNotFoundError("Run matching experiments first: " + str(database))
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = read_sources(database)
    audit = audit_decisions(sources["decisions"])
    reports = make_reports(sources)
    reports.update(primary_reports(reports))
    reports["primary_timing_comparison"] = primary_timing_comparison(reports)
    diagnostics = candidate_diagnostics(database, reports["trials"])
    reports["candidate_diagnostics"] = diagnostics
    reports.update(publication_tables(reports["decisions"], diagnostics))
    # Compact exports retain the decision counts needed to independently audit
    # every summary; verbose timing repetitions remain in the experiment DB.
    skip = {"decisions", "primary_decisions", "practical", "primary_practical", "oracles", "primary_oracles",
            "timing_repeats", "cosine_timing_repeats", "baseline_timing_repeats", "run_costs", "trials"}
    for name, frame in reports.items():
        if name not in skip:
            frame = frame.drop(columns=[c for c in ("scores_path", "feature_source", "report_created_utc") if c in frame])
            _write_csv(frame, output_dir / (name + ".csv"))
    for name, frame in comparison_frames(reports).items():
        _write_csv(frame, output_dir / (name + ".csv.gz"))
    _write_csv(timing_coverage(reports), output_dir / "timing_coverage.csv")
    curves, availability = read_primary_threshold_curves(database, EXPERIMENT, PRIMARY_CELLS, ILLUSTRATION_MODEL)
    _write_csv(availability, output_dir / "threshold_curve_availability.csv")
    if curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure = plot_primary_threshold_curves(curves, PRIMARY_CELLS, ILLUSTRATION_MODEL)
        figure.savefig(output_dir / "primary_thresholds.png", dpi=180, facecolor="white")
        plt.close(figure)
    summary = {"coverage": reports["coverage"].to_dict("records"), "metric_audit": audit,
        "threshold_cells_available": len(curves), "complete_replication_verified": False,
        "interpretation": "Gold-scored development comparisons; NLL selects configurations only within panels."}
    fixed_summary = reports["primary_fixed_summary"]
    if not fixed_summary.empty and fixed_summary.complete_coverage.any():
        summary["best_shared_primary_rule"] = fixed_summary.loc[fixed_summary.complete_coverage,
            ["panel_name", "role", "method", "decision_label", "dataset_equal_f1", "ranking_uses_gold"]].iloc[0].to_dict()
    if blocking_database is not None and Path(blocking_database).is_file():
        summary["blocking"] = report_blocking(blocking_database, output_dir / "blocking",
            reference_dir=(reference_dir or Path(__file__).resolve().parents[1] / "results" / "reference")
            if compare_reference else None)
    if compare_reference:
        reference_dir = reference_dir or Path(__file__).resolve().parents[1] / "results" / "reference"
        summary["reference_comparison"] = compare_reports(reports, reference_dir)
        summary["metrics_replication_verified"] = True
        summary["blocking_replication_verified"] = summary.get("blocking", {}).get("reference_comparison", {}).get("passed", False)
        summary["complete_replication_verified"] = (summary["reference_comparison"]["timing_coverage"]["status"] == "complete"
            and len(curves) == len(PRIMARY_CELLS) and summary["blocking_replication_verified"])
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def read_blocking(database):
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as blocking_evidence_db:
        blocking_evidence_db.executescript("""
        CREATE TEMP VIEW blocking_evidence AS
        WITH params AS (
            SELECT trial_id, json_group_object(param_name,
                CASE WHEN json_extract(distribution_json, '$.name') = 'CategoricalDistribution'
                     THEN json_extract(distribution_json,
                          '$.attributes.choices[' || CAST(param_value AS INTEGER) || ']')
                     ELSE param_value END) AS p
            FROM trial_params GROUP BY trial_id
        ), attrs AS (
            SELECT trial_id, json_group_object(key, json(value_json)) AS a
            FROM trial_user_attributes GROUP BY trial_id
        ), named AS (
            SELECT study_id, CASE WHEN study_name GLOB '*_full'
                THEN substr(study_name, 1, length(study_name)-5) ELSE study_name END AS name
            FROM studies
        )
        SELECT substr(s.name, 1, length(s.name)-9) AS dataset,
            json_extract(p, '$.embedding_model') AS embedding_model,
            json_extract(p, '$.queried_side') AS side,
            CAST(json_extract(p, '$.top_k') AS INTEGER) AS k,
            json_extract(a, '$.candset_size') AS candidates,
            json_extract(a, '$.tp') AS gold_in_candidates,
            json_extract(a, '$.n_gold') AS gold_pairs,
            1.0*json_extract(a, '$.tp')/json_extract(a, '$.n_gold') AS PC,
            1.0*json_extract(a, '$.tp')/json_extract(a, '$.candset_size') AS PQ
        FROM named s JOIN trials t USING (study_id)
        JOIN params USING (trial_id) JOIN attrs USING (trial_id)
        WHERE s.name GLOB '*_blocking' AND t.state='COMPLETE';

        CREATE TEMP VIEW blocker_operating_evidence AS
        WITH eligible AS (
            SELECT *, row_number() OVER (PARTITION BY dataset, embedding_model
                ORDER BY candidates, PQ DESC, side, k) AS choice
            FROM blocking_evidence WHERE PC >= .90
        ), models AS (
            SELECT DISTINCT dataset, embedding_model FROM blocking_evidence
        )
        SELECT m.*, e.side, e.k, e.candidates, e.gold_in_candidates, e.gold_pairs, e.PC, e.PQ,
            rank() OVER (PARTITION BY m.dataset ORDER BY e.candidates IS NULL, e.candidates) AS rank
        FROM models m LEFT JOIN eligible e ON m.dataset=e.dataset
            AND m.embedding_model=e.embedding_model AND e.choice=1;

        CREATE TEMP VIEW blocker_ranking_scope AS
        SELECT dataset, count(*) AS models, sum(rank=1) AS tied_first,
            2*sum(rank=1)<=count(*) AS used_for_ranking
        FROM blocker_operating_evidence GROUP BY dataset;
        """)
        blocker_trials = pd.read_sql_query("SELECT * FROM blocking_evidence ORDER BY dataset,embedding_model,side,k", blocking_evidence_db)
        blocker_ranking = pd.read_sql_query("""
            SELECT o.embedding_model,
                round(avg(CASE WHEN s.used_for_ranking THEN o.rank END),4) AS mean_rank,
                max(CASE WHEN s.used_for_ranking THEN o.rank END) AS worst_rank,
                sum(s.used_for_ranking AND o.rank=1) AS wins,
                round(avg(o.PQ),4) AS mean_PQ,
                sum(o.candidates IS NULL) AS misses
            FROM blocker_operating_evidence o JOIN blocker_ranking_scope s USING (dataset)
            GROUP BY o.embedding_model
            ORDER BY mean_rank IS NULL, mean_rank, worst_rank, mean_PQ DESC, o.embedding_model
        """, blocking_evidence_db)
        blocker_scope = pd.read_sql_query("SELECT * FROM blocker_ranking_scope ORDER BY dataset", blocking_evidence_db)
        blocker_primary_points = pd.read_sql_query("""
            SELECT dataset, side, k, candidates, gold_in_candidates, gold_pairs, PC, PQ
            FROM blocker_operating_evidence WHERE embedding_model=? ORDER BY dataset
        """, blocking_evidence_db, params=(MODEL,))
        blocker_ladder_points = pd.read_sql_query("""
            SELECT dataset, side, k, candidates, gold_in_candidates, gold_pairs, PC, PQ
            FROM blocking_evidence WHERE embedding_model=?
        """, blocking_evidence_db, params=(MODEL,))

    ladder_cells = {(dataset, side, k) for dataset, cells in LADDER.items() for side, k in cells}
    blocker_primary_points["in_existing_ladder"] = [
        (row.dataset, row.side, row.k) in ladder_cells for row in blocker_primary_points.itertuples()]
    blocker_ladder_points = blocker_ladder_points.loc[[
        (row.dataset, row.side, row.k) in ladder_cells for row in blocker_ladder_points.itertuples()]]
    blocker_ranking["chosen_blocker"] = blocker_ranking.embedding_model.eq(MODEL)

    return dict(trials=blocker_trials, ranking=blocker_ranking, scope=blocker_scope,
                primary_points=blocker_primary_points, sensitivity_points=blocker_ladder_points)


def report_blocking(database, output_dir, reference_dir=None):
    """Rebuild minimum-cost PC>=.90 operating points and the blocker ranking."""
    frames = read_blocking(database)
    for name, frame in frames.items():
        _write_csv(frame, Path(output_dir) / (name + ".csv.gz" if name == "trials" else name + ".csv"))
    trials = frames["trials"]
    keys = ["dataset", "embedding_model", "side", "k"]
    if trials.duplicated(keys).any():
        raise ValueError("Blocking sweep has duplicate model/direction/k cells")
    complete = (len(trials) == 20000 and trials.dataset.nunique() == 5
                and trials.embedding_model.nunique() == 20 and trials.k.between(1, 100).all()
                and set(trials.side) == {"left", "right"})
    result = dict(trials=len(trials), expected_trials=20000,
                status="complete" if complete else "partial",
                ranking_uses_gold=True, fixed_matching_blocker=MODEL)
    if reference_dir is not None:
        expected = pd.read_csv(Path(reference_dir) / "blocking_trials.csv.gz")
        if len(trials) != len(expected):
            raise AssertionError("Blocking comparison requires all 20,000 trials; available {}".format(len(trials)))
        actual = trials.sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[keys], expected[keys], check_dtype=False)
        columns = ["candidates", "gold_in_candidates", "gold_pairs"]
        np.testing.assert_array_equal(actual[columns], expected[columns])
        np.testing.assert_allclose(actual[["PC", "PQ"]], expected[["PC", "PQ"]], rtol=0, atol=1e-12)
        result["reference_comparison"] = {"passed": True, "counts": "exact", "metric_atol": 1e-12}
    return result


def read_primary_threshold_curves(database, experiment, primary_cells, model):
    """Read one SQLite snapshot of existing selected-fit curves; never refit."""
    curves, availability = {}, []
    score_names = ("llr", "record_cos:" + model, "posterior")
    required = {(score, decoder) for score in score_names for decoder in ("threshold", "greedy")}
    if not Path(database).exists():
        return curves, pd.DataFrame([dict(dataset=d, side=s, k=k, available=False,
            status="unavailable: database missing") for d, s, k in primary_cells])
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True, timeout=120) as con:
        con.execute("BEGIN")
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"studies", "study_user_attributes", "trials", "unsupervised_thresholds"} <= tables:
            return curves, pd.DataFrame([dict(dataset=d, side=s, k=k, available=False,
                status="unavailable: stored threshold curves missing") for d, s, k in primary_cells])
        fits = pd.read_sql_query("""
            WITH attrs AS (
                SELECT study_id, json_group_object(key, json(value_json)) AS a
                FROM study_user_attributes GROUP BY study_id
            ), completed AS (
                SELECT study_id, number, row_number() OVER
                    (PARTITION BY study_id ORDER BY number DESC) AS retry_order
                FROM trials WHERE state='COMPLETE'
            )
            SELECT s.study_name, t.number AS trial_number,
                   json_extract(a, '$.dataset') AS dataset,
                   json_extract(a, '$.side') AS side,
                   json_extract(a, '$.k') AS k
            FROM studies s JOIN attrs USING (study_id) JOIN completed t USING (study_id)
            WHERE json_extract(a, '$.experiment')=?
              AND json_extract(a, '$.panel_name')=?
              AND json_extract(a, '$.phase')='full'
              AND json_extract(a, '$.role')='selected'
              AND coalesce(json_extract(a, '$.excluded'), 0)=0
              AND t.retry_order=1
        """, con, params=(experiment, "single:" + model))
        for dataset, side, k in primary_cells:
            rows = fits.loc[fits.dataset.eq(dataset) & fits.side.eq(side) & fits.k.eq(k)]
            info = dict(dataset=dataset, side=side, k=k, available=False)
            if len(rows) != 1:
                info["status"] = ("unavailable: selected refit pending or failed" if rows.empty
                                  else "unavailable: ambiguous selected studies")
                availability.append(info)
                continue
            fit = rows.iloc[0]
            stored = con.execute("""SELECT score_name, decoder, encoding, curve
                FROM unsupervised_thresholds WHERE study_name=? AND trial_number=?""",
                (fit.study_name, int(fit.trial_number))).fetchall()
            loaded, invalid = {}, []
            for score, decoder, encoding, payload in stored:
                if (score, decoder) not in required:
                    continue
                if encoding != "json+zlib":
                    invalid.append(score + "/" + decoder)
                    continue
                try:
                    curve = pd.DataFrame(json.loads(zlib.decompress(payload)))
                    cut = pd.to_numeric(curve["threshold"], errors="coerce")
                    curve = curve.assign(threshold=cut).loc[np.isfinite(cut)]
                    curve = curve.sort_values("threshold")
                    if curve.empty or not np.isfinite(curve["f1"]).all():
                        raise ValueError("no finite curve")
                except (KeyError, ValueError, TypeError, zlib.error):
                    invalid.append(score + "/" + decoder)
                    continue
                loaded[score, decoder] = curve
            missing = required - set(loaded)
            if missing or invalid:
                info["status"] = "unavailable: missing or invalid curve(s)"
            else:
                info.update(available=True, status="available", study_name=fit.study_name,
                            trial_number=int(fit.trial_number))
                curves[dataset, side, k] = loaded
            availability.append(info)
    return curves, pd.DataFrame(availability)


def plot_primary_threshold_curves(curves, primary_cells, model):
    """Make a standalone figure; absent points remain visibly unavailable."""
    import matplotlib.pyplot as plt
    labels = {"fodors_zagats": "Fodors–Zagats", "dblp_acm": "DBLP–ACM",
              "abt_buy": "Abt–Buy", "amazon_googleproducts": "Amazon–Google",
              "dblp_scholar": "DBLP–Scholar"}
    fig, axes = plt.subplots(len(primary_cells), 3, figsize=(16.5, 11.5), squeeze=False,
                             sharey=True, constrained_layout=True)
    for row, (dataset, side, k) in enumerate(primary_cells):
        data = curves.get((dataset, side, k), {})
        for col, (score, reference, axis_label, title, reference_label) in enumerate(
                (("llr", 0., "LLR threshold (symlog)", "Fitted match/nonmatch evidence", "LLR 0"),
                 ("record_cos:" + model, .7, "Record cosine threshold", "Whole-record cosine", "cosine .7"),
                 ("posterior", .5, "Posterior match probability threshold", "Fitted match probability", "posterior .5"))):
            ax = axes[row, col]
            for decoder, color, style, label in (
                    ("threshold", "#0072B2", "-", "Threshold only"),
                    ("greedy", "#D55E00", "--", "Threshold + greedy 1:1")):
                curve = data.get((score, decoder))
                if curve is not None:
                    # Inclusive cuts: a decision set at t persists until the
                    # previous distinct score when moving down the threshold.
                    cuts, f1 = curve.threshold.to_numpy(), curve.f1.to_numpy()
                    if score == "posterior":
                        # Show the complete probability domain, including
                        # reject-all at 1 only when no stored score equals 1.
                        if cuts[0] > 0:
                            cuts, f1 = np.r_[0., cuts], np.r_[f1[0], f1]
                        if cuts[-1] < 1:
                            cuts, f1 = np.r_[cuts, 1.], np.r_[f1, 0.]
                    ax.step(cuts, f1, where="pre", color=color,
                            linestyle=style, linewidth=1.5, label=label)
            ax.axvline(reference, color="0.4", linestyle=":", linewidth=1,
                       label="Reference: " + reference_label)
            if col == 0:
                ax.set_xscale("symlog", linthresh=1)
                ax.set_ylabel(labels.get(dataset, dataset) + "\n" + side + "/k" + str(k) + ": F1")
                ax.axvline(-5., color="#7B3294", linestyle="-.", linewidth=1,
                           label="Best shared rule: LLR −5")
            if score == "posterior":
                ax.set_xlim(0, 1)
                ax.set_xticks([0, .25, .5, .75, 1])
            if not data:
                ax.text(.5, .5, "Unavailable: selected fit or curve pending",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_xlabel(axis_label)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", color="0.9", linewidth=.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=11)
                ax.legend(loc="lower left", fontsize=8, framealpha=.9)
    model_label = model.split("_", 1)[-1]
    fig.suptitle(model_label + " matching scores at the five primary operating points\n"
                 "Gold-evaluated development comparison; stored fits, no refitting", fontsize=12)
    return fig

"""SQL reads and study summaries extracted from the October 20 results notebook.

These functions read raw Optuna and decision tables, never persisted report
snapshots. Reporting does not update the experiment database.
"""
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from exp.config import LADDER

EXPERIMENT = "unsupervised-refined-v1"
REFERENCE_EXPERIMENT = "unsupervised-baselines-v2"
EXPECTED_PANELS = 27
PRIMARY_CELLS = (("fodors_zagats", "right", 1), ("dblp_acm", "right", 1),
                 ("abt_buy", "left", 2), ("amazon_googleproducts", "left", 4),
                 ("dblp_scholar", "left", 6))
SENSITIVITY_CELLS = {(dataset, side, k) for dataset, rows in LADDER.items() for side, k in rows}
REPORT_CELLS = sorted(SENSITIVITY_CELLS | set(PRIMARY_CELLS))
EXPECTED_SEARCH_STUDIES = len(REPORT_CELLS) * EXPECTED_PANELS
EXPECTED_PRIMARY_SEARCH_STUDIES = len(PRIMARY_CELLS) * EXPECTED_PANELS
EXPECTED_SENSITIVITY_SEARCH_STUDIES = len(SENSITIVITY_CELLS) * EXPECTED_PANELS


def read_sources(database, experiment=EXPERIMENT, reference_experiment=REFERENCE_EXPERIMENT):
    """One read snapshot; optional results/timing tables may still be pending."""
    if not Path(database).exists():
        return {name: pd.DataFrame() for name in
                ("studies", "trials", "decisions", "timing_repeats")}
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True, timeout=120) as con:
        con.execute("BEGIN")
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"studies", "study_user_attributes", "trials", "trial_user_attributes", "trial_values", "trial_params"}
        if not required <= tables:
            return {name: pd.DataFrame() for name in
                    ("studies", "trials", "decisions", "timing_repeats")}
        studies = pd.read_sql_query("""
            WITH attrs AS (
                SELECT study_id, json_group_object(key, json(value_json)) AS a
                FROM study_user_attributes GROUP BY study_id
            )
            SELECT s.study_id, s.study_name,
                json_extract(a, '$.experiment') AS experiment,
                json_extract(a, '$.dataset') AS dataset,
                json_extract(a, '$.side') AS side,
                json_extract(a, '$.k') AS k,
                json_extract(a, '$.phase') AS phase,
                coalesce(json_extract(a, '$.role'), json_extract(a, '$.phase')) AS role,
                json_extract(a, '$.panel_name') AS panel_name,
                json_extract(a, '$.selection') AS selection,
                json_extract(a, '$.objective') AS objective,
                json_extract(a, '$.spec.panel.fusion') AS fusion,
                json_extract(a, '$.spec.panel.models') AS models_json,
                json_extract(a, '$.spec.panel.metrics') AS metrics_json,
                json_extract(a, '$.spec.panel.attributes') AS attributes_json,
                json_extract(a, '$.spec.input_hash') AS input_hash,
                json_extract(a, '$.spec.code_hash') AS code_hash,
                json_extract(a, '$.protocol_hash') AS protocol_hash,
                json_extract(a, '$.refit_hash') AS refit_hash,
                json_extract(a, '$.policy.config_override') AS config_override_json,
                json_extract(a, '$.spec.score_state') AS score_state,
                json_extract(a, '$.refinement_source_study') AS refinement_source_study,
                json_extract(a, '$.refinement_source_experiment') AS refinement_source_experiment,
                json_extract(a, '$.refinement_source_protocol_hash') AS refinement_source_protocol_hash,
                json_extract(a, '$.copied_trials') AS copied_trials,
                CASE WHEN json_extract(a, '$.phase') = 'full' THEN 1
                     ELSE (SELECT count(*) FROM json_each(json_extract(a, '$.spec.configs')))
                END AS expected_trials
            FROM studies s JOIN attrs USING (study_id)
            WHERE (json_extract(a, '$.experiment') = ?
               OR (json_extract(a, '$.experiment') = ? AND json_extract(a, '$.phase') = 'baseline'))
              AND coalesce(json_extract(a, '$.role'), '') <> 'excluded'
              AND coalesce(json_extract(a, '$.selection'), '') <> 'in_model_transitivity'
              AND coalesce(json_extract(a, '$.excluded'), 0) = 0
            ORDER BY dataset, side, k, panel_name, phase, selection
        """, con, params=(experiment, reference_experiment))
        trials = pd.read_sql_query("""
            WITH attrs AS (
                SELECT trial_id, json_group_object(key, json(value_json)) AS a
                FROM trial_user_attributes
                WHERE key IN ('config_name', 'config_hash', 'folds', 'n_errors', 'n_validation',
                    'validation_nll', 'source_study', 'source_trial', 'source_experiment', 'source_protocol_hash',
                    'copied', 'baseline', 'error', 'seconds',
                    'fit_seconds', 'score_seconds', 'preprocessing_seconds', 'loading_seconds',
                    'load_timings', 'n_iter', 'converged', 'init_fraction', 'pi', 'final_parameter_pi',
                    'kappa_capped', 'component_mean_difference', 'n_features', 'n_no_features', 'n_informative', 'candset_size',
                    'tp_blocking', 'n_gold', 'brier', 'label_log_loss', 'scores_path', 'feature_generation')
                GROUP BY trial_id
            ), parameters AS (
                SELECT trial_id, json_group_object(param_name,
                    CASE WHEN json_extract(distribution_json, '$.name') = 'CategoricalDistribution'
                         THEN json_extract(distribution_json,
                              '$.attributes.choices[' || CAST(param_value AS INTEGER) || ']')
                         ELSE param_value END) AS params_json
                FROM trial_params GROUP BY trial_id
            )
            SELECT t.study_id, t.trial_id, t.number AS trial_number, t.state,
                t.datetime_start, t.datetime_complete, v.value AS objective_value,
                p.params_json, json_extract(a, '$.config_name') AS config_name,
                json_extract(a, '$.config_hash') AS config_hash,
                json_extract(a, '$.source_study') AS source_study,
                json_extract(a, '$.source_trial') AS source_trial,
                json_extract(a, '$.source_experiment') AS source_experiment,
                json_extract(a, '$.source_protocol_hash') AS source_protocol_hash,
                coalesce(json_extract(a, '$.copied'), 0) AS copied,
                json_extract(a, '$.validation_nll') AS recorded_validation_nll,
                json_extract(a, '$.baseline') AS baseline,
                json_extract(a, '$.folds') AS folds_json,
                json_extract(a, '$.n_errors') AS n_errors,
                json_extract(a, '$.n_validation') AS n_validation,
                json_extract(a, '$.error') AS error,
                json_extract(a, '$.seconds') AS wall_seconds,
                json_extract(a, '$.fit_seconds') AS fit_seconds,
                json_extract(a, '$.score_seconds') AS score_seconds,
                json_extract(a, '$.preprocessing_seconds') AS preprocessing_seconds,
                coalesce(json_extract(a, '$.loading_seconds'),
                         json_extract(a, '$.load_timings.loading_seconds')) AS loading_seconds,
                json_extract(a, '$.load_timings.similarity_seconds') AS cached_similarity_seconds,
                json_extract(a, '$.load_timings.preprocessing_seconds') AS input_preprocessing_seconds,
                json_extract(a, '$.n_iter') AS n_iter,
                json_extract(a, '$.converged') AS converged,
                json_extract(a, '$.init_fraction') AS init_fraction,
                coalesce(json_extract(a, '$.pi'), json_extract(a, '$.final_parameter_pi')) AS pi,
                json_extract(a, '$.kappa_capped') AS kappa_capped,
                json_extract(a, '$.component_mean_difference') AS component_mean_difference_json,
                json_extract(a, '$.n_features') AS n_features,
                json_extract(a, '$.n_no_features') AS n_no_features,
                json_extract(a, '$.candset_size') AS candset_size,
                json_extract(a, '$.tp_blocking') AS tp_blocking,
                json_extract(a, '$.n_gold') AS n_gold,
                json_extract(a, '$.brier') AS brier,
                json_extract(a, '$.label_log_loss') AS label_log_loss,
                json_extract(a, '$.scores_path') AS scores_path,
                json_extract(a, '$.feature_generation') AS feature_generation
            FROM trials t JOIN study_user_attributes e ON e.study_id=t.study_id
                AND e.key='experiment' AND (json_extract(e.value_json, '$')=?
                    OR (json_extract(e.value_json, '$')=? AND EXISTS (
                        SELECT 1 FROM study_user_attributes ph WHERE ph.study_id=t.study_id
                        AND ph.key='phase' AND json_extract(ph.value_json, '$')='baseline')))
            LEFT JOIN attrs USING (trial_id)
            LEFT JOIN parameters p USING (trial_id)
            LEFT JOIN trial_values v ON v.trial_id=t.trial_id AND v.objective=0
            ORDER BY t.study_id, t.number
        """, con, params=(experiment, reference_experiment)).merge(studies, on="study_id", how="inner", validate="many_to_one")
        trials["source_experiment"] = trials.source_experiment.fillna(trials.experiment)
        trials["cv_validation_nll"] = trials.objective_value.where(trials.phase.eq("search"))
        trials["source_validation_nll"] = trials.recorded_validation_nll.where(trials.phase.eq("full"))
        trials["n_unconverged_folds"] = trials.folds_json.map(
            lambda value: sum(fold.get("converged") is False for fold in json.loads(value or "[]")))
        trials = trials.drop(columns=["objective_value", "recorded_validation_nll"])
        for parameter in ("c_bay", "init_statistic", "missing", "hard", "transitivity"):
            trials[parameter] = trials.params_json.map(
                lambda value: json.loads(value or "{}").get(parameter))
        trials["fit_policy_label"] = np.where(trials.hard.eq(1),
            "hard latent assignments during fitting; no in-fit matching constraint",
            "soft latent assignments during fitting; no in-fit matching constraint")
        trials["claim_scope"] = np.where(trials.role.eq("ablation"), "diagnostic_ablation",
            np.where(trials.role.eq("selected") & trials.hard.eq(0), "selected_soft_em",
                     np.where(trials.phase.eq("baseline"), "baseline_comparator", "configuration_search")))
        mean_differences = trials.component_mean_difference_json.map(
            lambda value: json.loads(value) if isinstance(value, str) else None)
        trials["record_component_mean_difference"] = mean_differences.map(
            lambda value: value.get("record") if value is not None else None)
        trials["negative_mean_block_count"] = mean_differences.map(
            lambda value: sum(difference < 0 for difference in value.values()) if value is not None else None)
        decisions = pd.DataFrame()
        if "unsupervised_decisions" in tables:
            decisions = pd.read_sql_query("""
                SELECT d.* FROM unsupervised_decisions d
                JOIN studies s USING (study_name)
                JOIN study_user_attributes e ON e.study_id=s.study_id
                    AND e.key='experiment' AND (json_extract(e.value_json, '$')=?
                        OR (json_extract(e.value_json, '$')=? AND EXISTS (
                            SELECT 1 FROM study_user_attributes ph WHERE ph.study_id=s.study_id
                            AND ph.key='phase' AND json_extract(ph.value_json, '$')='baseline')))
            """, con, params=(experiment, reference_experiment))
            metadata = trials.drop(columns=["folds_json", "error", "n_gold"])
            decisions = decisions.merge(metadata, on=["study_name", "trial_number"],
                                          how="inner", validate="many_to_one")
            decisions = decisions.loc[decisions.state.eq("COMPLETE")].copy()
            decisions["metrics_use_gold"] = 1
            decisions["blocking_recall"] = decisions.tp_blocking / decisions.n_gold.replace(0, np.nan)
            decisions["candidate_prevalence"] = decisions.tp_blocking / decisions.candset_size.replace(0, np.nan)
        timing = pd.DataFrame()
        if "unsupervised_timings" in tables:
            timing = pd.read_sql_query("""
                SELECT tm.* FROM unsupervised_timings tm
                JOIN studies s USING (study_name)
                JOIN study_user_attributes e ON e.study_id=s.study_id
                    AND e.key='experiment' AND (json_extract(e.value_json, '$')=?
                        OR (json_extract(e.value_json, '$')=? AND EXISTS (
                            SELECT 1 FROM study_user_attributes ph WHERE ph.study_id=s.study_id
                            AND ph.key='phase' AND json_extract(ph.value_json, '$')='baseline')))
            """, con, params=(experiment, reference_experiment))
            timing = timing.merge(studies, on="study_name", how="inner", validate="many_to_one")
            for key in ("source_study", "source_trial", "rule", "uses_gold", "similarity_scope",
                        "conservative_baseline_cost", "shared_costs_across_rules", "cache_state"):
                timing[key] = timing.metadata_json.map(lambda value: json.loads(value).get(key))
        cosine_timing = pd.DataFrame()
        if "unsupervised_cosine_timings" in tables:
            cosine_timing = pd.read_sql_query("SELECT * FROM unsupervised_cosine_timings", con)
            cells = pd.DataFrame(REPORT_CELLS, columns=["dataset", "side", "k"])
            cosine_timing = cosine_timing.merge(cells, on=["dataset", "side", "k"], validate="many_to_one")
            for key in ("rule", "uses_gold", "cost_scope", "cache_state", "shared_costs_across_rules", "protocol_hash"):
                cosine_timing[key] = cosine_timing.metadata_json.map(lambda value: json.loads(value).get(key))
            cosine_timing["experiment_scope"] = "shared candidate-cell timing; source has no experiment field"
        baseline_timing = pd.DataFrame()
        if "unsupervised_baseline_timings" in tables:
            baseline_timing = pd.read_sql_query("SELECT * FROM unsupervised_baseline_timings", con).rename(
                columns={"queried_side": "side", "top_k": "k"})
            for key in ("rule", "uses_gold", "cache_state", "shared_costs_across_rules",
                        "shared_similarity_across_baselines", "protocol_hash", "max_absolute_difference"):
                baseline_timing[key] = baseline_timing.metadata_json.map(lambda value: json.loads(value).get(key))
            baseline_timing["experiment_scope"] = "shared candidate-cell timing; source has no experiment field"
        return dict(studies=studies, trials=trials, decisions=decisions,
                    timing_repeats=timing, cosine_timing_repeats=cosine_timing,
                    baseline_timing_repeats=baseline_timing)

def terminal_outcomes(frame, keys):
    """One outcome per configuration/refit; a successful retry supersedes failure."""
    if frame.empty:
        return pd.DataFrame(columns=keys + ["complete", "terminal", "failed"])
    result = frame.groupby(keys, dropna=False).state.agg(list).reset_index()
    result["complete"] = result.state.map(lambda states: "COMPLETE" in states)
    result["terminal"] = result.complete | result.state.map(
        lambda states: all(state in ("COMPLETE", "FAIL", "PRUNED") for state in states))
    result["failed"] = result.terminal & ~result.complete
    return result.drop(columns="state")


def completion_status(expected_searches, finished_searches, no_valid_searches,
                      completed_refits, terminal_refits, completed_baselines,
                      terminal_baselines, expected_baselines):
    finished = (finished_searches == expected_searches
                and terminal_refits + no_valid_searches == expected_searches
                and terminal_baselines == expected_baselines)
    if not finished:
        return "partial"
    return ("complete" if completed_refits == expected_searches
            and completed_baselines == expected_baselines else "finished_with_failures")


def make_reports(sources, experiment=EXPERIMENT):
    studies, trials, decisions, timing = (sources[key].copy() for key in
                                          ("studies", "trials", "decisions", "timing_repeats"))
    cosine_timing = sources.get("cosine_timing_repeats", pd.DataFrame()).copy()
    baseline_timing = sources.get("baseline_timing_repeats", pd.DataFrame()).copy()
    reports = {}
    if studies.empty:
        reports["coverage"] = pd.DataFrame([dict(experiment=experiment,
            expected_search_studies=EXPECTED_SEARCH_STUDIES,
            expected_primary_search_studies=EXPECTED_PRIMARY_SEARCH_STUDIES,
            expected_sensitivity_search_studies=EXPECTED_SENSITIVITY_SEARCH_STUDIES, search_studies=0,
            studies_with_selection=0, search_studies_without_valid_configuration=0,
            selected_refits=0, selected_refits_terminal=0, selected_refits_failed=0,
            baseline_trials=0, baseline_trials_terminal=0, baseline_trials_failed=0,
            expected_baseline_trials=2 * len(REPORT_CELLS),
            timed_studies=0, status="partial")])
        for name in ("studies", "trials", "selection", "failures", "run_costs", "decisions", "practical",
                     "oracles", "baselines", "dataset_summary", "development_summary", "timing_repeats", "timings", "cosine_timing_repeats", "cosine_timings", "baseline_timing_repeats", "baseline_timings", "calibration"):
            reports[name] = pd.DataFrame(columns=["experiment", "dataset", "study_name"])
        return reports
    counts = trials.groupby(["study_name", "state"]).size().unstack(fill_value=0)
    for state in ("COMPLETE", "FAIL", "RUNNING", "WAITING"):
        studies["n_" + state.lower()] = studies.study_name.map(
            counts[state] if state in counts else pd.Series(dtype=int)).fillna(0).astype(int)
    studies["n_trials"] = studies.study_name.map(trials.groupby("study_name").size()).fillna(0).astype(int)
    configurations = terminal_outcomes(trials, ["study_name", "config_hash"])
    terminal_counts = configurations.loc[configurations.terminal.astype(bool)].groupby("study_name").size()
    studies["n_terminal_configurations"] = studies.study_name.map(terminal_counts).fillna(0).astype(int)
    studies["finished"] = studies.n_terminal_configurations >= studies.expected_trials
    eligible = trials.loc[trials.phase.eq("search") & trials.state.eq("COMPLETE")
                          & np.isfinite(pd.to_numeric(trials.cv_validation_nll, errors="coerce"))]
    # Sorting is scoped to study_id. No NLL ranking across panels or cells.
    choices = eligible.sort_values(["study_id", "cv_validation_nll", "trial_number"]).drop_duplicates("study_id")
    choice_columns = ["study_id", "trial_id", "trial_number", "config_name", "params_json",
                      "cv_validation_nll", "n_validation", "n_unconverged_folds", "c_bay", "init_statistic", "missing",
                      "copied", "source_experiment", "source_study", "source_trial"]
    selection = studies.loc[studies.phase.eq("search")].merge(
        choices[choice_columns], on="study_id", how="left", validate="one_to_one")
    selection["choice_status"] = np.where(selection.trial_id.notna(),
        np.where(selection.finished, "selected", "provisional: search running"),
        np.where(selection.finished, "no eligible configuration", "pending"))
    trials["selected_within_study"] = trials.trial_id.isin(choices.trial_id).astype(int)
    failures = []
    for row in trials.itertuples(index=False):
        metadata = {key: getattr(row, key) for key in ("experiment", "dataset", "side", "k",
            "panel_name", "phase", "role", "selection", "study_name", "trial_number", "config_name", "state")}
        for fold in json.loads(row.folds_json or "[]"):
            if "error" in fold:
                failures.append(dict(metadata, fold=fold.get("fold"), error=fold["error"],
                    n_train=fold.get("n_train"), n_validation=fold.get("n_validation")))
        if pd.notna(row.error):
            failures.append(dict(metadata, fold=None, error=row.error, n_train=None, n_validation=None))
    failure_columns = ["experiment", "dataset", "side", "k", "panel_name", "phase", "role", "selection",
                       "study_name", "trial_number", "config_name", "state", "fold", "error", "n_train", "n_validation"]
    reports.update(studies=studies, trials=trials.drop(columns=["folds_json"]), selection=selection,
                   failures=pd.DataFrame(failures, columns=failure_columns))
    refit_outcomes = terminal_outcomes(trials.loc[trials.phase.eq("full") & trials.role.eq("selected")],
                                       ["study_name"])
    baseline_outcomes = terminal_outcomes(trials.loc[trials.phase.eq("baseline")],
                                          ["study_name", "config_hash"])
    no_valid_searches = int((selection.finished & selection.trial_id.isna()).sum())
    finished_searches = int((studies.phase.eq("search") & studies.finished).sum())
    coverage_status = completion_status(EXPECTED_SEARCH_STUDIES, finished_searches, no_valid_searches,
        int(refit_outcomes.complete.sum()), int(refit_outcomes.terminal.sum()),
        int(baseline_outcomes.complete.sum()), int(baseline_outcomes.terminal.sum()), 2 * len(REPORT_CELLS))
    reports["coverage"] = pd.DataFrame([dict(experiment=experiment,
        expected_search_studies=EXPECTED_SEARCH_STUDIES,
        expected_primary_search_studies=EXPECTED_PRIMARY_SEARCH_STUDIES,
        expected_sensitivity_search_studies=EXPECTED_SENSITIVITY_SEARCH_STUDIES,
        search_studies=int(studies.phase.eq("search").sum()),
        finished_search_studies=finished_searches,
        studies_with_selection=len(choices), search_studies_without_valid_configuration=no_valid_searches,
        selected_refits=int(refit_outcomes.complete.sum()),
        selected_refits_terminal=int(refit_outcomes.terminal.sum()),
        selected_refits_failed=int(refit_outcomes.failed.sum()),
        baseline_trials=int(baseline_outcomes.complete.sum()),
        baseline_trials_terminal=int(baseline_outcomes.terminal.sum()),
        baseline_trials_failed=int(baseline_outcomes.failed.sum()),
        expected_baseline_trials=2 * len(REPORT_CELLS),
        timed_studies=timing.study_name.nunique() if not timing.empty else 0,
        direct_cosine_cells=len(cosine_timing[["dataset", "side", "k"]].drop_duplicates()) if not cosine_timing.empty else 0,
        fresh_baseline_cells=len(baseline_timing[["dataset", "side", "k"]].drop_duplicates()) if not baseline_timing.empty else 0,
        status=coverage_status)])
    # Fold preprocessing is shared across configurations; refit wall time includes reporting/writes.
    cost_rows = []
    cost_columns = ["experiment", "dataset", "side", "k", "panel_name", "phase", "role", "selection",
                    "study_name", "trial_number", "config_name", "state", "copied", "source_experiment", "source_study", "source_trial",
                    "fold", "timing_kind",
                    "loading_seconds", "similarity_seconds", "preprocessing_seconds", "fit_seconds",
                    "score_seconds", "wall_seconds"]
    for row in trials.itertuples(index=False):
        base = {key: getattr(row, key) for key in cost_columns[:16]}
        if row.phase == "search":
            for fold in json.loads(row.folds_json or "[]"):
                cost_rows.append(dict(base, fold=fold.get("fold"), timing_kind="search_fold",
                    **{key: fold.get(key) for key in ("preprocessing_seconds", "fit_seconds", "score_seconds")}))
        else:
            cost_rows.append(dict(base, fold=None, timing_kind="cached_refit_or_baseline",
                loading_seconds=row.loading_seconds, similarity_seconds=row.cached_similarity_seconds,
                preprocessing_seconds=row.preprocessing_seconds, fit_seconds=row.fit_seconds,
                score_seconds=row.score_seconds, wall_seconds=row.wall_seconds))
    reports["run_costs"] = pd.DataFrame(cost_rows, columns=cost_columns)
    reports["run_costs"]["preprocessing_shared_across_configurations"] = reports["run_costs"].phase.eq("search").astype(int)
    calibration_columns = ["experiment", "dataset", "side", "k", "panel_name", "role", "selection",
        "study_name", "trial_number", "source_study", "source_trial", "source_validation_nll",
        "converged", "n_iter", "pi", "init_fraction", "kappa_capped", "n_no_features",
        "candset_size", "brier", "label_log_loss", "fit_policy_label", "claim_scope",
        "component_mean_difference_json", "record_component_mean_difference", "negative_mean_block_count"]
    reports["calibration"] = trials.loc[trials.phase.eq("full") & trials.state.eq("COMPLETE"),
                                        calibration_columns].copy()
    reports["calibration"]["pi_times_k"] = reports["calibration"].pi * reports["calibration"].k
    reports["calibration"]["metrics_use_gold"] = 1
    reports["calibration"]["population"] = "candidate pairs"
    if not decisions.empty:
        fixed = decisions.loc[decisions.uses_gold.eq(0)].copy()
        cut = pd.to_numeric(fixed.threshold_json, errors="coerce")
        primary = ((fixed.score_name.eq("llr") & cut.eq(0))
                   | (fixed.score_name.isin(["posterior", "legacy_posterior"]) & cut.eq(.5)))
        practical = fixed.loc[(fixed.rule.eq("fixed") & primary)
                             | (fixed.decoder.eq("greedy") & fixed.rule.eq("pure_transitivity"))].copy()
        practical["decision_label"] = np.where(practical.rule.eq("pure_transitivity"),
            practical.score_name + ":greedy:no-cut",
            practical.score_name + ":" + practical.decoder + ":" + practical.threshold_json)
        oracle = decisions.loc[decisions.uses_gold.eq(1)].copy()
        join_keys = ["study_name", "trial_number", "decoder"]
        reference_specs = [("llr0", "llr", 0.), ("posterior05", "posterior", .5),
                           ("legacy_posterior05", "legacy_posterior", .5)]
        for label, score, threshold in reference_specs:
            reference = fixed.loc[fixed.rule.eq("fixed") & fixed.score_name.eq(score) & cut.eq(threshold),
                                  join_keys + ["f1"]].drop_duplicates(join_keys)
            oracle = oracle.merge(reference.rename(columns={"f1": label + "_f1"}),
                                  on=join_keys, how="left", validate="many_to_one")
            oracle["gain_vs_" + label] = oracle.f1 - oracle[label + "_f1"]
        for score in ("llr", "posterior"):
            no_cut_reference = fixed.loc[fixed.rule.eq("pure_transitivity") & fixed.score_name.eq(score),
                ["study_name", "trial_number", "f1"]].drop_duplicates(["study_name", "trial_number"])
            oracle = oracle.merge(no_cut_reference.rename(columns={"f1": score + "_nocut_f1"}),
                on=["study_name", "trial_number"], how="left", validate="many_to_one")
            oracle["gain_vs_" + score + "_nocut"] = oracle.f1 - oracle[score + "_nocut_f1"]
        same_keys = ["study_name", "trial_number", "score_name"]
        no_cut = fixed.loc[fixed.rule.eq("pure_transitivity"), same_keys + ["f1"]].drop_duplicates(same_keys)
        oracle = oracle.merge(no_cut.rename(columns={"f1": "same_score_nocut_f1"}),
                              on=same_keys, how="left", validate="many_to_one")
        oracle["gain_vs_same_score_nocut"] = oracle.f1 - oracle.same_score_nocut_f1
        oracle["comparison_note"] = "gold-selected oracle diagnostic; no fit or panel selection"
        reports.update(decisions=decisions, practical=practical, oracles=oracle,
                       baselines=decisions.loc[decisions.phase.eq("baseline")].copy())
        # Keep each method, panel and ablation separate. Equal dataset weights,
        # after averaging the declared cells within each dataset.
        practical["method"] = np.where(practical.phase.eq("baseline"), practical.baseline,
                                        practical.selection.fillna("cv_nll"))
        keys = ["panel_name", "role", "method", "decision_label"]
        sensitivity = practical.merge(pd.DataFrame(sorted(SENSITIVITY_CELLS),
            columns=["dataset", "side", "k"]), on=["dataset", "side", "k"], how="inner", validate="many_to_one")
        dataset = sensitivity.groupby(keys + ["dataset"], dropna=False).agg(
            mean_f1=("f1", "mean"), mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"), cells=("study_name", "nunique")).reset_index()
        dataset["expected_cells"] = dataset.dataset.map({name: len(rows) for name, rows in LADDER.items()})
        dataset["complete_dataset"] = dataset.cells.eq(dataset.expected_cells)
        development = dataset.groupby(keys, dropna=False).agg(
            dataset_equal_f1=("mean_f1", "mean"), dataset_equal_precision=("mean_precision", "mean"),
            dataset_equal_recall=("mean_recall", "mean"), datasets=("dataset", "nunique"),
            cells=("cells", "sum"), complete_datasets=("complete_dataset", "sum")).reset_index()
        development["expected_datasets"] = len(LADDER)
        development["complete_coverage"] = development.complete_datasets.eq(len(LADDER))
        development["interpretation"] = "19-cell sensitivity ranking using gold; not an unsupervised method selection"
        reports.update(dataset_summary=dataset, development_summary=development)
    reports["timing_repeats"] = timing
    if not timing.empty:
        timing_keys = ["experiment", "dataset", "side", "k", "panel_name", "role", "selection",
                       "study_name", "score_name", "decoder", "threshold_json", "rule", "similarity_scope",
                       "conservative_baseline_cost", "shared_costs_across_rules", "cache_state"]
        stages = ["loading_seconds", "similarity_seconds", "preprocessing_seconds", "fit_seconds",
                  "score_seconds", "decode_seconds", "matching_seconds", "total_seconds"]
        aggregates = {"repeats": ("repeat", "nunique")}
        aggregates.update({stage + "_median": (stage, "median") for stage in stages})
        aggregates.update(matching_seconds_min=("matching_seconds", "min"),
                          matching_seconds_max=("matching_seconds", "max"))
        reports["timings"] = timing.groupby(timing_keys, dropna=False).agg(**aggregates).reset_index()
    reports["cosine_timing_repeats"] = cosine_timing
    if not cosine_timing.empty:
        keys = ["dataset", "side", "k", "score_name", "models_json", "decoder", "threshold_json",
                "rule", "cost_scope", "cache_state", "shared_costs_across_rules", "experiment_scope"]
        stages = ["loading_seconds", "similarity_seconds", "preprocessing_seconds", "aggregation_seconds",
                  "decode_seconds", "matching_seconds", "total_seconds"]
        aggregates = {"repeats": ("repeat", "nunique")}
        aggregates.update({stage + "_median": (stage, "median") for stage in stages})
        aggregates.update(matching_seconds_min=("matching_seconds", "min"),
                          matching_seconds_max=("matching_seconds", "max"))
        reports["cosine_timings"] = cosine_timing.groupby(keys, dropna=False).agg(**aggregates).reset_index()
    reports["baseline_timing_repeats"] = baseline_timing
    if not baseline_timing.empty:
        keys = ["dataset", "side", "k", "baseline", "score_name", "decoder", "threshold_json", "rule",
                "cache_state", "shared_costs_across_rules", "shared_similarity_across_baselines", "experiment_scope"]
        stages = ["loading_seconds", "similarity_seconds", "preprocessing_seconds", "fit_seconds",
                  "decode_seconds", "matching_seconds", "total_seconds"]
        aggregates = {"repeats": ("repeat", "nunique")}
        aggregates.update({stage + "_median": (stage, "median") for stage in stages})
        aggregates.update(matching_seconds_min=("matching_seconds", "min"),
                          matching_seconds_max=("matching_seconds", "max"))
        reports["baseline_timings"] = baseline_timing.groupby(keys, dropna=False).agg(**aggregates).reset_index()
    # Even pending sections have a queryable table after RUN_REPORT.
    for name in ("decisions", "practical", "oracles", "baselines", "dataset_summary",
                 "development_summary", "timing_repeats", "timings", "cosine_timing_repeats", "cosine_timings", "baseline_timing_repeats", "baseline_timings", "calibration"):
        if name not in reports or not len(reports[name].columns):
            reports[name] = pd.DataFrame(columns=["experiment", "dataset", "study_name"])
    for frame in reports.values():
        if "experiment" not in frame:
            frame.insert(0, "experiment", experiment)
    return reports


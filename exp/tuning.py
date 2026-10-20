"""Run the visible tuning lifecycle for an editable experiment design.

search: train each configuration on record-disjoint folds and measure NLL.
refine: reuse the initial trials and fit only the additional configurations.
refit: choose minimum NLL within each panel, fit all candidates, save predictor.
report: export configurations, fold objectives, failures and fitted artifacts.

Gold evaluation is an explicit refit option. Neither selection nor prediction
needs match labels. Publication reproduction remains in exp.protocol/main_zeroer.
"""
import contextlib
import json
from pathlib import Path
import re

import optuna
from optuna.trial import TrialState
from joblib import Parallel, delayed

from exp.refinement import run_refinement
from exp.runner import run_search, run_refit, select_best_trial, _storage, _hash
from utils.tuning_reports import export_tuning_results


STAGES = ("search", "refine", "refit", "report", "all")


def _existing_search(storage, experiment, cell, panel, configurations, split):
    """Find the exact preceding stage; never start training implicitly."""
    matches = []
    for summary in optuna.get_all_study_summaries(storage=_storage(storage)):
        attrs = summary.user_attrs
        if (attrs.get("phase") == "search" and attrs.get("experiment") == experiment
                and attrs.get("panel_name") == panel["name"]
                and (attrs.get("dataset"), attrs.get("side"), attrs.get("k")) == tuple(cell)):
            matches.append(summary.study_name)
    if len(matches) != 1:
        raise ValueError("Expected one {} search for {} / {}; found {}. Run the preceding stage first."
                         .format(experiment, cell, panel["name"], len(matches)))
    study = optuna.load_study(study_name=matches[0], storage=_storage(storage))
    spec = study.user_attrs["spec"]
    expected_panel = dict(panel, models=sorted(panel["models"]))
    if (spec["configs"] != configurations or spec["split_spec"] != split
            or any(spec["panel"].get(key) != value for key, value in expected_panel.items())):
        raise ValueError("The stored search uses a different design; choose a new experiment/output directory")
    return study


def _require_complete_search(study):
    """Refit only after every declared configuration has one terminal result."""
    configurations = study.user_attrs.get("spec", {}).get("configs", {})
    expected = {_hash(config) for config in configurations.values()}
    trials = study.trials
    actual = [_hash(trial.params) for trial in trials]
    if (not configurations or len(expected) != len(configurations)
            or len(trials) != len(configurations) or len(set(actual)) != len(trials)
            or set(actual) != expected
            or any(trial.state not in (TrialState.COMPLETE, TrialState.FAIL)
                   or trial.user_attrs.get("config_hash") != fingerprint
                   for trial, fingerprint in zip(trials, actual))):
        raise ValueError("Refit requires one COMPLETE or FAIL trial for every declared configuration, "
                         "with matching parameters and no duplicates. Complete the search/refine stage first.")


def _run_cell(plan, cell, panel, stage, storage, directory, evaluate):
    dataset, side, k = cell
    label = "{}__{}_k{}__{}".format(dataset, side, k, panel["name"])
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)
    log_path = directory / "logs" / (label + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(dataset=dataset, side=side, k=k, panel=panel["name"],
                  stage=stage, studies=[], refits=[], log=str(log_path))
    with log_path.open("a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        if stage in ("search", "all"):
            initial = run_search(dataset, side, k, panel, plan.configurations,
                                 plan.split, storage, experiment=plan.experiment)
        else:
            initial = _existing_search(storage, plan.experiment, cell, panel,
                                       plan.configurations, plan.split)
        result["studies"].append(initial.study_name)
        selected_source = initial
        if plan.refinements and stage != "search":
            if stage in ("refine", "all"):
                selected_source = run_refinement(initial, storage, plan.refinements,
                                                  experiment=plan.final_experiment)
            elif stage == "refit":
                selected_source = _existing_search(storage, plan.final_experiment, cell, panel,
                    dict(plan.configurations, **plan.refinements), plan.split)
            else:
                # A report after only the initial search is useful too.
                summaries = optuna.get_all_study_summaries(storage=_storage(storage))
                if any(s.user_attrs.get("refinement_source_study") == initial.study_name for s in summaries):
                    selected_source = _existing_search(storage, plan.final_experiment, cell, panel,
                        dict(plan.configurations, **plan.refinements), plan.split)
            if selected_source.study_name != initial.study_name:
                result["studies"].append(selected_source.study_name)
        if stage in ("refit", "all"):
            _require_complete_search(selected_source)
            try:
                selected = select_best_trial(selected_source)
            except ValueError:
                result["status"] = "no_finite_configuration"
            else:
                result["selected_configuration"] = dict(selected.params)
                result["validation_nll"] = selected.value
                for selection, overrides in [("cv_nll", None), *plan.ablations.items()]:
                    full = run_refit(selected_source, storage, plan.thresholds,
                        directory / "predictors", config_override=overrides, selection=selection,
                        evaluate=evaluate, save_predictor=True)
                    trial = full.trials[-1]
                    result["refits"].append(dict(study=full.study_name, selection=selection,
                        state=trial.state.name, predictor=trial.user_attrs.get("predictor_path"),
                        error=trial.user_attrs.get("error")))
                result["status"] = ("complete" if all(row["state"] == "COMPLETE" for row in result["refits"])
                                    else "finished_with_failed_refits")
        else:
            result["status"] = "stage_complete"
    print("{} {}: {}".format(label, stage, result["status"]), flush=True)
    return result


def run_tuning(plan, storage, output_dir, *, stage="all", workers=1, evaluate=False):
    """Execute one explicit stage or the whole chain; resume verified finished work.

    Change the experiment name or output directory when changing the design.
    Set ``evaluate`` on the first refit if gold metrics/oracles are wanted;
    changing that policy later requires a separate run, just like new thresholds.
    """
    if stage not in STAGES or type(workers) is not int or workers < 1:
        raise ValueError("Choose a supported stage and a positive integer worker count")
    if stage == "refine" and not plan.refinements:
        raise ValueError("The design has no refinement grid")
    if evaluate and stage not in ("refit", "all"):
        raise ValueError("--evaluate applies to refit/all, after label-free selection")
    directory = Path(output_dir) / "tuning" / plan.experiment
    directory.mkdir(parents=True, exist_ok=True)
    design_path = directory / "design.json"
    design = json.dumps(plan.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if design_path.exists() and design_path.read_text() != design:
        raise ValueError("This experiment already has a different design; choose a new name/output directory")
    design_path.write_text(design)
    _storage(storage)
    rows = Parallel(n_jobs=workers)(delayed(_run_cell)(
        plan, cell, panel, stage, storage, directory, evaluate)
        for cell in plan.cells for panel in plan.panels)
    names = list(dict.fromkeys(name for row in rows for name in row["studies"]))
    summary = export_tuning_results(storage, names, directory / "reports")
    result = dict(experiment=plan.experiment, stage=stage, evaluate=evaluate,
                  design=str(design_path), results=rows, reports=summary)
    (directory / (stage + "_summary.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result

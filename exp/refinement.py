"""Extend a completed six-configuration grid without refitting its existing trials."""
from hashlib import sha256
from pathlib import Path
import re

import numpy as np
import optuna
from optuna.trial import TrialState

import exp.runner as core


def run_refinement(source_study, storage, configs, experiment="unsupervised-refined-v1"):
    """Copy verified original fits, then evaluate only additional configurations.

    The source's panel, inputs, folds and frozen fitting code must still match.
    Trial values remain within-panel selection-CV NLL; no gold is opened here.
    Copied timing attributes describe the original work, not new compute.
    """
    core._database(storage)
    source = (optuna.load_study(study_name=source_study, storage=core._storage(storage))
              if isinstance(source_study, str) else source_study)
    attrs, trials = source.user_attrs, source.get_trials(deepcopy=True)
    if attrs.get("phase") != "search" or attrs.get("objective") != "validation_nll":
        raise ValueError("Refinement requires a label-free search study")
    if source.direction != optuna.study.StudyDirection.MINIMIZE:
        raise ValueError("Source study must minimise validation NLL")
    spec = attrs["spec"]
    original = core._configurations(spec["configs"])
    if len(original) != 6 or len(trials) != 6 or any(
            trial.state not in (TrialState.COMPLETE, TrialState.FAIL) for trial in trials):
        raise ValueError("Source must contain all six finished configurations and no running trials")
    expected = {core._hash(config) for config in original.values()}
    actual = {core._hash(trial.params) for trial in trials}
    if len(expected) != 6 or actual != expected or any(
            trial.user_attrs.get("config_hash") != core._hash(trial.params) for trial in trials):
        raise ValueError("Source trials do not match its declared configurations")
    if any(trial.state == TrialState.COMPLETE and
           (trial.value is None or not np.isfinite(trial.value)) for trial in trials):
        raise ValueError("Source contains a completed trial without finite validation NLL")
    if attrs.get("protocol_hash") != core._hash(spec) or spec["code_hash"] != core._code_hash():
        raise ValueError("Source protocol or frozen fitting code changed")
    if experiment == spec["experiment"]:
        raise ValueError("Refinement needs a new experiment namespace")
    extra = core._configurations(configs)
    if original.keys() & extra.keys():
        raise ValueError("Additional configurations need distinct names")
    extra_hashes = {core._hash(config) for config in extra.values()}
    if len(extra_hashes) != len(extra) or expected & extra_hashes:
        raise ValueError("Additional configurations must be new and unique")
    merged = core._configurations(dict(original, **extra))
    distributions = core._distributions(merged)  # also checks identical parameter keys

    panel = spec["panel"]
    cell = core.load_cell(spec["dataset"], spec["side"], spec["k"], panel["models"],
                          metrics=panel["metrics"], include_record=panel.get("include_record", True))
    if cell["input_hash"] != spec["input_hash"]:
        raise ValueError("Source candidate inputs changed")
    splits = [split["metadata"] for split in core.record_disjoint_splits(cell["ids"], **spec["split_spec"])]
    if core._json(splits) != core._json(attrs.get("splits")):
        raise ValueError("Source record-disjoint split metadata changed")
    new_spec = dict(spec, configs=merged, experiment=experiment)
    protocol_hash = core._hash(new_spec)
    panel_name = panel.get("name", "panel")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", panel_name)
    name = "{}_{}_{}_k{}_{}_{}".format(spec["dataset"], experiment, spec["side"],
                                      spec["k"], label, core._hash(panel)[:8])
    target = optuna.create_study(study_name=name, storage=core._storage(storage),
                                 direction="minimize", load_if_exists=True)
    refinement = dict(source_study=source.study_name, source_experiment=spec["experiment"],
                      source_protocol_hash=attrs["protocol_hash"],
                      code_hash=sha256(Path(__file__).read_bytes()).hexdigest())
    old_hash = target.user_attrs.get("protocol_hash")
    if old_hash is not None and (old_hash != protocol_hash or target.user_attrs.get("refinement") != refinement):
        raise ValueError("Existing refinement protocol or provenance changed")
    if old_hash is None:
        if target.trials:
            raise ValueError("Existing refinement has trials without protocol metadata")
        metadata = dict(spec=new_spec, protocol_hash=protocol_hash, phase="search",
                        dataset=spec["dataset"], side=spec["side"], k=spec["k"],
                        panel_name=panel_name, experiment=experiment, objective="validation_nll",
                        splits=splits, load_timings=cell.get("timings", {}), refinement=refinement,
                        refinement_source_study=source.study_name,
                        refinement_source_experiment=spec["experiment"],
                        refinement_source_protocol_hash=attrs["protocol_hash"], copied_trials=6)
        for key, value in metadata.items():
            target.set_user_attr(key, value)
    finished = {trial.user_attrs.get("config_hash") for trial in target.trials}
    for trial in trials:
        if trial.user_attrs["config_hash"] in finished:
            continue
        copied = dict(trial.user_attrs, copied=True, source_study=source.study_name,
                      source_trial=trial.number, source_experiment=spec["experiment"],
                      source_protocol_hash=attrs["protocol_hash"])
        target.add_trial(optuna.trial.create_trial(params=trial.params, distributions=distributions,
                         value=trial.value, state=trial.state, user_attrs=copied))
    return core.run_search(spec["dataset"], spec["side"], spec["k"], panel, merged,
                           spec["split_spec"], storage, experiment=experiment)

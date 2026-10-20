"""Editable experiment design for unsupervised, within-panel grid search.

``configs/unsupervised.json`` separates candidate cells, feature panels, fitting
defaults, search/refinement grids, record-disjoint folds, and decision rules.
Grid values override fitting defaults. A different panel or missingness policy
defines a different likelihood; their NLLs must not be ranked together.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
from pathlib import Path
import re

from data.benchmarks import DATASETS
from exp import protocol


FIT_DEFAULTS = dict(c_bay=.01, init_statistic="sum", init_threshold=.5,
                    missing="ignore", hard=False, transitivity=False,
                    pi_M=None, tol=1e-5, max_iter=200)


def validate_fit(config):
    """Validate a complete learning policy before any expensive input loading."""
    config = dict(config)
    if set(config) != set(FIT_DEFAULTS):
        raise ValueError("Fitting settings must be: " + ", ".join(FIT_DEFAULTS))
    for key in ("c_bay", "init_threshold", "pi_M", "tol"):
        value = config[key]
        if value is None and key in ("pi_M", "tol"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + " must be a finite number")
        # Optuna treats numerically equal categorical choices as one value.
        # Canonical floats keep duplicate detection and persisted hashes aligned.
        config[key] = 0.0 if value == 0 else float(value)
    if config["c_bay"] < 0 or not 0 <= config["init_threshold"] <= 1:
        raise ValueError("c_bay must be nonnegative; init_threshold must lie in [0, 1]")
    if config["tol"] is not None and config["tol"] < 0:
        raise ValueError("tol must be nonnegative or null")
    if config["pi_M"] is not None and not 0 < config["pi_M"] < 1:
        raise ValueError("pi_M must lie strictly between 0 and 1, or be null")
    if type(config["max_iter"]) is not int or config["max_iter"] < 1:
        raise ValueError("max_iter must be a positive integer")
    if type(config["hard"]) is not bool or config["transitivity"] is not False:
        raise ValueError("hard must be boolean; transitivity must be false (decode after fitting)")
    if config["missing"] not in ("ignore", "zero"):
        raise ValueError("missing must be ignore or zero")
    if config["init_statistic"] not in ("sum", "present_mean"):
        raise ValueError("init_statistic must be sum or present_mean")
    return dict(config)


def expand_grid(defaults, grid, prefix):
    """Cartesian product in JSON field/value order, with stable trial labels."""
    if not isinstance(grid, dict) or set(grid) - set(FIT_DEFAULTS):
        raise ValueError("A search grid must map fitting parameter names to value lists")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError("Grid values must be nonempty lists: " + key)
    configurations = {}
    seen = set()
    for index, values in enumerate(product(*grid.values())):
        config = validate_fit(dict(defaults, **dict(zip(grid, values))))
        identity = json.dumps(config, sort_keys=True)
        if identity in seen:
            raise ValueError("The grid contains duplicate configurations")
        seen.add(identity)
        configurations["{}-{:03d}".format(prefix, index)] = config
    return configurations


def _cells(value):
    if isinstance(value, str):
        return protocol.select_cells(value)
    if not isinstance(value, list) or not value:
        raise ValueError("cells must be a nonempty list, or primary/sensitivity/all")
    cells = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"dataset", "side", "k"}:
            raise ValueError("Each cell needs dataset, side, and k")
        dataset, side, k = item["dataset"], item["side"], item["k"]
        if (not isinstance(dataset, str) or dataset not in DATASETS
                or side not in ("left", "right") or type(k) is not int or k < 1):
            raise ValueError("A cell needs a known dataset, left/right side, and positive integer k")
        cells.append((dataset, side, k))
    if len(set(cells)) != len(cells):
        raise ValueError("Candidate cells must be distinct")
    return cells


def _panels(value, cells):
    if value == "all":
        return deepcopy(protocol.PANELS)
    if not isinstance(value, list) or not value:
        raise ValueError("panels must be a nonempty list of names/objects, or all")
    panels = []
    allowed = {"name", "models", "fusion", "metrics", "attributes", "include_record"}
    for item in value:
        panel = protocol.select_panels([item])[0] if isinstance(item, str) else deepcopy(item)
        if not isinstance(panel, dict) or set(panel) - allowed:
            raise ValueError("Unknown panel settings; allowed: " + ", ".join(sorted(allowed)))
        if not isinstance(panel.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:+-]*", panel["name"]):
            raise ValueError("Each panel needs a simple, nonempty name")
        models = panel.get("models")
        if (not isinstance(models, list) or not models or not all(isinstance(model, str) for model in models)
                or len(set(models)) != len(models)
                or set(models) - set(protocol.MODELS)):
            raise ValueError("Panel models must be distinct names from the supplied encoder pool")
        panel.setdefault("fusion", "separate")
        panel.setdefault("metrics", ["cos"])
        panel.setdefault("include_record", True)
        if panel["fusion"] not in ("separate", "mean", "zmean", "family"):
            raise ValueError("Unsupported feature fusion")
        metrics = panel["metrics"]
        if (not isinstance(metrics, list) or not metrics or not all(isinstance(metric, str) for metric in metrics)
                or len(set(metrics)) != len(metrics)
                or set(metrics) - {"cos", "linf"}):
            raise ValueError("Panel metrics must select cos and/or linf without duplicates")
        if type(panel["include_record"]) is not bool:
            raise ValueError("include_record must be boolean")
        if "attributes" in panel:
            attributes = panel["attributes"]
            if (not isinstance(attributes, list) or not attributes or not all(isinstance(a, str) for a in attributes)
                    or len(set(attributes)) != len(attributes)):
                raise ValueError("Panel attributes must be a nonempty list of distinct names")
            for dataset, _, _ in cells:
                if set(attributes) - set(DATASETS[dataset]["attributes"]):
                    raise ValueError("Panel attributes are not available in " + dataset)
        panels.append(panel)
    if len({panel["name"] for panel in panels}) != len(panels):
        raise ValueError("Panel names must be distinct")
    return panels


@dataclass
class TuningPlan:
    """Validated assembly choices; model computation and runtime paths live elsewhere."""
    experiment: str
    cells: list
    panels: list
    split: dict
    configurations: dict
    refinements: dict
    thresholds: dict
    ablations: dict

    @property
    def models(self):
        selected = {model for panel in self.panels for model in panel["models"]}
        return [model for model in protocol.MODELS if model in selected]

    @property
    def final_experiment(self):
        return self.experiment + "-refined" if self.refinements else self.experiment

    def to_dict(self):
        return asdict(self)


def load_tuning_plan(path):
    """Read an editable JSON design, rejecting misspellings and incomparable grids."""
    with Path(path).open() as stream:
        source = json.load(stream)
    allowed = {"experiment", "cells", "panels", "split", "fit", "search", "refinement", "thresholds", "ablations"}
    if not isinstance(source, dict) or set(source) - allowed:
        raise ValueError("Unknown tuning fields; allowed: " + ", ".join(sorted(allowed)))
    required = {"experiment", "cells", "panels", "search"}
    if required - set(source):
        raise ValueError("Missing tuning fields: " + ", ".join(sorted(required - set(source))))
    for name in ("fit", "split", "refinement"):
        if name in source and not isinstance(source[name], dict):
            raise ValueError(name + " must be a JSON object")
    experiment = source["experiment"]
    if not isinstance(experiment, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", experiment):
        raise ValueError("experiment must be a simple name of at most 64 characters")
    reserved = {protocol.EXPERIMENT, protocol.FINAL_EXPERIMENT, protocol.BASELINE_EXPERIMENT}
    if experiment in reserved or experiment + "-refined" in reserved:
        raise ValueError("Choose a custom experiment name outside the publication namespaces")
    cells = _cells(source["cells"])
    panels = _panels(source["panels"], cells)
    split = dict(protocol.SPLIT, **source.get("split", {}))
    if (set(split) != {"n_splits", "seed"} or type(split["n_splits"]) is not int
            or split["n_splits"] < 2 or type(split["seed"]) is not int):
        raise ValueError("split requires integer n_splits >= 2 and integer seed")
    defaults = validate_fit(dict(FIT_DEFAULTS, **source.get("fit", {})))
    configurations = expand_grid(defaults, source["search"], "initial")
    refinements = expand_grid(defaults, source["refinement"], "extra") if source.get("refinement") else {}
    all_configs = list(configurations.values()) + list(refinements.values())
    if len({json.dumps(config, sort_keys=True) for config in all_configs}) != len(all_configs):
        raise ValueError("Refinement configurations must be new, not repeats of the initial grid")
    if len({config["missing"] for config in all_configs}) != 1:
        raise ValueError("Keep one missingness policy per NLL search; use ablations for the comparison")
    thresholds = deepcopy(source.get("thresholds", protocol.THRESHOLDS))
    if not isinstance(thresholds, dict) or set(thresholds) != {"llr", "posterior", "cosine"}:
        raise ValueError("thresholds must specify llr, posterior and cosine lists")
    for name, cuts in thresholds.items():
        if not isinstance(cuts, list) or not cuts:
            raise ValueError("Threshold grids must be nonempty lists")
        if any(isinstance(cut, bool) or not isinstance(cut, (int, float)) or not math.isfinite(cut) for cut in cuts):
            raise ValueError("Thresholds must be finite numbers")
        if len(set(cuts)) != len(cuts) or (name == "posterior" and any(not 0 <= cut <= 1 for cut in cuts)):
            raise ValueError("Thresholds must be distinct, and posteriors must lie in [0, 1]")
    ablations = source.get("ablations", {})
    if not isinstance(ablations, dict):
        raise ValueError("ablations must map diagnostic names to fitting overrides")
    for name, overrides in ablations.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name == "cv_nll":
            raise ValueError("Ablation names must be distinct simple names other than cv_nll")
        if not isinstance(overrides, dict) or not overrides:
            raise ValueError("Each ablation needs fitting overrides")
        for config in all_configs:
            validate_fit(dict(config, **overrides))
    return TuningPlan(experiment, cells, panels, split, configurations,
                      refinements, thresholds, ablations)

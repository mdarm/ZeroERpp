"""Label-free candidate inputs and train-fitted embedding feature panels.

Only IDs, cached similarities, presence masks and raw vectors are read here.
Candidate rows retain their stored order. A fitted ``PanelTransform`` owns every
population-dependent statistic and keeps the same output columns at prediction.
"""
from __future__ import annotations

import hashlib
import json
import time
from os.path import join

import numpy as np
import pandas as pd

from exp.config import ATTR_DIR, BENCH_DIR, RECORD_DIR
from data.benchmarks import DATASETS, FAMILIES
from data.features.bank import load_bank, load_presence
from data.features.embeddings import FUSIONS, METRICS


def _timings():
    return dict(loading_seconds=0.0, similarity_seconds=0.0,
                preprocessing_seconds=0.0)


def _candidate_input(dataset, side, k):
    metadata = load_presence(dataset, side, k)
    ids = metadata[["ltable_id", "rtable_id"]].astype(str).reset_index(drop=True)
    positions = []
    limit = 0
    for table, column in (("left", "ltable_id"), ("right", "rtable_id")):
        full_ids = pd.read_csv(join(BENCH_DIR, dataset, table + ".csv"),
                               usecols=["id"], dtype=str)["id"]
        if not full_ids.is_unique:
            raise ValueError("{} {} IDs are not unique".format(dataset, table))
        index = pd.Series(np.arange(len(full_ids)), index=full_ids)
        rows = index.loc[ids[column]].to_numpy(dtype=np.int64)
        positions.append(rows if table == "left" else rows + limit)
        if table == "left":
            limit = len(full_ids)
    return ids, np.column_stack(positions), metadata, limit


def _raw_scores(dataset, model, pairs, limit, attribute=None, metrics=("cos",)):
    """Read only bounded batches of mmap vectors; time reads and math separately.

    Attribute scores reproduce the stored float32 bank. Record cosine divides
    by float64-accumulated norms, also for the legacy float16 vector files.
    """
    timing = _timings()
    started = time.perf_counter()
    base = RECORD_DIR if attribute is None else ATTR_DIR
    stem = model if attribute is None else "{}_{}".format(model, attribute)
    left, right = [np.load(join(base, dataset, stem + "_" + side + ".npy"),
                           mmap_mode="r") for side in ("left", "right")]
    timing["loading_seconds"] += time.perf_counter() - started
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("incompatible vector shapes for " + stem)
    # Two gathered arrays plus intermediate arrays stay within roughly 32 MiB.
    chunk_size = max(1, min(4096, (32 * 1024**2) // max(1, left.shape[1] * 16)))
    values = {metric: np.empty(len(pairs), dtype=float) for metric in metrics}
    present = np.empty(len(pairs), dtype=bool)
    for start in range(0, len(pairs), chunk_size):
        stop = min(start + chunk_size, len(pairs))
        started = time.perf_counter()
        u = np.asarray(left[pairs[start:stop, 0]], dtype=np.float32)
        v = np.asarray(right[pairs[start:stop, 1] - limit], dtype=np.float32)
        timing["loading_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        there = u.any(axis=1) & v.any(axis=1)
        present[start:stop] = there
        for metric in metrics:
            if attribute is None:
                numerator = np.einsum("ij,ij->i", u, v, dtype=np.float64)
                denominator = np.sqrt(
                    np.einsum("ij,ij->i", u, u, dtype=np.float64)
                    * np.einsum("ij,ij->i", v, v, dtype=np.float64))
                score = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                  where=denominator > 0)
            else:
                score = METRICS[metric](u, v)
            values[metric][start:stop] = np.where(there, score, 0.0)
        timing["similarity_seconds"] += time.perf_counter() - started
    return values, present, timing


def _fingerprint(ids, bank, present):
    digest = hashlib.sha256()
    for frame in (ids, bank, present):
        digest.update(json.dumps(list(frame.columns)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def load_cell(dataset, side, k, models, metrics=("cos",), include_record=True):
    """Load a fixed candidate cell without reading match labels or full records.

    ``pairs`` address full CSV row positions, with right rows offset by the left
    table size. ``present`` contains attribute names, plus ``record`` when used.
    The record mask is common to all selected models; disagreement is rejected.
    Loading timings include mmap page reads for computed record similarities.
    """
    models, metrics = tuple(models), tuple(metrics)
    if not models or not metrics or any(metric not in METRICS for metric in metrics):
        raise ValueError("models and supported metrics must be nonempty")
    timing = _timings()
    started = time.perf_counter()
    ids, pairs, metadata, limit = _candidate_input(dataset, side, k)
    attributes = list(DATASETS[dataset]["attributes"])
    bank = load_bank(dataset, side, k, models, attributes=attributes,
                     metrics=metrics).astype(np.float32)
    present = metadata[["present_" + attr for attr in attributes]].copy()
    present.columns = attributes
    present = present.reset_index(drop=True).astype(bool)
    timing["loading_seconds"] = time.perf_counter() - started
    if len(bank) != len(ids):
        raise ValueError("bank and candidate rows differ")
    records = {}
    for model in models if include_record else ():
        scores, there, measured = _raw_scores(dataset, model, pairs, limit)
        for key in timing:
            timing[key] += measured[key]
        if "record" in present and not np.array_equal(present["record"], there):
            raise ValueError("record presence differs across selected models")
        present["record"] = there
        records[model] = scores["cos"]
    started = time.perf_counter()
    record_scores = pd.DataFrame(records, index=np.arange(len(ids)))
    if records:
        bank = pd.concat([bank.reset_index(drop=True), record_scores.rename(
            columns=lambda model: "record_emb_cos__" + model).astype(np.float32)], axis=1)
    input_hash = _fingerprint(ids, bank, present)
    timing["preprocessing_seconds"] += time.perf_counter() - started
    return dict(ids=ids, pairs=pairs, bank=bank, present=present,
                record_scores=record_scores, attributes=attributes,
                input_hash=input_hash, timings=timing)


class PanelTransform:
    """Fit fusion/scaling on training rows, then reuse a fixed feature schema.

    Separate and mean panels retain raw similarity units. Zmean and family
    panels standardize each source on present training rows and rescale the
    fused training range to [0, 1]. Prediction neither refits nor clips values.
    Empty or constant observed training columns are removed before fusion;
    constant fused columns are removed too. Missing observations always map to 0.
    """

    def __init__(self, models, attributes, fusion="separate", metrics=("cos",)):
        self.models = tuple(models)
        self.attributes = tuple(attributes)
        self.fusion = fusion
        self.metrics = tuple(metrics)
        if fusion not in FUSIONS or not self.models or not self.metrics:
            raise ValueError("nonempty models/metrics and a supported fusion are required")
        if any(metric not in METRICS for metric in self.metrics):
            raise ValueError("unsupported metric")

    def fit(self, bank, present):
        if len(bank) != len(present):
            raise ValueError("bank and presence rows differ")
        self._specs = []
        blocks = [(attr, metric) for attr in self.attributes for metric in self.metrics]
        if any("record_emb_cos__" + model in bank for model in self.models):
            blocks.append(("record", "cos"))
        for attr, metric in blocks:
            there = np.asarray(present[attr], dtype=bool)
            if not there.any():
                continue
            columns, members = [], []
            for model in self.models:
                column = "{}_emb_{}__{}".format(attr, metric, model)
                observed = bank[column].to_numpy(dtype=float)[there]
                if not np.isfinite(observed).all():
                    raise ValueError("nonfinite observed values in " + column)
                if np.ptp(observed) > 1e-12:
                    columns.append(column)
                    members.append(model)
            if not columns:
                continue
            values = bank[columns].to_numpy(dtype=float)[there]
            means, scales = values.mean(axis=0), values.std(axis=0)
            if self.fusion == "separate":
                groups = [(column, [i]) for i, column in enumerate(columns)]
            elif self.fusion == "family":
                unknown = set(members) - {m for group in FAMILIES.values() for m in group}
                if unknown:
                    raise ValueError("models lack a family: " + ", ".join(sorted(unknown)))
                groups = [("{}_emb_{}__{}".format(attr, metric, family),
                           [i for i, model in enumerate(members) if model in group])
                          for family, group in FAMILIES.items()]
            else:
                groups = [("{}_emb_{}__{}".format(attr, metric, self.fusion),
                           list(range(len(columns))))]
            standardized = self.fusion in ("zmean", "family")
            for name, indices in groups:
                if not indices:
                    continue
                selected = values[:, indices]
                if standardized:
                    selected = (selected - means[indices]) / scales[indices]
                score = selected.mean(axis=1)
                low, span = score.min(), np.ptp(score)
                if span <= 1e-12:
                    continue
                self._specs.append(dict(name=name, attribute=attr,
                                        columns=[columns[i] for i in indices],
                                        means=means[indices], scales=scales[indices],
                                        standardized=standardized, low=low, span=span))
        self.columns_ = [spec["name"] for spec in self._specs]
        if not self.columns_:
            raise ValueError("no nonconstant observed training features")
        return self

    def transform(self, bank, present):
        if not hasattr(self, "columns_") or not self.columns_:
            raise ValueError("fit a nonempty feature panel before transform")
        if len(bank) != len(present):
            raise ValueError("bank and presence rows differ")
        output = {}
        for spec in self._specs:
            there = np.asarray(present[spec["attribute"]], dtype=bool)
            values = bank[spec["columns"]].to_numpy(dtype=float)
            if not np.isfinite(values[there]).all():
                raise ValueError("nonfinite observed prediction features")
            if spec["standardized"]:
                values = (values - spec["means"]) / spec["scales"]
            score = values.mean(axis=1)
            if spec["standardized"]:
                score = (score - spec["low"]) / spec["span"]
            output[spec["name"]] = np.where(there, score, 0.0)
        return pd.DataFrame(output, index=bank.index)


def record_disjoint_splits(ids, n_splits=3, seed=0):
    """Deterministic folds with no left or right record shared across a split.

    A record's assignment depends only on seed, side and external ID, so the
    same dataset records keep their folds across directional candidate cells.
    Validation uses two endpoints assigned to the fold; training uses neither.
    Crossing pairs are excluded and retained fractions are reported explicitly.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    assignments = []
    for side, column in (("left", "ltable_id"), ("right", "rtable_id")):
        def assign(value):
            key = json.dumps([seed, side, str(value)], separators=(",", ":")).encode()
            return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % n_splits
        values = ids[column].astype(str)
        mapping = {value: assign(value) for value in values.unique()}
        assignments.append(values.map(mapping).to_numpy())
    left, right = assignments
    output = []
    for fold in range(n_splits):
        train_mask = (left != fold) & (right != fold)
        validation_mask = (left == fold) & (right == fold)
        excluded_mask = ~(train_mask | validation_mask)
        indices = {"train": np.flatnonzero(train_mask),
                   "validation": np.flatnonzero(validation_mask),
                   "excluded": np.flatnonzero(excluded_mask)}
        counts = {"n_" + name: len(rows) for name, rows in indices.items()}
        metadata = dict(seed=seed, fold=fold, n_splits=n_splits, n_pairs=len(ids), **counts)
        for name, rows in indices.items():
            metadata[name + "_fraction"] = len(rows) / len(ids) if len(ids) else 0.0
            metadata[name + "_left_records"] = int(ids.iloc[rows]["ltable_id"].nunique())
            metadata[name + "_right_records"] = int(ids.iloc[rows]["rtable_id"].nunique())
        metadata["retained_fraction"] = metadata["train_fraction"] + metadata["validation_fraction"]
        output.append(dict(indices, metadata=metadata))
    return output


def benchmark_similarity(dataset, side, k, models, metrics=("cos",), repeats=3,
                         include_record=True):
    """Recompute a selected panel from raw mmap arrays without changing caches.

    Similarities are computed afresh; the OS page cache is not controlled. Timings
    exclude encoding, cache validation and model fitting. Attribute parity uses
    the existing bank; record parity uses the normalized loader path.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    models, metrics = tuple(models), tuple(metrics)
    expected = load_cell(dataset, side, k, models, metrics, include_record)
    results, worst = [], 0.0
    for _ in range(repeats):
        timing = _timings()
        started = time.perf_counter()
        ids, pairs, metadata, limit = _candidate_input(dataset, side, k)
        timing["loading_seconds"] += time.perf_counter() - started
        columns = {}
        for model in models:
            for attr in list(expected["attributes"]) + ([None] if include_record else []):
                active_metrics = ("cos",) if attr is None else metrics
                scores, there, measured = _raw_scores(dataset, model, pairs, limit,
                                                     attribute=attr, metrics=active_metrics)
                for key in timing:
                    timing[key] += measured[key]
                group = "record" if attr is None else attr
                np.testing.assert_array_equal(there, expected["present"][group])
                for metric, score in scores.items():
                    columns["{}_emb_{}__{}".format(group, metric, model)] = score
        started = time.perf_counter()
        fresh = pd.DataFrame(columns)[expected["bank"].columns]
        timing["preprocessing_seconds"] += time.perf_counter() - started
        np.testing.assert_array_equal(ids.to_numpy(), expected["ids"].to_numpy())
        np.testing.assert_allclose(fresh, expected["bank"], rtol=1e-6, atol=1e-6)
        if fresh.size:
            worst = max(worst, float(np.abs(fresh.to_numpy() - expected["bank"].to_numpy()).max()))
        results.append(timing)
    return dict(repeats=results, max_absolute_difference=worst,
                input_hash=expected["input_hash"], n_pairs=len(expected["ids"]),
                n_features=expected["bank"].shape[1], cache_state="OS cache unspecified")

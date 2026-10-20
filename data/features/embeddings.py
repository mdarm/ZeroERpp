"""Pair-local attribute similarities and embedding-feature fusion helpers.

Cosine is the inner product of normalized vectors. L-infinity distance is
oriented as ``1 / (1 + distance)`` so larger values consistently indicate
similarity. An exact zero vector marks an absent attribute; any pair with an
absent value receives zero similarity and a separate presence mask.

Feature names begin with the attribute name, which places the columns in that
attribute's covariance block. Similarities depend only on their pair, allowing
smaller candidate sets to reuse generated rows by neighbour rank.

The publication selection/refit path uses ``data.features.panels.PanelTransform``
for train-fitted fusion and scaling, and that module adds the record-cosine
block. The helpers here operate on attribute vectors and complete input frames.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from exp.config import ATTR_DIR
from data.benchmarks import FAMILIES, load_embeddings

__all__ = ["METRICS", "FUSIONS", "build_embedding_features", "embedding_feature_builder", "fuse"]


def _cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine, i.e. the plain inner product — the vectors are already L2-normalised."""
    return np.einsum("ij,ij->i", u, v)


def _linf(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``1/(1+‖u-v‖∞)``: the largest single-dimension disagreement, oriented."""
    return 1.0 / (1.0 + np.abs(u - v).max(axis=1))


# Extension point. A new entry must be increasing in match likelihood and depend
# on its own pair only; see the module docstring on both counts.
METRICS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "cos": _cos,
    "linf": _linf,
}


def build_embedding_features(
    pairs,
    dataset: str,
    model: str,
    attributes: Sequence[str],
    data,
    metrics: Iterable[str] = ("cos",),
    base: str = ATTR_DIR,
) -> pd.DataFrame:
    """``{attr}_emb_{metric}`` columns for ``pairs``, rows aligned with ``pairs``.

    pairs: (left_row, right_row) positional indices into ``data.entities``, right
        rows offset by ``data.dataset_limit`` — the same form
        ``data.features.strings.build_zeroer_features`` takes, so the matrices
        share their row-ID contract.
    dataset, model: names in the attribute-embedding directory selected by
        ``base``; a vector file for each ``(model, attr, side)`` must exist.
    attributes: matchable fields from ``data.benchmarks.DATASETS``.
    data: ``data.benchmarks.BenchmarkData`` providing tables and row-ID mappings.
    metrics: names from ``METRICS``. One column per (attribute, metric).

    Vectors are loaded and released one attribute at a time. The experiment
    preparation command materializes pair-local similarities for reuse.
    """
    if getattr(data, "is_dirty_er", False):
        raise ValueError("per-attribute embedding features support clean-clean ER only")
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise KeyError(f"unknown metric(s) {unknown}; have {sorted(METRICS)}")
    pairs = np.asarray(pairs)
    if len(pairs) == 0:
        return pd.DataFrame()

    limit = data.dataset_limit
    li, ri = pairs[:, 0], pairs[:, 1] - limit

    columns = {}
    for attr in attributes:
        left, right = load_embeddings(dataset, f"{model}_{attr}", data, base=base)
        u, v = left[li], right[ri]
        # an exact zero row is the encoders' "value absent" marker
        present = u.any(axis=1) & v.any(axis=1)
        for name in metrics:
            columns[f"{attr}_emb_{name}"] = np.where(present, METRICS[name](u, v), 0.0)
        del left, right, u, v
    return pd.DataFrame(columns)


def embedding_feature_builder(
    dataset: str,
    model: str,
    attributes: Sequence[str],
    data,
    metrics: Iterable[str] = ("cos",),
    base: str = ATTR_DIR,
) -> Callable[[object], pd.DataFrame]:
    """Bind the dataset, model and attributes into a ``pairs -> features`` callable."""
    metrics = tuple(metrics)

    def build(pairs) -> pd.DataFrame:
        return build_embedding_features(pairs, dataset, model, attributes, data,
                                       metrics=metrics, base=base)

    build.__doc__ = (f"{list(attributes)} x {list(metrics)} embedding features "
                     f"from {model} on {dataset}")
    return build


FUSIONS = ("separate", "mean", "zmean", "family")


def fuse(bank, models, attributes, present, fusion="separate", metric="cos"):
    """Fuse attribute features of one complete input frame.

    Selection with held-out rows uses ``PanelTransform`` instead, so validation
    reuses training statistics and the same feature schema.

    ``bank``: ``load_bank`` rows of exactly the pairs being fitted; ``present``:
    {attribute: boolean array}. One column group per attribute, named into that
    attribute's block:

    * ``separate`` — one column per model; the EM sees every contrast.
    * ``mean`` — the raw mean of the cosines: for L2-normalised vectors this *is*
      the cosine of the concatenated embeddings.
    * ``zmean`` — the mean of the per-model z-scored cosines.
    * ``family`` — one ``zmean`` per encoder family present in ``models``.

    z-scales are fitted here, on the fitted pairs' present rows — never once on a
    larger cell and reused. A fused z-column is mapped back onto [0, 1] over the
    present rows, with 0 where the attribute is absent: upstream's convention for
    every similarity feature, so the initialisation reads it the same way.
    """
    if fusion not in FUSIONS:
        raise ValueError("fusion must be one of {}".format(FUSIONS))
    out = {}
    for attr in attributes:
        there = np.asarray(present[attr], dtype=bool)
        columns = bank[["{}_emb_{}__{}".format(attr, metric, m) for m in models]]
        if fusion == "separate":
            out.update({c: columns[c].to_numpy() for c in columns})
            continue
        if fusion == "mean":
            out["{}_emb_{}__mean".format(attr, metric)] = columns.mean(axis=1).to_numpy()
            continue
        z = (columns - columns[there].mean()) / columns[there].std().replace(0, 1)
        z.columns = models
        groups = {"zmean": models} if fusion == "zmean" else \
            {family: [m for m in members if m in models] for family, members in FAMILIES.items()}
        for name, members in groups.items():
            if members:
                score = z[members].mean(axis=1).to_numpy()
                low, high = score[there].min(), score[there].max()
                out["{}_emb_{}__{}".format(attr, metric, name)] = \
                    np.where(there, (score - low) / ((high - low) or 1.0), 0.0)
    return pd.DataFrame(out)

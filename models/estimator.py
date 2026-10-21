"""Assemble an embedding panel and ZeroER for full-data refitting."""
from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd

from data.features.panels import PanelTransform
from training.unsupervised import fit_and_score


def _aligned_inputs(bank, present):
    if not isinstance(bank, pd.DataFrame) or not isinstance(present, pd.DataFrame):
        raise TypeError("bank and present must be pandas DataFrames")
    if not bank.index.equals(present.index):
        raise ValueError("bank and presence indices must match in row order")
    if not bank.columns.is_unique or not present.columns.is_unique:
        raise ValueError("bank and presence column names must be unique")
    if present.isna().any().any() or any(dtype.kind != "b" for dtype in present.dtypes):
        raise ValueError("presence columns must contain booleans without missing values")


def prepare_panel(cell, panel, train, validation):
    """Fit a transform on training rows, then transform both selected splits.

    ``cell`` contains row-aligned ``bank`` and ``present`` DataFrames and an
    ``attributes`` list; no labels or dataset handles are required. Raw bank
    columns are ``{attribute}_emb_{metric}__{model}``. Presence columns name
    the attribute blocks. ``train`` and ``validation`` select positional rows.
    Return ``(transformer, x_train, x_valid, seconds)``. Transform outputs keep
    their original row indices and share one fitted column order.
    """
    start = perf_counter()
    bank, present = cell["bank"], cell["present"]
    _aligned_inputs(bank, present)
    selected = panel.get("attributes")
    attributes = [a for a in (cell["attributes"] if selected is None else selected)
                  if a != "record"]
    if any(not isinstance(a, str) or "_" in a for a in attributes):
        raise ValueError("attribute block names must be strings without underscores")
    if len(set(attributes)) != len(attributes):
        raise ValueError("attribute block names must be unique")
    if not panel.get("include_record", True):
        bank = bank.loc[:, ~bank.columns.str.startswith("record_emb_")]
    transformer = PanelTransform(panel["models"], attributes,
                                 fusion=panel.get("fusion", "separate"),
                                 metrics=panel.get("metrics", ["cos"]))
    transformer.fit(bank.iloc[train], present.iloc[train])
    x_train = transformer.transform(bank.iloc[train], present.iloc[train])
    x_valid = transformer.transform(bank.iloc[validation], present.iloc[validation])
    return transformer, x_train, x_valid, perf_counter() - start


class ZeroEREstimator:
    """Fit a label-free embedding panel and retain full-data scores."""

    def __init__(self, panel, config):
        self.panel = deepcopy(dict(panel))
        self.config = deepcopy(dict(config))

    def fit(self, pairs, bank, present, attributes=None):
        """Fit and return self, retaining transformed rows and final scores.

        ``pairs`` is an integer (n_pairs, 2) array aligned positionally with
        ``bank``; callers retain the same endpoint IDs for deterministic tie
        ordering. ``attributes`` supplies the default block order when panel
        attributes are omitted; otherwise presence-column order is used.
        ``features_``, ``scores_`` and ``diagnostics_`` expose fit artifacts.
        """
        _aligned_inputs(bank, present)
        pairs = np.asarray(pairs)
        if pairs.shape != (len(bank), 2) or pairs.dtype.kind not in "iu":
            raise ValueError("pairs must be a row-aligned integer array with shape (n, 2)")
        attributes = ([name for name in present if name != "record"]
                      if attributes is None else list(attributes))
        cell = dict(bank=bank, present=present, attributes=attributes)
        rows = np.arange(len(bank))
        transformer, features, _, seconds = prepare_panel(cell, self.panel, rows, rows)
        matcher, scores, diagnostics = fit_and_score(
            self.config, pairs, features, present, features, present)
        # Publish fitted state together only after both fitting and scoring succeed.
        self.transformer_, self.matcher_ = transformer, matcher
        self.features_, self.scores_ = features, scores
        self.diagnostics_ = dict(diagnostics, preprocessing_seconds=seconds)
        return self

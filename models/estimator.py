"""Assemble a fitted embedding panel and ZeroER into a reusable predictor."""
from copy import deepcopy
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import pandas as pd

from data.features.panels import PanelTransform
from models.matcher import ZeroERMatcher
from training.unsupervised import fit_and_score, observed_blocks, score_features
from utils.decisions import decode


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
    """A label-free embedding-panel fit with a complete inference interface.

    ``panel`` chooses models, metrics, fusion, optional attributes, and whether
    to include available record-cosine features. ``config`` supplies the
    ZeroERMatcher learning parameters. Fit learns feature filtering/scaling
    only from its input rows. Score and predict reuse that transformation.

    Inputs are a numeric bank DataFrame of shape (n_pairs, n_raw_features) and
    a boolean presence DataFrame of shape (n_pairs, n_attribute_blocks), with
    identical indices. Missing attributes are represented by False masks and
    become zeros after transformation; raw absent values may be nonfinite.
    Extra raw columns are ignored; all fitted source columns must be present.
    Attribute block names cannot contain underscores. Neither fit nor
    inference reads datasets, labels, experiment storage, or configuration files.
    """

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

    def _features(self, bank, present):
        if not hasattr(self, "transformer_"):
            raise RuntimeError("fit or load an estimator before inference")
        _aligned_inputs(bank, present)
        return self.transformer_.transform(bank, present)

    def score(self, bank, present):
        """Return final-parameter density/LLR/posterior vectors in input order.

        The output dictionary contains vectors ``log_match``, ``log_nonmatch``,
        ``llr``, ``p_match``, ``log_density``, ``n_blocks`` and scalar ``pi``.
        These are generative scores before thresholding or greedy decoding.
        Reordered raw columns are accepted by name; scaling and model state
        remain fixed. Empty input frames return empty score vectors.
        """
        features = self._features(bank, present)
        return score_features(self.matcher_, features, present)

    def predict(self, pairs, bank, present, score="llr", decoder="greedy", threshold=0.):
        """Return an (n_pairs,) boolean match mask without labels or refitting.

        ``pairs`` supplies (left_id, right_id) rows in the bank's row order;
        use the same endpoint ID convention as during fitting.
        Use score ``'llr'`` or ``'posterior'`` (alias ``'p_match'``), with an
        explicit cut in that score's units. Cuts are inclusive; ``threshold=None``
        enables no-cut greedy matching. Greedy orders ties by left then right
        ID, using each endpoint at most once. Pairs with no observed fitted
        block always abstain, including under the zero-fill model policy.
        """
        if score not in ("llr", "posterior", "p_match"):
            raise ValueError("score must be 'llr' or 'posterior'")
        scores = self.score(bank, present)
        observed, _ = observed_blocks(present, self.transformer_.columns_)
        key = "p_match" if score == "posterior" else score
        return decode(pairs, scores[key], decoder=decoder, threshold=threshold,
                      eligible=observed.any(axis=1))

    def save(self, path):
        """Persist config, fitted transform, model, and diagnostics as a pickle.

        Fit-only transformed features and score tables are not duplicated in
        this predictor artifact. Restore with ``ZeroEREstimator.load(path)``.
        """
        if not hasattr(self, "transformer_"):
            raise RuntimeError("fit an estimator before saving it")
        path = Path(path)
        payload = dict(version=1, panel=self.panel, config=self.config,
                       transformer=self.transformer_, model=self.matcher_.model,
                       diagnostics=self.diagnostics_)
        with path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        """Restore a trusted predictor artifact; no training data is required.

        The loaded estimator exposes its transformer_, matcher_, diagnostics_,
        score and predict methods. Fit-only features_ and scores_ are omitted.
        """
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("version") != 1:
            raise ValueError("unsupported estimator artifact version")
        estimator = cls(payload["panel"], payload["config"])
        estimator.transformer_ = payload["transformer"]
        estimator.matcher_ = ZeroERMatcher(**estimator.config)
        estimator.matcher_.model = payload["model"]
        estimator.diagnostics_ = payload["diagnostics"]
        return estimator

"""Fit ZeroER and evaluate its label-free marginal likelihood.

Features are float DataFrames with candidate rows and named feature columns.
Presence is a boolean DataFrame with the same index and one column per
attribute block. This module does not load datasets or select experiments.
"""
from time import perf_counter

import numpy as np

from models.matcher import ZeroERMatcher


def observed_blocks(present, columns):
    """Return the (n_pairs, n_blocks) mask in first feature-block order.

    Feature names begin with ``attribute_``. A block is observed when both
    records have that attribute. Only blocks retained by the fitted panel count.
    """
    blocks = list(dict.fromkeys(name.split("_")[0] for name in columns))
    return present[blocks].to_numpy(dtype=bool), blocks


def marginal_nll(scores, reduction="mean"):
    """Reduce final-parameter marginal log densities over every candidate row.

    ``reduction='sum'`` supports row-weighted aggregation across unequal folds:
    add fold sums, then divide by the total validation-row count. Rows with no
    observed block contribute zero under missing-aware scoring and remain in
    that count. An empty, nonfinite, or overflowing objective is rejected.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("NLL reduction must be 'mean' or 'sum'")
    values = np.asarray(scores["log_density"])
    if values.ndim != 1 or not len(values):
        raise ValueError("NLL requires a nonempty vector of row log densities")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite validation log density")
    total = float(-values.sum())
    if not np.isfinite(total):
        raise ValueError("Non-finite marginal NLL")
    return total if reduction == "sum" else total / len(values)


def _diagnostics(matcher, scores, present, columns):
    observed, blocks = observed_blocks(present, columns)
    block_counts = observed.sum(axis=1)
    predicted = np.asarray(scores["p_match"]) >= .5
    strata, missingness = [], []
    nll = -np.asarray(scores["log_density"])
    for count in np.unique(block_counts):
        keep = block_counts == count
        strata.append(dict(n_blocks=int(count), n=int(keep.sum()),
                           nll_sum=float(nll[keep].sum()), nll=float(nll[keep].mean())))
    for i, block in enumerate(blocks):
        mask = observed[:, i]
        a, b = int((predicted & mask).sum()), int((predicted & ~mask).sum())
        c, d = int((~predicted & mask).sum()), int((~predicted & ~mask).sum())
        denom = float((a + b) * (c + d) * (a + c) * (b + d)) ** .5
        missingness.append(dict(attribute=block, present=int(mask.sum()), n=len(mask),
                                predicted_present=a, predicted_absent=b,
                                phi=(a * d - b * c) / denom if denom else None))
    model = matcher.model
    informed = observed.any(axis=1)
    mean_difference = np.asarray(model.Mu_M) - np.asarray(model.Mu_U)
    order = {}
    for block in blocks:
        indices = [j for j, name in enumerate(columns) if name.split("_")[0] == block]
        order[block] = float(np.mean(mean_difference[indices]))
    return dict(pi=float(scores["pi"]), init_fraction=float(np.mean(model.y)),
                converged=bool(model.converged), n_iter=int(model.n_iter),
                kappa_capped=float(np.mean(np.asarray(model.kappas) >= 1)),
                n_features=len(columns), feature_names=list(columns),
                n_no_features=int((block_counts == 0).sum()),
                n_informative=int(informed.sum()),
                informative_nll=float(nll[informed].mean()) if informed.any() else None,
                component_mean_difference=order, presence_strata=strata, missingness=missingness)


def score_features(matcher, features, present):
    """Score a prepared panel without fitting or changing model parameters.

    ``features`` has shape (n_pairs, n_fitted_features), with the exact fitted
    column order; its index must equal the boolean ``present`` frame's index.
    Returns row vectors ``log_match``, ``log_nonmatch``, ``llr``, ``p_match``,
    ``log_density``, ``n_blocks``, and scalar fitted match prior ``pi``. These
    scores use the final M-step parameters, not legacy last-E-step posteriors.
    """
    if not features.index.equals(present.index):
        raise ValueError("feature and presence indices must match in row order")
    if list(features.columns) != list(matcher.model.feature_names):
        raise ValueError("scoring columns must match the fitted feature order")
    scores = matcher.model.score_components(features.to_numpy(), present=present)
    if len(features):
        marginal_nll(scores, reduction="sum")
    if not 0 < scores["pi"] < 1:
        raise ValueError("Mixture collapsed to a single component")
    return scores


def fit_and_score(config, pairs, x_train, present_train, x_valid, present_valid):
    """Fit on training candidates and score a prepared validation panel.

    ``pairs`` is an integer (n_train, 2) array identifying left/right records.
    Features and presence frames must share an index within each split;
    validation must have the same named feature columns as training. No gold
    labels are accepted. Return ``(matcher, scores, diagnostics)``; the latter
    includes fit and score seconds and observed-block diagnostics.
    """
    if not len(x_train) or not len(x_valid):
        raise ValueError("training and validation panels must be nonempty")
    if not x_train.index.equals(present_train.index):
        raise ValueError("training feature and presence indices must match")
    if not x_train.columns.is_unique:
        raise ValueError("training feature names must be unique")
    start = perf_counter()
    matcher = ZeroERMatcher(**config)
    matcher.match_pairs(pairs, x_train, present=present_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    scores = score_features(matcher, x_valid, present_valid)
    score_seconds = perf_counter() - start
    attrs = _diagnostics(matcher, scores, present_valid, x_train.columns)
    attrs.update(fit_seconds=fit_seconds, score_seconds=score_seconds)
    return matcher, scores, attrs

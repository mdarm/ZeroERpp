"""Post-fit thresholds and greedy one-to-one decisions on fixed pair scores.

The no-cut greedy rule is the experiment's ``pure_transitivity`` arm: take
eligible pairs in descending score order, using each endpoint at most once.
It is not transitive closure, and it does not force a maximum-cardinality
matching. Gold is used only by the reporting helpers, never by ``decode``.
"""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np


def _inputs(pairs, scores, eligible):
    pairs = [tuple(pair) for pair in pairs]
    scores = np.asarray(scores, dtype=float)
    if any(len(pair) != 2 for pair in pairs) or scores.shape != (len(pairs),):
        raise ValueError("pairs and one-dimensional scores must be row-aligned")
    eligible = (np.ones(len(pairs), dtype=bool) if eligible is None
                else np.asarray(eligible, dtype=bool))
    if eligible.shape != scores.shape:
        raise ValueError("eligible must be row-aligned with scores")
    if not np.isfinite(scores[eligible]).all():
        raise ValueError("eligible scores must be finite")
    return pairs, scores, eligible


def _order(pairs, scores, eligible):
    # IDs must be mutually sortable within each side. Separate sides can use
    # overlapping ID spaces; (left=1) and (right=1) are different records.
    return np.array(sorted(np.flatnonzero(eligible),
                           key=lambda i: (-scores[i], pairs[i][0], pairs[i][1])),
                    dtype=int)


def _greedy(pairs, order):
    keep = np.zeros(len(pairs), dtype=bool)
    used_left, used_right = set(), set()
    for i in order:
        left, right = pairs[i]
        if left not in used_left and right not in used_right:
            keep[i] = True
            used_left.add(left)
            used_right.add(right)
    return keep


def _threshold_value(threshold):
    value = float(threshold)
    if math.isnan(value):
        raise ValueError("threshold must not be NaN")
    return value


def _reported_threshold(threshold):
    """Infinity strings preserve the reject-all sentinel in strict JSON."""
    if threshold is None:
        return None
    value = _threshold_value(threshold)
    return ("+inf" if value > 0 else "-inf") if math.isinf(value) else value


def decode(pairs, scores, decoder="threshold", threshold=0.0, eligible=None):
    """Return a row-aligned accepted-pair mask without inspecting labels.

    ``threshold`` accepts numeric cuts and the strings ``'+inf'``/``'-inf'``.
    Cuts are inclusive. ``decoder='greedy', threshold=None`` applies no cut,
    including to negative scores. ``eligible=False`` always abstains (for
    example, pairs without an observed feature block). Equal scores are
    ordered by left ID, then right ID, independently of input row order.
    """
    pairs, scores, eligible = _inputs(pairs, scores, eligible)
    if decoder not in ("threshold", "greedy"):
        raise ValueError("decoder must be 'threshold' or 'greedy'")
    if threshold is None and decoder == "threshold":
        raise ValueError("threshold-only decoding requires a threshold")
    active = eligible.copy()
    if threshold is not None:
        active &= scores >= _threshold_value(threshold)
    if decoder == "threshold":
        return active
    return _greedy(pairs, _order(pairs, scores, active))


def _gold(is_gold, n_gold, size):
    is_gold = np.asarray(is_gold, dtype=bool)
    if is_gold.shape != (size,):
        raise ValueError("is_gold must be row-aligned with scores")
    if int(n_gold) != n_gold or n_gold < int(is_gold.sum()):
        raise ValueError("n_gold must count all gold pairs, including blocked-out pairs")
    return is_gold, int(n_gold)


def _counts(tp, predicted, n_gold):
    tp, predicted = int(tp), int(predicted)
    return dict(tp=tp, predicted=predicted, n_gold=n_gold,
                precision=tp / predicted if predicted else 0.0,
                recall=tp / n_gold if n_gold else 0.0,
                f1=2 * tp / (predicted + n_gold) if predicted + n_gold else 0.0)


def threshold_curve(pairs, scores, is_gold, n_gold, decoder="threshold", eligible=None):
    """Exact gold diagnostic curve, one row per eligible distinct score.

    Rows descend from the JSON-safe ``'+inf'`` reject-all sentinel. Tied
    scores enter together. Greedy acceptance is computed once: filtering its
    no-cut result at any inclusive threshold is identical to rerunning the
    greedy rule at that threshold. Complexity is O(n log n), not one sort per
    possible cut. The curve must never be used for unsupervised fit selection.
    """
    pairs, scores, eligible = _inputs(pairs, scores, eligible)
    is_gold, n_gold = _gold(is_gold, n_gold, len(scores))
    if decoder not in ("threshold", "greedy"):
        raise ValueError("decoder must be 'threshold' or 'greedy'")
    order = _order(pairs, scores, eligible)
    rows = [dict(decoder=decoder, threshold="+inf", **_counts(0, 0, n_gold))]
    if not len(order):
        return rows
    accepted = (eligible if decoder == "threshold" else _greedy(pairs, order))
    predicted = np.cumsum(accepted[order])
    true_positive = np.cumsum(accepted[order] & is_gold[order])
    ordered_scores = scores[order]
    ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(order) - 1]
    for end in ends:
        rows.append(dict(decoder=decoder, threshold=float(scores[order[end]]),
                         **_counts(true_positive[end], predicted[end], n_gold)))
    return rows


def decision_rows(pairs, scores, is_gold, n_gold, score_name, thresholds, eligible=None):
    """Report declared cuts, no-cut greedy, and separately tagged gold oracles.

    ``uses_gold`` describes how the *decision rule was chosen*. Precision,
    recall and F1 are always gold-based diagnostics. Oracle ties select the
    highest cut (fewest predictions), including rejecting all when tied at
    zero F1. Each row's ``decode_seconds`` times only decoding, not the oracle
    search or gold metric calculation. All returned values are strict-JSON
    serializable, including infinite threshold sentinels.
    """
    pairs, scores, eligible = _inputs(pairs, scores, eligible)
    is_gold, n_gold = _gold(is_gold, n_gold, len(scores))
    thresholds = list(thresholds)
    rows = []

    def report(decoder, threshold, rule, uses_gold=False):
        started = perf_counter()
        keep = decode(pairs, scores, decoder, threshold, eligible)
        elapsed = perf_counter() - started
        rows.append(dict(score=score_name, decoder=decoder,
                         threshold=_reported_threshold(threshold), rule=rule,
                         uses_gold=uses_gold, decode_seconds=elapsed,
                         **_counts((keep & is_gold).sum(), keep.sum(), n_gold)))

    for threshold in thresholds:
        if threshold is None:
            raise ValueError("declared thresholds must be numeric; no-cut greedy is automatic")
        for decoder in ("threshold", "greedy"):
            report(decoder, threshold, "fixed")
    report("greedy", None, "pure_transitivity")
    for decoder in ("threshold", "greedy"):
        curve = threshold_curve(pairs, scores, is_gold, n_gold, decoder, eligible)
        best = max(curve, key=lambda row: row["f1"])
        report(decoder, best["threshold"], "oracle_threshold", uses_gold=True)
    return rows

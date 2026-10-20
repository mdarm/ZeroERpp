"""End-to-end and blocking-stage quality metrics.

Both are scored against the *complete* gold standard, so they are comparable
across blocking configurations. Depends on nothing else in the repo.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _count_gold_pairs_present(adj, data) -> int:
    """Count gold-standard pairs present in an adjacency-like container.

    Works for both a networkx ``Graph`` (the match graph) and a ``{id: set}``
    blocks dict — both answer ``i1 in adj`` and ``i2 in adj[i1]`` — so the two
    metrics below share exactly one membership test.
    """
    n = 0
    for _, (id1, id2) in data.ground_truth.iterrows():
        i1 = data._ids_mapping_1[id1]
        i2 = data._ids_mapping_2[id2]
        if (i1 in adj and i2 in adj[i1]) or (i2 in adj and i1 in adj[i2]):
            n += 1
    return n


def precision_recall_f1(graph, data) -> Tuple[float, float, float]:
    """End-to-end P/R/F1 of a match graph against the full gold standard.

    Recall denominator is the *complete* ground truth, so matches dropped by
    blocking count against recall (the statistically sound estimand when the
    blocker is itself a variable being compared).
    """
    tp = _count_gold_pairs_present(graph, data)
    pred = graph.number_of_edges()
    truth = len(data.ground_truth)
    p = tp / pred if pred else 0.0
    r = tp / truth if truth else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def blocking_recall(blocks, data) -> float:
    """Pair completeness: the fraction of gold matches that survive blocking.

    Computed on the candidate-pair structure *before* the matcher runs, so it
    isolates the matches the blocker discards (unrecoverable) from those the
    matcher later rejects. It is therefore the ceiling that end-to-end recall
    can reach for a given blocking configuration — the "don't drop true pairs to
    begin with" objective of the sweep.
    """
    if not blocks:
        return 0.0
    truth = len(data.ground_truth)
    return _count_gold_pairs_present(blocks, data) / truth if truth else 0.0


def gold_ranks(nbrs, data, queried_side, limit):
    """1-based rank at which each gold pair enters the candidate set.

    ``nbrs.shape[1] + 1`` means "never retrieved within the searched range".
    k-NN lists are nested in k, so a pair is in the candidate set at every
    ``k >= its rank`` — one pass therefore yields PC at *every* k.

    Shared by the blocking sweep and record-space export.
    """
    n_k = nbrs.shape[1]
    ranks = np.full(len(data.ground_truth), n_k + 1, dtype=np.int64)
    for r, (_, (id1, id2)) in enumerate(data.ground_truth.iterrows()):
        i1 = data._ids_mapping_1[id1]              # left-table row index
        i2 = data._ids_mapping_2[id2] - limit      # right-table row index
        row, target = (i2, i1) if queried_side == "right" else (i1, i2)
        hit = np.flatnonzero(nbrs[row] == target)
        if hit.size:
            ranks[r] = hit[0] + 1
    return ranks

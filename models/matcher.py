"""Fit a ZeroER mixture on prepared candidate features without a host framework.

The original last E-step probabilities remain in ``posteriors`` and the match
graph. Use ``model.score_components`` for non-mutating final-parameter scores.
Feature transformation and post-fit decision policy belong to the experiment.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from networkx import Graph

from models.zeroer import ZeroerModel, initial_labels

__all__ = ["ZeroERMatcher"]


class ZeroERMatcher:
    """Prepare initial assignments and fit row-aligned candidate features."""

    def __init__(self, c_bay: float = 0.1, max_iter: int = 40,
                 init_threshold: float = 0.8, tol=None,
                 transitivity: bool = False, pi_M=None, hard: bool = False,
                 init_statistic: str = "sum", missing: str = "zero"):
        if missing not in ("zero", "ignore"):
            raise ValueError("missing must be 'zero' or 'ignore'")
        self.c_bay = c_bay
        self.max_iter = max_iter
        # The defaults above and below are the released code's. The paper's
        # stated configuration (sections 3.3, 5.1) is c_bay=0.01,
        # init_threshold=0.5, tol=1e-5, max_iter=200.
        self.init_threshold = init_threshold
        self.tol = tol
        # ZeroER's in-model transitivity for duplicate-free tables — the only
        # variant one cross-table candidate set can serve; the paper's full
        # Algorithm 2 also needs within-table candidate sets and features.
        self.transitivity = transitivity
        # pi_M: the initial match prior (None = the share of initial labels);
        # hard: hard-assignment EM. Both are upstream run_em arguments.
        self.pi_M = pi_M
        self.hard = hard
        # Not upstream — see models.zeroer.initial_labels and ZeroerModel.__init__.
        # init_statistic 'present_mean' seeds EM from the blocks a pair has;
        # missing='ignore' leaves absent attributes out of the likelihood instead
        # of reading them as zero similarity. Both need match_pairs(present=).
        self.init_statistic = init_statistic
        self.missing = missing
        self.model: ZeroerModel = None
        self.feature_matrix: pd.DataFrame = None
        self.pairs: Graph = None
        # row-aligned pair list and the EM's full posterior over it — the graph
        # above drops everything below 0.5, these do not
        self.candidate_pairs: list | None = None
        self.posteriors: np.ndarray | None = None
        self.execution_time = 0.0
        self.em_time = 0.0

    def match_pairs(self, candidate_pairs, feature_matrix: pd.DataFrame,
                    present=None, trace: bool = False, y_init=None) -> Graph:
        """Fit prepared features and retain all legacy last-E-step probabilities.

        ``present``: {attribute block: boolean array, True where both records
        have the value}, row-aligned — what ``init_statistic='present_mean'`` and
        ``missing='ignore'`` read. ``trace`` keeps the per-iteration state in
        ``self.model.history``. ``y_init`` optionally supplies binary initial
        labels; otherwise they are computed once using the selected statistic."""
        if self.missing == "ignore" and present is None:
            raise ValueError("missing='ignore' needs present=")
        if feature_matrix.shape[1] == 0:
            raise RuntimeError("Feature matrix is empty after dropping constant columns.")
        candidate_pairs = list(candidate_pairs)
        if len(candidate_pairs) != len(feature_matrix) or any(len(pair) != 2 for pair in candidate_pairs):
            raise ValueError("candidate pairs and feature rows must be aligned")
        self.feature_matrix = feature_matrix

        em_t0 = time.time()
        names = feature_matrix.columns.tolist()
        if y_init is None:
            y_init = initial_labels(feature_matrix.values, names, self.init_threshold,
                                    self.init_statistic, present)
        # transitivity looks pairs up by id, so it needs the real ones
        id_df = pd.DataFrame(list(candidate_pairs)) if self.transitivity else None
        # P_M is the last E-step's posterior; the model's parameters are one
        # M-step ahead of it (run_em ends on an M-step, as upstream does)
        self.model, P_M = ZeroerModel.run_em(
            similarity_matrixs=(feature_matrix.values, None, None),
            feature_names=names,
            y_inits=(y_init, None, None),
            id_dfs=(id_df, None, None),
            LR_dup_free=self.transitivity,
            LR_identical=False,
            run_trans=self.transitivity,
            c_bay=self.c_bay,
            max_iter=self.max_iter,
            init_threshold=self.init_threshold,
            tol=self.tol,
            pi_M=self.pi_M,
            hard=self.hard,
            init_statistic=self.init_statistic,
            present=present if self.missing == "ignore" else None,
            trace=trace,
        )
        self.em_time = time.time() - em_t0

        P_M = np.asarray(P_M, dtype=float).ravel()
        # zip() below would silently truncate to the shorter of the two, hiding a
        # misalignment as a plausible-looking edge set.
        if len(P_M) != len(candidate_pairs):
            raise RuntimeError(
                f"EM returned {len(P_M)} posteriors for {len(candidate_pairs)} "
                "candidate pairs — feature matrix and pair list are misaligned")
        self.candidate_pairs = candidate_pairs
        self.posteriors = P_M

        self.pairs = Graph()
        for (id1, id2), p in zip(candidate_pairs, P_M):
            if p >= 0.5:
                self.pairs.add_edge(int(id1), int(id2), weight=float(p))
        self.execution_time = time.time() - em_t0
        return self.pairs

    def posterior_frame(self) -> pd.DataFrame:
        """All candidate row IDs with legacy probabilities, without gold labels."""
        if self.posteriors is None:
            raise RuntimeError("no posteriors yet — run match_pairs()")
        return pd.DataFrame({
            "id1": [int(a) for a, _ in self.candidate_pairs],
            "id2": [int(b) for _, b in self.candidate_pairs],
            "p_match": self.posteriors,
        })

    def _configuration(self) -> dict:
        return {
            "c_bay": self.c_bay,
            "max_iter": self.max_iter,
            "init_threshold": self.init_threshold,
            "tol": self.tol,
            "transitivity": self.transitivity,
            "pi_M": self.pi_M,
            "hard": self.hard,
            "init_statistic": self.init_statistic,
            "missing": self.missing,
        }


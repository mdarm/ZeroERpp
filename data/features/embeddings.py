"""Pair-local embedding similarities and supported panel fusions."""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["METRICS", "FUSIONS"]


def _cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine, i.e. the plain inner product — the vectors are already L2-normalised."""
    return np.einsum("ij,ij->i", u, v)


def _linf(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``1/(1+‖u-v‖∞)``: the largest single-dimension disagreement, oriented."""
    return 1.0 / (1.0 + np.abs(u - v).max(axis=1))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "cos": _cos,
    "linf": _linf,
}


FUSIONS = ("separate", "mean", "zmean", "family")

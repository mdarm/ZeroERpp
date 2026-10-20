"""Raw benchmark tables prepared for the original ZeroER string features.

Preparation reads the downloaded CSV files, renames Amazon/Google's right-side
title field and appends Buy's manufacturer to its description. Numeric scalar
types are retained separately for the two tables.
"""
from os.path import join

import pandas as pd

from exp.config import BENCH_DIR

PREP = {
    "fodors_zagats": {},
    "dblp_acm": {},
    "dblp_scholar": {},
    "abt_buy": {"concat": ("description", "manufacturer")},
    "amazon_googleproducts": {"rename": {"name": "title"}},
}


def load_reference_tables(dataset, bench_dir=BENCH_DIR):
    """Read UTF-8 tables and apply the baseline's dataset-specific field rules."""
    spec = PREP[dataset]
    left = pd.read_csv(join(bench_dir, dataset, "left.csv"), encoding="utf-8")
    right = pd.read_csv(join(bench_dir, dataset, "right.csv"), encoding="utf-8")
    if "rename" in spec:
        right = right.rename(columns=spec["rename"])
    if "concat" in spec:
        target, source = spec["concat"]
        right[target] = right[target] + " " + right[source]
    return left, right

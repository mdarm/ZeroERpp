"""Publication input/output locations.

Set ZEROER_DATA_DIR and ZEROER_OUTPUT_DIR before importing experiment modules.
The root CLI does this for its --data-dir and --output-dir arguments.
"""
import os
from os.path import abspath, dirname, join

ROOT_DIR = dirname(dirname(abspath(__file__)))
BENCH_DIR = abspath(os.environ.get("ZEROER_DATA_DIR", join(ROOT_DIR, "data", "benchmarks")))
EMB_DIR = join(BENCH_DIR, "embeddings")
RECORD_DIR = join(EMB_DIR, "record")
ATTR_DIR = join(EMB_DIR, "attribute")
OUTPUT_DIR = abspath(os.environ.get("ZEROER_OUTPUT_DIR", join(ROOT_DIR, "results", "runs")))
PAIRS_DIR = join(OUTPUT_DIR, "ladder_pairs")
FEATURES_DIR = join(OUTPUT_DIR, "magellan-features")
BANK_DIR = join(OUTPUT_DIR, "cosine_bank")
STORAGE = "sqlite:///" + join(OUTPUT_DIR, "optuna.db")
UNSUPERVISED_STORAGE = "sqlite:///" + join(OUTPUT_DIR, "optuna_unsupervised.db")
# Fresh string-feature timings for every cell stay outside the published timing coverage.
FEATURE_TIMING_STORAGE = "sqlite:///" + join(OUTPUT_DIR, "magellan_timings.db")
# A reserved path used by the runtime's guard against accidental legacy writes.
MATCHING_STORAGE = "sqlite:///" + join(OUTPUT_DIR, "optuna_matching.db")

MODEL = "tencent_KaLM-Embedding-Gemma3-12B-2511"
DISTANCE = "cosine"

LADDER = {
    "fodors_zagats": [("right", 4), ("left", 1), ("right", 1)],
    "dblp_acm": [("right", 12), ("right", 5), ("left", 2), ("left", 1)],
    "abt_buy": [("left", 13), ("right", 5), ("left", 2), ("left", 1)],
    "amazon_googleproducts": [("right", 5), ("right", 2), ("left", 2), ("left", 1)],
    "dblp_scholar": [("right", 1), ("left", 10), ("left", 5), ("left", 1)],
}


def cell_name(dataset, side, k):
    return "{}__{}_k{}".format(dataset, side, k)

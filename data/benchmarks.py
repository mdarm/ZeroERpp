"""Benchmark/encoder metadata, full-table loading and vector alignment."""

from __future__ import annotations

import os
from typing import Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from exp.config import BENCH_DIR, RECORD_DIR

# Matchable attributes and renames that align the two tables on shared names:
#   - fodors_zagats: 'type'/'class' dropped — 'class' is a ground-truth leak.
#   - amazon_googleproducts: Amazon's 'title' and Google's 'name' are the same
#     field under different names, so the right table is renamed 'name'->'title'.
DATASETS = {
    "fodors_zagats":         dict(attributes=["name", "addr", "city", "phone"]),
    "dblp_acm":              dict(attributes=["title", "authors", "venue", "year"]),
    "dblp_scholar":          dict(attributes=["title", "authors", "venue", "year"]),
    "abt_buy":               dict(attributes=["name", "description", "price"]),
    "amazon_googleproducts": dict(attributes=["title", "description", "manufacturer", "price"],
                                  rename_2={"name": "title"})
}

# Original string baselines exclude these known nonmatching/leaking blocks.
# Embedding serialization already excludes them through the attribute lists.
DROP_GROUPS = {"fodors_zagats": ["type", "class"]}


# Encoder families used by panel fusion.
FAMILIES = {
    "classic": ["minilm6", "minilm12", "mpnet", "distilroberta"],
    "t5": ["st5_xl", "gtr_t5_large"],
    "retriever": ["e5_large_v2", "e5_instruct", "bge_large", "gte_large"],
    "llm": ["Qwen_Qwen3-Embedding-4B", "Qwen_Qwen3-Embedding-8B",
            "codefuse-ai_F2LLM-v2-4B", "codefuse-ai_F2LLM-v2-8B", "codefuse-ai_F2LLM-v2-14B",
            "Octen_Octen-Embedding-8B", "nvidia_llama-embed-nemotron-8b",
            "microsoft_harrier-oss-v1-0.6b", "microsoft_harrier-oss-v1-27b",
            "tencent_KaLM-Embedding-Gemma3-12B-2511"],
}

def load_embeddings(name: str, model: str, data,
                    base: str = RECORD_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Record vectors for ``data``'s two tables, as float32 (faiss-ready).

    The .npy files follow the row order of left.csv / right.csv. Table IDs map
    the requested rows to their vector positions; an incompatible row count
    raises before retrieval or feature construction.
    """
    d = os.path.join(base, name)
    out = []
    for side, table, id_col in (("left", data.dataset_1, data.id_column_name_1),
                                ("right", data.dataset_2, data.id_column_name_2)):
        vec = np.load(os.path.join(d, f"{model}_{side}.npy"), mmap_mode="r")
        full_ids = pd.read_csv(os.path.join(BENCH_DIR, name, f"{side}.csv"),
                               usecols=[id_col]).astype(str)[id_col]
        if len(vec) != len(full_ids):
            raise ValueError(
                f"{model}_{side}.npy has {len(vec)} rows but {name}/{side}.csv "
                f"has {len(full_ids)} — embeddings not row-aligned")
        pos = pd.Series(np.arange(len(full_ids)), index=full_ids)
        rows = pos[table[id_col].astype(str)].to_numpy()
        out.append(np.asarray(vec[rows], dtype=np.float32))
    return tuple(out)


@dataclass
class BenchmarkData:
    """The two benchmark tables and their row-ID contract.

    Candidate indices address left rows first and right rows after
    ``dataset_limit``. Gold is optional and is never needed for feature input.
    The fields used by retrieval and evaluation are explicit; this is not an
    entity-resolution framework or a training-data abstraction.
    """

    name: str
    dataset_1: pd.DataFrame
    dataset_2: pd.DataFrame
    attributes_1: list[str]
    attributes_2: list[str]
    ground_truth: pd.DataFrame | None = None
    id_column_name_1: str = field(default="id", init=False)
    id_column_name_2: str = field(default="id", init=False)
    is_dirty_er: bool = field(default=False, init=False)

    def __post_init__(self):
        for table, attributes in ((self.dataset_1, self.attributes_1),
                                  (self.dataset_2, self.attributes_2)):
            if "id" not in table or not table["id"].is_unique:
                raise ValueError("benchmark tables require unique id columns")
            missing = set(attributes) - set(table.columns)
            if missing:
                raise ValueError("benchmark is missing attributes: " + ", ".join(sorted(missing)))
        self.dataset_limit = self.num_of_entities_1 = len(self.dataset_1)
        self.num_of_entities_2 = len(self.dataset_2)
        self.num_of_entities = self.num_of_entities_1 + self.num_of_entities_2
        self.dataset_name_1 = self.name + ":left"
        self.dataset_name_2 = self.name + ":right"
        self.entities = pd.concat([self.dataset_1, self.dataset_2], ignore_index=True)
        self._ids_mapping_1 = dict(zip(self.dataset_1["id"], range(self.dataset_limit)))
        self._ids_mapping_2 = dict(zip(self.dataset_2["id"],
                                      range(self.dataset_limit, self.num_of_entities)))
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.astype(str).drop_duplicates().reset_index(drop=True)


def load_data(name: str, base: str = BENCH_DIR,
              include_gold: bool = True) -> BenchmarkData:
    """Load the full clean-clean benchmark with stable positional IDs.

    ``include_gold=False`` leaves ``matches.csv`` unopened. Matching selection
    creates record-disjoint candidate folds in ``data.features.panels``.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    spec = DATASETS[name]
    directory = os.path.join(base, name)
    left = pd.read_csv(os.path.join(directory, "left.csv")).astype(str)
    right = pd.read_csv(os.path.join(directory, "right.csv")).astype(str)
    if spec.get("rename_1"):
        left = left.rename(columns=spec["rename_1"])
    if spec.get("rename_2"):
        right = right.rename(columns=spec["rename_2"])
    truth = (pd.read_csv(os.path.join(directory, "matches.csv")).astype(str)
             if include_gold else None)
    attrs = list(spec["attributes"])
    return BenchmarkData(name, left, right, attrs, list(attrs), truth)


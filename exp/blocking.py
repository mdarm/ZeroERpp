"""Exact FAISS blocking sweep retained from the source experiments.

The full design is five datasets, twenty frozen models, both query directions,
and k=1..100 (20,000 stored trials). A single k_max search supplies every prefix.
"""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time

import faiss
import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.trial import TrialState

from exp.config import BENCH_DIR, RECORD_DIR, STORAGE
from exp.protocol import MODELS
from data.benchmarks import load_data, load_embeddings


def nn_search(v1, v2, k: int, similarity_distance: str) -> np.ndarray:
    """Return each v2 row's ordered k nearest v1 row indices, nearest first.

Copies and normalizes float32 arrays exactly as in the original experiment.
FAISS controls ordering among exactly tied neighbors; data preparation preserves
the original larger-k prefixes where the historical caches used them.
"""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    v1, v2 = (np.asarray(v).astype(np.float32) for v in (v1, v2))
    if v1.ndim != 2 or v2.ndim != 2 or v1.shape[1] != v2.shape[1] or not len(v1):
        raise ValueError("nonempty, compatible two-dimensional vector arrays are required")
    if not np.isfinite(v1).all() or not np.isfinite(v2).all():
        raise ValueError("embedding vectors must be finite")
    if similarity_distance == "cosine":
        faiss.normalize_L2(v1)
        faiss.normalize_L2(v2)
        index = faiss.IndexFlatIP(v1.shape[1])
    elif similarity_distance == "euclidean":
        index = faiss.IndexFlatL2(v1.shape[1])
    else:
        raise ValueError("similarity_distance must be 'cosine' or 'euclidean'")
    index.add(v1)
    _, neighbors = index.search(v2, min(k, len(v1)))
    return neighbors


def gold_ranks(neighbors, data, queried_side, limit):
    """One-based first retrieval ranks; k_max+1 denotes a blocked-out match."""
    ranks = np.full(len(data.ground_truth), neighbors.shape[1] + 1, dtype=np.int64)
    for row, (_, (left_id, right_id)) in enumerate(data.ground_truth.iterrows()):
        left = data._ids_mapping_1[left_id]
        right = data._ids_mapping_2[right_id] - limit
        query, target = (right, left) if queried_side == "right" else (left, right)
        hits = np.flatnonzero(neighbors[query] == target)
        if hits.size:
            ranks[row] = hits[0] + 1
    return ranks


def _vector_fingerprint(path):
    """Hash vector contents on every run, independent of location or mtime."""
    digest = sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return dict(bytes=size, sha256=digest.hexdigest())


def _spec(dataset, models, k_max, queried_sides):
    files = {}
    for name in ("left.csv", "right.csv", "matches.csv"):
        path = Path(BENCH_DIR) / dataset / name
        files[name] = dict(sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    for model in models:
        for side in ("left", "right"):
            path = Path(RECORD_DIR) / dataset / (model + "_" + side + ".npy")
            files[path.name] = _vector_fingerprint(path)
    return dict(dataset=dataset, models=models, k_max=k_max, queried_sides=queried_sides,
                distance="cosine", files=files, code_hash=sha256(Path(__file__).read_bytes()).hexdigest())


def run_blocking_sweep(dataset, k_max=100, queried_sides=("right", "left"),
                       storage=STORAGE, models=None):
    """Persist all k values, resuming only complete model/direction sweeps."""
    models = sorted(MODELS if models is None else models)
    queried_sides = list(queried_sides)
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError("models must select distinct members of the frozen twenty-model pool")
    if not queried_sides or len(set(queried_sides)) != len(queried_sides) or set(queried_sides) - {"left", "right"}:
        raise ValueError("queried_sides must contain left and/or right without duplicates")
    data = load_data(dataset)
    if not 1 <= k_max <= min(data.num_of_entities_1, data.num_of_entities_2):
        raise ValueError("k_max must be positive and no larger than either table")
    limit, n_gold = data.dataset_limit, len(data.ground_truth)
    if not n_gold:
        raise ValueError("blocking evaluation requires nonempty gold matches")
    n_side = {"right": data.num_of_entities_2, "left": data.num_of_entities_1}
    spec = _spec(dataset, models, k_max, queried_sides)
    protocol_hash = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    study = optuna.create_study(study_name=dataset + "_blocking", storage=storage,
                               load_if_exists=True, directions=["maximize", "minimize"])
    old_hash = study.user_attrs.get("protocol_hash")
    if old_hash is not None and old_hash != protocol_hash:
        raise ValueError("Blocking settings, sources or code changed; use a fresh --output-dir")
    if old_hash is None:
        if study.trials:
            raise ValueError("Existing blocking trials lack publication protocol metadata; use fresh outputs")
        study.set_user_attr("spec", spec)
        study.set_user_attr("protocol_hash", protocol_hash)
    groups = defaultdict(list)
    for trial in study.trials:
        groups[trial.params["embedding_model"], trial.params["queried_side"]].append(trial)
    for key, trials in groups.items():
        if (len(trials) != k_max or {t.params["top_k"] for t in trials} != set(range(1, k_max + 1))
                or any(t.state != TrialState.COMPLETE for t in trials)):
            raise RuntimeError("Incomplete or incompatible blocking sweep {} in {}; use fresh outputs".format(
                key, study.study_name))
    distributions = dict(embedding_model=CategoricalDistribution(models),
                         top_k=IntDistribution(1, k_max),
                         queried_side=CategoricalDistribution(queried_sides))
    print("{}: {} models, {} gold matches, {}/{} complete directions".format(
        dataset, len(models), n_gold, len(groups), len(models) * len(queried_sides)), flush=True)
    for model in models:
        vectors = None
        for side in queried_sides:
            if (model, side) in groups:
                continue
            if vectors is None:
                started = time.time()
                vectors = load_embeddings(dataset, model, data)
                load_seconds = round(time.time() - started, 3)
            left, right = vectors
            index, query = (left, right) if side == "right" else (right, left)
            started = time.time()
            neighbors = nn_search(index, query, k_max, "cosine")
            faiss_seconds = round(time.time() - started, 3)
            sorted_neighbors = np.sort(neighbors, axis=1)
            if (sorted_neighbors[:, 1:] == sorted_neighbors[:, :-1]).any():
                raise ValueError("FAISS returned duplicate neighbors")
            tp_at_k = np.cumsum(np.bincount(gold_ranks(neighbors, data, side, limit),
                                           minlength=k_max + 2)[1:k_max + 1])
            trials = []
            for k, tp in enumerate(tp_at_k, start=1):
                tp, count = int(tp), int(n_side[side]) * k
                trials.append(optuna.trial.create_trial(
                    params=dict(embedding_model=model, top_k=k, queried_side=side),
                    distributions=distributions, values=[tp / n_gold, float(count)],
                    user_attrs=dict(tp=tp, n_gold=n_gold, candset_size=count,
                                    pair_completeness=round(tp / n_gold, 6),
                                    pair_quality=round(tp / count, 6), load_seconds=load_seconds,
                                    faiss_seconds=faiss_seconds, dim=int(left.shape[1]),
                                    n_left=int(data.num_of_entities_1), n_right=int(data.num_of_entities_2))))
            study.add_trials(trials)
            print("  {} {}: PC@1={:.6f} PC@{}={:.6f}".format(
                model, side, tp_at_k[0] / n_gold, k_max, tp_at_k[-1] / n_gold), flush=True)
        del vectors
    return study

"""Fresh record-cosine and greedy baseline timings, without attribute vectors.

Every model's normalized record cosine is computed once per cell/repeat.
Single-model and pooled scores are charged only for their own member models;
the shared candidate-ID load is charged once per reported pipeline. Encoding,
blocking, artifact validation and result writes are excluded from timings.
"""
from hashlib import sha256
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd

from utils.decisions import decode, _reported_threshold
from exp.runner import _database, _hash, _json
import data.features.panels as feature_inputs
from data.features.panels import _candidate_input, _raw_scores


DEFAULT_COMBINATIONS = {"classic:pair": ["minilm6", "mpnet"],
                        "retriever:pair": ["e5_large_v2", "gte_large"]}


def _timing_code_hash():
    root = Path(__file__).resolve().parents[1]
    return sha256(Path(__file__).read_bytes()
                  + Path(feature_inputs.__file__).read_bytes()
                  + (root / "utils/decisions.py").read_bytes()).hexdigest()


def _source_signature(dataset, models):
    signatures = {}
    for model in models:
        for side in ("left", "right"):
            name = model + "_" + side + ".npy"
            path = Path(feature_inputs.RECORD_DIR) / dataset / name
            digest = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            signatures[name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    return signatures


def _tables(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS unsupervised_cosine_timings (
            dataset TEXT, side TEXT, k INTEGER, repeat INTEGER, score_name TEXT,
            models_json TEXT, decoder TEXT, threshold_json TEXT,
            loading_seconds REAL, similarity_seconds REAL, preprocessing_seconds REAL,
            aggregation_seconds REAL, decode_seconds REAL, matching_seconds REAL,
            total_seconds REAL, metadata_json TEXT,
            PRIMARY KEY(dataset, side, k, repeat, score_name, decoder, threshold_json));
        CREATE TABLE IF NOT EXISTS unsupervised_cosine_timing_runs (
            dataset TEXT, side TEXT, k INTEGER, repeat INTEGER, protocol_hash TEXT,
            n_rows INTEGER, PRIMARY KEY(dataset, side, k, repeat));
    """)


def _read(connection, dataset, side, k, repeats):
    return pd.read_sql_query("""SELECT * FROM unsupervised_cosine_timings
        WHERE dataset=? AND side=? AND k=? AND repeat<?
        ORDER BY repeat, score_name, decoder, threshold_json""",
        connection, params=(dataset, side, k, repeats))


def benchmark_cosine_cell(dataset, side, k, models, storage, thresholds,
                          repeats=3, combinations=None):
    """Persist exact record-only pipeline costs for singles and declared means.

    ``models`` are computed once per repeat. ``all:mean`` uses all supplied
    models. By default, the classic/retriever pairs are also included whenever
    both members are supplied; ``combinations`` can override those named pools.
    Completed repeat numbers resume without loading vector arrays. Changing the
    panel, cuts, code or input files requires a separate timing database.

    ``matching_seconds`` sums similarity, preprocessing, aggregation and decode
    time; ``total_seconds`` additionally includes file loading/ID mapping.
    The OS cache is unspecified. No labels, attribute vectors or EM fits are used.
    """
    models = sorted(set(models))
    thresholds = tuple(thresholds)
    if not models or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("nonempty models and a positive repeat count are required")
    if any(cut is None for cut in thresholds):
        raise ValueError("thresholds must be numeric; greedy no-cut is automatic")
    cuts = list(dict.fromkeys(_reported_threshold(cut) for cut in thresholds))
    if combinations is None:
        combinations = {name: members for name, members in DEFAULT_COMBINATIONS.items()
                        if set(members) <= set(models)}
    groups = {"record_cos:" + model: [model] for model in models}
    groups["all:mean"] = models
    for name, members in combinations.items():
        members = sorted(set(members))
        if name in groups or not members or not set(members) <= set(models):
            raise ValueError("invalid or reserved cosine combination " + str(name))
        groups[name] = members

    database = _database(storage)
    started = perf_counter()
    initial = _candidate_input(dataset, side, k)
    initial_loading = perf_counter() - started
    ids, pairs, _, limit = initial
    pair_hash = sha256(pd.util.hash_pandas_object(ids, index=False).to_numpy().tobytes()
                       + pairs.tobytes()).hexdigest()
    sources = _source_signature(dataset, models)
    protocol = dict(dataset=dataset, side=side, k=int(k), groups=groups, thresholds=cuts,
                    pairs_hash=pair_hash, sources=sources, code_hash=_timing_code_hash())
    protocol_hash = _hash(protocol)
    with sqlite3.connect(database, timeout=120) as connection:
        _tables(connection)
        completed = connection.execute("""SELECT repeat, protocol_hash, n_rows
            FROM unsupervised_cosine_timing_runs WHERE dataset=? AND side=? AND k=?""",
            (dataset, side, k)).fetchall()
        if any(previous != protocol_hash for _, previous, _ in completed):
            raise ValueError("cosine timing protocol or inputs changed; use a separate database")
        for repeat, _, count in completed:
            actual = connection.execute("""SELECT COUNT(*) FROM unsupervised_cosine_timings
                WHERE dataset=? AND side=? AND k=? AND repeat=?""",
                (dataset, side, k, repeat)).fetchone()[0]
            if count != actual:
                raise ValueError("incomplete stored cosine timing repeat " + str(repeat))
        pending = [i for i in range(repeats) if i not in {row[0] for row in completed}]
        if not pending:
            return _read(connection, dataset, side, k, repeats)

    for offset, repeat in enumerate(pending):
        if offset:
            started = perf_counter()
            current_ids, current_pairs, _, current_limit = _candidate_input(dataset, side, k)
            common_loading = perf_counter() - started
            if not ids.equals(current_ids) or not np.array_equal(pairs, current_pairs) or limit != current_limit:
                raise ValueError("candidate IDs changed during timing")
        else:
            common_loading = initial_loading
        scores, measured, present = {}, {}, None
        for model in models:
            values, there, measured[model] = _raw_scores(dataset, model, pairs, limit)
            if present is not None and not np.array_equal(present, there):
                raise ValueError("record presence differs across selected models")
            present = there
            scores[model] = values["cos"]
        rows = []
        for name, members in groups.items():
            if len(members) == 1:
                values, aggregation_seconds = scores[members[0]], 0.0
            else:
                started = perf_counter()
                values = np.column_stack([scores[model] for model in members]).mean(axis=1)
                aggregation_seconds = perf_counter() - started
            loading_seconds = common_loading + sum(measured[m]["loading_seconds"] for m in members)
            similarity_seconds = sum(measured[m]["similarity_seconds"] for m in members)
            preprocessing_seconds = sum(measured[m]["preprocessing_seconds"] for m in members)
            rules = [(decoder, cut) for cut in cuts for decoder in ("threshold", "greedy")]
            rules.append(("greedy", None))
            for decoder, threshold in rules:
                started = perf_counter()
                accepted = decode(pairs, values, decoder, threshold, eligible=present)
                decode_seconds = perf_counter() - started
                matching_seconds = similarity_seconds + preprocessing_seconds + aggregation_seconds + decode_seconds
                metadata = dict(protocol_hash=protocol_hash, code_hash=protocol["code_hash"],
                    pairs_hash=pair_hash, n_pairs=len(pairs), predicted=int(accepted.sum()),
                    rule="pure_transitivity" if threshold is None else "fixed", uses_gold=False,
                    cost_scope="record-only selected models", cache_state="OS cache unspecified",
                    common_loading_seconds=common_loading,
                    model_timings={model: measured[model] for model in members},
                    sources={key: value for key, value in sources.items()
                             if any(key == model + "_" + table + ".npy"
                                    for model in members for table in ("left", "right"))},
                    shared_costs_across_rules=True,
                    excluded=["embedding generation", "candidate blocking", "attribute vectors",
                              "EM fitting", "artifact validation", "artifact writes"])
                rows.append((dataset, side, k, repeat, name, _json(members), decoder,
                             _json(_reported_threshold(threshold)), loading_seconds,
                             similarity_seconds, preprocessing_seconds, aggregation_seconds,
                             decode_seconds, matching_seconds, loading_seconds + matching_seconds,
                             _json(metadata)))
        if _source_signature(dataset, models) != sources:
            raise ValueError("record vector sources changed during timing")
        with sqlite3.connect(database, timeout=120) as connection:
            connection.executemany("""INSERT OR REPLACE INTO unsupervised_cosine_timings
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            connection.execute("INSERT OR REPLACE INTO unsupervised_cosine_timing_runs VALUES (?,?,?,?,?,?)",
                               (dataset, side, k, repeat, protocol_hash, len(rows)))
    with sqlite3.connect(database, timeout=120) as connection:
        return _read(connection, dataset, side, k, repeats)

"""Repeat selected matching pipelines from precomputed vectors, without gold.

Each SQL row is the cost of one decision rule. Similarities and one EM refit
are shared across thresholds within a repeat; rows must not be summed as jobs.
Embedding generation, candidate blocking, oracle search and artifact writes are
excluded. Raw similarities are recomputed; the OS page cache is uncontrolled.
"""
from hashlib import sha256
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from utils.decisions import decode, _reported_threshold
from exp.runner import (_baselines, _code_hash, _database, _fit, _hash, _json,
                          _observed, _prepare, _storage)
from data.features.panels import benchmark_similarity, load_cell


def _timing_code_hash():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _tables(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS unsupervised_timings (
            study_name TEXT, repeat INTEGER, score_name TEXT, decoder TEXT,
            threshold_json TEXT, loading_seconds REAL, similarity_seconds REAL,
            preprocessing_seconds REAL, fit_seconds REAL, score_seconds REAL,
            decode_seconds REAL, matching_seconds REAL, total_seconds REAL,
            metadata_json TEXT,
            PRIMARY KEY(study_name, repeat, score_name, decoder, threshold_json));
        CREATE TABLE IF NOT EXISTS unsupervised_timing_runs (
            study_name TEXT, repeat INTEGER, protocol_hash TEXT, n_rows INTEGER,
            PRIMARY KEY(study_name, repeat));
    """)


def _read(connection, study_name, repeats):
    return pd.read_sql_query("""SELECT * FROM unsupervised_timings
        WHERE study_name=? AND repeat<?
        ORDER BY repeat, score_name, decoder, threshold_json""",
        connection, params=(study_name, repeats))


def benchmark_study(study_name, storage, repeats=3):
    """Time and persist missing repeats of a completed full Optuna study.

    The result is a DataFrame from ``unsupervised_timings`` in the supplied new
    SQLite database. Repeating the call skips completed repeat numbers; raising
    ``repeats`` appends only new ones. Changed timing code or source selection is
    rejected rather than mixed with previous measurements.

    LLR/posterior rows include fusion, EM fitting and final mixture scoring.
    Cosine rows omit fusion/EM and time their own aggregation. They still carry
    the measured whole-panel similarity cost, explicitly marked as conservative
    when the baseline uses only part of that panel. Loading is excluded from
    ``matching_seconds`` and included in ``total_seconds``.
    """
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    database = _database(storage)
    study = optuna.load_study(study_name=study_name, storage=_storage(storage))
    if study.user_attrs.get("phase") != "full":
        raise ValueError("benchmark_study requires a completed full study")
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not complete:
        raise ValueError("full study has no completed refit")
    selected = max(complete, key=lambda trial: trial.number)
    spec, policy = study.user_attrs["spec"], study.user_attrs["policy"]
    if _code_hash() != spec["code_hash"]:
        raise ValueError("matching code changed since the selected refit")
    protocol = dict(refit_hash=study.user_attrs["refit_hash"],
                    trial_number=selected.number, config=selected.params,
                    timing_code_hash=_timing_code_hash())
    protocol_hash = _hash(protocol)
    with sqlite3.connect(database, timeout=120) as connection:
        _tables(connection)
        recorded = connection.execute("""SELECT repeat, protocol_hash, n_rows
            FROM unsupervised_timing_runs WHERE study_name=?""", (study_name,)).fetchall()
        if any(previous != protocol_hash for _, previous, _ in recorded):
            raise ValueError("timing protocol or source refit changed; keep measurements separate")
        done = set()
        for repeat, _, n_rows in recorded:
            count = connection.execute("""SELECT COUNT(*) FROM unsupervised_timings
                WHERE study_name=? AND repeat=?""", (study_name, repeat)).fetchone()[0]
            if count != n_rows:
                raise ValueError("incomplete stored timing rows for repeat " + str(repeat))
            done.add(repeat)
        pending = [repeat for repeat in range(repeats) if repeat not in done]
        if not pending:
            return _read(connection, study_name, repeats)

    panel = spec["panel"]
    arguments = dict(dataset=spec["dataset"], side=spec["side"], k=spec["k"],
                     models=panel["models"], metrics=panel.get("metrics", ["cos"]),
                     include_record=panel.get("include_record", True))
    cell = load_cell(**arguments)
    if cell["input_hash"] != spec["input_hash"]:
        raise ValueError("inputs changed since the selected refit")
    fresh = benchmark_similarity(**arguments, repeats=len(pending))
    if fresh["input_hash"] != cell["input_hash"]:
        raise ValueError("inputs changed during the fresh-similarity benchmark")
    all_rows = np.arange(len(cell["ids"]))
    thresholds = policy["thresholds"]
    for repeat, raw_timing in zip(pending, fresh["repeats"]):
        features, _, fusion_seconds = _prepare(cell, panel, all_rows, all_rows)
        _, scores, attrs = _fit(selected.params, cell["pairs"], features,
                                cell["present"], features, cell["present"])
        started = perf_counter()
        observed, _ = _observed(cell["present"], features.columns)
        eligible = observed.any(axis=1)
        fusion_seconds += perf_counter() - started
        score_specs = [("llr", scores["llr"], eligible, thresholds["llr"],
                        attrs["fit_seconds"], attrs["score_seconds"], fusion_seconds),
                       ("posterior", scores["p_match"], eligible, thresholds["posterior"],
                        attrs["fit_seconds"], attrs["score_seconds"], fusion_seconds)]
        baselines = iter(_baselines(cell, panel))
        while True:
            started = perf_counter()
            try:
                name, values, valid = next(baselines)
            except StopIteration:
                break
            score_specs.append((name, values, valid, thresholds["cosine"],
                                0.0, perf_counter() - started, 0.0))
        rows = []
        for name, values, valid, cuts, fit_seconds, score_seconds, panel_seconds in score_specs:
            rules = [(decoder, cut) for cut in cuts for decoder in ("threshold", "greedy")]
            rules.append(("greedy", None))
            # Declared duplicate thresholds should still make one SQL result row.
            seen = set()
            for decoder, threshold in rules:
                threshold_json = _json(_reported_threshold(threshold))
                if (decoder, threshold_json) in seen:
                    continue
                seen.add((decoder, threshold_json))
                started = perf_counter()
                keep = decode(cell["pairs"], values, decoder, threshold, eligible=valid)
                decode_seconds = perf_counter() - started
                preprocessing_seconds = raw_timing["preprocessing_seconds"] + panel_seconds
                matching_seconds = (raw_timing["similarity_seconds"] + preprocessing_seconds
                                    + fit_seconds + score_seconds + decode_seconds)
                metadata = dict(protocol, protocol_hash=protocol_hash,
                    source_study=selected.user_attrs.get("source_study"),
                    source_trial=selected.user_attrs.get("source_trial"),
                    input_hash=cell["input_hash"], n_pairs=len(cell["ids"]),
                    n_features=len(features.columns), predicted=int(keep.sum()),
                    rule="pure_transitivity" if threshold is None else "fixed",
                    uses_gold=False, cache_state="OS cache unspecified",
                    similarity_scope="whole selected panel",
                    cost_scope=("selected matching pipeline" if name in ("llr", "posterior")
                                else "panel-wide upper-bound, shared feature work"),
                    conservative_baseline_cost=name not in ("llr", "posterior"),
                    shared_costs_across_rules=True,
                    max_absolute_difference=fresh["max_absolute_difference"],
                    excluded=["embedding generation", "candidate blocking", "oracle search",
                              "artifact validation", "artifact writes"])
                rows.append((study_name, repeat, name, decoder, threshold_json,
                    raw_timing["loading_seconds"], raw_timing["similarity_seconds"],
                    preprocessing_seconds, fit_seconds, score_seconds, decode_seconds,
                    matching_seconds, raw_timing["loading_seconds"] + matching_seconds,
                    _json(metadata)))
        with sqlite3.connect(database, timeout=120) as connection:
            connection.executemany("""INSERT OR REPLACE INTO unsupervised_timings
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            connection.execute("INSERT OR REPLACE INTO unsupervised_timing_runs VALUES (?,?,?,?)",
                               (study_name, repeat, protocol_hash, len(rows)))
    with sqlite3.connect(database, timeout=120) as connection:
        return _read(connection, study_name, repeats)

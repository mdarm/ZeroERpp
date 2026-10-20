"""Optional fresh string-feature timing on fixed candidates; never runs on import.

Costs shared across baselines and decision rules must not be summed as jobs.
Amazon-Google's smallest cell is documented to take about 12 minutes just for
string features; that dataset requires an explicit ``allow_expensive=True``.
"""
from hashlib import sha256
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd

from utils.decisions import decode, _reported_threshold
from data.benchmarks import DROP_GROUPS
from models.matcher import ZeroERMatcher
from data.reference import load_reference_tables
from exp.runner import _database, _hash, _json
from exp.baselines import BASELINES, _code_hash as _baseline_code_hash, _load_cell
from data.features.strings import build_zeroer_features

THRESHOLDS = (.1, .25, .5, .75, .9)


def _code_hash():
    root = Path(__file__).resolve().parents[1]
    return sha256(_baseline_code_hash().encode() + b"".join((root / name).read_bytes()
        for name in ("exp/baseline_timing.py",
                     "data/features/strings.py", "data/reference.py"))).hexdigest()


def _tables_hash(left, right):
    digest = sha256()
    for table in (left, right):
        digest.update(_json(list(table.columns)).encode())
        digest.update(_json([str(dtype) for dtype in table.dtypes]).encode())
        digest.update(pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _tables(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS unsupervised_baseline_timings (
            dataset TEXT, queried_side TEXT, top_k INTEGER, baseline TEXT,
            repeat INTEGER, score_name TEXT, decoder TEXT, threshold_json TEXT,
            loading_seconds REAL, similarity_seconds REAL, preprocessing_seconds REAL,
            fit_seconds REAL, decode_seconds REAL, matching_seconds REAL,
            total_seconds REAL, metadata_json TEXT,
            PRIMARY KEY(dataset, queried_side, top_k, baseline, repeat, decoder, threshold_json));
        CREATE TABLE IF NOT EXISTS unsupervised_baseline_timing_runs (
            dataset TEXT, queried_side TEXT, top_k INTEGER, repeat INTEGER,
            protocol_hash TEXT, n_rows INTEGER,
            PRIMARY KEY(dataset, queried_side, top_k, repeat));
    """)


def _read(connection, dataset, side, k, repeats):
    return pd.read_sql_query("""SELECT * FROM unsupervised_baseline_timings
        WHERE dataset=? AND queried_side=? AND top_k=? AND repeat<?
        ORDER BY repeat, baseline, decoder, threshold_json""", connection,
        params=(dataset, side, k, repeats))


def benchmark_baseline_cell(dataset, side, k, storage, repeats=1,
                            n_jobs=1, allow_expensive=False):
    """Measure fresh Magellan-style features plus original B-paper/B-code fits.

    No blocking, embeddings, gold reads or final-E-step refresh. Each repeat
    generates similarities once and fits each fixed baseline once; eleven
    declared decisions share that work. Returned SQL rows report feature,
    preprocessing, fit and decode time separately. ``matching_seconds`` excludes
    raw-table loading; ``total_seconds`` includes it. Reference loading/parity
    checks and database writes are excluded. Completed repeats resume unchanged.
    """
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    if dataset == "amazon_googleproducts" and not allow_expensive:
        raise ValueError("Amazon-Google fresh string features can take hours across cells; "
                         "pass allow_expensive=True to benchmark this cell explicitly")
    database = _database(storage)
    cell = _load_cell(dataset, side, k)
    left, right = load_reference_tables(dataset)
    raw_hash = _tables_hash(left, right)
    protocol = dict(dataset=dataset, side=side, k=int(k), input_hash=cell["input_hash"],
                    raw_tables_hash=raw_hash, code_hash=_code_hash(), n_jobs=n_jobs,
                    configurations=BASELINES, thresholds=list(THRESHOLDS),
                    dtype_policy="preserve_per_side_scalars", score_state="legacy_last_e_step")
    protocol_hash = _hash(protocol)
    with sqlite3.connect(database, timeout=120) as connection:
        _tables(connection)
        recorded = connection.execute("""SELECT repeat, protocol_hash, n_rows
            FROM unsupervised_baseline_timing_runs
            WHERE dataset=? AND queried_side=? AND top_k=?""", (dataset, side, k)).fetchall()
        if any(previous != protocol_hash for _, previous, _ in recorded):
            raise ValueError("Baseline timing inputs, code or policy changed; keep measurements separate")
        done = set()
        for repeat, _, expected in recorded:
            count = connection.execute("""SELECT COUNT(*) FROM unsupervised_baseline_timings
                WHERE dataset=? AND queried_side=? AND top_k=? AND repeat=?""",
                (dataset, side, k, repeat)).fetchone()[0]
            if count != expected:
                raise ValueError("Incomplete stored baseline timing repeat " + str(repeat))
            done.add(repeat)
        pending = [repeat for repeat in range(repeats) if repeat not in done]
        if not pending:
            return _read(connection, dataset, side, k, repeats)

    # Compare even constant columns before dropping them. Unknown columns or
    # known concatenation/dtype discrepancies cannot silently pass as timing.
    reference = cell["raw_features"]
    for repeat in pending:
        started = perf_counter()
        left, right = load_reference_tables(dataset)
        loading_seconds = perf_counter() - started
        if _tables_hash(left, right) != raw_hash:
            raise ValueError("Raw tables changed during the baseline benchmark")
        started = perf_counter()
        attributes = [attr for attr in left.columns if attr in right.columns
                      and attr not in DROP_GROUPS.get(dataset, [])]
        # The shared feature API takes one frame. Object arrays retain int
        # values on one side and float values on the other (DBLP-Scholar year),
        # preserving Magellan's separate-table scalar types.
        entities = pd.concat([left.astype(object), right.astype(object)], ignore_index=True)
        preprocessing_seconds = perf_counter() - started
        started = perf_counter()
        fresh = build_zeroer_features(cell["pairs"], entities, attributes,
            dataset_limit=len(left), drop_zero_variance=False, n_jobs=n_jobs)
        similarity_seconds = perf_counter() - started
        if set(fresh.columns) != set(reference.columns) or len(fresh) != len(reference):
            raise ValueError("Fresh and cached baseline feature schemas differ")
        fresh = fresh.loc[:, reference.columns]
        difference = np.abs(fresh.to_numpy(float) - reference.to_numpy(float))
        if not np.isfinite(difference).all() or (difference > 1e-15).any():
            raise ValueError("Fresh and cached baseline feature values differ (max absolute error {})".format(
                float(np.max(difference))))
        max_difference = float(np.max(difference)) if difference.size else 0.
        started = perf_counter()
        features = fresh.loc[:, fresh.nunique() > 1]
        preprocessing_seconds += perf_counter() - started
        if list(features.columns) != list(cell["features"].columns):
            raise ValueError("Fresh and cached baseline constant-column selections differ")

        rows = []
        for baseline, config in BASELINES.items():
            started = perf_counter()
            matcher = ZeroERMatcher(**config)
            matcher.match_pairs(cell["pairs"], features)
            posterior = np.asarray(matcher.posteriors, dtype=float)
            fit_seconds = perf_counter() - started
            if posterior.shape != (len(cell["pairs"]),) or not np.isfinite(posterior).all():
                raise ValueError("Baseline posterior must be finite and candidate-aligned")
            rules = [(decoder, cut) for cut in THRESHOLDS for decoder in ("threshold", "greedy")]
            rules.append(("greedy", None))
            for decoder, cut in rules:
                started = perf_counter()
                keep = decode(cell["pairs"], posterior, decoder, cut)
                decode_seconds = perf_counter() - started
                matching_seconds = similarity_seconds + preprocessing_seconds + fit_seconds + decode_seconds
                metadata = dict(protocol, protocol_hash=protocol_hash, baseline=baseline,
                    config=config, predicted=int(keep.sum()), n_pairs=len(cell["pairs"]),
                    n_features=features.shape[1], max_absolute_difference=max_difference,
                    uses_gold=False, rule="pure_transitivity" if cut is None else "fixed",
                    shared_costs_across_rules=True, shared_similarity_across_baselines=True,
                    cache_state="OS cache unspecified", source=cell["source"],
                    excluded=["candidate blocking", "cached reference loading", "parity validation",
                              "gold evaluation", "artifact writes"])
                rows.append((dataset, side, k, baseline, repeat, "legacy_posterior", decoder,
                    _json(_reported_threshold(cut)), loading_seconds, similarity_seconds,
                    preprocessing_seconds, fit_seconds, decode_seconds, matching_seconds,
                    loading_seconds + matching_seconds, _json(metadata)))
        with sqlite3.connect(database, timeout=120) as connection:
            connection.executemany("""INSERT OR REPLACE INTO unsupervised_baseline_timings
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            connection.execute("INSERT OR REPLACE INTO unsupervised_baseline_timing_runs VALUES (?,?,?,?,?,?)",
                               (dataset, side, k, repeat, protocol_hash, len(rows)))
    with sqlite3.connect(database, timeout=120) as connection:
        return _read(connection, dataset, side, k, repeats)

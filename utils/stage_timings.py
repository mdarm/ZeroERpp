"""Publication stage costs, keeping historical evidence separate from fresh runs.

Readers accept stored measurements only. They never encode, fit, or benchmark,
and coverage does not assert equality between runtimes on different machines.
"""
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from exp.protocol import CELLS, PRIMARY_CELLS


EVIDENCE = Path(__file__).resolve().parents[1] / "data" / "publication-evidence.json"
CELL = ["dataset", "side", "k"]
STAGES = ["loading_seconds", "similarity_seconds", "preprocessing_seconds",
          "fit_seconds", "decode_seconds", "matching_seconds", "total_seconds"]
RULES = {(baseline, decoder, str(cut)) for baseline in ("B-paper", "B-code")
         for decoder in ("threshold", "greedy") for cut in (.1, .25, .5, .75, .9)}
RULES |= {(baseline, "greedy", "null") for baseline in ("B-paper", "B-code")}


def _historical(evidence_path):
    evidence = json.loads(Path(evidence_path).read_text())
    calls = pd.DataFrame(evidence["encoding_calls"])
    source = evidence["sources"]["encoding"]
    for key in ("path", "sha256", "measurement_population", "hardware_provenance", "scope"):
        calls["source_" + key] = source[key]
    calls["historical_measurement"] = True
    rows = []
    for item in evidence["blocking_trials"]:
        attrs = item["attributes"]
        rows.append({**{key: item[key] for key in CELL}, "model": evidence["model"],
                     "trial_id": item["trial_id"], "trial_number": item["trial_number"],
                     "study_name": item["study_name"], "recorded_at": item["datetime_start"],
                     "retrieval_depth": item["retrieval_depth"],
                     "retrieval_seconds": attrs["faiss_seconds"],
                     "loading_seconds": attrs["load_seconds"], "historical_measurement": True})
    blocking = pd.DataFrame(rows)
    source = evidence["sources"]["blocking"]
    for key in ("path", "sha256", "measurement_population", "hardware_provenance", "scope"):
        blocking["source_" + key] = source[key]
    primary = []
    for dataset, side, k in PRIMARY_CELLS:
        cell = dict(dataset=dataset, side=side, k=k, model=evidence["model"],
                    historical_measurement=True)
        encoded = calls.loc[calls.dataset.eq(dataset) & calls.phase.eq("attribute")]
        primary.append(dict(cell, stage="attribute_encoding", seconds=float(encoded.seconds.sum()),
                            measured_calls=len(encoded), status="historical",
                            measurement_population=evidence["sources"]["encoding"]["measurement_population"]))
        primary.append(dict(cell, stage="record_encoding", seconds=np.nan,
                            measured_calls=0, status="not_timed",
                            measurement_population="record vectors were cache hits"))
        retrieved = blocking.loc[blocking.dataset.eq(dataset)].iloc[0]
        primary.append(dict(cell, stage="blocking_depth100", seconds=retrieved.retrieval_seconds,
                            measured_calls=1, status="historical",
                            measurement_population=source["measurement_population"]))
    return {"historical_encoding_calls": calls, "historical_blocking_timings": blocking,
            "primary_historical_stage_timings": pd.DataFrame(primary)}


def _feature_repeats(database):
    """Read complete, internally consistent repeats from the fresh-feature DB."""
    database = Path(database)
    if not database.is_file():
        return []
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"unsupervised_baseline_timings", "unsupervised_baseline_timing_runs"} <= tables:
            return []
        raw = pd.read_sql_query("SELECT * FROM unsupervised_baseline_timings", connection).rename(
            columns={"queried_side": "side", "top_k": "k"})
        runs = pd.read_sql_query("SELECT * FROM unsupervised_baseline_timing_runs", connection).rename(
            columns={"queried_side": "side", "top_k": "k"})
    repeats = []
    keys = CELL + ["repeat"]
    if runs.duplicated(keys).any():
        raise ValueError("Duplicate fresh-feature timing run markers")
    for item in runs.to_dict("records"):
        frame = raw.loc[(raw[keys] == pd.Series({key: item[key] for key in keys})).all(axis=1)]
        # A partially written run cannot satisfy coverage.
        rules = set(frame[["baseline", "decoder", "threshold_json"]].itertuples(index=False, name=None))
        if len(frame) != len(RULES) or item["n_rows"] != len(RULES) or rules != RULES:
            continue
        if (item["repeat"] < 0 or not np.isfinite(frame[STAGES]).all().all()
                or (frame[STAGES] < 0).any().any()):
            raise ValueError("Fresh-feature timing durations and repeat indices must be finite and nonnegative")
        metadata = [json.loads(value) for value in frame.metadata_json]
        for row, details in zip(frame.to_dict("records"), metadata):
            if (details.get("protocol_hash") != item["protocol_hash"]
                    or not isinstance(details.get("input_hash"), str) or not details["input_hash"]
                    or details.get("n_jobs") != 1 or details.get("uses_gold") is not False
                    or details.get("shared_costs_across_rules") is not True
                    or details.get("shared_similarity_across_baselines") is not True
                    or not 0 <= details.get("max_absolute_difference", float("inf")) <= 1e-15):
                raise ValueError("Fresh-feature timing provenance or feature parity is invalid")
            expected = sum(row[key] for key in ("similarity_seconds", "preprocessing_seconds",
                                                "fit_seconds", "decode_seconds"))
            if (not np.isclose(row["matching_seconds"], expected, rtol=1e-12, atol=1e-12)
                    or not np.isclose(row["total_seconds"], expected + row["loading_seconds"],
                                      rtol=1e-12, atol=1e-12)):
                raise ValueError("Fresh-feature timing stage totals are inconsistent")
        shared = ["loading_seconds", "similarity_seconds", "preprocessing_seconds"]
        if frame[shared].nunique().gt(1).any() or frame.groupby("baseline").fit_seconds.nunique().gt(1).any():
            raise ValueError("Shared feature and fit costs disagree across decision rows")
        if any(details["input_hash"] != metadata[0]["input_hash"]
               or details["n_pairs"] != metadata[0]["n_pairs"]
               or details["n_features"] != metadata[0]["n_features"] for details in metadata):
            raise ValueError("Fresh-feature timing input dimensions disagree")
        fits = frame.groupby("baseline").fit_seconds.first()
        repeats.append({**{key: item[key] for key in keys},
                        "protocol_hash": item["protocol_hash"], "n_pairs": metadata[0]["n_pairs"],
                        "input_hash": metadata[0]["input_hash"],
                        "n_features": metadata[0]["n_features"],
                        **{key: frame.iloc[0][key] for key in shared},
                        "feature_seconds": frame.iloc[0].similarity_seconds,
                        "b_paper_fit_seconds": fits["B-paper"], "b_code_fit_seconds": fits["B-code"]})
    return repeats


def stage_timing_reports(feature_database, evidence_path=None, *, expected_trials=None):
    """Return historical evidence tables and fresh coverage for all 22 cells.

    ``feature_database`` is the separate ``magellan_timings.db`` produced by
    ``timings --kind features``. One complete serial repeat per cell is required;
    eleven decision rows share each baseline fit and the common feature cost.
    When supplied, ``expected_trials`` is the matching report's trial frame:
    feature timings must share its completed baseline candidate/feature hash.
    """
    reports = _historical(evidence_path or EVIDENCE)
    repeats = _feature_repeats(feature_database)
    coverage, stages = [], []
    for dataset, side, k in CELLS:
        cell = dict(dataset=dataset, side=side, k=k)
        measured = [row for row in repeats if all(row[key] == value for key, value in cell.items())]
        complete = any(row["repeat"] == 0 for row in measured)
        status = "complete" if complete else "missing"
        if measured and expected_trials is not None:
            expected = expected_trials
            if not expected.empty:
                expected = expected.loc[expected.phase.eq("baseline") & expected.state.eq("COMPLETE")
                    & (expected[CELL] == pd.Series(cell)).all(axis=1)]
            if expected.empty:
                status = "missing_matching_baseline"
            else:
                hashes = expected.input_hash.dropna().unique()
                if len(hashes) != 1 or any(row["input_hash"] != hashes[0] for row in measured):
                    raise ValueError("Fresh-feature timing inputs differ from the matching baseline: " + str(cell))
        coverage.append(dict(cell, required_repeats=1, complete_repeats=len(measured),
                             status=status))
        if not measured:
            continue
        frame = pd.DataFrame(measured)
        if frame[["protocol_hash", "input_hash", "n_pairs", "n_features"]].nunique().gt(1).any():
            raise ValueError("Fresh-feature timing protocols or dimensions changed between repeats")
        values = ["loading_seconds", "preprocessing_seconds", "feature_seconds",
                  "b_paper_fit_seconds", "b_code_fit_seconds"]
        stages.append(dict(cell, n_pairs=measured[0]["n_pairs"], n_features=measured[0]["n_features"],
                           input_hash=measured[0]["input_hash"],
                           repeats=len(measured), **{key: frame[key].median() for key in values},
                           feature_seconds_min=frame.feature_seconds.min(),
                           feature_seconds_max=frame.feature_seconds.max(),
                           status=status, historical_measurement=False,
                           scope="fresh string features and both code-profile fits; shared costs counted once"))
    reports["feature_timing_coverage"] = pd.DataFrame(coverage)
    reports["feature_stage_timings"] = pd.DataFrame(stages, columns=CELL + [
        "n_pairs", "n_features", "input_hash", "repeats", "loading_seconds", "preprocessing_seconds", "feature_seconds",
        "b_paper_fit_seconds", "b_code_fit_seconds", "feature_seconds_min", "feature_seconds_max",
        "status", "historical_measurement", "scope"])
    return reports

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from exp.protocol import CELLS
from utils.stage_timings import RULES, stage_timing_reports


def make_database(path, cells):
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE unsupervised_baseline_timing_runs (
                dataset TEXT, queried_side TEXT, top_k INTEGER, repeat INTEGER,
                protocol_hash TEXT, n_rows INTEGER);
            CREATE TABLE unsupervised_baseline_timings (
                dataset TEXT, queried_side TEXT, top_k INTEGER, baseline TEXT,
                repeat INTEGER, decoder TEXT, threshold_json TEXT,
                loading_seconds REAL, similarity_seconds REAL, preprocessing_seconds REAL,
                fit_seconds REAL, decode_seconds REAL, matching_seconds REAL,
                total_seconds REAL, metadata_json TEXT);
        """)
        for dataset, side, k in cells:
            metadata = json.dumps(dict(protocol_hash="same-input-and-code", n_jobs=1, uses_gold=False,
                input_hash="matching-candidate-feature-hash",
                shared_costs_across_rules=True, shared_similarity_across_baselines=True,
                max_absolute_difference=0., n_pairs=100, n_features=4))
            connection.execute("INSERT INTO unsupervised_baseline_timing_runs VALUES (?,?,?,?,?,?)",
                               (dataset, side, k, 0, "same-input-and-code", 22))
            for baseline, decoder, threshold in RULES:
                fit = 2. if baseline == "B-paper" else 3.
                connection.execute("INSERT INTO unsupervised_baseline_timings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (dataset, side, k, baseline, 0, decoder, threshold, .2, 10., .1, fit, .05,
                     10.15 + fit, 10.35 + fit, metadata))


class StageTimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "features.db"

    def test_historical_evidence_preserves_untimed_record_encoding(self):
        reports = stage_timing_reports(self.database)
        self.assertFalse(self.database.exists())
        calls = reports["historical_encoding_calls"]
        self.assertEqual(len(calls), 38)
        self.assertAlmostEqual(calls.seconds.sum(), 509.303)
        self.assertEqual(set(calls.phase), {"attribute"})
        primary = reports["primary_historical_stage_timings"]
        untimed = primary.loc[primary.stage.eq("record_encoding")]
        self.assertEqual(len(untimed), 5)
        self.assertTrue(untimed.seconds.isna().all())
        self.assertEqual(set(untimed.status), {"not_timed"})
        blocking = reports["historical_blocking_timings"].set_index("dataset")
        self.assertEqual(blocking.loc["dblp_scholar", "retrieval_seconds"], 10.056)
        self.assertEqual(set(blocking.retrieval_depth), {100})
        self.assertEqual(set(reports["feature_timing_coverage"].status), {"missing"})

    def test_complete_coverage_counts_shared_work_once(self):
        make_database(self.database, CELLS)
        reports = stage_timing_reports(self.database)
        coverage = reports["feature_timing_coverage"]
        self.assertEqual(len(coverage), 22)
        self.assertEqual(set(coverage.status), {"complete"})
        times = reports["feature_stage_timings"]
        self.assertEqual(times.feature_seconds.sum(), 220.)
        self.assertEqual(times.b_paper_fit_seconds.sum(), 44.)
        self.assertEqual(times.b_code_fit_seconds.sum(), 66.)

    def test_incomplete_repeat_does_not_satisfy_coverage(self):
        make_database(self.database, CELLS[:1])
        with sqlite3.connect(self.database) as connection:
            connection.execute("DELETE FROM unsupervised_baseline_timings WHERE rowid=1")
        reports = stage_timing_reports(self.database)
        self.assertEqual(set(reports["feature_timing_coverage"].status), {"missing"})
        self.assertTrue(reports["feature_stage_timings"].empty)

    def test_missing_final_cell_is_visible(self):
        make_database(self.database, CELLS[:-1])
        coverage = stage_timing_reports(self.database)["feature_timing_coverage"]
        self.assertEqual(coverage.status.eq("complete").sum(), 21)
        self.assertEqual(coverage.status.eq("missing").sum(), 1)

    def test_invalid_parity_is_rejected(self):
        make_database(self.database, CELLS[:1])
        with sqlite3.connect(self.database) as connection:
            details = json.loads(connection.execute(
                "SELECT metadata_json FROM unsupervised_baseline_timings LIMIT 1").fetchone()[0])
            details["max_absolute_difference"] = .01
            connection.execute("UPDATE unsupervised_baseline_timings SET metadata_json=? WHERE rowid=1",
                               (json.dumps(details),))
        with self.assertRaisesRegex(ValueError, "parity"):
            stage_timing_reports(self.database)

    def test_inconsistent_shared_cost_is_rejected(self):
        make_database(self.database, CELLS[:1])
        with sqlite3.connect(self.database) as connection:
            connection.execute("""UPDATE unsupervised_baseline_timings
                SET similarity_seconds=similarity_seconds+1, matching_seconds=matching_seconds+1,
                    total_seconds=total_seconds+1 WHERE rowid=1""")
        with self.assertRaisesRegex(ValueError, "Shared"):
            stage_timing_reports(self.database)

    def test_timings_must_match_the_matching_baseline_inputs(self):
        make_database(self.database, CELLS[:1])
        dataset, side, k = CELLS[0]
        expected = pd.DataFrame([dict(dataset=dataset, side=side, k=k,
            phase="baseline", state="COMPLETE", input_hash="matching-candidate-feature-hash")])
        coverage = stage_timing_reports(self.database, expected_trials=expected)["feature_timing_coverage"]
        self.assertEqual(coverage.status.eq("complete").sum(), 1)
        expected["input_hash"] = "different-inputs"
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            stage_timing_reports(self.database, expected_trials=expected)

    def test_timings_without_matching_baseline_do_not_satisfy_coverage(self):
        make_database(self.database, CELLS[:1])
        coverage = stage_timing_reports(self.database, expected_trials=pd.DataFrame())["feature_timing_coverage"]
        self.assertFalse(coverage.status.eq("complete").any())
        self.assertEqual(coverage.status.eq("missing_matching_baseline").sum(), 1)


if __name__ == "__main__":
    unittest.main()

"""Small contract tests; optional acceptance reads existing publication outputs."""
import os
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from exp.config import MODEL
from utils.publication_diagnostics import (align_presence, binary_association,
    publication_diagnostics, raw_feature_mean, same_model_comparisons,
    compare_publication_diagnostics, REFERENCE_KEYS)


class PublicationDiagnosticsTest(unittest.TestCase):
    def test_raw_mean_uses_observed_attributes_and_always_record(self):
        result = raw_feature_mean([.6, .9, 0], [2, 1, 0], [.9, .5, .7])
        np.testing.assert_allclose(result, [.7, .7, .7])
        with self.assertRaisesRegex(ValueError, "feature counts"):
            raw_feature_mean([.6], [1.5], [.9])

    def test_presence_reorders_by_both_ids_and_rejects_mismatch(self):
        scores = pd.DataFrame({"ltable_id": ["a", "b"], "rtable_id": ["c", "d"]})
        masks = scores.iloc[::-1].assign(present_price=[True, False])
        self.assertEqual(align_presence(scores, masks).present_price.tolist(), [False, True])
        with self.assertRaisesRegex(ValueError, "different candidate IDs"):
            align_presence(scores, masks.assign(rtable_id=["d", "x"]))
        with self.assertRaisesRegex(ValueError, "unique"):
            align_presence(scores, pd.concat([masks.iloc[:1]] * 2))

    def test_phi_reports_cross_tab_and_undefined_constant_margin(self):
        result = binary_association([1, 1, 0, 0], [1, 0, 0, 0])
        self.assertEqual([result[k] for k in ("n11", "n10", "n01", "n00")], [1, 1, 0, 2])
        self.assertAlmostEqual(result["phi"], 1 / np.sqrt(3))
        self.assertTrue(np.isnan(binary_association([0, 0], [0, 1])["phi"]))

    def test_controls_are_same_model_and_missing_cells_remain_missing(self):
        rows = []
        for model, delta in (("a", .001), ("b", .0004), ("c", -.001)):
            base = dict(dataset="d", side="left", k=1, role="selected", panel_name="single:" + model,
                        decoder="greedy", rule="pure_transitivity", trial_number=0,
                        study_name=model, n_gold=2)
            rows.extend([dict(base, score_name="llr", f1=.5 + delta),
                         dict(base, score_name="record_cos:" + model, f1=.5),
                         dict(base, score_name="record_cos:wrong", f1=.9)])
        frame = same_model_comparisons(pd.DataFrame(rows), [("d", "left", 1)], ["a", "b", "c", "d"])
        self.assertEqual(frame.outcome.iloc[:3].tolist(), ["win", "tie", "loss"])
        self.assertFalse(frame.available.iloc[3])

    def test_relocated_scores_and_numeric_tie_order_without_refitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data" / "fodors_zagats"
            data.mkdir(parents=True)
            pd.DataFrame({"id": ["z", "a"], "class": ["c", "d"]}).to_csv(data / "left.csv", index=False)
            pd.DataFrame({"id": ["r"], "class": ["c"]}).to_csv(data / "right.csv", index=False)
            pd.DataFrame({"left": ["z"], "right": ["r"]}).to_csv(data / "matches.csv", index=False)
            score_dir = root / "unsupervised_scores"
            score_dir.mkdir()
            scores = pd.DataFrame({"ltable_id": ["a", "z"], "rtable_id": ["r", "r"],
                "llr": [3., 2.], "p_match": [0., 0.], "eligible": [True, True],
                "attribute_cos:mean": [.8, 0.], "record_cos:" + MODEL: [.7, .8]})
            scores.to_parquet(score_dir / "saved.parquet", index=False)
            bank = root / "cosine_bank" / "fodors_zagats__right_k4"
            bank.mkdir(parents=True)
            scores.iloc[::-1][["ltable_id", "rtable_id"]].assign(rank=0,
                present_name=[False, True], present_addr=False, present_city=False,
                present_phone=False).to_parquet(bank / "pairs.parquet", index=False)
            trials = pd.DataFrame([dict(dataset="fodors_zagats", side="right", k=1, state="COMPLETE",
                phase="full", role="selected", panel_name="single:" + MODEL,
                trial_number=0, study_name="saved", scores_path="/app/old/saved.parquet", baseline=None)])
            reports = publication_diagnostics(root / "saved.db", trials, pd.DataFrame(), data_dir=root / "data")
            row = reports["primary_posterior_saturation"].iloc[0]
            self.assertEqual(row.posterior_zero, 2)
            self.assertEqual(row.accepted_mask_xor, 2)
            self.assertEqual(row.xor_at_zero, 2)
            self.assertEqual(row.posterior_f1, 1.)
            self.assertEqual(row.llr_f1, 0.)
            self.assertEqual(reports["primary_raw_mean"].iloc[0].f1, 1.)
            self.assertTrue(reports["fz_class_diagnostic"].iloc[0].exact_gold_reconstruction)
            self.assertFalse(reports["publication_diagnostic_availability"].available.all())

    def test_absent_inputs_report_unavailable_without_creating_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = publication_diagnostics(root / "missing.db", pd.DataFrame(), pd.DataFrame(), data_dir=root)
            self.assertFalse(reports["publication_diagnostic_availability"].available.any())
            self.assertEqual(list(root.iterdir()), [])

    def test_reference_comparison_requires_complete_matching_diagnostics(self):
        frames = {name: pd.DataFrame([dict({key: 1 if key == "k" else "sample" for key in keys}, metric=1.)])
                  for name, keys in REFERENCE_KEYS.items()}
        frames["publication_diagnostic_availability"] = pd.DataFrame([dict(available=True, diagnostic="all")])
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "reference.json"
            reference.write_text(json.dumps(dict(schema="zeroerpp-publication-diagnostics-v1",
                provenance=dict(database_sha256="test"),
                reports={name: frames[name].to_dict("records") for name in REFERENCE_KEYS})))
            self.assertTrue(compare_publication_diagnostics(frames, reference)["passed"])
            frames["primary_raw_mean"].loc[0, "metric"] = 2.
            with self.assertRaises(AssertionError):
                compare_publication_diagnostics(frames, reference)
            frames["publication_diagnostic_availability"].loc[0, "available"] = False
            with self.assertRaisesRegex(AssertionError, "unavailable"):
                compare_publication_diagnostics(frames, reference)


@unittest.skipUnless(os.environ.get("ZEROER_DIAGNOSTIC_ACCEPTANCE_DB"), "optional stored-run acceptance")
class PublicationAcceptanceTest(unittest.TestCase):
    def test_manuscript_diagnostics(self):
        from utils.report_tables import read_sources
        path = os.environ["ZEROER_DIAGNOSTIC_ACCEPTANCE_DB"]
        source = read_sources(path)
        reports = publication_diagnostics(path, source["trials"], source["decisions"])
        self.assertTrue(reports["publication_diagnostic_availability"].available.all())
        self.assertTrue(compare_publication_diagnostics(reports)["passed"])
        ordered = ["fodors_zagats", "dblp_acm", "dblp_scholar", "abt_buy", "amazon_googleproducts"]
        raw = reports["primary_raw_mean"].set_index("dataset").loc[ordered]
        np.testing.assert_allclose(raw.f1, [.681, .991, .598, .905, .634], atol=.0005, rtol=0)
        np.testing.assert_allclose(raw.fit_minus_raw_mean_f1,
                                   [-.0182, -.0009, -.0004, .0101, .0058], atol=.00005, rtol=0)
        primary = reports["primary_same_model_summary"].iloc[0]
        self.assertEqual((primary.wins, primary.ties, primary.losses), (42, 19, 39))
        union = reports["kalm_union_summary"].iloc[0]
        self.assertEqual((union.wins, union.ties, union.losses), (4, 6, 12))
        saturation = reports["primary_posterior_saturation"].set_index("dataset").loc["fodors_zagats"]
        self.assertEqual((saturation.posterior_zero, saturation.accepted_mask_xor, saturation.xor_at_zero), (72, 8, 8))
        presence = reports["primary_presence_diagnostics"].set_index(["dataset", "method", "indicator"])
        for dataset, attribute, phi in (("abt_buy", "price", (.787, -.036, -.10)),
                                       ("amazon_googleproducts", "manufacturer", (.716, .123, .01))):
            self.assertAlmostEqual(presence.loc[(dataset, "B-paper", attribute + "_present"), "phi"], phi[0], delta=.0005)
            self.assertAlmostEqual(presence.loc[(dataset, "B-paper", "gold"), "phi"], phi[1], delta=.0005)
            self.assertAlmostEqual(presence.loc[(dataset, "KaLM", attribute + "_present"), "phi"], phi[2], delta=.005)


if __name__ == "__main__":
    unittest.main()

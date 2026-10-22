"""Publication selection, refinement and replay contracts without large assets.

Run with ``python -m unittest discover -s tests`` in the pinned environment.
The deterministic matcher isolates experiment policy; the real transforms,
training adapter, SQLite studies, scores and decision exports remain in use.
"""
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib
import json

import numpy as np
import optuna
import pandas as pd

from data.features.panels import record_disjoint_splits
from exp import protocol, runner
from exp.refinement import run_refinement
from training import unsupervised
from utils.report_tables import read_sources
from utils.search_audit import compare_search_trials, search_trial_frame


class OptunaReproductionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fitted = []
        ids = pd.DataFrame({
            "ltable_id": ["l0", "l0", "l1", "l1", "l2", "l2", "l3", "l3"],
            "rtable_id": ["r0", "r1", "r0", "r1", "r2", "r3", "r2", "r3"]})
        bank = pd.DataFrame({"name_emb_cos__tiny": [.1, .2, .3, .4, .6, .7, .8, .9]})
        self.cell = dict(ids=ids,
                         pairs=np.array([[int(a[1:]), 4 + int(b[1:])] for a, b in ids.to_numpy()]),
                         bank=bank, present=pd.DataFrame({"name": [True] * 8}),
                         record_scores=pd.DataFrame(index=bank.index),
                         attributes=["name"], input_hash="fixture")
        self.splits = [dict(train=np.arange(3), validation=np.arange(3, 8),
                            excluded=np.array([], dtype=int), metadata={"fold": 0}),
                       dict(train=np.arange(3, 8), validation=np.arange(3),
                            excluded=np.array([], dtype=int), metadata={"fold": 1})]
        fitted = self.fitted

        class Matcher:
            def __init__(self, **config):
                self.config = config

            def match_pairs(self, pairs, features, present=None):
                fitted.append(dict(self.config))
                if self.config["c_bay"] == -.5 and features.iloc[:, 0].mean() > .5:
                    raise RuntimeError("bad second fold")
                width = features.shape[1]
                self.model = SimpleNamespace(
                    y=np.array([0, 1]), converged=True, n_iter=3,
                    kappas=np.zeros(width), Mu_M=np.ones(width), Mu_U=np.zeros(width),
                    feature_names=list(features), score_components=self.score)

            def score(self, features, present=None):
                p = features[:, 0]
                return dict(log_match=np.log(p), log_nonmatch=np.log1p(-p),
                            p_match=p, llr=np.log(p) - np.log1p(-p),
                            log_density=-self.config["c_bay"] - p,
                            n_blocks=present.sum(axis=1).to_numpy(), pi=.3)

        for target, value in (
                ("exp.runner.load_cell", lambda *args, **kwargs: self.cell),
                ("exp.runner._code_hash", lambda: "code-fixture"),
                ("exp.runner.record_disjoint_splits", lambda *args, **kwargs: self.splits),
                ("training.unsupervised.ZeroERMatcher", Matcher)):
            mock = patch(target, value)
            mock.start()
            self.addCleanup(mock.stop)
        self.args = dict(dataset="fixture", side="right", k=2,
                         panel=dict(name="single:tiny", models=["tiny"], fusion="separate",
                                    metrics=["cos"], include_record=False),
                         configs=protocol.CONFIGS, split_spec=dict(n_splits=2, seed=0),
                         storage="sqlite:///" + str(self.root / "search.db"))

    def test_six_plus_four_grid_fits_without_gold_and_resumes(self):
        with patch.object(runner, "_load_gold", side_effect=AssertionError("selection opened gold")):
            source = runner.run_search(**self.args)
            self.assertEqual(len(self.fitted), 6 * 2)
            refined = run_refinement(source, self.args["storage"], protocol.EXTRA_CONFIGS)
            self.assertEqual(len(self.fitted), 10 * 2)
            self.assertEqual(len(refined.trials), 10)
            for original, copied in zip(source.trials, refined.trials[:6]):
                self.assertEqual(copied.params, original.params)
                self.assertEqual(copied.value, original.value)
                self.assertEqual(copied.user_attrs["folds"], original.user_attrs["folds"])
                self.assertTrue(copied.user_attrs["copied"])
            self.assertEqual(runner.select_best_trial(refined).params["c_bay"], 0)
            runner.run_search(**self.args)
            resumed = run_refinement(source, self.args["storage"], protocol.EXTRA_CONFIGS)
            self.assertEqual(len(resumed.trials), 10)
            self.assertEqual(len(self.fitted), 10 * 2)

    def test_validation_is_pair_weighted_and_partial_failure_cannot_win(self):
        common = next(iter(protocol.CONFIGS.values()))
        configs = {"good": dict(common, c_bay=.1), "bad": dict(common, c_bay=-.5)}
        study = runner.run_search(**dict(self.args, configs=configs))
        expected = .1 + self.cell["bank"].iloc[:, 0].mean()
        self.assertAlmostEqual(study.best_value, expected)
        folds = study.trials[0].user_attrs["folds"]
        self.assertNotAlmostEqual(study.best_value, np.mean([row["nll"] for row in folds]))
        failed = study.trials[1]
        self.assertEqual(failed.state, optuna.trial.TrialState.FAIL)
        self.assertNotIn("error", failed.user_attrs["folds"][0])
        self.assertIn("error", failed.user_attrs["folds"][1])
        self.assertEqual(runner.select_best_trial(study).params, configs["good"])

    def test_refit_preserves_selection_exports_scores_and_resumes(self):
        study = runner.run_search(**self.args)
        selected = runner.select_best_trial(study)
        gold = np.array([False, True, True, False, False, True, True, False])

        def load_gold(dataset, ids):
            self.assertEqual(self.fitted[-1], selected.params)
            self.assertEqual(len(self.fitted), 6 * 2 + 1)
            return gold, 5  # One true pair lies outside the candidate set.

        with patch.object(runner, "_load_gold", side_effect=load_gold) as gold_reader:
            full = runner.run_refit(study, self.args["storage"], protocol.THRESHOLDS, self.root)
            replay = runner.run_refit(study, self.args["storage"], protocol.THRESHOLDS, self.root)
            self.assertEqual(len(replay.trials), 1)
            self.assertEqual(gold_reader.call_count, 1)
        trial = full.trials[0]
        self.assertEqual(trial.params, selected.params)
        self.assertEqual(trial.user_attrs["source_trial"], selected.number)
        self.assertEqual(trial.user_attrs["tp_blocking"], 4)
        self.assertEqual(trial.user_attrs["n_gold"], 5)
        scores = pd.read_parquet(trial.user_attrs["scores_path"])
        self.assertEqual(len(scores), 8)
        self.assertTrue({"llr", "p_match", "log_density", "eligible"}.issubset(scores))
        with sqlite3.connect(self.root / "search.db") as connection:
            count = connection.execute("SELECT count(*) FROM unsupervised_decisions").fetchone()[0]
            self.assertEqual(count, len(trial.user_attrs["decisions"]))
            curve = connection.execute("SELECT curve FROM unsupervised_thresholds LIMIT 1").fetchone()[0]
            self.assertEqual(json.loads(zlib.decompress(curve))[0]["threshold"], "+inf")
        with self.assertRaisesRegex(ValueError, "threshold policy"):
            runner.run_refit(study, self.args["storage"], dict(protocol.THRESHOLDS, llr=[0]), self.root)

    def test_changed_inputs_code_and_split_evidence_cannot_resume(self):
        source = runner.run_search(**self.args)
        count = len(self.fitted)
        with self.assertRaisesRegex(ValueError, "protocol"):
            runner.run_search(**dict(self.args, split_spec=dict(n_splits=2, seed=1)))
        with patch.object(runner, "_code_hash", return_value="changed-code"):
            with self.assertRaisesRegex(ValueError, "changed"):
                run_refinement(source, self.args["storage"], protocol.EXTRA_CONFIGS)
        self.cell["input_hash"] = "changed-input"
        with self.assertRaisesRegex(ValueError, "changed"):
            run_refinement(source, self.args["storage"], protocol.EXTRA_CONFIGS)
        self.cell["input_hash"] = "fixture"
        source.set_user_attr("splits", [])
        with self.assertRaisesRegex(ValueError, "changed"):
            run_refinement(source, self.args["storage"], protocol.EXTRA_CONFIGS)
        self.assertEqual(len(self.fitted), count)

    def test_real_split_has_no_shared_endpoints_and_is_order_independent(self):
        ids = pd.DataFrame([(str(i), str(j)) for i in range(25) for j in range(20)],
                           columns=["ltable_id", "rtable_id"])
        splits = record_disjoint_splits(ids, **protocol.SPLIT)
        reversed_ids = ids.iloc[::-1].reset_index(drop=True)
        reversed_splits = record_disjoint_splits(reversed_ids, **protocol.SPLIT)
        for split, reversed_split in zip(splits, reversed_splits):
            train, validation = ids.iloc[split["train"]], ids.iloc[split["validation"]]
            for column in ids:
                self.assertFalse(set(train[column]) & set(validation[column]))
            for partition in ("train", "validation", "excluded"):
                self.assertEqual(set(map(tuple, ids.iloc[split[partition]].to_numpy())),
                                 set(map(tuple, reversed_ids.iloc[reversed_split[partition]].to_numpy())))
            self.assertEqual(sum(len(split[key]) for key in ("train", "validation", "excluded")), len(ids))

    def test_nll_keeps_no_evidence_rows_and_rejects_nonfinite_values(self):
        scores = dict(log_density=np.array([-2., 0., -4.]))
        self.assertEqual(unsupervised.marginal_nll(scores), 2.)
        self.assertEqual(unsupervised.marginal_nll(scores, reduction="sum"), 6.)
        for values in ([], [np.nan], [np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                unsupervised.marginal_nll(dict(log_density=np.array(values)))

    def test_search_audit_checks_every_trial_and_fold(self):
        self.splits = [dict(train=np.flatnonzero(np.arange(8) % 3 != fold),
                            validation=np.flatnonzero(np.arange(8) % 3 == fold),
                            excluded=np.array([], dtype=int), metadata={"fold": fold})
                       for fold in range(3)]
        runner.run_search(**self.args)
        sources = read_sources(self.root / "search.db", experiment=protocol.EXPERIMENT)
        actual = search_trial_frame(sources["trials"])
        reference = self.root / "search.csv"
        actual.to_csv(reference, index=False, float_format="%.17g")
        self.assertEqual(compare_search_trials(actual, reference)["trials"], 6)
        with self.assertRaisesRegex(AssertionError, "coverage"):
            compare_search_trials(actual.iloc[:-1], reference)
        for column, value in (("state", "FAIL"), ("fold_1_nll", 1000.),
                              ("cv_validation_nll", 1000.), ("fold_metadata_json", "[]")):
            corrupted = actual.copy()
            corrupted.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(AssertionError):
                compare_search_trials(corrupted, reference)


if __name__ == "__main__":
    unittest.main()

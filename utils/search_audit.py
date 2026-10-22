"""Compare every publication search trial, including failed fits and CV folds."""
import json

import numpy as np
import pandas as pd


KEY = ["dataset", "side", "k", "panel_name", "config_hash"]
NUMERIC = ["cv_validation_nll"] + ["fold_{}_nll".format(i) for i in range(3)]
FOLD_FIELDS = ["seed", "fold", "n_splits", "n_pairs", "n_train", "n_validation", "n_excluded",
               "train_left_records", "train_right_records", "validation_left_records",
               "validation_right_records", "excluded_left_records", "excluded_right_records",
               "n_features", "n_iter", "converged"]
EXACT = ["trial_number", "state", "params_json", "copied", "n_errors", "n_validation",
         "n_unconverged_folds", "fold_metadata_json"]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def search_trial_frame(trials):
    """Normalize ``read_sources()['trials']`` into one row per grid setting.

    Timing measurements and platform-specific exception text are excluded.
    Fold sizes, observed-feature counts and convergence remain exact evidence;
    objectives are separate numeric columns to permit numerical tolerances.
    """
    rows = []
    for trial in trials.loc[trials.phase.eq("search")].to_dict("records") if not trials.empty else []:
        folds = json.loads(trial["folds_json"] or "[]")
        if len(folds) != 3:
            raise ValueError("Publication search trials require three recorded folds")
        row = {key: trial[key] for key in KEY + EXACT if key != "fold_metadata_json"}
        params = json.loads(trial["params_json"])
        # SQLite's JSON extraction represents categorical booleans as 0/1.
        for key in ("hard", "transitivity"):
            params[key] = bool(params[key])
        row["params_json"] = _json(params)
        row["fold_metadata_json"] = _json([
            dict({key: fold.get(key) for key in FOLD_FIELDS}, failed="error" in fold)
            for fold in folds])
        row["cv_validation_nll"] = trial["cv_validation_nll"]
        row.update({"fold_{}_nll".format(i): fold.get("nll", np.nan)
                    for i, fold in enumerate(folds)})
        rows.append(row)
    frame = pd.DataFrame(rows, columns=KEY + EXACT + NUMERIC)
    if frame.duplicated(KEY).any():
        raise ValueError("Publication search contains duplicate configurations")
    return frame.sort_values(KEY).reset_index(drop=True)


def compare_search_trials(actual, reference_path):
    """Require the full grid and compare settings, fold evidence and NLLs.

    Finite objective differences permit rtol=1e-12 and atol=1e-10; missing values
    must coincide. A partial run cannot pass this complete-replication check.
    """
    expected = pd.read_csv(reference_path, keep_default_na=False, float_precision="round_trip")
    if len(actual) != len(expected):
        raise AssertionError("Search trial coverage: {} rows, expected {}".format(len(actual), len(expected)))
    actual, expected = [frame.sort_values(KEY).reset_index(drop=True) for frame in (actual, expected)]
    pd.testing.assert_frame_equal(actual[KEY + EXACT].fillna(""), expected[KEY + EXACT].fillna(""),
                                  check_dtype=False, check_exact=True)
    errors = []
    for column in NUMERIC:
        left = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-10, equal_nan=True,
                                   err_msg="Search objective differs: " + column)
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            errors.append(float(np.abs(left[finite] - right[finite]).max()))
    return dict(passed=True, trials=len(actual), studies=len(actual[KEY[:4]].drop_duplicates()),
                failed_trials=int(actual.state.eq("FAIL").sum()),
                complete_trials_with_unconverged_folds=int(
                    (actual.state.eq("COMPLETE") & actual.n_unconverged_folds.gt(0)).sum()),
                max_absolute_nll_difference=max(errors, default=0.), rtol=1e-12, atol=1e-10)

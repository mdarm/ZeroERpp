"""Export one explicitly named tuning workflow without publication assumptions.

The database is opened read-only. Configuration choices compare held-out NLL
only inside each search study; gold-based decision metrics remain separate.
"""
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _readable(value):
    """Keep invalid recorded numbers visible without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return "non-finite: " + str(value)
    if isinstance(value, dict):
        return {str(key): _readable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_readable(item) for item in value]
    return value


def _database(storage):
    value = str(storage)
    if value.startswith("sqlite:"):
        from sqlalchemy.engine import make_url

        url = make_url(value)
        if url.drivername != "sqlite" or not url.database or url.database == ":memory:":
            raise ValueError("Tuning reports require a file-backed SQLite database")
        value = url.database
    elif "://" in value:
        raise ValueError("Tuning reports require a file-backed SQLite database")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _metadata(connection):
    studies = {}
    for row in connection.execute("SELECT study_id, study_name FROM studies ORDER BY study_name"):
        studies[row[0]] = dict(id=row[0], name=row[1], attrs={}, directions=[])
    for row in connection.execute("SELECT study_id, key, value_json FROM study_user_attributes"):
        studies[row[0]]["attrs"][row[1]] = json.loads(row[2])
    for row in connection.execute("SELECT study_id, direction FROM study_directions ORDER BY objective"):
        studies[row[0]]["directions"].append(row[1])
    return {study["name"]: study for study in studies.values()}


def _load_trials(connection, studies):
    """Read requested study IDs in bounded batches, including typed parameters."""
    by_id, result = {}, {study["name"]: [] for study in studies}
    ids = sorted({study["id"] for study in studies})
    names = {study["id"]: study["name"] for study in studies}
    for start in range(0, len(ids), 400):
        batch = ids[start:start + 400]
        placeholders = ",".join("?" for _ in batch)
        trial_filter = "SELECT trial_id FROM trials WHERE study_id IN ({})".format(placeholders)
        for row in connection.execute("""SELECT t.trial_id, t.study_id, t.number, t.state,
                t.datetime_start, t.datetime_complete, v.value, v.value_type
                FROM trials t LEFT JOIN trial_values v ON v.trial_id=t.trial_id AND v.objective=0
                WHERE t.study_id IN ({}) ORDER BY t.study_id, t.number""".format(placeholders), batch):
            value = row[6]
            if row[7] in ("INF_POS", "INF_NEG"):
                value = math.inf if row[7] == "INF_POS" else -math.inf
            trial = dict(number=row[2], state=row[3], datetime_start=row[4],
                         datetime_complete=row[5], value=value, params={}, attrs={})
            by_id[row[0]] = trial
            result[names[row[1]]].append(trial)
        for row in connection.execute("SELECT trial_id, key, value_json FROM trial_user_attributes "
                                      "WHERE trial_id IN ({})".format(trial_filter), batch):
            by_id[row[0]]["attrs"][row[1]] = json.loads(row[2])
        for row in connection.execute("SELECT trial_id, param_name, param_value, distribution_json "
                                      "FROM trial_params WHERE trial_id IN ({})".format(trial_filter), batch):
            distribution = json.loads(row[3])
            name, value = distribution["name"], row[2]
            if name == "CategoricalDistribution":
                value = distribution["attributes"]["choices"][int(value)]
            elif name.startswith("Int"):
                value = int(value)
            by_id[row[0]]["params"][row[1]] = value
    return result


def _source_studies(metadata, names):
    sources = {}
    for name in names:
        if name not in metadata:
            raise ValueError("Requested tuning study does not exist: " + name)
        study = metadata[name]
        attrs = study["attrs"]
        if (attrs.get("phase") != "search" or attrs.get("objective") != "validation_nll"
                or study["directions"] != ["MINIMIZE"]):
            raise ValueError("Expected a search study minimizing validation NLL: " + name)
        spec = attrs.get("spec", {})
        if not spec.get("configs") or spec.get("experiment") != attrs.get("experiment"):
            raise ValueError("Missing or inconsistent tuning specification: " + name)
        sources[name] = study
    return sources


def _related_refits(metadata, sources):
    """Require source provenance as well as namespace, including legacy refits."""
    related = {}
    for name, study in metadata.items():
        attrs = study["attrs"]
        if attrs.get("phase") != "full" or attrs.get("excluded") or attrs.get("role") == "excluded":
            continue
        explicit = attrs.get("source_study_name", attrs.get("source_study"))
        candidates = [explicit] if explicit in sources else []
        if explicit is None:
            selection = attrs.get("selection", "cv_nll")
            suffix = "_full" if selection == "cv_nll" and not attrs.get("policy", {}).get("config_override") else (
                "_{}_full".format(re.sub(r"[^A-Za-z0-9_.-]+", "-", selection)))
            candidates = [source for source in sources if name == source + suffix]
        for source in candidates:
            source_attrs = sources[source]["attrs"]
            if (attrs.get("experiment") != source_attrs["experiment"]
                    or attrs.get("spec") != source_attrs["spec"]
                    or attrs.get("policy", {}).get("source_protocol") != source_attrs.get("protocol_hash")):
                raise ValueError("Refit source provenance does not match requested study: " + name)
            related[name] = source
    return related


def _identity(study):
    attrs = study["attrs"]
    return dict(study_name=study["name"], experiment=attrs["experiment"],
                dataset=attrs.get("dataset"), side=attrs.get("side"), k=attrs.get("k"),
                panel_name=attrs.get("panel_name"))


def _selection(study, trials):
    attrs, groups = study["attrs"], {}
    configs = attrs["spec"]["configs"]
    if not isinstance(configs, dict):
        raise ValueError("Tuning configuration metadata must use named configurations")
    for name, config in configs.items():
        fingerprint = sha256(_json(config).encode()).hexdigest()
        if fingerprint in groups:
            raise ValueError("Declared tuning configurations must be unique: " + study["name"])
        groups[fingerprint] = dict(name=name, config=config, trials=[])
    for trial in trials:
        fingerprint = sha256(_json(trial["params"]).encode()).hexdigest()
        if fingerprint not in groups or trial["attrs"].get("config_hash") != fingerprint:
            raise ValueError("Trial parameters differ from the declared tuning design: " + study["name"])
        groups[fingerprint]["trials"].append(trial)
    valid = [trial for trial in trials if trial["state"] == "COMPLETE" and _finite(trial["value"])]
    chosen = min(valid, key=lambda trial: (trial["value"], trial["number"])) if valid else None
    completed = sum(any(t["state"] == "COMPLETE" and _finite(t["value"]) for t in group["trials"])
                    for group in groups.values())
    terminal = sum(bool(group["trials"]) and all(t["state"] not in {"RUNNING", "WAITING"}
                                                for t in group["trials"]) for group in groups.values())
    finished = terminal == len(groups)
    status = ("selected" if finished else "provisional") if chosen else (
        "no_valid_configuration" if finished else "no_valid_configuration_yet")
    row = dict(_identity(study), expected_configurations=len(groups),
               attempted_configurations=sum(bool(group["trials"]) for group in groups.values()),
               terminal_configurations=terminal, complete_configurations=completed,
               failed_configurations=sum(bool(group["trials"]) and all(
                   t["state"] in {"FAIL", "PRUNED"} or (t["state"] == "COMPLETE" and not _finite(t["value"]))
                   for t in group["trials"]) for group in groups.values()),
               pending_configurations=len(groups) - terminal, trial_count=len(trials),
               search_status="complete" if finished else "partial", selection_status=status,
               selected_trial=chosen["number"] if chosen else None,
               validation_nll=chosen["value"] if chosen else None,
               selection_uses_gold=False, protocol_hash=attrs.get("protocol_hash"),
               refinement_source_study=attrs.get("refinement_source_study"))
    selection = dict(row, config_name=groups[chosen["attrs"]["config_hash"]]["name"] if chosen else None,
                     config_hash=chosen["attrs"]["config_hash"] if chosen else None,
                     config=chosen["params"] if chosen else None, panel=attrs["spec"].get("panel"),
                     split_spec=attrs["spec"].get("split_spec"), input_hash=attrs["spec"].get("input_hash"),
                     code_hash=attrs["spec"].get("code_hash"),
                     objective="validation_nll", comparison_scope="within this study only",
                     tie_break="lowest trial number")
    return row, selection


def _decision_rows(connection, related, trials):
    table_exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                      "AND name='unsupervised_decisions'").fetchone()
    rows = []
    for name in sorted(related):
        records = [dict(row) for row in connection.execute(
            "SELECT * FROM unsupervised_decisions WHERE study_name=? ORDER BY trial_number, row_number", (name,))
            ] if table_exists else []
        present = {row["trial_number"] for row in records}
        for trial in trials[name]:
            if trial["number"] not in present:
                for number, decision in enumerate(trial["attrs"].get("decisions", [])):
                    record = dict(decision, study_name=name, trial_number=trial["number"], row_number=number)
                    record["score_name"] = record.pop("score", record.get("score_name"))
                    record["threshold_json"] = _json(record.pop("threshold", None))
                    records.append(record)
        for record in records:
            record.update(source_study=related[name], metric_uses_gold=True,
                          threshold_uses_gold=bool(record.get("uses_gold")),
                          decision_role="oracle_diagnostic" if record.get("uses_gold") else "fixed_rule")
            rows.append(record)
    return rows


def _write_csv(path, rows, first):
    columns = list(dict.fromkeys(first + sorted({key for row in rows for key in row})))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row = _readable(row)
            writer.writerow({key: _json(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def export_tuning_results(storage, study_names, output_dir):
    """Write inspectable results for explicit search/refined names and their refits.

    A partial search has a provisional candidate, never a final selection. No
    expectations are inferred for unrequested cells, panels, refits or ablations.
    Empty evaluation tables mean evaluation has not been recorded for these fits.
    """
    if isinstance(study_names, str):
        study_names = [study_names]
    names = sorted(set(study_names))
    database = _database(storage)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=120) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        metadata = _metadata(connection)
        sources = _source_studies(metadata, names)
        related = _related_refits(metadata, sources)
        included = list(sources.values()) + [metadata[name] for name in sorted(related)]
        trials = _load_trials(connection, included)
        decisions = _decision_rows(connection, related, trials)

    study_rows, selections, trial_rows, folds, failures, refits, evaluations = [], [], [], [], [], [], []
    for name, study in sources.items():
        study_row, selected = _selection(study, trials[name])
        study_rows.append(study_row)
        selections.append(selected)
        for trial in trials[name]:
            attrs = trial["attrs"]
            common = dict(_identity(study), trial_number=trial["number"], state=trial["state"],
                          config_name=attrs.get("config_name"), config_hash=attrs.get("config_hash"),
                          copied=bool(attrs.get("copied", False)))
            trial_rows.append(dict(common, config=trial["params"],
                validation_nll=trial["value"], selected_within_study=trial["number"] == selected["selected_trial"],
                selection_status=selected["selection_status"], selection_uses_gold=False,
                datetime_start=trial["datetime_start"], datetime_complete=trial["datetime_complete"],
                **{key: attrs.get(key) for key in ("n_errors", "n_validation", "nll_sum", "seconds",
                   "source_study", "source_trial", "source_experiment", "source_protocol_hash")}))
            for fold in attrs.get("folds", []):
                folds.append(dict(fold, **common))
                if "error" in fold:
                    failures.append(dict(common, stage="search", fold=fold.get("fold"), error=fold["error"]))
            if attrs.get("error") or (trial["state"] == "FAIL" and not any("error" in f for f in attrs.get("folds", []))):
                failures.append(dict(common, stage="search", error=attrs.get("error", "Failed trial; no error detail recorded")))
            if trial["state"] == "COMPLETE" and not _finite(trial["value"]):
                failures.append(dict(common, stage="search", error="Completed trial has no finite validation NLL"))

    decision_counts = {}
    for row in decisions:
        key = (row["study_name"], row["trial_number"])
        decision_counts[key] = decision_counts.get(key, 0) + 1
    gold_keys = ("n_gold", "tp_blocking", "brier", "label_log_loss", "calibration_population")
    for name, source in sorted(related.items()):
        study, study_attrs = metadata[name], metadata[name]["attrs"]
        for trial in trials[name]:
            attrs = trial["attrs"]
            if attrs.get("source_study", source) != source:
                raise ValueError("Refit trial references another source study: " + name)
            count = decision_counts.get((name, trial["number"]), 0)
            recorded = bool(count or any(key in attrs for key in gold_keys))
            requested = study_attrs.get("evaluate", attrs.get("evaluate", recorded))
            if recorded and requested is False:
                raise ValueError("Unevaluated refit contains gold-based results: " + name)
            identity = dict(_identity(study), trial_number=trial["number"], state=trial["state"],
                            source_study=source, source_trial=attrs.get("source_trial"),
                            role=study_attrs.get("role"), selection=study_attrs.get("selection"))
            diagnostics = {key: value for key, value in attrs.items() if key not in set(gold_keys) | {
                "decisions", "source_study", "source_trial", "config_hash", "validation_nll",
                "scores_path", "predictor_path", "predictor_sha256", "evaluate", "save_predictor"}}
            refits.append(dict(identity, config=trial["params"], config_hash=attrs.get("config_hash"),
                source_validation_nll=attrs.get("validation_nll"), evaluation_requested=bool(requested),
                evaluation_status="recorded" if recorded else ("not_recorded" if requested else "not_requested"),
                decision_rows=count, fitting_uses_gold=False, diagnostics=diagnostics,
                **{key: attrs.get(key) for key in ("converged", "n_iter", "n_features", "candset_size",
                    "n_no_features", "pi", "fit_seconds", "score_seconds", "scores_path",
                    "predictor_path", "predictor_sha256")}))
            if recorded:
                evaluations.append(dict(identity, metric_uses_gold=True, decision_rows=count,
                    **{key: attrs[key] for key in ("candset_size",) + gold_keys if key in attrs}))
            if trial["state"] == "FAIL" or attrs.get("error"):
                failures.append(dict(identity, stage="refit", error=attrs.get("error", "Failed refit; no error detail recorded")))

    known_trials = {(row["study_name"], row["trial_number"]) for row in refits}
    if any((row["study_name"], row["trial_number"]) not in known_trials for row in decisions):
        raise ValueError("Decision rows refer to an unrecorded refit trial")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"studies.csv": study_rows, "trial_configs.csv": trial_rows, "fold_metrics.csv": folds,
              "failures.csv": failures, "refits.csv": refits, "evaluation.csv": evaluations,
              "decisions.csv": decisions}
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows, ["study_name", "trial_number"] if filename != "studies.csv" else ["study_name"])
    documents = {"selected_configs.json": selections,
                 "study_specs.json": [dict(study_name=study["name"], **study["attrs"]) for study in included]}
    for filename, value in documents.items():
        (output_dir / filename).write_text(json.dumps(_readable(value), indent=2, sort_keys=True,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    summary = dict(schema_version=1, kind="custom_tuning_results", database=str(database),
        requested_study_names=names, status="exported" if names else "no_studies_requested",
        selection_objective="validation_nll", selection_comparison="within each study only",
        selection_uses_gold=False, fitting_uses_gold=False,
        search_studies=len(sources), finished_search_studies=sum(row["search_status"] == "complete" for row in study_rows),
        selected_configurations=sum(row["selection_status"] == "selected" for row in study_rows),
        provisional_configurations=sum(row["selection_status"] == "provisional" for row in study_rows),
        searches_without_valid_configuration=sum(row["selected_trial"] is None for row in study_rows),
        discovered_refit_studies=len(related), refit_trials=len(refits),
        completed_refit_trials=sum(row["state"] == "COMPLETE" for row in refits),
        failed_refit_trials=sum(row["state"] == "FAIL" for row in refits),
        evaluated_refit_trials=len(evaluations), decision_rows=len(decisions),
        oracle_decision_rows=sum(row["threshold_uses_gold"] for row in decisions),
        files={filename: str(output_dir / filename) for filename in sorted(set(tables) | set(documents) | {"summary.json"})})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True,
                                                      allow_nan=False) + "\n", encoding="utf-8")
    return summary

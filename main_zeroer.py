"""Publication replication, editable unsupervised tuning, and saved prediction.

The publication ``match`` command keeps the frozen protocol. For new searches,
edit configs/unsupervised.json and use ``tune``; its stages expose the complete
search -> refinement -> NLL selection -> refit -> prediction/evaluation path.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def _parser():
    common = argparse.ArgumentParser(add_help=False)
    # Suppressed defaults let the same options work before or after a command.
    common.add_argument("--data-dir", default=argparse.SUPPRESS,
                        help="raw benchmark CSV root, containing embeddings/ (default: data/benchmarks)")
    common.add_argument("--output-dir", default=argparse.SUPPRESS,
                        help="generated caches, databases and reports (default: results/runs)")
    common.add_argument("--threads", type=int, default=argparse.SUPPRESS,
                        help="BLAS/OpenMP threads per process (default: 1)")
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                        help="print the resolved design without reading inputs or running experiments")
    parser = argparse.ArgumentParser(description=__doc__, parents=[common])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text, scope=False, models=False, panels=False):
        sub = commands.add_parser(name, help=help_text, parents=[common])
        if scope:
            sub.add_argument("--scope", choices=("primary", "sensitivity", "all"), default="all",
                             help="five primary, nineteen sensitivity or their 22-cell union")
            sub.add_argument("--datasets", nargs="+", help="restrict to named benchmark datasets")
        if models:
            sub.add_argument("--models", nargs="+", help="restrict to named frozen encoder stems")
        if panels:
            sub.add_argument("--panels", nargs="+", help="restrict to exact panel names, e.g. single:minilm6")
        return sub

    download = command("download", "download pinned raw data and embedding assets", scope=True, models=True)
    download.add_argument("--kind", choices=("all", "raw", "records", "attributes"), default="all")
    verify = command("verify-data", "verify required raw inputs against their manifests", scope=True, models=True)
    verify.add_argument("--kind", choices=("all", "raw", "records", "attributes"), default="all")
    verify.add_argument("--raw-only", action="store_true", help="verify CSVs without requiring embeddings")
    prepare = command("prepare", "regenerate candidate, similarity and string-feature caches", scope=True,
                      models=True, panels=True)
    prepare.add_argument("--workers", type=int, default=1, help="string-feature extraction workers")
    prepare.add_argument("--config", help="prepare only the embedding inputs of a custom tuning JSON design")
    blocking = command("blocking", "run the frozen record-embedding blocking grid", models=True)
    blocking.add_argument("--datasets", nargs="+")
    blocking.add_argument("--k-max", type=int, default=100)
    blocking.add_argument("--directions", nargs="+", choices=("right", "left"), default=["right", "left"])
    match = command("match", "run six initial configs, four refinements, refits and ablations",
                    scope=True, panels=True)
    match.add_argument("--workers", type=int, default=1, help="independent cell/panel workers")
    match.add_argument("--no-ablations", action="store_true", help="omit MiniLM6 zero-fill/hard-EM ablations")
    tune = command("tune", "run an editable unsupervised grid and its post-selection stages")
    tune.add_argument("--config", required=True, help="JSON design, e.g. configs/unsupervised.json")
    tune.add_argument("--stage", choices=("search", "refine", "refit", "report", "all"), default="all",
                      help="run a stage explicitly, or the full search/refinement/refit chain")
    tune.add_argument("--workers", type=int, default=1, help="independent cell/panel workers")
    tune.add_argument("--prepare", action="store_true", help="prepare candidate/similarity inputs first")
    tune.add_argument("--evaluate", action="store_true",
                      help="after refitting, read gold for metrics/oracles; never used for NLL selection")
    tune.epilog = ("Edit the JSON grids, then run tune --config configs/unsupervised.json --prepare. "
                   "Stages can also run individually in order: search, refine, refit. "
                   "Each stage exports trials and selections under OUTPUT_DIR/tuning/EXPERIMENT. "
                   "Set --evaluate on the first refit if metrics are wanted. "
                   "Changed designs or refit policies require a new experiment/output directory.")
    predict = command("predict", "restore a fitted predictor and decode candidates without labels")
    predict.add_argument("--predictor", required=True, help="saved predictor .pkl from a tuning refit")
    predict.add_argument("--dataset", required=True)
    predict.add_argument("--side", choices=("left", "right"), required=True)
    predict.add_argument("--k", type=int, required=True)
    predict.add_argument("--score", choices=("llr", "posterior"), default="llr")
    predict.add_argument("--decoder", choices=("threshold", "greedy"), default="greedy")
    predict.add_argument("--threshold", type=float, help="inclusive cut (default: LLR 0, posterior 0.5)")
    predict.add_argument("--no-cut", action="store_true", help="greedy matching without a score cut")
    predict.add_argument("--predictions", help="output Parquet path (default: OUTPUT_DIR/predictions.parquet)")
    command("baselines", "fit original B-paper and B-code on identical candidates", scope=True)
    timings = command("timings", "rerun the published serial timing protocol", scope=True, panels=True)
    timings.add_argument("--kind", choices=("published", "selected", "cosine", "strings"), default="published",
                         help="strings measures only the published Fodors-Zagats right/k1 cell")
    timings.add_argument("--repeats", type=int, default=3, help="selected-fit and direct-cosine repetitions")
    timings.add_argument("--string-repeats", type=int, default=1)
    report = command("report", "export tables and compare generated results to the frozen references")
    report.add_argument("--compare-reference", action="store_true")
    report.add_argument("--blocking-only", action="store_true")
    return parser


def _design(args):
    from exp import config, protocol
    if args.command == "tune" or (args.command == "prepare" and args.config):
        from exp.tuning_config import load_tuning_plan
        if args.command == "prepare" and (args.datasets or args.models or args.panels or args.scope != "all"):
            raise ValueError("With --config, candidate cells and panels come from the JSON design")
        plan = load_tuning_plan(args.config)
        args.tuning_plan = plan
        if args.command == "tune":
            if args.stage == "refine" and not plan.refinements:
                raise ValueError("The design has no refinement grid")
            if args.evaluate and args.stage not in ("all", "refit"):
                raise ValueError("--evaluate applies only to the refit/all stage")
        return dict(command=args.command, data_dir=config.BENCH_DIR, output_dir=config.OUTPUT_DIR,
                    **plan.to_dict(), models=plan.models, selected_studies=len(plan.cells) * len(plan.panels),
                    initial_configurations=len(plan.configurations), additional_configurations=len(plan.refinements))
    if args.command == "predict":
        import math
        if args.dataset not in protocol.DATASETS or args.k < 1:
            raise ValueError("Prediction requires a known dataset and a positive k")
        if args.no_cut and args.decoder != "greedy":
            raise ValueError("--no-cut requires greedy decoding")
        if args.threshold is None:
            args.threshold = 0.0 if args.score == "llr" else 0.5
        if not math.isfinite(args.threshold):
            raise ValueError("Prediction threshold must be finite; use --no-cut for greedy without a cut")
        return dict(command="predict", data_dir=config.BENCH_DIR, output_dir=config.OUTPUT_DIR,
                    dataset=args.dataset, side=args.side, k=args.k, predictor=args.predictor,
                    score=args.score, decoder=args.decoder, threshold=None if args.no_cut else args.threshold)
    datasets = getattr(args, "datasets", None) or protocol.DATASETS
    unknown = set(datasets) - set(protocol.DATASETS)
    if unknown:
        raise ValueError("Unknown datasets: " + ", ".join(sorted(unknown)))
    if len(set(datasets)) != len(datasets):
        raise ValueError("Datasets must be distinct")
    if len(set(getattr(args, "directions", []))) != len(getattr(args, "directions", [])):
        raise ValueError("Blocking directions must be distinct")
    cells = protocol.select_cells(getattr(args, "scope", "all"), datasets)
    panels = protocol.select_panels(getattr(args, "panels", None))
    models = getattr(args, "models", None)
    if models is None:
        models = protocol.panel_models(getattr(args, "panels", None))
    if not models or len(set(models)) != len(models) or set(models) - set(protocol.MODELS):
        raise ValueError("Models must be distinct members of the frozen twenty-encoder pool")
    if getattr(args, "models", None) and getattr(args, "panels", None):
        raise ValueError("Choose --models or --panels, not both")
    return dict(command=args.command, data_dir=config.BENCH_DIR, output_dir=config.OUTPUT_DIR,
                datasets=list(datasets), cells=cells, panels=[p["name"] for p in panels], models=models,
                split=protocol.SPLIT, initial_configurations=len(protocol.CONFIGS),
                additional_configurations=len(protocol.EXTRA_CONFIGS), thresholds=protocol.THRESHOLDS,
                selected_studies=len(cells) * len(panels),
                blocking_trials=len(datasets) * len(models) * getattr(args, "k_max", 100)
                                * len(getattr(args, "directions", ["right", "left"])))


def _save_summary(name, payload):
    from exp.config import OUTPUT_DIR
    payload = dict(payload, recorded_at=datetime.now(timezone.utc).isoformat())
    target = Path(OUTPUT_DIR) / (name + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _match_one(cell, panel, ablations=True):
    import contextlib
    from exp import config, protocol
    from exp.runner import run_search, run_refit
    from exp.refinement import run_refinement
    dataset, side, k = cell
    logs = Path(config.OUTPUT_DIR) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    label = "{}__{}_k{}__{}".format(dataset, side, k, panel["name"]).replace("/", "-")
    log_path = logs / (label + ".log")
    scores = Path(config.OUTPUT_DIR) / "unsupervised_scores"
    with log_path.open("a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        source = run_search(dataset, side, k, panel, protocol.CONFIGS, protocol.SPLIT,
                            config.UNSUPERVISED_STORAGE, experiment=protocol.EXPERIMENT)
        refined = run_refinement(source, config.UNSUPERVISED_STORAGE, protocol.EXTRA_CONFIGS,
                                 experiment=protocol.FINAL_EXPERIMENT)
        counts = Counter(t.state.name for t in refined.trials)
        result = dict(dataset=dataset, side=side, k=k, panel=panel["name"],
                      study=refined.study_name, search_states=dict(counts), log=str(log_path))
        if counts.get("COMPLETE", 0):
            full = run_refit(refined, config.UNSUPERVISED_STORAGE, protocol.THRESHOLDS, scores)
            result["refit"] = full.trials[-1].state.name
            if result["refit"] == "FAIL":
                result["refit_error"] = full.trials[-1].user_attrs.get("error")
            if ablations and panel["name"] == "single:minilm6":
                result["ablations"] = {}
                for selection, override in protocol.ABLATIONS.items():
                    arm = run_refit(refined, config.UNSUPERVISED_STORAGE, protocol.THRESHOLDS,
                                    scores, config_override=override, selection=selection)
                    result["ablations"][selection] = arm.trials[-1].state.name
        else:
            result["refit"] = "UNAVAILABLE: all search configurations failed"
    print("{} {}/{} {}: {}".format(dataset, side, k, panel["name"], result["refit"]), flush=True)
    return result


def _run_match(args, design):
    from exp import config, protocol
    from exp.runner import _storage
    from joblib import Parallel, delayed
    _storage(config.UNSUPERVISED_STORAGE)
    panels = protocol.select_panels(args.panels)
    results = Parallel(n_jobs=args.workers)(
        delayed(_match_one)(cell, panel, not args.no_ablations)
        for cell in design["cells"] for panel in panels)
    states = Counter(row["refit"] for row in results)
    failures = [row for row in results if row["refit"] != "COMPLETE"]
    failed_ablations = [dict(study=row["study"], ablation=name, state=state)
                        for row in results for name, state in row.get("ablations", {}).items()
                        if state != "COMPLETE"]
    failed_searches = sum(row["search_states"].get("FAIL", 0) for row in results)
    _save_summary("matching_run_summary", dict(
        design=design, selected_refit_states=dict(states), failed_or_unavailable=failures,
        search_failed_trials=failed_searches, failed_ablations=failed_ablations, results=results,
        status="finished_with_failed_fits" if failures or failed_searches or failed_ablations else "finished"))


def _run_baselines(args, design):
    from exp import config, protocol
    from exp.baselines import run_baselines
    rows = []
    for dataset, side, k in design["cells"]:
        study = run_baselines(dataset, side, k, config.UNSUPERVISED_STORAGE,
                              Path(config.OUTPUT_DIR) / "unsupervised_scores",
                              thresholds=protocol.THRESHOLDS["posterior"],
                              experiment=protocol.BASELINE_EXPERIMENT)
        rows.extend(dict(study=study.study_name, trial=t.number,
                         baseline=t.user_attrs.get("baseline"), state=t.state.name,
                         error=t.user_attrs.get("error")) for t in study.trials)
    failures = [row for row in rows if row["state"] != "COMPLETE"]
    _save_summary("baseline_run_summary", dict(design=design, trials=rows,
                  states=dict(Counter(row["state"] for row in rows)),
                  status="finished_with_failed_fits" if failures else "finished"))


def _run_tuning(args):
    from data.prepare import prepare_inputs
    from exp import config
    from exp.tuning import run_tuning
    plan = args.tuning_plan
    if args.prepare:
        prepare_inputs(cells=plan.cells, models=plan.models, include_strings=False)
    result = run_tuning(plan, config.UNSUPERVISED_STORAGE, config.OUTPUT_DIR,
                        stage=args.stage, workers=args.workers, evaluate=args.evaluate)
    print(json.dumps(result, indent=2, sort_keys=True))


def _run_prediction(args):
    from data.features.panels import load_cell
    from exp.config import OUTPUT_DIR
    from models.estimator import ZeroEREstimator
    predictor = ZeroEREstimator.load(args.predictor)
    panel = predictor.panel
    cell = load_cell(args.dataset, args.side, args.k, panel["models"],
                     metrics=panel.get("metrics", ["cos"]), include_record=panel.get("include_record", True))
    scores = predictor.score(cell["bank"], cell["present"])
    keep = predictor.predict(cell["pairs"], cell["bank"], cell["present"],
                             score=args.score, decoder=args.decoder,
                             threshold=None if args.no_cut else args.threshold)
    output = cell["ids"].assign(llr=scores["llr"], p_match=scores["p_match"], match=keep)
    path = Path(args.predictions) if args.predictions else Path(OUTPUT_DIR) / "predictions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False)
    print(json.dumps(dict(predictions=str(path), candidates=len(output), matches=int(keep.sum()),
                          labels_used=False, model_refitted=False), indent=2))


def _run_timings(args, design):
    import optuna
    from exp import config, protocol
    from exp.runner import _storage
    from exp.timing import benchmark_study
    from exp.cosine_timing import benchmark_cosine_cell
    from exp.baseline_timing import benchmark_baseline_cell
    selected, skipped = [], []
    cells, panel_names = set(map(tuple, design["cells"])), set(design["panels"])
    if args.kind in ("published", "selected"):
        for summary in optuna.get_all_study_summaries(storage=_storage(config.UNSUPERVISED_STORAGE)):
            attrs = summary.user_attrs
            if (attrs.get("experiment") != protocol.FINAL_EXPERIMENT or attrs.get("phase") != "full"
                    or attrs.get("role") != "selected" or attrs.get("excluded")
                    or attrs.get("panel_name") not in panel_names
                    or (attrs.get("dataset"), attrs.get("side"), attrs.get("k")) not in cells):
                continue
            if summary.best_trial is None:
                skipped.append(summary.study_name)
            else:
                benchmark_study(summary.study_name, config.UNSUPERVISED_STORAGE, repeats=args.repeats)
                selected.append(summary.study_name)
        if not selected:
            raise ValueError("No completed selected refits to time; run match for this scope first")
    cosine = []
    if args.kind in ("published", "cosine"):
        for dataset, side, k in design["cells"]:
            benchmark_cosine_cell(dataset, side, k, design["models"], config.UNSUPERVISED_STORAGE,
                                  protocol.THRESHOLDS["cosine"], repeats=args.repeats)
            cosine.append((dataset, side, k))
    strings = []
    string_cell = ("fodors_zagats", "right", 1)
    if args.kind == "strings" and string_cell not in cells:
        raise ValueError("Published string timing requires the Fodors-Zagats right/k1 cell")
    if args.kind in ("published", "strings") and string_cell in cells:
        benchmark_baseline_cell(*string_cell, config.UNSUPERVISED_STORAGE,
                                repeats=args.string_repeats, n_jobs=1)
        strings.append(string_cell)
    _save_summary("timing_run_summary", dict(design=design, selected_studies=selected,
                  skipped_failed_refits=skipped, cosine_cells=cosine, string_cells=strings,
                  repeats=args.repeats, string_repeats=args.string_repeats,
                  timing_scope="serial matching only; embedding generation and blocking excluded"))


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    for argument, variable in (("data_dir", "ZEROER_DATA_DIR"), ("output_dir", "ZEROER_OUTPUT_DIR")):
        if hasattr(args, argument):
            os.environ[variable] = str(Path(getattr(args, argument)).expanduser().resolve())
    threads = getattr(args, "threads", 1)
    if (threads < 1 or getattr(args, "workers", 1) < 1 or getattr(args, "repeats", 1) < 1
            or getattr(args, "string_repeats", 1) < 1 or getattr(args, "k_max", 1) < 1):
        parser.error("threads, workers, k_max and repetition counts must be positive")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(threads)
    # Paths and thread limits are established before importing NumPy, FAISS or
    # any module whose function defaults capture a configured input/output path.
    from exp import config
    try:
        design = _design(args)
    except ValueError as error:
        parser.error(str(error))
    if getattr(args, "dry_run", False):
        print(json.dumps(design, indent=2))
        return 0
    if args.command == "download":
        from data.download import download_data
        download_data(datasets=design["datasets"], models=design["models"], scope=args.scope, kind=args.kind)
    elif args.command == "verify-data":
        from data.download import verify_data
        result = verify_data(datasets=design["datasets"], models=design["models"], scope=args.scope,
                             kind=args.kind, verify_embeddings=not args.raw_only)
        print(json.dumps(result, indent=2))
    elif args.command == "prepare":
        from data.prepare import prepare_inputs
        if args.config:
            prepare_inputs(cells=args.tuning_plan.cells, models=args.tuning_plan.models,
                           n_jobs=args.workers, include_strings=False)
        else:
            prepare_inputs(scope=args.scope, datasets=design["datasets"], models=design["models"], n_jobs=args.workers)
    else:
        Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if args.command == "blocking":
            from exp.blocking import run_blocking_sweep
            for dataset in design["datasets"]:
                run_blocking_sweep(dataset, k_max=args.k_max, queried_sides=args.directions,
                                   models=design["models"])
        elif args.command == "match":
            _run_match(args, design)
        elif args.command == "tune":
            _run_tuning(args)
        elif args.command == "predict":
            _run_prediction(args)
        elif args.command == "baselines":
            _run_baselines(args, design)
        elif args.command == "timings":
            _run_timings(args, design)
        elif args.command == "report":
            from utils.reporting import report, report_blocking
            report_dir = Path(config.OUTPUT_DIR) / "reports"
            blocking_database = Path(config.OUTPUT_DIR) / "optuna.db"
            if args.blocking_only:
                result = report_blocking(blocking_database, report_dir,
                                        reference_dir=(Path(config.ROOT_DIR) / "results" / "reference"
                                                       if args.compare_reference else None))
            else:
                result = report(Path(config.OUTPUT_DIR) / "optuna_unsupervised.db", report_dir,
                                blocking_database=blocking_database,
                                compare_reference=args.compare_reference,
                                reference_dir=Path(config.ROOT_DIR) / "results" / "reference")
            print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

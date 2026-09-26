"""The cleo command: implemented, bounded research workflows only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import ResearchConfig, config_diff, config_hash, load_config


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True))


def _config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    cfg = commands.add_parser("config", help="validate, inspect, diff or export JSON schema")
    sub = cfg.add_subparsers(dest="operation", required=True)
    for name in ("validate", "inspect"):
        child = sub.add_parser(name)
        child.add_argument("path", nargs="?", type=Path)
        child.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    sub.add_parser("schema")
    child = sub.add_parser("diff")
    child.add_argument("left", type=Path)
    child.add_argument("right", type=Path)
    for name in ("evaluate", "simulate"):
        child = commands.add_parser(name, help="run a frozen agent/seed comparison and save evidence")
        _config_args(child)
        child.add_argument("--out", type=Path, default=Path("results/research"))
    child = commands.add_parser("design", help="preview bounded finite-factor configurations without executing")
    _config_args(child)
    child.add_argument("--factors", type=Path, required=True, help="JSON mapping of dotted fields to factor levels")
    child.add_argument("--budget", choices=("tiny", "small", "medium", "large", "exhaustive"), default="tiny")
    child.add_argument("--seed", type=int, default=0)
    for name in ("validate-data", "reconstruct", "replay"):
        child = commands.add_parser(name, help="validate canonical CSV/JSONL or reconstruct recorded order events")
        child.add_argument("path", type=Path)
        child.add_argument("--max-events", type=int, default=100_000)
    child = commands.add_parser("download-sample", help="download bounded public first-of-month Tardis samples")
    child.add_argument("--exchange", required=True)
    child.add_argument("--symbol", required=True)
    child.add_argument("--date", required=True, help="YYYY-MM-01")
    child.add_argument("--type", required=True, choices=("incremental_book_L2", "book_snapshot_5", "trades"))
    child.add_argument("--out", type=Path, default=Path("data/public"))
    child.add_argument("--max-bytes", type=int, default=64 * 1024**2)
    child = commands.add_parser("assess-l2", help="compare Tardis L2 reconstruction with published top-five snapshots")
    child.add_argument("--updates", type=Path, required=True)
    child.add_argument("--snapshots", type=Path, required=True)
    child.add_argument("--trades", type=Path)
    child.add_argument("--max-rows", type=int, default=20_000_000)
    child.add_argument("--out", type=Path, default=Path("results/historical"))
    child = commands.add_parser("calibrate", help="fit frozen L2 observables and evaluate purged chronological holdouts")
    child.add_argument("--train", type=Path, required=True)
    child.add_argument("--validation", type=Path, help="optional separate validation day; otherwise split training day 60/20/20")
    child.add_argument("--test", type=Path, required=True)
    child.add_argument("--out", type=Path, default=Path("results/calibration"))
    child = commands.add_parser("stress", help="run a registered six-scenario execution stress family")
    _config_args(child)
    child.add_argument("--out", type=Path, default=Path("results/stress"))
    child.add_argument("--max-episodes", type=int, default=2000)
    child = commands.add_parser("portfolio", help="run a portfolio accounting, limits and shock assessment")
    child.add_argument("--config", type=Path)
    child.add_argument("--out", type=Path, default=Path("results/portfolio"))
    child = commands.add_parser("benchmark", help="run the measured Python matching microbenchmark")
    child.add_argument("--pairs", type=int, default=2000)
    child.add_argument("--repeats", type=int, default=3)
    child = commands.add_parser("benchmark-suite", help="profile and measure L2/MBO/simulation research workloads")
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--operations", type=int, default=200)
    child.add_argument("--repeats", type=int, default=3)
    child.add_argument("--warmup", type=int, default=20)
    child = commands.add_parser("scaling-study", help="registered synthetic workload-size scaling study")
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--config", type=Path)
    child = commands.add_parser("verify-evidence", help="verify v0.4 registration, source, result and checkpoint bindings")
    child.add_argument("path", type=Path)
    child = commands.add_parser("mbo-replay", help="replay genuine order-ID or clearly synthetic MBO JSONL")
    child.add_argument("path", type=Path)
    child.add_argument("--max-events", type=int, default=100_000)
    child = commands.add_parser("validate-mbo", help="validate mapped order-identity source and optional aggregate references")
    child.add_argument("source", type=Path)
    child.add_argument("--manifest", type=Path, required=True)
    child.add_argument("--references", type=Path)
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--max-events", type=int, default=100_000)
    child = commands.add_parser("multiperiod-study", help="physically isolated chronological calibration and diagnostics")
    child.add_argument("--config", type=Path)
    child.add_argument("--data-root", type=Path, default=Path("."))
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--verify", action="store_true")
    child = commands.add_parser("calibration-study", help="registered chronological observable family comparison")
    child.add_argument("--config", type=Path, required=True)
    child.add_argument("--out", type=Path, required=True)
    child = commands.add_parser("policy-study", help="registered multi-seed PPO/DQN studies; requires rl extra")
    child.add_argument("operation", choices=("register", "train", "evaluate", "verify"))
    child.add_argument("--config", type=Path)
    child.add_argument("--out", type=Path, required=True)
    child = commands.add_parser("smoke", help="offline generated L2/MBO/calibration/execution reproduction smoke")
    child.add_argument("--out", type=Path, required=True)
    child = commands.add_parser("verify-artifact", help="verify a compact portable artifact's exact byte integrity")
    child.add_argument("path", type=Path)
    for name in ("report", "verify"):
        child = commands.add_parser(name, help="verify a saved research run and locate its offline report")
        child.add_argument("experiment", type=Path)
    child = commands.add_parser("reproduce", help="verify provenance and rerun the exact frozen settings")
    child.add_argument("experiment", type=Path)
    child.add_argument("--out", type=Path, default=Path("results/research"))
    child = commands.add_parser("experiment", help="compare saved experiment configuration and outcomes")
    op = child.add_subparsers(dest="operation", required=True).add_parser("diff")
    op.add_argument("left", type=Path)
    op.add_argument("right", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "config":
            if args.operation == "schema":
                _print(ResearchConfig.model_json_schema())
            elif args.operation == "diff":
                _print(config_diff(load_config(args.left).model_dump(mode="json"), load_config(args.right).model_dump(mode="json")))
            else:
                config = load_config(args.path, tuple(args.set))
                _print(config.model_dump(mode="json") if args.operation == "inspect" else {
                    "status": "VALID", "config_sha256": config_hash(config), "episodes": config.episode_count,
                    "estimated_events": config.estimated_events, "warnings": config.warnings()})
        elif args.command in {"evaluate", "simulate"}:
            from .experiments import read_experiment, run_experiment
            config = load_config(args.config, tuple(args.set))
            run = run_experiment(config, args.out)
            stored = read_experiment(run)
            _print({"experiment": str(run), "report": str(run / "report.html"),
                    "status": stored["result"]["status"], "outcomes": stored["result"]["outcome_counts"]})
            return 0 if stored["result"]["status"] in {"VALID", "WARNING"} else 1
        elif args.command == "design":
            from .experiments.design import design_experiment
            if args.factors.stat().st_size > 1_048_576:
                raise ValueError("factor file exceeds 1 MiB")
            factors = json.loads(args.factors.read_text(encoding="utf-8"))
            _print(design_experiment(load_config(args.config, tuple(args.set)), factors, budget=args.budget, seed=args.seed))
        elif args.command in {"validate-data", "reconstruct", "replay"}:
            from .replay import replay_file, validate_file
            if args.command == "validate-data":
                quality = validate_file(args.path, max_events=args.max_events)
                _print(quality.to_dict())
                return 0 if quality.valid else 1
            _print(replay_file(args.path, max_events=args.max_events))
        elif args.command == "benchmark":
            from .benchmarks import benchmark_matching
            _print(benchmark_matching(args.pairs, args.repeats))
        elif args.command == "benchmark-suite":
            from .performance import main as benchmark_main
            return benchmark_main(["--out", str(args.out), "--operations", str(args.operations),
                                   "--repeats", str(args.repeats), "--warmup", str(args.warmup)])
        elif args.command == "scaling-study":
            from .scaling import run_scaling
            config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
            result = run_scaling(args.out, config)
            _print({"status": result["status"], "workloads": len(result["metrics"]["measurements"]),
                    "out": str(args.out)})
        elif args.command == "verify-evidence":
            from .evidence import verify_evidence
            result = verify_evidence(args.path)
            _print(result)
            return 0 if result["valid"] else 1
        elif args.command == "mbo-replay":
            from .mbo import MBOReplay
            replay = MBOReplay(args.path, max_events=args.max_events)
            replay.run()
            _print(replay.summary())
        elif args.command == "validate-mbo":
            from .mbo_validation import run_mbo_validation
            result = run_mbo_validation(args.source, args.manifest, args.out,
                                       reference_path=args.references, max_events=args.max_events)
            _print(result)
            return 0 if result["status"] == "ESTABLISHED" else 1
        elif args.command == "multiperiod-study":
            from .multiperiod import run_study, verify_study
            if args.verify:
                result = verify_study(args.out)
                _print(result)
                return 0 if result["valid"] else 1
            if args.config is None:
                raise ValueError("--config is required when running a multi-period study")
            result = run_study(json.loads(args.config.read_text(encoding="utf-8")), args.out,
                               data_root=args.data_root)
            _print({key: value for key, value in result.items() if key != "periods"})
        elif args.command == "calibration-study":
            from .generalization import run_study
            _print(run_study(json.loads(args.config.read_text(encoding="utf-8")), args.out))
        elif args.command == "policy-study":
            from .policy_study import main as policy_main
            arguments = [args.operation, "--out", str(args.out)]
            if args.config:
                arguments += ["--config", str(args.config)]
            policy_main(arguments)
        elif args.command == "smoke":
            from .smoke import run_smoke
            _print(run_smoke(args.out))
        elif args.command == "verify-artifact":
            from .artifacts import verify_artifacts
            result = verify_artifacts(args.path)
            _print(result)
            return 0 if result["valid"] else 1
        elif args.command == "download-sample":
            from .data_download import download_tardis_sample
            path = download_tardis_sample(args.exchange, args.symbol, args.date, args.type,
                                          args.out, args.max_bytes)
            _print({"status": "DOWNLOADED_OR_VERIFIED", "path": str(path)})
        elif args.command == "calibrate":
            from .robustness import run_chronological_calibration
            run = run_chronological_calibration(args.train, args.validation, args.test, args.out)
            _print({"status": "COMPLETE", "study": str(run), "result": str(run / "result.json")})
        elif args.command == "stress":
            from .robustness import run_stress_suite
            run = run_stress_suite(load_config(args.config, tuple(args.set)), args.out,
                                   max_total_episodes=args.max_episodes)
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            _print({"status": result["status"], "study": str(run), "outcomes": result["outcomes"],
                    "family_inference": result["family_inference"]})
            return 0 if result["status"] == "COMPLETE" else 1
        elif args.command == "portfolio":
            from .portfolio import load_portfolio_config, run_portfolio_demo
            from .robustness import _new_run, _seal
            from .experiments.registry import PROJECT_ROOT, write_json, source_manifest
            config = load_portfolio_config(args.config or PROJECT_ROOT / "configs/portfolio_example.json")
            run, source = _new_run(args.out, {"kind": "portfolio_risk_assessment", "config": config.to_dict()})
            result = run_portfolio_demo(config)
            if source != source_manifest():
                raise ValueError("portfolio implementation changed during assessment")
            write_json(run / "result.json", result)
            _seal(run)
            _print({"status": "COMPLETE", "assessment": str(run), "result": str(run / "result.json")})
        elif args.command == "assess-l2":
            from .replay.assessment import assess_l2, save_assessment
            result = assess_l2(args.updates, args.snapshots, args.trades, max_rows=args.max_rows)
            run = save_assessment(result, args.out)
            _print({"status": result["status"], "assessment": str(run),
                    "comparison": result["reference_comparison"], "replay": result["replay"]})
            return 0 if result["status"] == "PASS" else 1
        elif args.command in {"verify", "report"}:
            from .experiments import verify_experiment
            is_study = (args.experiment / "plan.json").is_file()
            if is_study:
                from .robustness import verify_study
                verified = verify_study(args.experiment)
            else:
                verified = verify_experiment(args.experiment)
            if args.command == "report":
                name = "report.md" if is_study else "report.html"
                if is_study and not (args.experiment / name).is_file():
                    name = "result.json"
                verified["report"] = str(args.experiment.resolve() / name)
            _print(verified)
            return 0 if verified["valid"] else 1
        elif args.command == "reproduce":
            from .experiments import reproduce
            result = reproduce(args.experiment, args.out)
            _print(result)
            return 0 if result["valid_reproduction"] else 1
        elif args.command == "experiment":
            from .experiments import read_experiment
            left, right = read_experiment(args.left), read_experiment(args.right)
            _print({"configuration": config_diff(left["resolved_config"], right["resolved_config"]),
                    "provenance": config_diff(left["metadata"], right["metadata"]),
                    "results": config_diff(left["result"], right["result"])})
        return 0
    except (ValueError, TypeError, OSError, KeyError, yaml.YAMLError, RecursionError) as exc:
        # Typed data validation errors preserve machine-readable quality details.
        report = getattr(exc, "report", None)
        _print({"status": "INVALID", "error": str(exc), **({"quality": report.to_dict()} if report else {})})
        return 1


if __name__ == "__main__":
    sys.exit(main())

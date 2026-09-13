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
    child = commands.add_parser("benchmark", help="run the measured Python matching microbenchmark")
    child.add_argument("--pairs", type=int, default=2000)
    child.add_argument("--repeats", type=int, default=3)
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
        elif args.command in {"verify", "report"}:
            from .experiments import verify_experiment
            verified = verify_experiment(args.experiment)
            if args.command == "report":
                verified["report"] = str(args.experiment.resolve() / "report.html")
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

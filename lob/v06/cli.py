"""v0.6 command-line workflows, attached to ``cleo`` by ``lob.cli``.

Every command validates its inputs, fails with a machine-readable INVALID or
NOT_AVAILABLE status instead of falling back silently, and writes write-once,
protocol-bound run directories under ``results/v06`` (never overwriting).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True, default=str))


def add_commands(commands: argparse._SubParsersAction) -> None:
    child = commands.add_parser("protocol-v06", help="verify the frozen v0.6 protocol, ledger and holdout status")
    child.add_argument("operation", choices=("verify", "status"))
    child.add_argument("--root", type=Path, default=Path("."))
    child = commands.add_parser("realism-study", help="M1-M4: freeze the observable design, scales and margins")
    child.add_argument("phase", choices=("design",))
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--no-seal", action="store_true", help="labelled pilot: do not seal in the ledger")
    child = commands.add_parser("calibration-v3", help="M6: calibration v3 development search or sealed selection")
    child.add_argument("phase", choices=("develop", "select"))
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--from", dest="source", type=Path, help="development run (select phase)")
    child.add_argument("--pilot", action="store_true", help="labelled pilot: reduced budget, never sealed")
    child = commands.add_parser("verify-v06", help="verify sealed v0.6 run directories (bytes + binding; not replication)")
    child.add_argument("path", type=Path)


def _protocol(args) -> int:
    from .protocol import verify_protocol_files
    result = verify_protocol_files(args.root)
    if args.operation == "verify":
        result = {k: result[k] for k in ("valid", "issues", "protocol_sha256", "ledger_entries", "ledger_head_sha256",
                                         "fresh", "holdouts")}
    _print(result)
    return 0 if result["valid"] else 1


def _verify(args) -> int:
    from .evidence import verify_run, verify_tree
    if not args.path.exists():
        raise FileNotFoundError(f"no such run directory: {args.path}")
    result = verify_run(args.path) if (args.path / "binding.json").is_file() else verify_tree(args.path)
    if "reports" in result:
        result = {**result, "reports": {k: {"valid": r["valid"], "issues": r["issues"]} for k, r in result["reports"].items()}}
    _print(result)
    return 0 if result["valid"] else 1


def _realism(args) -> int:
    from .realism_study import design
    result = design(args.out, seal=not args.no_seal)
    _print({"scale_native_per_lot": result["scale_native_per_lot"], "tick_bps": result["tick_bps"],
            "objective_scales": result["objective_scales"], "equivalence_margins": result["equivalence_margins"],
            "sealed": not args.no_seal})
    return 0


PILOT_BUDGET = {"starts": 2, "global_draws": 8, "rounds": [0.25], "per_round": 8, "seconds": 600.0, "rescore": 6,
                "advance": 4, "rescore_seconds": 600.0, "selection_seconds": 600.0}


def _calibration(args) -> int:
    from .calibration_study import develop, select
    if args.phase == "develop":
        result = develop(args.out, budget=PILOT_BUDGET if args.pilot else None,
                         label="pilot" if args.pilot else "registered")
        _print({k: result[k] for k in ("label", "candidates", "failed_candidates", "start_best")}
               | {"best_search_objective": (result["best_search"] or {}).get("objective")})
        return 0
    if args.source is None:
        raise ValueError("calibration-v3 select requires --from <development run>")
    result = select(args.source, args.out, seal=not args.pilot)
    _print({"selected": result["selected"]["key"], "selection_objective": result["selected"]["selection_objective"],
            "control_selection_objective": result["control"]["selection_objective"],
            "near_optimal": len(result["near_optimal"]), "distinct": len(result["distinct_near_optimal"]),
            "ensemble": result["ensemble_members"], "sealed": not args.pilot})
    return 0


HANDLERS: dict[str, Callable] = {"calibration-v3": _calibration, "protocol-v06": _protocol, "verify-v06": _verify, "realism-study": _realism}


def dispatch(args) -> int | None:
    """Run a v0.6 command; None when the command is not a v0.6 command."""
    handler = HANDLERS.get(args.command)
    return None if handler is None else handler(args)

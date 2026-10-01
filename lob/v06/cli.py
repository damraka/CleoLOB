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
    child = commands.add_parser("identifiability-study", help="M7: identifiability, sensitivity and profiles (H4)")
    child.add_argument("--selection", type=Path, default=Path("results/v06/m6/select"))
    child.add_argument("--develop", type=Path, default=Path("results/v06/m6/develop"))
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--pilot", action="store_true")
    child = commands.add_parser("regime-calibration", help="M12: volatility-regime-conditioned calibration (seals)")
    child.add_argument("--selection", type=Path, default=Path("results/v06/m6/select"))
    child.add_argument("--develop", type=Path, default=Path("results/v06/m6/develop"))
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--pilot", action="store_true")
    child = commands.add_parser("execution-study", help="M8-M11: environment freeze, policy training, world evaluation")
    child.add_argument("phase", choices=("freeze", "train", "evaluate"))
    child.add_argument("--selection", type=Path, default=Path("results/v06/m6/select"))
    child.add_argument("--regime", type=Path, help="sealed regime-calibration run (freeze phase)")
    child.add_argument("--policies", type=Path, help="policy training run (evaluate phase)")
    child.add_argument("--out", type=Path)
    child.add_argument("--pilot", action="store_true")
    child = commands.add_parser("holdout-study", help="M5/M12/M13/M20: simulation bank, sealed holdout design, evaluation")
    child.add_argument("phase", choices=("bank", "seal", "evaluate"))
    child.add_argument("--selection", type=Path, default=Path("results/v06/m6/select"))
    child.add_argument("--regime", type=Path)
    child.add_argument("--bank", type=Path, default=Path("results/v06/m13/bank"))
    child.add_argument("--dataset")
    child.add_argument("--out", type=Path)
    child = commands.add_parser("claim-audit-v06", help="check claims against sealed results and documentation")
    child.add_argument("--claims", type=Path, default=Path("examples/studies/v06/evidence/claims.json"))
    child.add_argument("--docs", type=Path, nargs="*", default=[Path("docs/v06-final-report.md")])
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


PILOT_SEARCH = {"starts": 1, "global_draws": 6, "rounds": [0.25], "per_round": 4, "seconds": 600.0, "rescore": 4,
                "advance": 2, "rescore_seconds": 600.0, "selection_seconds": 600.0}


def _identifiability(args) -> int:
    from .identifiability_study import run
    budget = {"profile_draws": 2, "morris_trajectories": 2, "seconds": 600.0} if args.pilot else None
    result = run(args.selection, args.develop, args.out, budget=budget, label="pilot" if args.pilot else "registered")
    _print({"H4": result["H4_identifiability"]["status"], "distinct": result["H4_identifiability"]["distinct_near_optimal"],
            "flat_profiles": [k for k, v in result["profiles"].items() if v["flat_within_noise"]],
            "failed_evaluations": result["failed_evaluations"]})
    return 0


def _regime(args) -> int:
    from .regime_study import run
    result = run(args.develop, args.selection, args.out, budget=PILOT_SEARCH if args.pilot else None,
                 label="pilot" if args.pilot else "registered", seal=not args.pilot)
    _print({k: (v and {"key": v["key"], "selection_objective": v["selection_objective"]}) for k, v in
            result["models"].items()} | {"details": result["details"]})
    return 0


def _execution(args) -> int:
    from . import execution_study as es
    if args.phase == "freeze":
        regime = None
        if args.regime:
            from .regime_study import read_regime
            regime = read_regime(args.regime)["models"]
        document = es.freeze(selection_run=str(args.selection), regime=regime, seal=not args.pilot)
        _print({"mandate": document["mandate"], "worlds": [w["name"] for w in document["worlds"]],
                "ac": {k: v["fit"].get("status") for k, v in document["ac_parameters"].items()}})
        return 0
    if args.out is None:
        raise ValueError("--out is required")
    if args.phase == "train":
        summary = es.train(args.out, pilot={"seeds": 1, "timesteps": 512} if args.pilot else None)
        _print({"models": len(summary["models"]), "failed": summary["failed"]})
        return 0
    if args.policies is None:
        raise ValueError("execution-study evaluate requires --policies")
    result = es.evaluate(args.policies, args.out, markets=8 if args.pilot else None,
                         label="pilot" if args.pilot else "registered")
    _print({k: result[k]["status"] for k in ("H5_equifinality", "H6_rank_stability", "H7_execution_sensitive_realism")}
           | {"episodes": result["episodes"], "invalid_episodes": result["invalid_episodes"]})
    return 0


def _holdout(args) -> int:
    from . import holdout_study as hs
    if args.phase == "bank":
        result = hs.bank(args.out or args.bank, selection_run=str(args.selection),
                         regime_run=str(args.regime) if args.regime else None)
        _print({"tasks": result["tasks"], "models": list(result["models"])})
    elif args.phase == "seal":
        document = hs.seal(bank_run=str(args.bank), selection_run=str(args.selection),
                           regime_run=str(args.regime) if args.regime else None)
        _print({"sealed": True, "datasets": document["datasets"]})
    else:
        if not args.dataset or args.out is None:
            raise ValueError("holdout-study evaluate requires --dataset and --out")
        result = hs.evaluate(args.dataset, args.out, bank_run=str(args.bank))
        _print({"improvement": result["improvement"], "domain_gap": result["domain_gap"]["status"],
                "support": result["support"].get("label"), "regime": {k: v for k, v in result.get("regime", {}).items()
                                                                   if k in ("H9", "H10")}})
    return 0


def _claims(args) -> int:
    from .claims import audit
    result = audit(args.claims, args.docs)
    _print(result)
    return 0 if result["valid"] else 1


HANDLERS: dict[str, Callable] = {"identifiability-study": _identifiability, "regime-calibration": _regime,
                                 "execution-study": _execution, "holdout-study": _holdout, "claim-audit-v06": _claims,
                                 "calibration-v3": _calibration, "protocol-v06": _protocol, "verify-v06": _verify, "realism-study": _realism}


def dispatch(args) -> int | None:
    """Run a v0.6 command; None when the command is not a v0.6 command."""
    handler = HANDLERS.get(args.command)
    return None if handler is None else handler(args)

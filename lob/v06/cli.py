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


HANDLERS: dict[str, Callable] = {"protocol-v06": _protocol, "verify-v06": _verify}


def dispatch(args) -> int | None:
    """Run a v0.6 command; None when the command is not a v0.6 command."""
    handler = HANDLERS.get(args.command)
    return None if handler is None else handler(args)

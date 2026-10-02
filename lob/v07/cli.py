"""v0.7 command-line workflows, attached to ``cleo`` by ``lob.cli``.

Commands validate their inputs and fail with a machine-readable INVALID or
NOT_AVAILABLE status instead of falling back silently. Run directories live under
``results/v07`` and are write-once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True, default=str))


def add_commands(commands: argparse._SubParsersAction) -> None:
    child = commands.add_parser("protocol-v07", help="verify the frozen v0.7 protocol, ledger and holdout status")
    child.add_argument("operation", choices=("verify", "status"))
    child.add_argument("--root", type=Path, default=Path("."))


def _protocol(args) -> int:
    from .protocol.core import verify_protocol_files
    result = verify_protocol_files(args.root)
    _print(result)
    return 0 if result["valid"] else 1


HANDLERS: dict[str, Callable] = {"protocol-v07": _protocol}


def dispatch(args) -> int | None:
    """Run a v0.7 command; None when the command is not a v0.7 command."""
    handler = HANDLERS.get(args.command)
    return None if handler is None else handler(args)

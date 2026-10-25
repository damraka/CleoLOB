"""Importable task functions for the v1 runner/DAG tests (worker processes import them by name)."""
from __future__ import annotations

import json
from pathlib import Path


def square(x: int) -> int:
    return x * x


def fail(x: int) -> int:
    raise RuntimeError(f"boom {x}")


def make_run(out: Path, inputs: dict, value: float = 1.0) -> None:
    from lob.v1.evidence import runs as R
    total = value + sum(json.loads((p / "result.json").read_text())["total"] for p in inputs.values())
    run = R.new_run(out)
    R.finalize(run, analysis="dag-test", config={"value": value}, result={"total": total}, binding={})


def make_invalid(out: Path, inputs: dict) -> None:
    from lob.v1.evidence import runs as R
    run = R.new_run(out)
    R.finalize(run, analysis="dag-test", config={}, result={}, binding={}, status="INVALID", reason="test")

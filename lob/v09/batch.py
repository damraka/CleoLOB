"""Fail-closed serialized batch execution for v0.9 study runs (engineering safeguard).

A batch is an ordered list of steps ``{"name", "argv" | "cli", "out", "needs", "design"}``. Rules:

* a step succeeds only when its child exits 0 **and** its ``out`` run is sealed, verifies and is COMPLETED (a
  zero exit without a valid seal is a failure);
* a nonzero exit (or a missing/invalid seal) is a failure: it is recorded in the ledger as a retained ``attempt``
  (outcome INVALID, or RESOURCE_EXHAUSTED when the run says so) with the exit code, log and directory, and every
  step that ``needs`` it - directly or transitively - is SKIPPED and never started;
* the batch result is ``valid`` only when every step succeeded; ``run_batch`` returns the summary and the caller
  exits nonzero otherwise;
* a step whose ``out`` already holds a sealed valid run is ``ALREADY_SEALED`` (never re-run, never overwritten);
* before each step the runner waits until at least ``min_free_gb`` of memory is free.

The child exit code is taken directly from the process (no shell expansion between the child and the check).
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

from . import artifacts, protocol as pr


def free_gb() -> float:
    try:
        import ctypes

        class MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEM()
        m.dwLength = ctypes.sizeof(MEM)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise OSError
        return m.ullAvailPhys / 2 ** 30
    except (AttributeError, OSError):
        try:
            import os
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
        except (ValueError, OSError, AttributeError):
            return float("inf")


def sealed_ok(root: Path, out: str) -> bool:
    d = Path(root) / out
    if not (d / "binding.json").is_file():
        return False
    try:
        return artifacts.verify_run(d, root=Path(root))["valid"] and artifacts.run_status(d) == "COMPLETED"
    except (OSError, ValueError, KeyError):
        return False


def _design_of(step: dict) -> str:
    if step.get("design"):
        return step["design"]
    argv = step.get("cli") or []
    if "--design" in argv:
        return argv[argv.index("--design") + 1]
    ns = argv[argv.index("--namespace") + 1] if "--namespace" in argv else "dev"
    return f"{ns}-{argv[0] if argv else step['name']}"


def record_failure(root: Path, step: dict, *, exit_code: int, log: str, reason: str) -> dict:
    """Retained engineering attempt in the ledger (the directory, if any, is kept as is)."""
    out = step.get("out")
    exists = bool(out) and (Path(root) / out).exists()
    status = None
    if exists and (Path(root) / out / "status.json").is_file():
        status = json.loads((Path(root) / out / "status.json").read_text(encoding="utf-8")).get("status")
    outcome = "RESOURCE_EXHAUSTED" if status == "RESOURCE_EXHAUSTED" else "INVALID"
    payload = {"outcome": outcome, "directory": out if exists else None, "exit_code": int(exit_code), "log": log,
               "note": f"batch step {step['name']} failed: {reason}", "kind": "engineering"}
    if not exists:
        payload["no_directory_reason"] = "the step failed before creating its run directory"
    protocol = pr.load_protocol(Path(root) / pr.PROTOCOL_PATH)
    return pr.append_event(Path(root) / pr.LEDGER_PATH, "attempt", design=_design_of(step),
                           reason=f"engineering failure of batch step {step['name']}", protocol=protocol,
                           root=Path(root), payload=payload)


def run_batch(root: Path, steps: list[dict], *, logdir: Path, cwd: Path | None = None, min_free_gb: float = 8.0,
              python: str = sys.executable, echo: Callable[[str], None] = lambda m: print(m, flush=True),
              wait: float = 60.0) -> dict:
    root = Path(root).resolve()
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    state: dict[str, str] = {}
    rows = []
    for step in steps:
        name = step["name"]
        blocked = [n for n in step.get("needs", []) if state.get(n) not in ("SUCCEEDED", "ALREADY_SEALED")]
        if blocked:
            state[name] = "SKIPPED"
            rows.append({"name": name, "status": "SKIPPED", "blocked_by": blocked})
            echo(f"=== {time.strftime('%H:%M:%S')} skip {name} (needs {blocked})")
            continue
        if step.get("out") and sealed_ok(root, step["out"]):
            state[name] = "ALREADY_SEALED"
            rows.append({"name": name, "status": "ALREADY_SEALED", "out": step["out"]})
            echo(f"=== {time.strftime('%H:%M:%S')} {name} already sealed")
            continue
        while free_gb() < min_free_gb:
            echo(f"=== {time.strftime('%H:%M:%S')} waiting for memory ({free_gb():.1f} GB free)")
            time.sleep(wait)
        argv = step.get("argv") or [python, "-m", "lob.cli", *step["cli"], "--root", str(root)]
        log = logdir / f"{name}.log"
        echo(f"=== {time.strftime('%H:%M:%S')} start {name} (free {free_gb():.1f} GB)")
        t0 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            rc = subprocess.call(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd) if cwd else None)
        sealed = sealed_ok(root, step["out"]) if step.get("out") else rc == 0
        if rc == 0 and sealed:
            state[name] = "SUCCEEDED"
            rows.append({"name": name, "status": "SUCCEEDED", "exit_code": rc, "out": step.get("out"),
                         "seconds": round(time.time() - t0, 1)})
            echo(f"=== {time.strftime('%H:%M:%S')} end {name} exit=0 sealed")
            continue
        reason = f"exit {rc}" if rc != 0 else "exit 0 but no sealed, valid, COMPLETED run"
        entry = record_failure(root, step, exit_code=rc if rc != 0 else -1, log=log.as_posix(), reason=reason)
        state[name] = "FAILED"
        rows.append({"name": name, "status": "FAILED", "exit_code": rc, "reason": reason, "ledger_index": entry["index"],
                     "out": step.get("out")})
        echo(f"=== {time.strftime('%H:%M:%S')} FAILED {name} ({reason}; ledger attempt {entry['index']})")
    valid = all(r["status"] in ("SUCCEEDED", "ALREADY_SEALED") for r in rows)
    summary = {"valid": valid, "steps": rows,
               "failed": [r["name"] for r in rows if r["status"] == "FAILED"],
               "skipped": [r["name"] for r in rows if r["status"] == "SKIPPED"]}
    echo(f"=== {time.strftime('%H:%M:%S')} batch {'SUCCEEDED' if valid else 'FAILED'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="run a v0.9 batch spec (JSON list of steps) fail-closed")
    p.add_argument("spec", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--logdir", type=Path, required=True)
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--summary", type=Path, default=None)
    a = p.parse_args(argv)
    steps = json.loads(a.spec.read_text(encoding="utf-8"))
    out = run_batch(a.root, steps, logdir=a.logdir, min_free_gb=a.min_free_gb)
    if a.summary:
        a.summary.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0 if out["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

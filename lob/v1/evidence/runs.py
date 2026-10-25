"""Write-once evidence runs (M5, M151 foundation).

A run is a directory (``results/v1/<namespace>/<name>`` by convention) written once:

* ``config.json`` - the exact study configuration (including the design name and its sealed hash);
* ``result.json`` - the study output; values must be JSON-portable (no NaN, no absolute paths);
* ``binding.json`` - protocol hash, ledger head and length at finalisation, namespace, design, datasets read (each
  with the source hashes the ledger recorded at acquisition) and upstream runs (each with its result hash);
* ``provenance.json`` - code (git commit, dirty flag, SHA-256 of the ``lob`` source tree), environment (Python,
  platform, key package versions, CPU/GPU) and wall time;
* ``status.json`` - ``COMPLETED``, ``INVALID``, ``ABORTED``, ``RESOURCE_EXHAUSTED`` or ``NOT_AVAILABLE`` with a reason;
* ``checksums.json`` - SHA-256 of every other file.

Failed runs are finalised too (with their status), never deleted: a retry is a new run directory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

from ..protocol import ledger as L

SCHEMA = "cleolob-v1-run-1"
RUN_STATUSES = ("COMPLETED", "INVALID", "ABORTED", "RESOURCE_EXHAUSTED", "NOT_AVAILABLE")
FILES = ("config.json", "result.json", "binding.json", "provenance.json", "status.json")
_ABSOLUTE = re.compile(r"(^|[\s\"'=(])([A-Za-z]:[\\/]|/home/|/Users/|\\\\)")
PACKAGE_ROOT = Path(__file__).resolve().parents[2]          # the ``lob`` package directory


class RunError(RuntimeError):
    pass


def _portable(value: Any, where: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RunError(f"non-finite number at {where}")
    if isinstance(value, str) and len(value) < 4096 and _ABSOLUTE.search(value):
        raise RunError(f"absolute path at {where}: {value[:80]}")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise RunError(f"non-string key at {where}")
            _portable(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _portable(v, f"{where}[{i}]")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_digest(package: Path = PACKAGE_ROOT) -> str:
    """SHA-256 over the sorted relative paths and bytes of every ``.py`` file of the package."""
    h = hashlib.sha256()
    for p in sorted(package.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(package).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def _versions() -> dict:
    out = {}
    for name in ("numpy", "pandas", "torch", "pyarrow", "numba", "fastapi"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        out[name] = getattr(mod, "__version__", "?")
    return out


def environment() -> dict:
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(),
           "platform": platform.platform(terse=True), "machine": platform.machine(), "cpu_count": os.cpu_count(),
           "packages": _versions()}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            env["cuda"] = bool(torch.cuda.is_available())
            if env["cuda"]:
                env["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            env["cuda"] = False
    return env


def write_json(path: Path, value: Any) -> None:
    _portable(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def new_run(out: Path) -> Path:
    out = Path(out)
    if out.exists():
        raise RunError(f"{out} exists; runs are write-once (a retry is a new run directory)")
    out.mkdir(parents=True)
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Binding:
    protocol_sha256: str | None
    ledger_head: str
    ledger_entries: int
    namespace: str
    design: str | None
    design_sha256: str | None
    datasets: dict[str, dict]
    upstream: dict[str, dict]


def bind(*, ws, namespace: str, design: str | None, datasets: list[str], upstream: dict[str, Path],
         root: Path) -> dict:
    """Binding of a run to the protocol, ledger, design, datasets and upstream runs (verified, never trusted)."""
    from ..protocol import document as D
    state = ws.state() if ws is not None else None
    if state is not None:
        for d in datasets:
            ok = any(a["namespace"] == namespace for a in state.accesses.get(d, []))
            if not ok:
                raise RunError(f"dataset {d} has no ledgered access in namespace {namespace}")
        if design is not None and design not in state.designs:
            raise RunError(f"design {design} is not sealed")
    up = {}
    for key, path in upstream.items():
        path = Path(path)
        st = read_json(path / "status.json")
        up[key] = {"run": path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix(),
                   "result_sha256": sha256_file(path / "result.json"), "status": st["status"]}
    protocol_sha = D.canonical_sha256(ws.load()) if ws is not None and ws.protocol_path.exists() else None
    return {"schema": SCHEMA, "protocol_sha256": protocol_sha,
            "ledger_head": state.head if state else L.GENESIS, "ledger_entries": state.entries if state else 0,
            "namespace": namespace, "design": design,
            "design_sha256": state.designs[design]["sha256"] if (state and design) else None,
            "datasets": {d: {"source_sha256": dict(state.acquired.get(d, {})) if state else {}} for d in datasets},
            "upstream": up}


def finalize(out: Path, *, analysis: str, config: dict, result: Any, binding: dict, status: str = "COMPLETED",
             reason: str | None = None, started: float | None = None, root: Path | None = None) -> dict:
    if status not in RUN_STATUSES:
        raise RunError(f"unknown status {status}")
    out = Path(out)
    if not out.is_dir():
        raise RunError(f"{out} is not a run directory (call new_run first)")
    if (out / "status.json").exists():
        raise RunError(f"{out} is already finalised")
    git = L.git_state(root or out)
    write_json(out / "config.json", {"analysis": analysis, **config})
    write_json(out / "result.json", result)
    write_json(out / "binding.json", binding)
    write_json(out / "provenance.json", {"schema": SCHEMA, "analysis": analysis, "git_commit": git["commit"],
                                         "git_dirty": git["dirty"], "code_sha256": code_digest(),
                                         "environment": environment(), "finalized_at": L.utc_now(),
                                         "wall_seconds": (time.time() - started) if started else None})
    write_json(out / "status.json", {"status": status, "reason": reason, "analysis": analysis})
    sums = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "checksums.json"}
    write_json(out / "checksums.json", sums)
    return {"run": out.as_posix(), "status": status, "result_sha256": sums["result.json"]}


def verify_run(out: Path, *, ws=None) -> dict:
    out = Path(out)
    issues = []
    try:
        sums = read_json(out / "checksums.json")
    except (OSError, ValueError) as exc:
        return {"valid": False, "issues": [f"checksums unreadable: {exc}"]}
    for name in FILES:
        if name not in sums:
            issues.append(f"{name} missing from checksums")
    for name, digest in sums.items():
        p = out / name
        if not p.exists() or sha256_file(p) != digest:
            issues.append(f"{name} changed or missing")
    extra = {p.name for p in out.iterdir() if p.is_file()} - set(sums) - {"checksums.json"}
    if extra:
        issues.append(f"unlisted files {sorted(extra)}")
    if not issues and ws is not None:
        b = read_json(out / "binding.json")
        entries = L.read(ws.ledger_path)
        n = b["ledger_entries"]
        if n > len(entries) or (n and entries[n - 1]["sha256"] != b["ledger_head"]):
            issues.append("binding does not match a prefix of the ledger")
    status = read_json(out / "status.json")["status"] if (out / "status.json").exists() else None
    return {"valid": not issues, "issues": issues, "status": status}


def runs_under(base: Path) -> list[Path]:
    return sorted(p.parent for p in Path(base).rglob("checksums.json"))


__all__ = ["SCHEMA", "RUN_STATUSES", "RunError", "new_run", "bind", "finalize", "verify_run", "runs_under",
           "code_digest", "environment", "sha256_file", "read_json", "write_json"]

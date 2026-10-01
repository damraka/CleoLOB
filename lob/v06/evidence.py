"""Write-once v0.6 run directories bound to the v0.6 protocol and consumption ledger.

A sealed run holds ``config.json``, ``result.json``, optional detail files,
``provenance.json`` (portable runtime, package version, git commit and source
hashes), ``binding.json`` (protocol hash, ledger anchor, dataset identities and
roles, config/result/source hashes, seeds) and ``checksums.json`` over every
file. ``verify_run`` re-derives the binding and checks bytes: that is byte
integrity plus binding consistency, never independent scientific replication.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..artifacts import portable_provenance, seal_artifacts, verify_artifacts
from ..config import canonical_json
from ..experiments.registry import PROJECT_ROOT, source_manifest
from . import protocol as pr

RESULTS_ROOT = "results/v06"
_STARTED: dict[Path, dict] = {}
MEANING = "byte integrity plus protocol/ledger/dataset/config/result binding; not independent scientific replication"


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")


def new_run(out: str | Path) -> Path:
    """Create a run directory; refuses to reuse one (attempts are never overwritten)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    _STARTED[out.resolve()] = source_manifest()
    return out


def write_json(out: Path, name: str, value: Any) -> Path:
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    _write(target, json.loads(canonical_json(value)))
    return target


def finalize(out: Path, *, analysis: str, dataset_ids: Iterable[str], config: Any, result: Any,
             root: Path = PROJECT_ROOT, seeds: dict | None = None, extra: dict | None = None,
             check_source: bool = True) -> dict:
    """Bind and seal; refuses if the implementation changed while the run executed."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    provenance = portable_provenance()
    started = _STARTED.pop(Path(out).resolve(), None)
    if check_source and started is not None and provenance["source_files"] != started:
        changed = sorted(k for k in set(started) | set(provenance["source_files"])
                         if started.get(k) != provenance["source_files"].get(k))
        raise ValueError(f"implementation changed during the run ({changed}); attempt retained unsealed")
    config = json.loads(canonical_json(config))
    result = json.loads(canonical_json(result))
    write_json(out, "config.json", config)
    write_json(out, "result.json", result)
    write_json(out, "provenance.json", provenance)
    binding = pr.bind_evidence(protocol=protocol, ledger_path=root / pr.LEDGER_PATH, analysis=analysis,
                               dataset_ids=dataset_ids, config=config, result=result, provenance=provenance,
                               extra={"seeds": seeds or {}, **(extra or {})})
    write_json(out, "binding.json", binding)
    seal_artifacts(out)
    return binding


def read_run(out: str | Path) -> dict:
    out = Path(out)
    return {name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("config", "result", "binding", "provenance")}


def verify_run(out: str | Path, *, root: Path = PROJECT_ROOT, check_source: bool = False) -> dict:
    out = Path(out)
    issues: list[str] = []
    analysis = None
    try:
        issues.extend(verify_artifacts(out)["issues"])
        stored = read_run(out)
        analysis = stored["binding"].get("analysis")
        protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
        checked = pr.verify_binding(stored["binding"], protocol=protocol, ledger_path=root / pr.LEDGER_PATH,
                                    config=stored["config"], result=stored["result"],
                                    source_sha256=stored["provenance"]["source_sha256"])
        issues.extend(checked["issues"])
        if check_source and stored["provenance"]["source_files"] != source_manifest():
            issues.append("current implementation differs from the run's recorded source")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues, "analysis": analysis, "meaning": MEANING}


def verify_tree(base: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    """Verify every sealed v0.6 run (directories holding binding.json) below ``base``."""
    base = Path(base)
    runs = sorted(p.parent for p in base.rglob("binding.json"))
    reports = {p.relative_to(base).as_posix() or ".": verify_run(p, root=root) for p in runs}
    return {"valid": bool(reports) and all(r["valid"] for r in reports.values()), "runs": len(reports),
            "invalid": sorted(k for k, r in reports.items() if not r["valid"]), "reports": reports,
            "meaning": MEANING}

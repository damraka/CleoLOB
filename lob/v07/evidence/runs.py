"""Write-once v0.7 run directories bound to the v0.7 protocol and ledger (workstreams 65, 72).

Same structure as v0.6 (config, result, provenance, binding, checksums), with
one fix of the v0.6 provenance debt: source files are hashed after normalizing
line endings (CRLF -> LF), so a checkout's line-ending conversion no longer
changes the recorded implementation hash. Raw byte hashes are still recorded for
the files a run produced. ``verify_run`` checks byte integrity and binding
consistency; it is never independent scientific replication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ...artifacts import portable_provenance, seal_artifacts, verify_artifacts
from ...config import canonical_json
from ...experiments.registry import PROJECT_ROOT
from ..protocol import core as pr

RESULTS_ROOT = "results/v07"
MEANING = "byte integrity plus protocol/ledger/dataset/config/result binding; not independent scientific replication"
_STARTED: dict[Path, dict] = {}
SOURCE_GLOBS = ("lob/**/*.py",)
SOURCE_FILES = ("pyproject.toml",)


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Line-ending-normalized hashes of the implementation (fixes the v0.6 CRLF provenance debt)."""
    paths = sorted({p for pattern in SOURCE_GLOBS for p in root.glob(pattern)})
    paths += [root / name for name in SOURCE_FILES]
    return {p.relative_to(root).as_posix(): normalized_sha256(p) for p in paths
            if p.is_file() and "__pycache__" not in p.parts}


def provenance(root: Path = PROJECT_ROOT) -> dict:
    base = portable_provenance()
    files = source_manifest(root)
    base.pop("source_files", None)
    base.pop("source_sha256", None)
    return {**base, "source_hashing": "sha256 of file bytes with CRLF normalized to LF", "source_files": files,
            "source_sha256": hashlib.sha256(canonical_json(files).encode()).hexdigest()}


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
    prov = provenance()  # implementation hashes always come from the package, not the protocol root
    started = _STARTED.pop(Path(out).resolve(), None)
    if check_source and started is not None and prov["source_files"] != started:
        changed = sorted(k for k in set(started) | set(prov["source_files"])
                         if started.get(k) != prov["source_files"].get(k))
        raise ValueError(f"implementation changed during the run ({changed}); attempt retained unsealed")
    config = json.loads(canonical_json(config))
    result = json.loads(canonical_json(result))
    write_json(out, "config.json", config)
    write_json(out, "result.json", result)
    write_json(out, "provenance.json", prov)
    binding = pr.bind_evidence(protocol=protocol, ledger_path=root / pr.LEDGER_PATH, analysis=analysis,
                               dataset_ids=dataset_ids, config=config, result=result, provenance=prov,
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
        issues.extend(pr.verify_binding(stored["binding"], protocol=protocol, ledger_path=root / pr.LEDGER_PATH,
                                        config=stored["config"], result=stored["result"]))
        recorded = stored["provenance"]
        if hashlib.sha256(canonical_json(recorded["source_files"]).encode()).hexdigest() != recorded["source_sha256"]:
            issues.append("provenance source manifest does not match its digest")
        if stored["binding"].get("source_sha256") != recorded["source_sha256"]:
            issues.append("binding and provenance disagree on the source digest")
        if check_source and recorded["source_files"] != source_manifest(root):
            issues.append("current implementation differs from the run's recorded source")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues, "analysis": analysis, "meaning": MEANING}


def verify_tree(base: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    base = Path(base)
    runs = sorted(p.parent for p in base.rglob("binding.json"))
    reports = {p.relative_to(base).as_posix() or ".": verify_run(p, root=root) for p in runs}
    return {"valid": bool(reports) and all(r["valid"] for r in reports.values()), "runs": len(reports),
            "invalid": sorted(k for k, r in reports.items() if not r["valid"]), "reports": reports,
            "meaning": MEANING}

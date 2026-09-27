"""Write-once v0.5 study directories bound to the frozen protocol and consumption ledger.

A run directory holds ``config.json``, ``result.json``, any detail files,
``provenance.json`` (portable runtime/source identity), ``binding.json``
(protocol hash, ledger anchor, dataset identities, config/result/source hashes)
and ``checksums.json`` over every file. ``verify_run`` re-derives each binding
and checks bytes. Detailed files derived from restricted provider data stay in
ignored local directories; public exports carry summaries and hashes only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .artifacts import portable_provenance, seal_artifacts, verify_artifacts
from .config import canonical_json
from .experiments.registry import PROJECT_ROOT, source_manifest
from . import preregistration as pr

PROTOCOL_PATH = "configs/v05/protocol.json"
LEDGER_PATH = "configs/v05/consumption-ledger.jsonl"


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")


def new_run(out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_json(out: Path, name: str, value: Any) -> None:
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    _write(target, json.loads(canonical_json(value)))


def finalize(out: Path, *, analysis: str, dataset_ids: Iterable[str], config: Any, result: Any,
             root: Path = PROJECT_ROOT, extra: dict | None = None) -> dict:
    """Bind and seal; refuses if the implementation changed during the run."""
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    provenance = portable_provenance()
    if provenance["source_files"] != source_manifest():
        raise ValueError("implementation changed during the run; attempt retained unsealed")
    config = json.loads(canonical_json(config))
    result = json.loads(canonical_json(result))
    write_json(out, "config.json", config)
    write_json(out, "result.json", result)
    write_json(out, "provenance.json", provenance)
    binding = pr.bind_evidence(protocol=protocol, ledger_path=root / LEDGER_PATH, analysis=analysis,
                               dataset_ids=dataset_ids, config=config, result=result,
                               provenance=provenance, extra=extra)
    write_json(out, "binding.json", binding)
    seal_artifacts(out)
    return binding


def verify_run(out: str | Path, *, root: Path = PROJECT_ROOT, check_source: bool = False) -> dict:
    out = Path(out)
    issues = []
    try:
        issues.extend(verify_artifacts(out)["issues"])
        binding = json.loads((out / "binding.json").read_text(encoding="utf-8"))
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        protocol = pr.load_protocol(root / PROTOCOL_PATH)
        checked = pr.verify_binding(binding, protocol=protocol, ledger_path=root / LEDGER_PATH, config=config,
                                    result=result, source_sha256=provenance["source_sha256"],
                                    expected_commit=provenance.get("git_commit"))
        issues.extend(checked["issues"])
        if check_source and provenance["source_files"] != source_manifest():
            issues.append("current implementation differs from the run's recorded source")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues,
            "meaning": "byte integrity plus protocol/ledger/dataset/config/result binding; not replication"}

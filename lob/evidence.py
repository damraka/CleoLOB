"""Write-once research registration with semantic bindings beyond file checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .artifacts import portable_provenance, seal_artifacts, verify_artifacts
from .config import canonical_json
from .experiments.registry import PROJECT_ROOT, sha256_file, source_manifest, utc_now, write_json

SCHEMA = "cleolob-research-v2"
STATUSES = frozenset({"ESTABLISHED", "NOT_ESTABLISHED", "FAILED", "NOT_AVAILABLE", "INVALID", "INCONCLUSIVE"})


def document_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _relative(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or any(p in {"", ".", ".."} for p in name.split("/"))):
        raise ValueError("artifact path must be a canonical portable relative name")
    path = root / name
    if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
            or any(p.is_symlink() for p in path.parents if p != root)):
        raise ValueError("unsafe artifact path")
    return path


def _validate_inputs(inputs: list[dict]) -> None:
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("at least one declared source or synthetic generator is required")
    identities = set()
    for row in inputs:
        for key in ("identity", "adapter_version", "role", "kind"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"dataset {key} is required")
        if row["identity"] in identities:
            raise ValueError("duplicate dataset identity")
        identities.add(row["identity"])
        if not _digest(row.get("sha256")):
            raise ValueError("source bytes or synthetic generator specification SHA-256 required")
        if row["kind"] not in {"historical", "synthetic", "retained_evidence"}:
            raise ValueError("unsupported dataset evidence kind")


def register_evidence(root: str | Path, *, kind: str, config: dict, inputs: list[dict],
                      seeds: dict[str, list[int]], hypotheses: list[str]) -> dict:
    """Freeze inputs/config/source before running; existing directories are refused."""
    config = json.loads(canonical_json(config))
    inputs = json.loads(canonical_json(inputs))
    seeds = json.loads(canonical_json(seeds))
    _validate_inputs(inputs)
    if not isinstance(kind, str) or not kind.strip() or not hypotheses or not all(
            isinstance(h, str) and h.strip() for h in hypotheses):
        raise ValueError("kind and explicit hypotheses are required")
    if not isinstance(seeds, dict) or any(
        not isinstance(values, list) or len(set(values)) != len(values)
        or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 for s in values)
        for values in seeds.values()
    ):
        raise ValueError("seed domains must contain unique nonnegative integers")
    provenance = portable_provenance()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    write_json(root / "datasets.json", inputs)
    write_json(root / "provenance.json", provenance)
    registration = {"schema": SCHEMA, "kind": kind, "created_at": utc_now(),
                    "config_sha256": document_hash(config), "datasets_sha256": document_hash(inputs),
                    "source_sha256": provenance["source_sha256"], "seeds": seeds,
                    "hypotheses": hypotheses, "registered_before_execution": True}
    write_json(root / "registration.json", registration)
    write_json(root / "registration-seal.json", {"files": {
        name: sha256_file(root / name) for name in
        ("config.json", "datasets.json", "provenance.json", "registration.json")}})
    for name, digest in provenance["source_files"].items():
        target = _relative(root, "source/" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write((PROJECT_ROOT / name).read_bytes())
        if sha256_file(target) != digest:
            raise ValueError("source changed during registration; incomplete attempt retained")
    return registration


def _registration(root: Path) -> tuple[dict, dict]:
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    registration, provenance = read("registration.json"), read("provenance.json")
    seal = read("registration-seal.json")
    required = {"config.json", "datasets.json", "provenance.json", "registration.json"}
    if set(seal["files"]) != required:
        raise ValueError("registration seal file set mismatch")
    for name, digest in seal["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("registration seal mismatch")
    if registration.get("schema") != SCHEMA or registration.get("registered_before_execution") is not True:
        raise ValueError("unsupported registration schema or execution order")
    config, inputs = read("config.json"), read("datasets.json")
    _validate_inputs(inputs)
    if document_hash(config) != registration["config_sha256"]:
        raise ValueError("config/registration mismatch")
    if document_hash(inputs) != registration["datasets_sha256"]:
        raise ValueError("dataset/registration mismatch")
    if (document_hash(provenance["source_files"]) != registration["source_sha256"]
            or provenance["source_sha256"] != registration["source_sha256"]):
        raise ValueError("source/registration mismatch")
    for name, digest in provenance["source_files"].items():
        target = _relative(root, "source/" + name)
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"registered source missing or changed: {name}")
    return registration, provenance


def finalize_evidence(root: str | Path, *, status: str, metrics: dict,
                      models: list[dict] | None = None) -> dict:
    """Bind results to registration and exact checkpoints, then seal once."""
    root = Path(root)
    registration, provenance = _registration(root)
    if source_manifest() != provenance["source_files"]:
        raise ValueError("implementation changed after registration; attempt retained")
    if status not in STATUSES:
        raise ValueError("explicit research validity status required")
    metrics = json.loads(canonical_json(metrics))
    models = json.loads(canonical_json(models or []))
    _verify_models(root, models)
    result = {"schema": SCHEMA, "status": status, "metrics": metrics, "models": models,
              "registration_sha256": sha256_file(root / "registration.json"),
              "config_sha256": registration["config_sha256"],
              "datasets_sha256": registration["datasets_sha256"],
              "source_sha256": registration["source_sha256"]}
    write_json(root / "result.json", result)
    seal_artifacts(root)
    return result


def _verify_models(root: Path, models: list[dict]) -> None:
    seen = set()
    for model in models:
        name = model["path"]
        if name in seen or not isinstance(model.get("identity"), str) or not model["identity"]:
            raise ValueError("model identity required and checkpoint paths must be unique")
        seen.add(name)
        target = _relative(root, name)
        if not _digest(model.get("sha256")) or not target.is_file() or sha256_file(target) != model["sha256"]:
            raise ValueError("stale or missing model checkpoint")


def verify_evidence(root: str | Path) -> dict:
    """Check exact files and semantic links without executing archived source."""
    issues = []
    root = Path(root)
    try:
        checked = verify_artifacts(root)
        issues.extend(checked["issues"])
        if not issues:
            registration, _ = _registration(root)
            result = json.loads((root / "result.json").read_text(encoding="utf-8"))
            if result.get("schema") != SCHEMA or result.get("status") not in STATUSES:
                raise ValueError("invalid research result schema/status")
            if result["registration_sha256"] != sha256_file(root / "registration.json"):
                raise ValueError("result/registration mismatch")
            for key in ("config_sha256", "datasets_sha256", "source_sha256"):
                if result[key] != registration[key]:
                    raise ValueError(f"result/{key} mismatch")
            _verify_models(root, result["models"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues, "schema": SCHEMA,
            "meaning": "byte integrity and registration consistency; not independent scientific validation"}

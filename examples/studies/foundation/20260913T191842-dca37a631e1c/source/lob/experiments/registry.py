"""Append-only run directories with artifact integrity and source provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ResearchConfig, canonical_json, config_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Write once; collision is an error, never an overwrite."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(data) + "\n")


def source_manifest() -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "lob").rglob("*.py"))
    paths += [PROJECT_ROOT / name for name in ("pyproject.toml", "requirements.txt", "evaluate.py", "train_rl.py")]
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in paths if p.is_file()}


def runtime_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "gymnasium", "stable-baselines3", "torch", "pydantic", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "executable": sys.executable, "packages": versions}


def create_experiment(config: ResearchConfig, out: str | Path,
                      reproduced_from: str | None = None) -> tuple[Path, dict[str, Any]]:
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    run = root / run_id
    run.mkdir(exist_ok=False)
    (run / "logs").mkdir()
    source = source_manifest()
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                            stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    model = None
    if "ppo" in config.evaluation.agents:
        path = Path(config.evaluation.model_path or "").resolve()
        if not path.is_file() and path.with_suffix(".zip").is_file():
            path = path.with_suffix(".zip")
        model = {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
    metadata = {
        "experiment_id": run_id, "created_at": utc_now(), "config_sha256": config_hash(config),
        "git_commit": commit, "git_dirty": dirty, "source_manifest": source,
        "source_sha256": hashlib.sha256(canonical_json(source).encode()).hexdigest(),
        "runtime": runtime_metadata(), "dataset": {"kind": "synthetic", "model": "poisson_fifo_v2", "calibrated": False},
        "model": model, "reproduced_from": reproduced_from,
        "planned_episodes": config.episode_count,
        "randomness": {"episode_seeds": list(config.evaluation.seeds),
                       "statistics_seed": config.evaluation.statistics_seed,
                       "engine_streams": "SeedSequence streams; see source/engine configuration for exact mapping",
                       "agent_rng": "legacy random policy uses episode_seed + 7"},
    }
    write_json(run / "resolved_config.json", config.model_dump(mode="json"))
    write_json(run / "metadata.json", metadata)
    # A compact source snapshot preserves dirty/untracked research code. Reproduce
    # verifies the current implementation; it never executes archived source.
    for name in source:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write((PROJECT_ROOT / name).read_bytes())
    return run, metadata


def seal_experiment(run: Path) -> None:
    files = {p.relative_to(run).as_posix(): sha256_file(p) for p in sorted(run.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    write_json(run / "manifest.json", {"algorithm": "sha256", "files": files})


def verify_experiment(path: str | Path) -> dict[str, Any]:
    run = Path(path).resolve(strict=True)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    issues = []
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["files"]):
        issues.append("artifact file set differs from sealed manifest")
    for name, expected in manifest["files"].items():
        target = (run / name).resolve()
        if not target.is_relative_to(run) or not target.is_file():
            issues.append(f"missing or unsafe artifact: {name}")
        elif sha256_file(target) != expected:
            issues.append(f"artifact checksum mismatch: {name}")
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    raw = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    config = ResearchConfig.model_validate(raw)
    if config_hash(config) != metadata["config_sha256"]:
        issues.append("resolved configuration hash differs from metadata")
    return {"valid": not issues, "issues": issues, "experiment_id": metadata["experiment_id"]}


def read_experiment(path: str | Path) -> dict[str, Any]:
    run = Path(path).resolve(strict=True)
    verified = verify_experiment(run)
    if not verified["valid"]:
        raise ValueError("experiment integrity failed: " + "; ".join(verified["issues"]))
    return {name: json.loads((run / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("metadata", "resolved_config", "result")}

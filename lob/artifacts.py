"""Compact, portable research evidence; no absolute machine paths in metadata."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .config import canonical_json
from .experiments.registry import PROJECT_ROOT, runtime_metadata, sha256_file, source_manifest, write_json


def portable_provenance() -> dict[str, Any]:
    runtime = runtime_metadata()
    runtime.pop("executable", None)
    runtime["cpu"] = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    runtime["logical_cpu_count"] = os.cpu_count()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                runtime["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    source = source_manifest()
    return {"git_commit": commit, "git_dirty": dirty, "runtime": runtime,
            "source_files": source,
            "source_sha256": hashlib.sha256(canonical_json(source).encode()).hexdigest()}


def seal_artifacts(root: str | Path) -> dict:
    root = Path(root).resolve(strict=True)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlinks are not supported")
        if path.is_file() and path != root / "checksums.json":
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not files:
        raise ValueError("cannot seal an empty artifact")
    result = {"format": "cleolob-compact-v1", "algorithm": "sha256", "files": files}
    write_json(root / "checksums.json", result)
    return result


def verify_artifacts(root: str | Path) -> dict:
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "cleolob-compact-v1" or manifest.get("algorithm") != "sha256"
            or not isinstance(manifest.get("files"), dict) or not manifest["files"]):
        raise ValueError("unsupported or empty artifact manifest")
    issues = []
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p != root / "checksums.json"}
    if actual != set(manifest["files"]):
        issues.append("artifact file set changed")
    for name, digest in manifest["files"].items():
        target = root / name
        # Require canonical portable relative names, even when reading on Windows.
        if (not name or "\\" in name or ":" in name or name.startswith("/")
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or not target.resolve().is_relative_to(root)
                or target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root)):
            issues.append(f"unsafe artifact: {name}")
        elif not target.is_file() or sha256_file(target) != digest:
            issues.append(f"checksum mismatch: {name}")
    return {"valid": not issues, "issues": issues, "files": len(manifest["files"]),
            "meaning": "byte integrity only; not independent scientific validation"}

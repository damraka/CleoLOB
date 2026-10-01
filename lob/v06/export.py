"""M15: compact public v0.6 evidence bundle.

Copies only redistributable material: the protocol, ledger, dataset facts,
registration artifacts (environment freeze, holdout and transfer designs) and,
for every sealed run, ``config.json``, ``result.json``, ``binding.json`` and the
portable ``provenance.json``. Detailed files derived from restricted provider
data (block sketches, observable references, episode tables, window features)
and large simulator outputs are excluded and listed with their SHA-256 in
``export.json``. A guard refuses absolute local paths and credential-like text.
The bundle is sealed with independent checksums (``cleo verify-artifact``).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from ..artifacts import seal_artifacts
from ..experiments.registry import PROJECT_ROOT, sha256_file

PUBLIC_RUN_FILES = ("config.json", "result.json", "binding.json", "provenance.json")
REGISTRATION = ("configs/v06/protocol.json", "configs/v06/datasets.json", "configs/v06/consumption-ledger.jsonl",
                "configs/v06/environment-freeze.json", "configs/v06/holdout-design.json",
                "configs/v06/transfer-design.json")
# Absolute Windows paths (single or JSON-escaped backslash, or slash), POSIX home paths, credential-like words.
FORBIDDEN = re.compile(r"([A-Za-z]:(\\{1,2}|/)(Users|home|Documents)|/home/|/Users/|api[_-]?key|secret|password|"
                       r"token\b|Bearer )", re.I)


def check_text(name: str, text: str) -> None:
    match = FORBIDDEN.search(text)
    if match:
        raise ValueError(f"{name}: refusing to export text matching {match.group(0)!r}")


def export(out: str | Path, *, root: Path = PROJECT_ROOT, runs_root: str = "results/v06",
           extra: dict[str, Path] | None = None) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError("public bundle exists; export to a new directory")
    out.mkdir(parents=True)
    included, excluded = [], []
    for name in REGISTRATION:
        source = root / name
        if source.is_file():
            target = out / "registration" / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            check_text(name, source.read_text(encoding="utf-8"))
            shutil.copyfile(source, target)
            included.append(target.relative_to(out).as_posix())
    base = root / runs_root
    for binding in sorted(base.rglob("binding.json")):
        run = binding.parent
        relative = run.relative_to(base).as_posix()
        if relative.startswith("pilot/"):
            continue
        for path in sorted(p for p in run.rglob("*") if p.is_file()):
            name = path.relative_to(run).as_posix()
            if name in PUBLIC_RUN_FILES:
                text = path.read_text(encoding="utf-8")
                check_text(f"{relative}/{name}", text)
                target = out / "runs" / relative / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                included.append(target.relative_to(out).as_posix())
            elif name != "checksums.json":
                excluded.append({"path": f"{runs_root}/{relative}/{name}", "sha256": sha256_file(path),
                                 "bytes": path.stat().st_size,
                                 "reason": "derived from restricted provider data or large simulator output"})
    for name, source in (extra or {}).items():
        check_text(name, source.read_text(encoding="utf-8"))
        shutil.copyfile(source, out / name)
        included.append(name)
    manifest = {"schema": "cleolob-v06-public-evidence-1", "included": sorted(included), "excluded": excluded,
                "meaning": ("Byte integrity of this bundle plus the run bindings it contains; not independent "
                            "scientific replication. Restricted raw data and detailed derived files are excluded.")}
    (out / "export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    seal_artifacts(out)
    return manifest

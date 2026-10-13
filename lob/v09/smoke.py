"""Offline generated-data v0.9 smoke: functionality evidence, not scientific validation.

The smoke never touches real data and never writes to the repository ledger. When the repository configs are
available it builds an isolated protocol root inside the output directory (copies of the pinned v0.8 protocol and
ledger and of the v0.9 configs, plus a fresh v0.9 ledger) so every step is sealed and bound exactly like a study run;
otherwise (an installed wheel without the repository) it runs standalone with byte-integrity seals only and reports
protocol binding as NOT_AVAILABLE.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from ..experiments.registry import PROJECT_ROOT

HISTORY_FILES = ("configs/v08/protocol.json", "configs/v08/consumption-ledger.jsonl")


def isolated_root(out: Path, *, repo: Path = PROJECT_ROOT, initialize: bool = True) -> Path | None:
    from . import protocol as pr
    needed = [repo / f for f in HISTORY_FILES] + [repo / pr.CONFIG_DIR / f for f in pr.REQUIRED_CONFIGS]
    needed.append(repo / pr.PROTOCOL_PATH)
    if not all(p.is_file() for p in needed):
        return None
    root = Path(out) / "root"
    for f in HISTORY_FILES:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / f, root / f)
    (root / pr.CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    for f in ("protocol.json",) + pr.REQUIRED_CONFIGS:
        shutil.copyfile(repo / pr.CONFIG_DIR / f, root / pr.CONFIG_DIR / f)
    if initialize:
        pr.initialize_ledger(root)
    return root


def run(args) -> dict:
    from .smoke_steps import run_smoke
    return run_smoke(Path(args.out), workers=args.workers)

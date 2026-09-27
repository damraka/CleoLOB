"""Ledger-gated acquisition of v0.5 Deribit datasets (Tardis public first-of-month samples).

For a fresh holdout the ledger ``access`` entry is written before the download
starts (the dataset is consumed from that moment), and a second entry records the
exact source hashes after integrity checks. Consumed development periods only
record hashes. Files go to the ignored ``data/v05/tardis`` directory beside
their provenance sidecars; nothing is redistributed.
"""
from __future__ import annotations

from pathlib import Path

from .data_download import download_tardis_sample
from .experiments.registry import PROJECT_ROOT, sha256_file
from . import preregistration as pr
from .v05_evidence import LEDGER_PATH, PROTOCOL_PATH

DATA_DIR = "data/v05/tardis"
LEGACY_DIR = "data/public"
_TYPES = {"incremental_book_L2": "l2", "trades": "trades", "book_snapshot_5": "snapshot5"}


def _file_name(venue: str, data_type: str, date: str, symbol: str) -> str:
    return f"{venue}_{data_type}_{date}_{symbol}.csv.gz"


def dataset_files(dataset_id: str, *, root: Path = PROJECT_ROOT) -> dict[str, Path]:
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    declaration = pr.dataset_declaration(protocol, dataset_id)
    files = {}
    for data_type in declaration["data_types"]:
        name = _file_name(declaration["venue"], data_type, declaration["date"], declaration["instrument"])
        legacy = root / LEGACY_DIR / name
        files[_TYPES[data_type]] = legacy if legacy.is_file() else root / DATA_DIR / name
    return files


def acquire(dataset_id: str, *, root: Path = PROJECT_ROOT, types: tuple[str, ...] = ("l2", "trades"),
            purpose: str) -> dict[str, Path]:
    """Download missing files (ledger first for fresh data) and record exact hashes."""
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    declaration = pr.dataset_declaration(protocol, dataset_id)
    ledger = root / LEDGER_PATH
    state = pr.replay_ledger(pr.read_ledger(ledger), protocol)
    files = dataset_files(dataset_id, root=root)
    wanted = {kind: path for kind, path in files.items() if kind in types}
    missing = [kind for kind, path in wanted.items() if not path.is_file()]
    if missing and state.freshness(dataset_id) == "fresh":
        pr.append_event(ledger, "access", {"dataset_id": dataset_id, "purpose": f"download start: {purpose}",
                                           "files": sorted(missing)}, protocol=protocol)
    inverse = {v: k for k, v in _TYPES.items()}
    for kind in missing:
        download_tardis_sample(declaration["venue"], declaration["instrument"], declaration["date"], inverse[kind],
                               root / DATA_DIR, max_bytes=256 * 1024**2)
    hashes = {path.name: sha256_file(path) for path in wanted.values()}
    pr.append_event(ledger, "access", {"dataset_id": dataset_id, "source_sha256": hashes, "purpose": purpose},
                    protocol=protocol)
    return wanted

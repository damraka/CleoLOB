"""Ledger-gated access to v0.6 datasets (Tardis public first-of-month samples).

Every read of a dataset is recorded in the v0.6 ledger with its dataset id,
analysis, *use* and exact source hashes; the ledger refuses uses not permitted
for the dataset's role. For a holdout the first ``download`` access is written
before any byte is requested, so the holdout is consumed from that moment.
A failed holdout download is retried once; a second failure is recorded as an
``attempt`` with outcome NOT_AVAILABLE and no other date or instrument is used.
Raw files stay in ignored directories and are never redistributed.
"""
from __future__ import annotations

from pathlib import Path

from ..data_download import download_tardis_sample
from ..experiments.registry import PROJECT_ROOT, sha256_file
from . import protocol as pr

DATA_DIR = "data/v06/tardis"
SEARCH_DIRS = ("data/public", "data/v05/tardis", DATA_DIR)
_TYPES = {"incremental_book_L2": "l2", "trades": "trades"}
MAX_BYTES = 512 * 1024**2
MAX_EXPANDED_BYTES = 6 * 1024**3


class DataNotAvailable(RuntimeError):
    """A registered dataset could not be obtained; dependent results are NOT_AVAILABLE."""


def file_name(venue: str, data_type: str, date: str, symbol: str) -> str:
    return f"{venue}_{data_type}_{date}_{symbol}.csv.gz"


def dataset_files(dataset_id: str, *, root: Path = PROJECT_ROOT) -> dict[str, Path]:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    declaration = pr.dataset_declaration(protocol, dataset_id)
    files = {}
    for data_type in declaration["data_types"]:
        name = file_name(declaration["venue"], data_type, declaration["date"], declaration["instrument"])
        found = [root / d / name for d in SEARCH_DIRS if (root / d / name).is_file()]
        files[_TYPES[data_type]] = found[0] if found else root / DATA_DIR / name
    return files


def acquire(dataset_id: str, *, use: str, analysis: str, purpose: str, root: Path = PROJECT_ROOT,
            downloader=download_tardis_sample) -> dict[str, Path]:
    """Return local paths after recording the access; downloads missing files (ledger first)."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    declaration = pr.dataset_declaration(protocol, dataset_id)
    ledger = root / pr.LEDGER_PATH
    if use not in pr.ROLE_USES[declaration["role"]] or use == "download":
        raise pr.ProtocolError(f"{dataset_id}: use {use!r} is not permitted for role {declaration['role']}")
    files = dataset_files(dataset_id, root=root)
    missing = [kind for kind, path in files.items() if not path.is_file()]
    if missing:
        if declaration["role"] in pr.HOLDOUT_ROLES:
            pr.append_event(ledger, "access", {"dataset_id": dataset_id, "use": "download", "analysis": analysis,
                                               "purpose": f"download start: {purpose}", "files": sorted(missing)},
                            protocol=protocol)
        inverse = {v: k for k, v in _TYPES.items()}
        errors = []
        for kind in missing:
            for attempt in (1, 2):
                try:
                    downloader(declaration["venue"], declaration["instrument"], declaration["date"], inverse[kind],
                               root / DATA_DIR, max_bytes=MAX_BYTES, max_expanded_bytes=MAX_EXPANDED_BYTES)
                    break
                except (OSError, ValueError) as exc:
                    errors.append(f"{kind} attempt {attempt}: {type(exc).__name__}: {exc}")
            else:
                pr.append_event(ledger, "attempt", {
                    "analysis": analysis, "outcome": "NOT_AVAILABLE", "directory": DATA_DIR,
                    "dataset_id": dataset_id, "note": "download failed after one retry; no substitution",
                    "errors": errors}, protocol=protocol)
                raise DataNotAvailable(f"{dataset_id} NOT_AVAILABLE: {errors}")
        files = dataset_files(dataset_id, root=root)
    hashes = {path.name: sha256_file(path) for path in files.values()}
    pr.append_event(ledger, "access", {"dataset_id": dataset_id, "use": use, "analysis": analysis,
                                       "purpose": purpose, "source_sha256": hashes}, protocol=protocol)
    return files

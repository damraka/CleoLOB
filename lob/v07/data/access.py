"""Ledger-gated, stage-recorded access to v0.7 datasets.

Stages, each a ledger ``access`` entry with the dataset role and sealed design:

* ``downloaded`` — written *before* any network request for a missing file
  (the holdout is consumed from this moment), or when existing local files are
  first used;
* ``opened`` — source bytes hashed (hashes recorded; they may never change);
* ``parsed`` — the tape was built; the quality status is recorded;
* ``evaluated`` — written by the study when results are computed.

A failed download is retried once; a second failure is an ``attempt`` with
outcome NOT_AVAILABLE and no other date, instrument or venue is substituted. A
file above the registered size limit is NOT_AVAILABLE, never truncated.
"""
from __future__ import annotations

import json
from pathlib import Path

from ...data_download import download_tardis_sample
from ...experiments.registry import PROJECT_ROOT, sha256_file
from ..protocol import core as pr
from .schema import venue_spec
from .tape import cached_build

DATA_DIR = "data/v07/tardis"
SEARCH_DIRS = ("data/public", "data/v05/tardis", "data/v06/tardis", DATA_DIR)
_TYPES = {"incremental_book_L2": "l2", "trades": "trades"}


class DataNotAvailable(RuntimeError):
    """A registered dataset could not be obtained; dependent results are NOT_AVAILABLE."""


def _registry(root: Path) -> dict:
    return json.loads((root / pr.CONFIG_DIR / "dataset-registry.json").read_text(encoding="utf-8"))


def file_name(venue: str, data_type: str, date: str, symbol: str) -> str:
    return f"{venue}_{data_type}_{date}_{symbol}.csv.gz"


def dataset_files(dataset_id: str, *, root: Path = PROJECT_ROOT) -> dict[str, Path]:
    declaration = pr.dataset_declaration(pr.load_protocol(root / pr.PROTOCOL_PATH), dataset_id)
    if declaration["capability_level"] != "aggregate_l2":
        raise pr.ProtocolError(f"{dataset_id} is not a Tardis aggregate-L2 dataset")
    files = {}
    for data_type in declaration["data_types"]:
        name = file_name(declaration["venue"], data_type, declaration["date"], declaration["instrument"])
        found = [root / d / name for d in SEARCH_DIRS if (root / d / name).is_file()]
        files[_TYPES[data_type]] = found[0] if found else root / DATA_DIR / name
    return files


def _record(root: Path, dataset_id: str, role: str, design: str, use: str, stage: str, reason: str, **payload) -> None:
    pr.append_event(root / pr.LEDGER_PATH, "access", dataset=dataset_id, role=role, design=design, reason=reason,
                    protocol=pr.load_protocol(root / pr.PROTOCOL_PATH), root=root,
                    payload={"use": use, "stage": stage, **payload})


def acquire(dataset_id: str, *, use: str, design: str, purpose: str, root: Path = PROJECT_ROOT,
            downloader=download_tardis_sample) -> dict[str, Path]:
    """Local file paths after the ``downloaded`` and ``opened`` stages are ledgered."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    declaration = pr.dataset_declaration(protocol, dataset_id)
    role = declaration["role"]
    if use not in pr.ROLE_USES[role] or use == "download":
        raise pr.ProtocolError(f"{dataset_id}: use {use!r} is not permitted for role {role}")
    files = dataset_files(dataset_id, root=root)
    missing = sorted(kind for kind, path in files.items() if not path.is_file())
    holdout = role in pr.HOLDOUT_ROLES
    # The first stage is ledgered before any byte is requested (and validated: unsealed access raises here).
    _record(root, dataset_id, role, design, "download" if holdout else use, "downloaded",
            f"{'download start' if missing else 'local files used'}: {purpose}", files=missing)
    if missing:
        source = _registry(root)["source"]
        inverse = {v: k for k, v in _TYPES.items()}
        errors = []
        for kind in missing:
            for attempt in range(1, source["retries"] + 2):
                try:
                    downloader(declaration["venue"], declaration["instrument"], declaration["date"], inverse[kind],
                               root / DATA_DIR, max_bytes=source["max_compressed_bytes_per_file"],
                               max_expanded_bytes=source["max_expanded_bytes_per_file"])
                    break
                except (OSError, ValueError) as exc:
                    errors.append(f"{kind} attempt {attempt}: {type(exc).__name__}: {exc}")
            else:
                pr.append_event(root / pr.LEDGER_PATH, "attempt", design=design, dataset=dataset_id, role=role,
                                reason="download failed after one retry; no substitution", protocol=protocol,
                                root=root, payload={"outcome": "NOT_AVAILABLE", "directory": DATA_DIR,
                                                    "note": "dependent hypotheses are NOT_AVAILABLE",
                                                    "errors": errors})
                raise DataNotAvailable(f"{dataset_id} NOT_AVAILABLE: {errors}")
        files = dataset_files(dataset_id, root=root)
    hashes = {path.name: sha256_file(path) for path in files.values()}
    _record(root, dataset_id, role, design, use, "opened", f"source bytes hashed: {purpose}", source_sha256=hashes)
    return files


def load_tape(dataset_id: str, *, use: str, design: str, purpose: str, root: Path = PROJECT_ROOT,
              scale: float | None = None, downloader=download_tardis_sample):
    """Native (or lot-scaled) tape, quality report and metadata; records the ``parsed`` stage."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    declaration = pr.dataset_declaration(protocol, dataset_id)
    spec = venue_spec(declaration["venue"], declaration["instrument"])
    files = acquire(dataset_id, use=use, design=design, purpose=purpose, root=root, downloader=downloader)
    tape, quality = cached_build(files, tick=float(spec.tick), root=root)
    _record(root, dataset_id, declaration["role"], design, use, "parsed", f"tape built: {purpose}",
            quality=quality["status"], samples=int(len(tape.t)))
    meta = {"dataset_id": dataset_id, "role": declaration["role"], "venue": spec.venue, "instrument": spec.instrument,
            "tick": float(spec.tick), "venue_spec_sha256": spec.to_dict()["sha256"],
            "files": {k: v.name for k, v in files.items()}, "quality": quality}
    return (tape if scale is None else tape.rescaled(scale)), meta


def mark_evaluated(dataset_id: str, *, use: str, design: str, run: str, root: Path = PROJECT_ROOT) -> None:
    declaration = pr.dataset_declaration(pr.load_protocol(root / pr.PROTOCOL_PATH), dataset_id)
    _record(root, dataset_id, declaration["role"], design, use, "evaluated", f"results computed in {run}", run=run)


def mark_inspected(dataset_id: str, *, run: str, root: Path = PROJECT_ROOT) -> None:
    pr.append_event(root / pr.LEDGER_PATH, "inspect", dataset=dataset_id, reason="first outcome inspection",
                    protocol=pr.load_protocol(root / pr.PROTOCOL_PATH), root=root, payload={"run": run})

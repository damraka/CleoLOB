"""Registered primary/backup acquisition with ledgered hashes and failover (M18), plus the source census.

``acquire`` fetches every source of one dataset into the content-addressed store and records an ``acquire`` ledger
entry with the raw SHA-256 of each source. Acquisition is not access: nothing is parsed, so the dataset stays fresh.
A required source the provider does not serve makes the dataset ``NOT_AVAILABLE``; ``fill_slots`` then moves to the
next *registered* backup of the same pool and stream, in registered order, and never invents a substitute.

``census`` is a HEAD-only availability survey (reachability and size; not quality) used for feasibility planning.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import store as S
from .connectors import base as CB


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    connector: str
    segment: str
    symbol: str
    date: str
    role: str
    pool: str
    slot: int
    slot_kind: str                 # "primary" | "backup"

    @property
    def stream(self) -> str:
        return f"{self.connector}-{self.segment}:{self.symbol}"


def dataset_id(connector: str, segment: str, symbol: str, day: str) -> str:
    return f"{connector}-{segment}-{symbol.lower()}-{day}"


@dataclass
class Acquisition:
    spec: DatasetSpec
    status: str                    # "AVAILABLE" | "NOT_AVAILABLE"
    files: dict[str, Path] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)
    reason: str | None = None


def acquire(spec: DatasetSpec, raw: S.RawStore, *, ws=None, max_bytes: int = 4 << 30, fetch=None) -> Acquisition:
    conn = CB.get(spec.connector)
    files, digests = {}, {}
    for src in conn.sources(spec.segment, spec.symbol, spec.date):
        try:
            checksum = S.published_sha256(src.checksum_url) if src.checksum_url else None
            got = (fetch or raw.fetch)(src.url, max_bytes=max_bytes, checksum=checksum)
        except S.NotAvailable as exc:
            if src.optional:
                continue
            if ws is not None:
                ws.append("attempt", reason=f"acquire {spec.id}: NOT_AVAILABLE", payload={
                    "study": "acquire", "run": spec.id, "outcome": "NOT_AVAILABLE", "namespace": "dev",
                    "detail": str(exc)})
            return Acquisition(spec, "NOT_AVAILABLE", reason=str(exc))
        files[src.name], digests[src.name] = got.path, got.sha256
    if ws is not None:
        ws.append("acquire", dataset=spec.id, reason=f"raw sources of {spec.id} (not parsed)",
                  payload={"sha256": digests, "role": spec.role, "pool": spec.pool})
    return Acquisition(spec, "AVAILABLE", files, digests)


def fill_slots(specs: list[DatasetSpec], needed: int, raw: S.RawStore, *, ws=None, fetch=None) -> list[Acquisition]:
    """Primaries first, then registered backups in order, until ``needed`` datasets are AVAILABLE."""
    ordered = sorted(specs, key=lambda s: (s.slot_kind != "primary", s.slot))
    got: list[Acquisition] = []
    for spec in ordered:
        if sum(a.status == "AVAILABLE" for a in got) >= needed:
            break
        got.append(acquire(spec, raw, ws=ws, fetch=fetch))
    return got


def census(urls: list[str], *, workers: int = 6) -> list[dict]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(S.probe, urls))


__all__ = ["DatasetSpec", "Acquisition", "dataset_id", "acquire", "fill_slots", "census"]

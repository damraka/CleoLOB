"""Ledgered day-cache store shared by every study (download -> normalize -> reconstruct -> quality -> cache).

``prepare`` is the only way a study obtains real data. For one registered dataset it:

1. acquires the raw sources (``sources.acquire``: hashing, checksums, ``downloaded`` ledger stage);
2. builds the day cache (grid normalization, reconstruction, event inference, quality counters);
3. applies the registered quality decision; a ``QUALITY_REJECTED`` day is retained as a ledgered attempt and
   ``DataNotAvailable`` is raised so the caller moves to the next registered backup slot;
4. records the ``parsed`` stage with the cache digest, quality status and grid anomalies.

A cache that already exists is used only if its digest equals the ledgered ``parsed`` digest (otherwise the
mismatch is an error, never silently rebuilt). Caches are derived data under ``data/v09/cache`` (gitignored).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .. import protocol as pr, quality
from . import daycache, sources
from .sources import DataNotAvailable


def _parsed(state: pr.LedgerState, dataset_id: str) -> dict | None:
    for access in state.accesses.get(dataset_id, []):
        if access.get("stage") == "parsed" and access.get("cache_sha256"):
            return access
    return None


def prepare(dataset_id: str, *, use: str, design: str, purpose: str, root: Path) -> tuple[dict, dict]:
    """Return (cache arrays, parsed-stage record) for one registered dataset."""
    root = Path(root)
    protocol, state = pr.state_of(root)
    d = pr.dataset_declaration(protocol, dataset_id)
    path = daycache.cache_path(root, dataset_id)
    record = _parsed(state, dataset_id)
    if record is not None:
        if not path.is_file():
            raise FileNotFoundError(f"{dataset_id}: ledgered cache missing; rebuild with rebuild_missing()")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["cache_sha256"]:
            raise ValueError(f"{dataset_id}: cache bytes differ from the ledgered parsed digest")
        return daycache.load(path), record
    raw = sources.acquire(dataset_id, use=use, design=design, purpose=purpose, root=root)
    if d["venue"].startswith("bybit"):
        cache = daycache.build_bybit(raw["l2"], raw["trades"], day=d["date"], tick=d["tick_size"],
                                     lot=d["lot_size"], segment=d["venue"].split("-")[1])
    else:
        cache = daycache.build_binance(raw["l1"], raw["trades"], day=d["date"], tick=d["tick_size"],
                                       lot=d["lot_size"])
    decision = quality.decide(cache["quality"], protocol["data_quality"]["thresholds"])
    cache["quality"]["decision"] = decision
    cache["quality"]["profile_v09"] = quality.profile(cache)
    if decision["status"] != "AVAILABLE":
        sources.quality_rejected(dataset_id, design=design, root=root,
                                 report={"failed": decision["failed"], "checks": decision["checks"]})
        raise DataNotAvailable(f"{dataset_id}: QUALITY_REJECTED ({decision['failed']})")
    digest = daycache.save(path, cache)
    entry = sources.mark(dataset_id, stage="parsed", use=use, design=design, root=root,
                         reason=f"day cache built for {purpose}", cache_sha256=digest,
                         quality_status=decision["status"], grid_anomalies=cache["quality"]["grid"]["anomalies"]
                         if "grid" in cache["quality"] else [], cache_schema=daycache.SCHEMA)
    return cache, entry["payload"]


def rebuild_missing(dataset_id: str, *, root: Path) -> str:
    """Rebuild a ledgered cache from the hash-verified raw files; must reproduce the ledgered digest."""
    root = Path(root)
    protocol, state = pr.state_of(root)
    d = pr.dataset_declaration(protocol, dataset_id)
    record = _parsed(state, dataset_id)
    if record is None:
        raise ValueError(f"{dataset_id} has no parsed stage to rebuild")
    known = sources.recorded_hashes(state, dataset_id)
    raw = {n: sources.local_path(root, d, n) for n in d["sources"]}
    for n, p in raw.items():
        if sources.sha256_file(p) != known.get(n):
            raise ValueError(f"{dataset_id}/{n}: raw bytes differ from the ledger")
    if d["venue"].startswith("bybit"):
        cache = daycache.build_bybit(raw["l2"], raw["trades"], day=d["date"], tick=d["tick_size"], lot=d["lot_size"],
                                     segment=d["venue"].split("-")[1])
    else:
        cache = daycache.build_binance(raw["l1"], raw["trades"], day=d["date"], tick=d["tick_size"], lot=d["lot_size"])
    cache["quality"]["decision"] = quality.decide(cache["quality"], protocol["data_quality"]["thresholds"])
    cache["quality"]["profile_v09"] = quality.profile(cache)
    return daycache.save(daycache.cache_path(root, dataset_id), cache)


def resolve_units(root: Path, pool: str, streams: list[str], n: int) -> dict:
    """Registered slot resolution for several streams (primaries in order, failures replaced by backups)."""
    protocol, state = pr.state_of(Path(root))
    return {s: sources.resolve(protocol, state, pool, s, n) for s in streams}

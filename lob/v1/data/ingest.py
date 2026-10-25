"""Streaming, chunked ingestion of raw archives into columnar canonical-event files (M15, M43).

``ingest`` drives a connector's event iterator chunk by chunk into a Parquet file (one row group per chunk, so memory
stays bounded by the chunk size), writes the day's trades to a second Parquet file and a metadata document with the
capability, grid counters, structural validation and the raw source hashes. Reading back is streaming too
(``iter_events``) or memory-mapped via Arrow (``read_events``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .schema import EVENT_DTYPE, empty, validate_events

SCHEMA = "cleolob-v1-events-1"


def _arrow():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


def _table(chunk: np.ndarray):
    pa, _ = _arrow()
    return pa.table({name: chunk[name] for name in EVENT_DTYPE.names})


def write_events(chunks, path: Path) -> dict:
    """Write canonical event chunks to ``path`` (Parquet, zstd); returns streaming validation counters."""
    pa, pq = _arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    writer = None
    total = {"events": 0, "ts_backwards": 0, "negative_sizes": 0, "nonfinite_sizes": 0, "by_kind": {},
             "chunks": 0, "first_ts": None, "last_ts": None}
    last = None
    try:
        for chunk in chunks:
            if not len(chunk):
                continue
            v = validate_events(chunk)
            total["events"] += v["events"]
            total["ts_backwards"] += v["ts_backwards"]
            total["negative_sizes"] += v["negative_sizes"]
            total["nonfinite_sizes"] += v["nonfinite_sizes"]
            for k, c in v["by_kind"].items():
                total["by_kind"][k] = total["by_kind"].get(k, 0) + c
            ts = chunk["ts_exchange"]
            if last is not None and ts[0] < last:
                total["ts_backwards"] += 1
            last = int(ts[-1])
            total["first_ts"] = int(ts[0]) if total["first_ts"] is None else total["first_ts"]
            total["last_ts"] = last
            t = _table(chunk)
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            writer.write_table(t)
            total["chunks"] += 1
        if writer is None:
            writer = pq.ParquetWriter(tmp, _table(empty(0)).schema, compression="zstd")
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(path)
    return total


def read_events(path: Path, *, columns: list[str] | None = None) -> np.ndarray:
    _, pq = _arrow()
    t = pq.read_table(path, columns=columns, memory_map=True)
    out = np.zeros(t.num_rows, dtype=EVENT_DTYPE) if columns is None else None
    if out is None:
        return {c: t.column(c).to_numpy() for c in columns}
    for name in EVENT_DTYPE.names:
        out[name] = t.column(name).to_numpy()
    return out


def iter_events(path: Path) -> Iterator[np.ndarray]:
    _, pq = _arrow()
    f = pq.ParquetFile(path, memory_map=True)
    for i in range(f.num_row_groups):
        t = f.read_row_group(i)
        out = np.zeros(t.num_rows, dtype=EVENT_DTYPE)
        for name in EVENT_DTYPE.names:
            out[name] = t.column(name).to_numpy()
        yield out


def ingest(connector, files: dict[str, Path], instrument, out_dir: Path, *, capability: dict, sources: dict[str, str],
           chunk: int = 200_000) -> dict:
    """Book events -> ``book.parquet``, trades -> ``trades.parquet``, plus ``meta.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    book = write_events(connector.events(files, instrument, chunk=chunk), out_dir / "book.parquet")
    trades = None
    if hasattr(connector, "trades") and "trades" in files:
        tr = connector.trades(files, instrument)
        trades = write_events([tr], out_dir / "trades.parquet")
    meta = {"schema": SCHEMA, "instrument": {"venue": instrument.venue, "segment": instrument.segment,
                                             "symbol": instrument.symbol, "tick": instrument.tick,
                                             "lot": instrument.lot, "kind": instrument.kind},
            "capability": capability, "raw_sha256": sources, "book": book, "trades": trades}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


__all__ = ["write_events", "read_events", "iter_events", "ingest", "SCHEMA"]

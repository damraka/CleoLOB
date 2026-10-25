"""Bybit public-archive connector (M11 linear MBP, M13 spot): order-book snapshot+delta archives and trades.

Order-book archive (``quote-saver.bycsi.com``): a zip with one JSON message per line,
``{"topic", "type": "snapshot"|"delta", "ts": ms, "data": {"s", "b": [[px, sz], ...], "a": [...], "u", "seq"}}``;
sizes are absolute level sizes (``"0"`` removes the level), ``u`` increments by one between deltas and a snapshot
resets the book.

**Depth eras** (verified by HTTP probes on 2026-10-25, see ``docs/v1-data-sources.md``): the linear archive is
``ob500`` up to 2025-08-20 and ``ob200`` from 2025-08-21 (``ob500`` returns 404 from that date); the spot archive is
``ob200``. The era is part of the dataset capability (``depth_levels``), so a 500-level and a 200-level day are never
pooled silently.

Trades (``public.bybit.com``): gzip CSV; linear ``timestamp`` (s), ``side`` (taker), ``size``, ``price``,
``trdMatchID``; spot ``id``, ``timestamp`` (ms), ``price``, ``volume``, ``side``.

Public L2 has no order ids: queue positions are never observed (``queue_observability = aggregate_level``) and
cancellations are inferred as depth decreases not explained by prints.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
from pathlib import Path
from typing import Iterator
import zipfile

import numpy as np

from ..schema import NO_TS, Capability, EventKind, Instrument, empty
from .base import Licence, Source, register

L2_URL = "https://quote-saver.bycsi.com/orderbook/{segment}/{sym}/{day}_{sym}_ob{depth}.data.zip"
TRADES_URL = {"linear": "https://public.bybit.com/trading/{sym}/{sym}{day}.csv.gz",
              "spot": "https://public.bybit.com/spot/{sym}/{sym}_{day}.csv.gz"}
LINEAR_OB200_FROM = "2025-08-21"

# registered instrument grids; the observed grid is verified at access (normalization records anomalies)
INSTRUMENTS = {
    ("linear", "BTCUSDT"): (0.1, 0.001, "linear_perpetual"),
    ("linear", "ETHUSDT"): (0.01, 0.01, "linear_perpetual"),
    ("linear", "SOLUSDT"): (0.01, 0.1, "linear_perpetual"),
    ("linear", "XRPUSDT"): (0.0001, 1.0, "linear_perpetual"),
    ("linear", "DOGEUSDT"): (0.00001, 1.0, "linear_perpetual"),
    ("spot", "BTCUSDT"): (0.01, 0.000001, "spot"),
    ("spot", "ETHUSDT"): (0.01, 0.00001, "spot"),
    ("spot", "SOLUSDT"): (0.01, 0.001, "spot"),
    ("spot", "XRPUSDT"): (0.0001, 0.01, "spot"),
    ("spot", "DOGEUSDT"): (0.00001, 0.1, "spot"),
}


class ParseError(ValueError):
    pass


def depth_for(segment: str, date: str) -> int:
    if segment == "spot":
        return 200
    if segment == "linear":
        return 200 if date >= LINEAR_OB200_FROM else 500
    raise ValueError(f"unknown Bybit segment {segment}")


def book_messages(path: Path) -> Iterator[dict]:
    """Raw messages in file order (never re-sorted: ordering problems are measured, not hidden)."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ParseError(f"expected one member in {Path(path).name}, found {len(names)}")
        with archive.open(names[0]) as handle:
            for line in io.BufferedReader(handle, buffer_size=1 << 20):
                if line.strip():
                    yield json.loads(line)


class Grid:
    """Decimal-string -> integer tick/lot mapping with off-grid counters (never silent)."""

    def __init__(self, tick: float, lot: float) -> None:
        self.tick, self.lot = float(tick), float(lot)
        self.off_tick = self.off_lot = self.nonfinite = self.negative = 0

    def price(self, s: str) -> int | None:
        x = float(s) / self.tick
        if not math.isfinite(x):
            self.nonfinite += 1
            return None
        k = round(x)
        if abs(x - k) > 1e-6 * max(1.0, abs(x)):
            self.off_tick += 1
        return int(k)

    def size(self, s: str) -> float:
        x = float(s) / self.lot
        if not math.isfinite(x):
            self.nonfinite += 1
            return 0.0
        if x < 0:
            self.negative += 1
            return 0.0
        k = round(x)
        if abs(x - k) > 1e-6 * max(1.0, x):
            self.off_lot += 1
            return x
        return float(k)

    def counters(self) -> dict:
        return {"off_tick": self.off_tick, "off_lot": self.off_lot, "nonfinite": self.nonfinite,
                "negative": self.negative}


def _levels(levels, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized decimal-string -> (ticks, lots) for one side of one message, with the grid counters."""
    a = np.asarray(levels, dtype=np.float64).reshape(-1, 2)
    x = a[:, 0] / grid.tick
    q = a[:, 1] / grid.lot
    finite = np.isfinite(x) & np.isfinite(q)
    if not finite.all():
        grid.nonfinite += int((~finite).sum())
        x, q = x[finite], q[finite]
    k = np.rint(x)
    grid.off_tick += int((np.abs(x - k) > 1e-6 * np.maximum(1.0, np.abs(x))).sum())
    neg = q < 0
    if neg.any():
        grid.negative += int(neg.sum())
        q = np.where(neg, 0.0, q)
    kq = np.rint(q)
    off = np.abs(q - kq) > 1e-6 * np.maximum(1.0, q)
    grid.off_lot += int(off.sum())
    return k.astype(np.int64), np.where(off, q, kq)


def book_events(messages, grid: Grid, *, chunk: int = 200_000) -> Iterator[np.ndarray]:
    """Canonical MBP events: SNAPSHOT_BEGIN, LEVEL_SET..., SNAPSHOT_END for snapshots; LEVEL_SET for deltas."""
    buf = empty(chunk)
    n = 0
    marker = empty(1)

    def flush():
        nonlocal n
        out = buf[:n].copy()
        n = 0
        return out

    for m in messages:
        ts = int(m["ts"]) * 1_000_000
        data = m["data"]
        seq = int(data.get("u", -1))
        snap = m.get("type") == "snapshot"
        parts = []
        for side, key in ((1, "b"), (-1, "a")):
            levels = data.get(key)
            if levels:
                parts.append((side,) + _levels(levels, grid))
        size = sum(len(p[1]) for p in parts) + (2 if snap else 0)
        if size > chunk:
            raise ParseError(f"one message holds {size} events, more than the chunk size {chunk}")
        if n + size > chunk:
            yield flush()
        if snap:
            marker["ts_exchange"], marker["seq"], marker["kind"] = ts, seq, EventKind.SNAPSHOT_BEGIN
            buf[n] = marker[0]
            n += 1
        for side, px, sz in parts:
            k = len(px)
            sl = buf[n:n + k]
            sl["ts_exchange"], sl["ts_receive"], sl["seq"], sl["kind"] = ts, NO_TS, seq, EventKind.LEVEL_SET
            sl["side"], sl["price"], sl["size"], sl["order_id"], sl["aux"] = side, px, sz, -1, -1
            n += k
        if snap:
            marker["ts_exchange"], marker["seq"], marker["kind"] = ts, seq, EventKind.SNAPSHOT_END
            buf[n] = marker[0]
            n += 1
    if n:
        yield flush()


def seconds_to_ns(text: str) -> int:
    """Exact decimal seconds -> integer ns (float multiplication shifts prints across bin edges)."""
    whole, _, frac = text.strip().partition(".")
    if len(frac) > 9:
        raise ParseError(f"timestamp {text!r} finer than 1 ns")
    return int(whole) * 1_000_000_000 + (int(frac.ljust(9, "0")) if frac else 0)


def trade_events(path: Path, segment: str, grid: Grid) -> np.ndarray:
    """All prints of a day as canonical TRADE events (side = aggressor; aux = per-file ordinal)."""
    ts, px, sz, side = [], [], [], []
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if segment == "linear":
                ts.append(seconds_to_ns(row["timestamp"]))
                sz.append(row["size"])
            else:
                ts.append(int(row["timestamp"]) * 1_000_000)
                sz.append(row["volume"])
            px.append(row["price"])
            s = row["side"].strip().lower()
            side.append(1 if s == "buy" else -1 if s == "sell" else 0)
    out = empty(len(ts))
    out["ts_exchange"] = np.asarray(ts, np.int64)
    out["kind"] = EventKind.TRADE
    out["side"] = np.asarray(side, np.int8)
    out["price"] = [grid.price(p) or 0 for p in px]
    out["size"] = [grid.size(s) for s in sz]
    out["aux"] = np.arange(len(ts), dtype=np.int64)
    return out


class BybitConnector:
    name = "bybit"

    def instrument(self, segment: str, symbol: str, date: str) -> Instrument:
        if (segment, symbol) not in INSTRUMENTS:
            raise KeyError(f"no registered Bybit instrument {segment}:{symbol}")
        tick, lot, kind = INSTRUMENTS[(segment, symbol)]
        return Instrument("bybit", segment, symbol, tick, lot, kind)

    def sources(self, segment: str, symbol: str, date: str) -> list[Source]:
        depth = depth_for(segment, date)
        return [Source("l2", L2_URL.format(segment=segment, sym=symbol, day=date, depth=depth), f"bybit-ob{depth}"),
                Source("trades", TRADES_URL[segment].format(sym=symbol, day=date), f"bybit-trades-{segment}")]

    def capability(self, segment: str, symbol: str, date: str) -> Capability:
        return Capability(book="l2", depth_levels=depth_for(segment, date), order_ids=False, trades=True,
                          trade_side=True, trade_ids=True, exchange_ts_resolution_ns=1_000_000, receive_ts=False,
                          sequence_numbers=True, snapshots=True, queue_observability="aggregate_level",
                          cancel_visibility="inferred_from_depth_minus_trades",
                          notes=(f"depth era: ob{depth_for(segment, date)}",
                                 "archive messages carry exchange time only; no receive time"))

    def licence(self) -> Licence:
        return Licence("Bybit public market-data archive", "https://www.bybit.com/en/announcement-info/terms/",
                       "2026-10-25", "non-commercial research; raw data never redistributed", redistribution=False,
                       commercial=False)

    def events(self, files: dict[str, Path], instrument: Instrument, *, chunk: int = 200_000) -> Iterator[np.ndarray]:
        grid = Grid(instrument.tick, instrument.lot)
        yield from book_events(book_messages(files["l2"]), grid, chunk=chunk)

    def trades(self, files: dict[str, Path], instrument: Instrument) -> np.ndarray:
        return trade_events(files["trades"], instrument.segment, Grid(instrument.tick, instrument.lot))


CONNECTOR = register(BybitConnector())

__all__ = ["BybitConnector", "CONNECTOR", "Grid", "book_messages", "book_events", "trade_events", "depth_for",
           "INSTRUMENTS", "LINEAR_OB200_FROM", "ParseError"]

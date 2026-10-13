"""Deterministic event-sourced L2 reconstruction (M20), replay consistency (M21) and quality counters (M13-M16).

Two implementations of the same semantics:

* ``ReferenceBook`` - a slow, obviously-correct dict book (price tick -> size lots);
* ``DenseBook`` - the optimized book: one dense float array per side indexed by tick offset from a movable base,
  with incrementally maintained best levels and re-centering when prices leave the window.

``replay`` drives either book through a message stream; the differential tests require identical touch prices,
visible depth and quality counters from both on real and fuzzed streams.

Semantics (registered): a ``snapshot`` replaces the whole book; a ``delta`` sets absolute level sizes (``0``
removes). Update-id continuity (``u == previous u + 1`` between deltas) is checked; a gap, a duplicate (same ``u``)
or an out-of-order message (smaller ``u``) is counted and the message is still applied in file order (nothing is
reordered or dropped silently). A crossed book (best bid >= best ask) after a message is counted. Prices and sizes
are mapped to integer ticks/lots on the registered grid; off-grid values are counted and rounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

VISIBLE = 10


@dataclass
class Counters:
    messages: int = 0
    snapshots: int = 0
    deltas: int = 0
    sequence_gaps: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    crossed: int = 0
    empty_side: int = 0
    off_tick: int = 0
    off_lot: int = 0
    negative_size: int = 0
    nonfinite: int = 0
    ts_backwards: int = 0
    recenters: int = 0
    truncated_levels: int = 0
    max_gap_ms: int = 0
    long_gap_ms_total: int = 0                          # summed duration of message gaps longer than 5 s
    first_ts: int | None = None
    last_ts: int | None = None
    gap_starts_ms: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["gap_starts_ms"] = list(self.gap_starts_ms[:50])
        return d


class _Grid:
    def __init__(self, tick: float, lot: float, counters: Counters) -> None:
        self.tick, self.lot, self.c = tick, lot, counters

    def price(self, s: str) -> int:
        x = float(s) / self.tick
        k = round(x)
        if not math.isfinite(x):
            self.c.nonfinite += 1
            raise ValueError("non-finite price")
        if abs(x - k) > 1e-6 * max(1.0, abs(x)):
            self.c.off_tick += 1
        return int(k)

    def size(self, s: str) -> float:
        x = float(s) / self.lot
        if not math.isfinite(x):
            self.c.nonfinite += 1
            return 0.0
        if x < 0:
            self.c.negative_size += 1
            return 0.0
        k = round(x)
        if abs(x - k) > 1e-6 * max(1.0, x):
            self.c.off_lot += 1
            return x
        return float(k)


class ReferenceBook:
    """Slow reference: dict price -> size per side."""

    def __init__(self) -> None:
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def set(self, side: int, price: int, size: float) -> None:
        book = self.bids if side > 0 else self.asks
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size

    def best(self, side: int) -> int | None:
        book = self.bids if side > 0 else self.asks
        if not book:
            return None
        return max(book) if side > 0 else min(book)

    def visible(self, side: int, k: int = VISIBLE) -> np.ndarray:
        """Sizes at touch, touch-1 tick, ... (bids) / touch, touch+1 tick, ... (asks); contiguous ticks."""
        best = self.best(side)
        out = np.zeros(k)
        if best is None:
            return out
        book = self.bids if side > 0 else self.asks
        step = -1 if side > 0 else 1
        for i in range(k):
            out[i] = book.get(best + step * i, 0.0)
        return out

    def levels(self, side: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        book = self.bids if side > 0 else self.asks
        prices = sorted(book, reverse=side > 0)[:n]
        return np.asarray(prices, np.int64), np.asarray([book[p] for p in prices], float)


class DenseBook:
    """Dense tick-indexed book with incremental best tracking and re-centering."""

    def __init__(self, width: int = 1 << 20, counters: Counters | None = None) -> None:
        self.width = width
        self.base: int | None = None
        self.bid = np.zeros(width)
        self.ask = np.zeros(width)
        self.best_bid = -1
        self.best_ask = width
        self.c = counters or Counters()

    def reset(self) -> None:
        self.bid[:] = 0.0
        self.ask[:] = 0.0
        self.best_bid, self.best_ask = -1, self.width

    def _index(self, price: int) -> int:
        if self.base is None:
            self.base = price - self.width // 2
        i = price - self.base
        if not 0 <= i < self.width:
            self._recenter(price)
            i = price - self.base
        return i

    def _recenter(self, around: int) -> None:
        """Shift the window so ``around`` is central; levels falling outside are dropped and counted."""
        new_base = around - self.width // 2
        shift = new_base - self.base
        for arr in (self.bid, self.ask):
            dropped = np.count_nonzero(arr[:shift]) if shift > 0 else np.count_nonzero(arr[self.width + shift:])
            self.c.truncated_levels += int(dropped)
            if shift > 0:
                arr[:-shift] = arr[shift:].copy()
                arr[-shift:] = 0.0
            else:
                arr[-shift:] = arr[:shift].copy()
                arr[:-shift] = 0.0
        self.base = new_base
        self.c.recenters += 1
        self._rescan()

    def _rescan(self) -> None:
        nz = np.flatnonzero(self.bid)
        self.best_bid = int(nz[-1]) if len(nz) else -1
        nz = np.flatnonzero(self.ask)
        self.best_ask = int(nz[0]) if len(nz) else self.width

    def set(self, side: int, price: int, size: float) -> None:
        if size <= 0 and (self.base is None or not 0 <= price - self.base < self.width):
            return          # removing a level outside the window: it is not present (never re-center for it)
        i = self._index(price)
        if side > 0:
            self.bid[i] = size if size > 0 else 0.0
            if size > 0 and i > self.best_bid:
                self.best_bid = i
            elif size <= 0 and i == self.best_bid:
                j = i
                while j >= 0 and self.bid[j] <= 0:
                    j -= 1
                    if i - j > 4096:
                        nz = np.flatnonzero(self.bid[:j + 1])
                        j = int(nz[-1]) if len(nz) else -1
                        break
                self.best_bid = j
        else:
            self.ask[i] = size if size > 0 else 0.0
            if size > 0 and i < self.best_ask:
                self.best_ask = i
            elif size <= 0 and i == self.best_ask:
                j = i
                while j < self.width and self.ask[j] <= 0:
                    j += 1
                    if j - i > 4096:
                        nz = np.flatnonzero(self.ask[j:])
                        j = j + int(nz[0]) if len(nz) else self.width
                        break
                self.best_ask = j

    def best(self, side: int) -> int | None:
        if side > 0:
            return None if self.best_bid < 0 else self.best_bid + self.base
        return None if self.best_ask >= self.width else self.best_ask + self.base

    def visible(self, side: int, k: int = VISIBLE) -> np.ndarray:
        out = np.zeros(k)
        if side > 0:
            if self.best_bid < 0:
                return out
            lo = max(0, self.best_bid - k + 1)
            seg = self.bid[lo:self.best_bid + 1][::-1]
        else:
            if self.best_ask >= self.width:
                return out
            seg = self.ask[self.best_ask:self.best_ask + k]
        out[: len(seg)] = seg
        return out

    def levels(self, side: int, n: int, span: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
        if side > 0:
            if self.best_bid < 0:
                return np.zeros(0, np.int64), np.zeros(0)
            lo = max(0, self.best_bid - span)
            idx = np.flatnonzero(self.bid[lo:self.best_bid + 1])[::-1][:n] + lo
            return idx.astype(np.int64) + self.base, self.bid[idx].copy()
        if self.best_ask >= self.width:
            return np.zeros(0, np.int64), np.zeros(0)
        hi = min(self.width, self.best_ask + span + 1)
        idx = np.flatnonzero(self.ask[self.best_ask:hi])[:n] + self.best_ask
        return idx.astype(np.int64) + self.base, self.ask[idx].copy()


def apply_message(book, message: dict, grid: _Grid, state: dict) -> None:
    """Apply one archive message with the registered semantics, updating the quality counters."""
    c = grid.c
    c.messages += 1
    ts = int(message["ts"])
    if c.first_ts is None:
        c.first_ts = ts
    if c.last_ts is not None:
        if ts < c.last_ts:
            c.ts_backwards += 1
        gap = ts - c.last_ts
        if gap > c.max_gap_ms:
            c.max_gap_ms = gap
        if gap > 5_000:
            c.gap_starts_ms.append(c.last_ts)
            c.long_gap_ms_total += gap
    c.last_ts = max(ts, c.last_ts or ts)
    data = message["data"]
    u = int(data.get("u", -1))
    kind = message.get("type")
    prev = state.get("u")
    if kind == "snapshot":
        c.snapshots += 1
        book.reset()
    else:
        c.deltas += 1
        if prev is not None:
            if u == prev:
                c.duplicates += 1
            elif u < prev:
                c.out_of_order += 1
            elif u != prev + 1:
                c.sequence_gaps += 1
    state["u"] = u if prev is None else max(u, prev)
    for side, key in ((1, "b"), (-1, "a")):
        for px, sz in data.get(key, ()):
            try:
                price = grid.price(px)
            except ValueError:
                continue
            book.set(side, price, grid.size(sz))
    bb, ba = book.best(1), book.best(-1)
    if bb is None or ba is None:
        c.empty_side += 1
    elif bb >= ba:
        c.crossed += 1


def replay(messages, *, tick: float, lot: float, book=None, every=None) -> tuple[object, Counters, list]:
    """Replay a message stream; ``every(book, message)`` is called after each message (for differential tests)."""
    counters = Counters()
    grid = _Grid(tick, lot, counters)
    book = book if book is not None else DenseBook(counters=counters)
    if isinstance(book, DenseBook):
        book.c = counters
    state: dict = {}
    trace = []
    for m in messages:
        apply_message(book, m, grid, state)
        if every is not None:
            trace.append(every(book, m))
    return book, counters, trace

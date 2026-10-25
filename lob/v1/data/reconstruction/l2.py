"""Deterministic MBP/L2 reconstruction from canonical events (M21) with snapshot/delta reconciliation and gap
detection (M23).

Semantics (registered):

* ``SNAPSHOT_BEGIN`` .. ``SNAPSHOT_END`` replaces the whole book (a reconciliation point);
* ``LEVEL_SET`` sets the absolute size of (side, price); 0 removes the level;
* events sharing (ts_exchange, seq) form one venue message and are applied atomically;
* update-id continuity between deltas (``seq == previous + 1``) is checked. A **gap** (jump), a duplicate or an
  out-of-order message makes the book *unreliable until the next snapshot*: every bin in that interval is
  invalid. v0.9 counted gaps but kept the book valid; v1 never treats a possibly incomplete book as observed;
* a crossed or one-sided book after a message makes that state invalid (counted).

Two books implement the same semantics: ``ReferenceBook`` (dict, obviously correct) and ``DenseBook`` (dense arrays
over a movable tick window, vectorized per message). ``replay`` drives either; the differential tests require
identical states.

``frames`` turns a day into the v0.8 ``EventFrame`` layout at 100 ms (touch prices, 10 contiguous visible levels,
inferred event counts, traded volume, validity) plus deep (non-contiguous, top-``deep_levels``) snapshots once a
second inside registered windows - the same layout as the v0.9 day caches, so v0.9-compatible replay applies.
Inferred event counts are an inference from aggregate data: an increase at a visible level is an *add*; a decrease
is first explained by passive-side prints of the same or previous bin, the remainder is a *cancel*.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone

import numpy as np

from ..schema import EventKind

VISIBLE = 10
BIN_MS = 100
DAY_BINS = 864_000


@dataclass
class Counters:
    messages: int = 0
    snapshots: int = 0
    deltas: int = 0
    sequence_gaps: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    crossed: int = 0
    one_sided: int = 0
    unreliable_messages: int = 0
    recenters: int = 0
    truncated_levels: int = 0
    max_gap_ms: int = 0
    long_gap_ms_total: int = 0
    gap_intervals: list = field(default_factory=list)     # (from ts_ns, to ts_ns) unreliable intervals

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gap_intervals"] = [list(x) for x in self.gap_intervals[:100]]
        return d


class ReferenceBook:
    """Slow reference: dict price -> size per side."""

    def __init__(self) -> None:
        self.side = {1: {}, -1: {}}

    def reset(self) -> None:
        self.side[1].clear()
        self.side[-1].clear()

    def set_many(self, side: int, prices: np.ndarray, sizes: np.ndarray) -> None:
        book = self.side[side]
        for p, q in zip(prices.tolist(), sizes.tolist()):
            if q <= 0:
                book.pop(p, None)
            else:
                book[p] = q

    def best(self, side: int) -> int | None:
        b = self.side[side]
        if not b:
            return None
        return max(b) if side > 0 else min(b)

    def size_at(self, side: int, prices: np.ndarray) -> np.ndarray:
        b = self.side[side]
        return np.asarray([b.get(p, 0.0) for p in prices.tolist()], float)

    def visible(self, side: int, k: int = VISIBLE) -> np.ndarray:
        best = self.best(side)
        if best is None:
            return np.zeros(k)
        step = -1 if side > 0 else 1
        return self.size_at(side, np.asarray([best + step * i for i in range(k)], np.int64))

    def levels(self, side: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        b = self.side[side]
        prices = sorted(b, reverse=side > 0)[:n]
        return np.asarray(prices, np.int64), np.asarray([b[p] for p in prices], float)


class DenseBook:
    """Dense tick-window book; one vectorized assignment per message side, incremental best tracking."""

    def __init__(self, width: int = 1 << 20, counters: Counters | None = None) -> None:
        self.width = width
        self.base: int | None = None
        self.arr = {1: np.zeros(width), -1: np.zeros(width)}
        self.best_i = {1: -1, -1: width}
        self.c = counters or Counters()

    def reset(self) -> None:
        self.arr[1][:] = 0.0
        self.arr[-1][:] = 0.0
        self.best_i = {1: -1, -1: self.width}

    def _recenter(self, around: int) -> None:
        new_base = around - self.width // 2
        shift = new_base - self.base
        for arr in self.arr.values():
            if abs(shift) >= self.width:
                self.c.truncated_levels += int(np.count_nonzero(arr))
                arr[:] = 0.0
                continue
            if shift > 0:
                self.c.truncated_levels += int(np.count_nonzero(arr[:shift]))
                arr[:-shift] = arr[shift:].copy()
                arr[-shift:] = 0.0
            else:
                self.c.truncated_levels += int(np.count_nonzero(arr[self.width + shift:]))
                arr[-shift:] = arr[:shift].copy()
                arr[:-shift] = 0.0
        self.base = new_base
        self.c.recenters += 1
        for s in (1, -1):
            self._rescan(s)

    def _rescan(self, side: int) -> None:
        nz = np.flatnonzero(self.arr[side])
        if side > 0:
            self.best_i[1] = int(nz[-1]) if len(nz) else -1
        else:
            self.best_i[-1] = int(nz[0]) if len(nz) else self.width

    def set_many(self, side: int, prices: np.ndarray, sizes: np.ndarray) -> None:
        if not len(prices):
            return
        if self.base is None:
            self.base = int(prices[0]) - self.width // 2
        idx = prices - self.base
        live = sizes > 0
        if live.any() and (idx[live].min() < 0 or idx[live].max() >= self.width):
            self._recenter(int(np.median(prices[live])))
            idx = prices - self.base
        inside = (idx >= 0) & (idx < self.width)
        idx, sz = idx[inside], np.where(sizes[inside] > 0, sizes[inside], 0.0)
        arr = self.arr[side]
        arr[idx] = sz
        b = self.best_i[side]
        if side > 0:
            add = idx[sz > 0]
            if len(add) and add.max() > b:
                b = int(add.max())
            if b >= 0 and arr[b] <= 0:
                lo = max(0, b - 4096)
                nz = np.flatnonzero(arr[lo:b + 1])
                b = int(nz[-1]) + lo if len(nz) else -1
                if b < 0:
                    nz = np.flatnonzero(arr[:lo])
                    b = int(nz[-1]) if len(nz) else -1
        else:
            add = idx[sz > 0]
            if len(add) and add.min() < b:
                b = int(add.min())
            if b < self.width and arr[b] <= 0:
                hi = min(self.width, b + 4097)
                nz = np.flatnonzero(arr[b:hi])
                b = int(nz[0]) + b if len(nz) else self.width
                if b >= self.width:
                    nz = np.flatnonzero(arr[hi:])
                    b = int(nz[0]) + hi if len(nz) else self.width
        self.best_i[side] = b

    def best(self, side: int) -> int | None:
        b = self.best_i[side]
        if side > 0:
            return None if b < 0 else b + self.base
        return None if b >= self.width else b + self.base

    def size_at(self, side: int, prices: np.ndarray) -> np.ndarray:
        if self.base is None:
            return np.zeros(len(prices))
        idx = np.asarray(prices, np.int64) - self.base
        ok = (idx >= 0) & (idx < self.width)
        out = np.zeros(len(idx))
        out[ok] = self.arr[side][idx[ok]]
        return out

    def visible(self, side: int, k: int = VISIBLE) -> np.ndarray:
        out = np.zeros(k)
        b = self.best_i[side]
        arr = self.arr[side]
        if side > 0:
            if b < 0:
                return out
            seg = arr[max(0, b - k + 1):b + 1][::-1]
        else:
            if b >= self.width:
                return out
            seg = arr[b:b + k]
        out[:len(seg)] = seg
        return out

    def levels(self, side: int, n: int, span: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
        b, arr = self.best_i[side], self.arr[side]
        if side > 0:
            if b < 0:
                return np.zeros(0, np.int64), np.zeros(0)
            lo = max(0, b - span)
            idx = np.flatnonzero(arr[lo:b + 1])[::-1][:n] + lo
        else:
            if b >= self.width:
                return np.zeros(0, np.int64), np.zeros(0)
            idx = np.flatnonzero(arr[b:min(self.width, b + span + 1)])[:n] + b
        return idx.astype(np.int64) + self.base, arr[idx].copy()


def messages(chunks):
    """Group canonical book events into venue messages: yields (ts_ns, seq, is_snapshot, events)."""
    carry = None
    for ev in chunks:
        if carry is not None and len(carry):
            ev = np.concatenate([carry, ev])
            carry = None
        if not len(ev):
            continue
        brk = np.flatnonzero((np.diff(ev["seq"]) != 0) | (np.diff(ev["ts_exchange"]) != 0)) + 1
        starts = np.r_[0, brk]
        ends = np.r_[brk, len(ev)]
        # the last message of a chunk may continue in the next chunk
        for s, e in zip(starts[:-1], ends[:-1]):
            m = ev[s:e]
            yield int(m["ts_exchange"][0]), int(m["seq"][0]), bool((m["kind"] == EventKind.SNAPSHOT_BEGIN).any()), m
        carry = ev[starts[-1]:]
    if carry is not None and len(carry):
        yield int(carry["ts_exchange"][0]), int(carry["seq"][0]), bool(
            (carry["kind"] == EventKind.SNAPSHOT_BEGIN).any()), carry


def apply(book, m: np.ndarray, snapshot: bool) -> None:
    if snapshot:
        book.reset()
    lv = m[m["kind"] == EventKind.LEVEL_SET]
    for side in (1, -1):
        s = lv[lv["side"] == side]
        if len(s):
            book.set_many(side, s["price"], s["size"])


class Reliability:
    """Update-id continuity: a gap/duplicate/out-of-order delta makes the book unreliable until a snapshot."""

    def __init__(self, c: Counters) -> None:
        self.c, self.prev, self.reliable, self.since = c, None, False, None
        self.last_ts = None

    def step(self, ts: int, seq: int, snapshot: bool) -> bool:
        c = self.c
        c.messages += 1
        if self.last_ts is not None:
            gap = (ts - self.last_ts) // 1_000_000
            c.max_gap_ms = max(c.max_gap_ms, int(gap))
            if gap > 5_000:
                c.long_gap_ms_total += int(gap)
        self.last_ts = ts
        if snapshot:
            c.snapshots += 1
            if not self.reliable and self.since is not None:
                c.gap_intervals.append((self.since, ts))
            self.reliable, self.prev, self.since = True, seq, None
            return True
        c.deltas += 1
        if self.prev is not None and seq >= 0:
            bad = None
            if seq == self.prev:
                c.duplicates += 1
                bad = "dup"
            elif seq < self.prev:
                c.out_of_order += 1
                bad = "ooo"
            elif seq != self.prev + 1:
                c.sequence_gaps += 1
                bad = "gap"
            if bad and self.reliable:
                self.reliable, self.since = False, ts
            self.prev = max(seq, self.prev)
        elif seq >= 0:
            self.prev = seq
        if not self.reliable:
            c.unreliable_messages += 1
        return self.reliable


def replay(chunks, book=None, *, every=None) -> tuple[object, Counters, list]:
    c = Counters()
    book = book if book is not None else DenseBook(counters=c)
    if isinstance(book, DenseBook):
        book.c = c
    rel = Reliability(c)
    trace = []
    for ts, seq, snap, m in messages(chunks):
        ok = rel.step(ts, seq, snap)
        apply(book, m, snap)
        bb, ba = book.best(1), book.best(-1)
        if bb is None or ba is None:
            c.one_sided += 1
        elif bb >= ba:
            c.crossed += 1
        if every is not None:
            trace.append(every(book, ts, ok))
    return book, c, trace


def day_start_ns(day: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()) * 1_000_000_000


def frames(chunks, trades: np.ndarray, *, day: str, deep_seconds: np.ndarray | None = None, deep_levels: int = 200,
           bins: int = DAY_BINS) -> dict:
    """100 ms day frames in the v0.9 day-cache layout (see module docstring); returns plain arrays."""
    t0 = day_start_ns(day)
    c = Counters()
    book = DenseBook(counters=c)
    rel = Reliability(c)
    tr = trades[np.argsort(trades["ts_exchange"], kind="stable")]
    tb = (tr["ts_exchange"] - t0) // (BIN_MS * 1_000_000)
    tr_side = tr["side"].astype(np.int8)
    traded: dict[tuple[int, int, int], float] = {}
    for b, p, q, s in zip(tb.tolist(), tr["price"].tolist(), tr["size"].tolist(), tr_side.tolist()):
        key = (b, -1 if s > 0 else 1, p)                      # buy prints consume asks, sell prints bids
        traded[key] = traded.get(key, 0.0) + q
    counts = np.zeros((bins, 6), np.int32)
    bid_px = np.zeros(bins, np.int64)
    ask_px = np.zeros(bins, np.int64)
    depth = np.zeros((bins, 2, VISIBLE), np.float32)
    valid = np.zeros(bins, bool)
    captured = np.zeros(bins, bool)
    secs = np.zeros(0, np.int64) if deep_seconds is None else np.asarray(deep_seconds, np.int64)
    deep_bin = secs * 1000 // BIN_MS
    deep_px = np.zeros((len(secs), 2, deep_levels), np.int64)
    deep_q = np.zeros((len(secs), 2, deep_levels), np.float32)
    deep_ok = np.zeros(len(secs), bool)
    decreases: dict[tuple[int, int], float] = {}
    cur, reliable = -1, False

    def capture(lo: int, hi: int) -> None:
        if lo > hi or lo >= bins:
            return
        hi = min(hi, bins - 1)
        captured[lo] = True
        bb, ba = book.best(1), book.best(-1)
        if not reliable or bb is None or ba is None or bb >= ba:
            return
        bid_px[lo], ask_px[lo] = bb, ba
        depth[lo, 0], depth[lo, 1] = book.visible(1), book.visible(-1)
        valid[lo] = True
        a, z = np.searchsorted(deep_bin, lo, "left"), np.searchsorted(deep_bin, hi, "right")
        if z > a:
            lv = [book.levels(s, deep_levels) for s in (1, -1)]
            for i in range(a, z):
                for k, (p, q) in enumerate(lv):
                    deep_px[i, k, :len(p)] = p
                    deep_q[i, k, :len(q)] = q
                deep_ok[i] = True

    def close(b: int) -> None:
        for (side, price), dq in decreases.items():
            left = dq
            for key in ((b - 1, side, price), (b, side, price)):
                avail = traded.get(key, 0.0)
                use = min(avail, left)
                if use > 0:
                    traded[key] = avail - use
                    left -= use
            if left > 1e-9:
                counts[b, 4 if side > 0 else 5] += 1
        decreases.clear()

    for ts, seq, snap, m in messages(chunks):
        b = (ts - t0) // (BIN_MS * 1_000_000)
        if b >= bins:
            break
        if b > cur:
            if cur >= 0:
                close(cur)
            if b >= 0:
                capture(max(cur + 1, 0), b)
            cur = b
        ok = rel.step(ts, seq, snap)
        if b >= 0 and not snap and reliable and ok:
            lv = m[m["kind"] == EventKind.LEVEL_SET]
            for side, col in ((1, 2), (-1, 3)):
                best = book.best(side)
                s = lv[lv["side"] == side]
                if best is None or not len(s):
                    continue
                dist = (best - s["price"]) if side > 0 else (s["price"] - best)
                near = (dist < VISIBLE) & (dist > -VISIBLE)
                if not near.any():
                    continue
                px, new = s["price"][near], s["size"][near]
                old = book.size_at(side, px)
                inc = new > old + 1e-9
                counts[b, col] += int(inc.sum())
                for p, d in zip(px[new < old - 1e-9].tolist(), (old - new)[new < old - 1e-9].tolist()):
                    decreases[(side, p)] = decreases.get((side, p), 0.0) + d
        apply(book, m, snap)
        reliable = ok
        bb, ba = book.best(1), book.best(-1)
        if bb is None or ba is None:
            c.one_sided += 1
        elif bb >= ba:
            c.crossed += 1
    if cur >= 0:
        close(cur)
        capture(cur + 1, bins - 1)
    src = np.where(captured, np.arange(bins), -1)
    np.maximum.accumulate(src, out=src)
    has = src >= 0
    j = np.where(has, src, 0)
    stale = (np.arange(bins) - j) * BIN_MS > 5_000
    bid_px, ask_px, depth = bid_px[j], ask_px[j], depth[j]
    valid = has & valid[j] & ~stale
    inside = (tb >= 0) & (tb < bins)
    buy, sell = inside & (tr_side > 0), inside & (tr_side < 0)
    np.add.at(counts[:, 0], tb[buy], 1)
    np.add.at(counts[:, 1], tb[sell], 1)
    volume = np.zeros((bins, 2), np.float32)
    np.add.at(volume[:, 0], tb[buy], tr["size"][buy])
    np.add.at(volume[:, 1], tb[sell], tr["size"][sell])
    mid = np.where(valid, (bid_px + ask_px) / 2.0, np.nan)
    mid = np.r_[mid, mid[-1]]
    idx = np.where(np.isfinite(mid), np.arange(len(mid)), -1)
    np.maximum.accumulate(idx, out=idx)
    mid = np.where(idx >= 0, mid[np.maximum(idx, 0)], np.nan)
    first = np.flatnonzero(np.isfinite(mid))
    if len(first):
        mid[: first[0]] = mid[first[0]]
    return {"counts": counts, "bid_px": bid_px, "ask_px": ask_px, "depth": depth, "volume": volume, "mid": mid,
            "valid": valid, "deep_s": secs, "deep_px": deep_px, "deep_q": deep_q, "deep_ok": deep_ok,
            "trade_t": (tr["ts_exchange"] - t0) / 1e9, "trade_px": tr["price"].astype(np.int64),
            "trade_q": tr["size"].astype(float), "trade_s": tr_side, "counters": c.to_dict()}


__all__ = ["Counters", "ReferenceBook", "DenseBook", "messages", "apply", "replay", "frames", "Reliability",
           "VISIBLE", "BIN_MS", "DAY_BINS", "day_start_ns"]

"""Market-data quality engine (M3).

Input is a canonical columnar event table (``EventTable``): every row is a book
level update (absolute size at a price; size 0 deletes the level) or a trade,
with exchange and receive timestamps in integer microseconds. The engine
reconstructs the book incrementally and reports structured flags; it never
repairs data (the protocol permits no silent repairs). Each flag has a count, a
rate, examples (row indices) and a PASS/WARN/FAIL status against registered
thresholds; the quality score is ``1 - sum(weight * min(rate / fail, 1))``.

Flags: missing timestamps, duplicate events, out-of-order receive times,
exchange-timestamp regressions, crossed books, locked books, negative sizes,
impossible depth transitions (non-finite or non-positive prices, deletes of
absent levels), long gaps, stale periods, zero-volume trades, price-grid and
lot-grid violations, and trades inconsistent with the reconstructed book.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path

import numpy as np

BOOK, TRADE = 0, 1
FLAGS = ("missing_timestamp", "duplicate_event", "out_of_order", "timestamp_regression", "crossed_book",
         "locked_book", "negative_size", "impossible_transition", "long_gap", "stale_period", "zero_volume_trade",
         "price_grid", "lot_grid", "trade_book_inconsistent")
THRESHOLDS = {  # (warn rate, fail rate) per relevant denominator
    "missing_timestamp": (1e-6, 1e-4), "duplicate_event": (1e-5, 1e-3), "out_of_order": (1e-5, 1e-3),
    "timestamp_regression": (1e-4, 1e-2), "crossed_book": (1e-5, 1e-3), "locked_book": (1e-4, 1e-2),
    "negative_size": (1e-9, 1e-6), "impossible_transition": (1e-6, 1e-4), "long_gap": (1e-5, 1e-3),
    "stale_period": (1e-5, 1e-3), "zero_volume_trade": (1e-5, 1e-3), "price_grid": (1e-6, 1e-4),
    "lot_grid": (1e-4, 1e-2), "trade_book_inconsistent": (1e-3, 5e-2)}
WEIGHTS = {f: 1.0 / len(FLAGS) for f in FLAGS}


@dataclass
class EventTable:
    kind: np.ndarray            # 0 book update, 1 trade
    exchange_us: np.ndarray     # int64 (0 or negative = missing)
    receive_us: np.ndarray
    side: np.ndarray            # book: +1 bid / -1 ask; trade: +1 buy / -1 sell aggressor
    price: np.ndarray
    size: np.ndarray
    snapshot: np.ndarray | None = None   # book rows that belong to a snapshot (reset)

    def __len__(self) -> int:
        return len(self.kind)


def from_tardis(l2: Path, trades: Path | None = None, *, max_rows: int = 2_000_000) -> EventTable:
    """Read Tardis incremental L2 (and trades) CSV(.gz) into an EventTable (bounded by ``max_rows`` each)."""
    import pandas as pd
    parts = []
    book = pd.read_csv(l2, nrows=max_rows)
    parts.append(pd.DataFrame({"kind": BOOK, "exchange_us": book["timestamp"], "receive_us": book["local_timestamp"],
                               "side": np.where(book["side"].str.lower() == "bid", 1, -1), "price": book["price"],
                               "size": book["amount"], "snapshot": book.get("is_snapshot", False).astype(bool)}))
    if trades is not None:
        tr = pd.read_csv(trades, nrows=max_rows)
        parts.append(pd.DataFrame({"kind": TRADE, "exchange_us": tr["timestamp"], "receive_us": tr["local_timestamp"],
                                   "side": np.where(tr["side"].str.lower() == "buy", 1, -1), "price": tr["price"],
                                   "size": tr["amount"], "snapshot": False}))
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("receive_us", kind="stable").reset_index(drop=True) if trades is not None else df
    return EventTable(df["kind"].to_numpy(np.int8), df["exchange_us"].fillna(-1).to_numpy(np.int64),
                      df["receive_us"].fillna(-1).to_numpy(np.int64), df["side"].to_numpy(np.int8),
                      df["price"].to_numpy(float), df["size"].to_numpy(float), df["snapshot"].to_numpy(bool))


def from_frame(frame, *, start_us: int = 1_600_000_000_000_000, tick: float = 1.0) -> EventTable:
    """Canonical events from an EventFrame: level updates where visible depth changed, plus recorded prints."""
    rows = []
    previous: dict[tuple[int, int], float] = {}
    for i in range(len(frame)):
        t = start_us + int(round((frame.t0 + i * 0.1) * 1e6))
        for side_index, sign in ((0, 1), (1, -1)):
            touch = frame.bid_px[i] if sign > 0 else frame.ask_px[i]
            current = {}
            for k in range(frame.depth.shape[2]):
                q = float(frame.depth[i, side_index, k])
                if q > 0:
                    current[(sign, int(touch - sign * k))] = q
            for key in sorted(set(previous) | set(current)):
                if key[0] != sign:
                    continue
                if previous.get(key) != current.get(key):
                    # removals sort before additions within a bin, so no transient locked/crossed state appears
                    rows.append((BOOK, t, t, sign, key[1] * tick, current.get(key, 0.0)))
            previous = {k: v for k, v in previous.items() if k[0] != sign} | current
    tt, tp, tq, ts = frame.trades()
    for a, b, c, d in zip(tt, tp, tq, ts):
        t = start_us + int(round(a * 1e6))
        rows.append((TRADE, t, t, int(d), float(b) * tick, float(c)))
    rows.sort(key=lambda r: (r[2], r[0], r[5] != 0))
    arr = np.asarray(rows, dtype=float).reshape(-1, 6)
    return EventTable(arr[:, 0].astype(np.int8), arr[:, 1].astype(np.int64), arr[:, 2].astype(np.int64),
                      arr[:, 3].astype(np.int8), arr[:, 4], arr[:, 5], np.zeros(len(arr), bool))


class _Side:
    """Book side with lazy-deletion heap for the best price."""

    def __init__(self, sign: int) -> None:
        self.sign, self.levels, self.heap = sign, {}, []

    def set(self, price: float, size: float) -> bool:
        existed = price in self.levels
        if size == 0:
            self.levels.pop(price, None)
            return existed
        self.levels[price] = size
        heapq.heappush(self.heap, -price if self.sign > 0 else price)
        return True

    def best(self) -> float | None:
        while self.heap:
            p = -self.heap[0] if self.sign > 0 else self.heap[0]
            if p in self.levels:
                return p
            heapq.heappop(self.heap)
        return None


def validate(table: EventTable, *, tick: float | None = None, lot: float | None = None, gap_s: float = 5.0,
             stale_s: float = 30.0, max_examples: int = 5) -> dict:
    n = len(table)
    counts = {f: 0 for f in FLAGS}
    examples: dict[str, list[int]] = {f: [] for f in FLAGS}

    def flag(name: str, index: int) -> None:
        counts[name] += 1
        if len(examples[name]) < max_examples:
            examples[name].append(int(index))

    missing = (table.exchange_us <= 0) | (table.receive_us <= 0)
    for i in np.nonzero(missing)[0]:
        flag("missing_timestamp", i)
    keys = np.column_stack([table.kind, table.exchange_us, table.receive_us, table.side, table.price, table.size])
    _, first = np.unique(keys, axis=0, return_index=True)
    for i in sorted(set(range(n)) - set(first.tolist())):
        flag("duplicate_event", i)
    for i in np.nonzero(np.diff(table.receive_us) < 0)[0] + 1:
        flag("out_of_order", i)
    for kind in (BOOK, TRADE):
        idx = np.nonzero(table.kind == kind)[0]
        ex = table.exchange_us[idx]
        for j in np.nonzero(np.diff(ex) < 0)[0] + 1:
            flag("timestamp_regression", idx[j])
    for i in np.nonzero(table.size < 0)[0]:
        flag("negative_size", i)
    for i in np.nonzero((table.kind == TRADE) & (table.size == 0))[0]:
        flag("zero_volume_trade", i)
    if tick:
        off = np.abs(table.price / tick - np.round(table.price / tick))
        for i in np.nonzero(off > 1e-6)[0]:
            flag("price_grid", i)
    if lot:
        off = np.abs(table.size / lot - np.round(table.size / lot))
        for i in np.nonzero((off > 1e-6) & (table.size != 0))[0]:
            flag("lot_grid", i)
    t = table.receive_us.astype(float) / 1e6
    if n > 1:
        gaps = np.diff(t)
        for i in np.nonzero(gaps > gap_s)[0] + 1:
            flag("long_gap", i)
    bids, asks = _Side(1), _Side(-1)
    last_change = t[0] if n else 0.0
    last_best = (None, None)
    trades_checked = 0
    for i in range(n):
        price, size = float(table.price[i]), float(table.size[i])
        if table.kind[i] == BOOK:
            if table.snapshot is not None and table.snapshot[i] and (i == 0 or not table.snapshot[i - 1]):
                bids, asks = _Side(1), _Side(-1)
            if not math.isfinite(price) or price <= 0 or not math.isfinite(size):
                flag("impossible_transition", i)
                continue
            side = bids if table.side[i] > 0 else asks
            if size >= 0 and not side.set(price, max(size, 0.0)) and size == 0:
                flag("impossible_transition", i)
            best = (bids.best(), asks.best())
            if best[0] is not None and best[1] is not None:
                if best[0] > best[1]:
                    flag("crossed_book", i)
                elif best[0] == best[1]:
                    flag("locked_book", i)
            if best != last_best:
                if t[i] - last_change > stale_s:
                    flag("stale_period", i)
                last_change, last_best = t[i], best
        else:
            bb, ba = bids.best(), asks.best()
            if bb is not None and ba is not None:
                trades_checked += 1
                # a buy aggressor cannot print below the bid, a sell aggressor not above the ask (sweeps are fine)
                slack = (tick or 0.0) * 1.0001
                if (table.side[i] > 0 and price < bb - slack - 1e-12) or (table.side[i] < 0 and price > ba + slack + 1e-12):
                    flag("trade_book_inconsistent", i)
    denominators = {f: max(n, 1) for f in FLAGS}
    denominators["trade_book_inconsistent"] = max(trades_checked, 1)
    report, score = {}, 1.0
    for f in FLAGS:
        rate = counts[f] / denominators[f]
        warn, fail = THRESHOLDS[f]
        status = "FAIL" if rate >= fail else "WARN" if rate >= warn else "PASS"
        if f in ("price_grid",) and tick is None or f == "lot_grid" and lot is None:
            status = "NOT_EVALUABLE"
        report[f] = {"count": counts[f], "rate": rate, "status": status, "examples": examples[f]}
        if status != "NOT_EVALUABLE":
            score -= WEIGHTS[f] * min(rate / fail, 1.0)
    statuses = [r["status"] for r in report.values()]
    overall = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    return {"rows": n, "book_rows": int(np.sum(table.kind == BOOK)), "trade_rows": int(np.sum(table.kind == TRADE)),
            "flags": report, "score": max(score, 0.0), "status": overall, "repairs": "none (not permitted)",
            "thresholds": {k: list(v) for k, v in THRESHOLDS.items()}}

"""100 ms event bundles extracted from a lot-scaled historical tape (development data only).

Aggregate L2 does not reveal individual orders, so the generative unit is a
*level change per 100 ms bin*: for each pair of consecutive valid samples and
each side, every visible price level whose size changed beyond what aggressive
prints consumed is one addition (net increase) or one cancellation (net
decrease). Aggressive events are the v0.5 grouped prints. Prints are allocated to
levels by walking the book from the touch (a sweep model). Prices that leave
the visible top 10 are not counted (unobserved, not cancelled).

Per bin the extractor records six counts — market buy/sell, add bid/ask, cancel
bid/ask — and the book state at the start of the bin (spread in ticks, top-5
imbalance, log L1 depth); it also collects size and tick-offset samples for each
event type. Cancellation "sizes" are fractions (0, 1] of the level volume left
after prints, which gives generated books the proportional decay real books have. These are the only empirical inputs of generators G1, G3 and G4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MARKS = ("market_buy", "market_sell", "add_bid", "add_ask", "cancel_bid", "cancel_ask")
STATE = ("spread_ticks", "imbalance5", "log_l1_depth")
OFFSET_RANGE = (-5, 30)
SPREAD_BUCKETS = ("1", "2+")


def spread_bucket(spread_ticks: float) -> str:
    """Offsets are conditioned on the spread: improvements inside the spread need a spread above one tick."""
    return "1" if spread_ticks <= 1.5 else "2+"


@dataclass
class Bundles:
    counts: np.ndarray          # (n, 6) int
    state: np.ndarray           # (n, 3) state at bin start
    valid: np.ndarray           # (n,) bool: both endpoints valid
    sizes: dict                 # mark -> sizes in lots (cancel marks: fraction of the level removed)
    offsets: dict               # 'mark@spread bucket' -> (k, 2) [tick offset from the touch, size or fraction] pairs
    tick: float

    def to_arrays(self) -> dict:
        out = {"counts": self.counts, "state": self.state, "valid": self.valid, "tick": np.asarray(self.tick)}
        out.update({f"size__{k}": v for k, v in self.sizes.items()})
        out.update({f"offset__{k}": v for k, v in self.offsets.items()})
        return out

    @classmethod
    def from_arrays(cls, a) -> Bundles:
        return cls(a["counts"], a["state"], a["valid"],
                   {k.split("__", 1)[1]: a[k] for k in a if k.startswith("size__")},
                   {k.split("__", 1)[1]: a[k] for k in a if k.startswith("offset__")}, float(a["tick"]))


def book_state(bp0: float, ap0: float, bq: np.ndarray, aq: np.ndarray, tick: float) -> tuple[float, float, float]:
    spread = (ap0 - bp0) / tick
    b5, a5 = float(np.sum(bq[:5])), float(np.sum(aq[:5]))
    imbalance = (b5 - a5) / (b5 + a5) if b5 + a5 > 0 else 0.0
    return float(min(max(round(spread), 1), 10)), imbalance, float(np.log(max(bq[0] + aq[0], 1e-6)))


def _side_changes(p0, q0, p1, q1, sweep: list[float], tick: float, sign: int):
    """Net additions/cancellations at visible levels of one side between two samples."""
    f0, f1 = np.isfinite(p0), np.isfinite(p1)
    if not f0.any() or not f1.any():
        return [], []
    prev = {round(p / tick): q for p, q in zip(p0[f0], q0[f0])}
    now = {round(p / tick): q for p, q in zip(p1[f1], q1[f1])}
    # Common visible range: for bids prices >= the shallower of the two deepest visible levels (asks mirrored).
    deep = max(min(prev), min(now)) if sign > 0 else min(max(prev), max(now))
    best_prev = max(prev) if sign > 0 else min(prev)
    consumed: dict[int, float] = {}
    for size in sweep:   # allocate each aggressive event from the touch outward
        remaining = size
        for price in sorted(prev, reverse=sign > 0):
            if remaining <= 0:
                break
            take = min(remaining, prev[price] - consumed.get(price, 0.0))
            if take > 0:
                consumed[price] = consumed.get(price, 0.0) + take
                remaining -= take
    adds, cancels = [], []
    for price in set(prev) | set(now):
        if (sign > 0 and price < deep) or (sign < 0 and price > deep):
            continue
        net = now.get(price, 0.0) - prev.get(price, 0.0) + consumed.get(price, 0.0)
        offset = (best_prev - price) * sign
        if net > 1e-9:
            adds.append((offset, net))
        elif net < -1e-9:
            # Cancellations are recorded as the fraction of the remaining (post-print) level they removed.
            available = prev.get(price, 0.0) - consumed.get(price, 0.0)
            cancels.append((offset, min(1.0, -net / available) if available > 1e-9 else 1.0))
    return adds, cancels


def extract(tape, *, max_bins: int | None = None) -> Bundles:
    """Bundles from a lot-scaled ``BookTape`` (v0.6 grid)."""
    n = len(tape.t) - 1 if max_bins is None else min(len(tape.t) - 1, max_bins)
    valid = tape.valid
    counts = np.zeros((n, len(MARKS)), dtype=np.int32)
    state = np.zeros((n, len(STATE)))
    ok = np.zeros(n, dtype=bool)
    sizes = {m: [] for m in MARKS}
    offsets = {f"{m}@{b}": [] for m in MARKS[2:] for b in SPREAD_BUCKETS}
    tt, ts, tq = tape.trade_t, tape.trade_s, tape.trade_q
    edges = np.searchsorted(tt, tape.t, side="right")
    tick = tape.tick
    for i in range(1, n + 1):
        if not (valid[i - 1] and valid[i]):
            continue
        b = i - 1
        ok[b] = True
        state[b] = book_state(tape.bp[b, 0], tape.ap[b, 0], tape.bq[b], tape.aq[b], tick)
        lo, hi = edges[i - 1], edges[i]
        buys = [float(q) for q, s in zip(tq[lo:hi], ts[lo:hi]) if s > 0]
        sells = [float(q) for q, s in zip(tq[lo:hi], ts[lo:hi]) if s < 0]
        counts[b, 0], counts[b, 1] = len(buys), len(sells)
        sizes["market_buy"].extend(buys)
        sizes["market_sell"].extend(sells)
        for sign, p0, q0, p1, q1, sweep, add_m, can_m, ai, ci in (
                (1, tape.bp[b], tape.bq[b], tape.bp[i], tape.bq[i], sells, "add_bid", "cancel_bid", 2, 4),
                (-1, tape.ap[b], tape.aq[b], tape.ap[i], tape.aq[i], buys, "add_ask", "cancel_ask", 3, 5)):
            adds, cancels = _side_changes(p0, q0, p1, q1, sweep, tick, sign)
            counts[b, ai], counts[b, ci] = len(adds), len(cancels)
            bucket = spread_bucket(state[b, 0])
            for offset, q in adds:
                offsets[f"{add_m}@{bucket}"].append((offset, q))
                sizes[add_m].append(q)
            for offset, q in cancels:
                offsets[f"{can_m}@{bucket}"].append((offset, q))
                sizes[can_m].append(q)
    lo, hi = OFFSET_RANGE
    pairs = {}
    for k, v in offsets.items():
        a = np.asarray(v, float).reshape(-1, 2)
        a[:, 0] = np.clip(a[:, 0], lo, hi)
        pairs[k] = a
    return Bundles(counts, state, ok, {k: np.asarray(v, float) for k, v in sizes.items()}, pairs, tick)

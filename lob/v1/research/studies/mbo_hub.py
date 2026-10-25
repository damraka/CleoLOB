"""MBO hub: registered unit value of H15 (genuine MBO narrows queue-position uncertainty versus L2).

On one captured Bitstamp day (prospective capture, exact order ids), a hypothetical order joins the back of each touch
level every ``every_s`` seconds. After 30 s (registered horizon):

* MBO (exact): size still ahead = remaining size of the orders that were ahead at joining and are still at that price
  without having lost priority (``MBOBook`` semantics: a size increase or a price change moves an order to the back);
* L2 view of the *same* day (aggregate level size and prints only): optimistic ahead = max(0, Q0 - all size decreases
  at the level) - every decrease came from the front; pessimistic ahead = min(level size, max(0, Q0 - volume printed
  at that price)) - cancellations all came from behind. A trade-through or a vanished level clears both (0).

H15 unit = mean over samples of (pessimistic - optimistic) / Q0 (the MBO interval has width 0, so the unit is the
uncertainty MBO removes). The share of samples with optimistic <= exact <= pessimistic is reported as the
reconstruction validation (it must be ~1; failures point to feed or reconstruction problems, never hidden). Samples
whose window contains a capture GAP or a re-anchoring census are discarded and counted.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

from ...data.schema import EventKind

HORIZON_NS = 30 * 10 ** 9


def capture_segments(root: Path, pair: str, day: str) -> list[Path]:
    """Capture files of one UTC day (segment directories are named by their UTC start time)."""
    base = root / "data/v1/capture/bitstamp" / pair
    tag = day.replace("-", "")
    # only finalized segments (manifest written at segment close); the live segment is still being appended
    return sorted(p / "capture.jsonl.gz" for p in base.glob(f"{tag}T*") if (p / "capture-manifest.json").exists())


def day_events(root: Path, pair: str, day: str):
    from ...data.connectors import bitstamp as BS
    inst = BS.BitstampConnector().instrument("exchange", pair, day)
    for f in capture_segments(root, pair, day):
        yield from BS.mbo_events(BS.segment_records(f), inst)


def _level_size(book, side: int, price: int) -> float:
    q = book.levels[side].get(price)
    return float(sum(book.orders[i].size for i in q)) if q else 0.0


def h15(chunks, *, every_s: float = 60.0, min_ahead: float = 1e-9) -> dict:
    from ...data.reconstruction import mbo as M
    every = int(every_s * 1e9)
    trades: list[tuple[int, int, int, float]] = []          # (ts, aggressor side, price, size), time order
    trade_ts: list[int] = []
    breaks: list[int] = []                                   # GAP / census timestamps

    def tee():
        for ch in chunks:
            k = ch["kind"]
            t = ch[k == EventKind.TRADE]
            trades.extend(zip(t["ts_exchange"].tolist(), t["side"].tolist(), t["price"].tolist(), t["size"].tolist()))
            trade_ts.extend(t["ts_exchange"].tolist())
            breaks.extend(ch["ts_exchange"][(k == EventKind.GAP) | (k == EventKind.SNAPSHOT_END)].tolist())
            yield ch

    active: list[dict] = []
    done: list[dict] = []
    nxt = [None]
    skipped = [0]

    def finish(v, book):
        p, s = v["price"], v["side"]
        lo, hi = bisect_left(trade_ts, v["t0"]), bisect_right(trade_ts, v["t1"])
        tr = [x for x in trades[lo:hi] if x[1] == -s]
        at_px = sum(x[3] for x in tr if x[2] == p)
        through = any((x[2] < p) if s > 0 else (x[2] > p) for x in tr)
        level = _level_size(book, s, p)
        exact = 0.0
        for oid, size0 in v["ahead"]:
            o = book.orders.get(oid)
            if o is not None and o.price == p and o.size <= size0 + 1e-12:
                exact += o.size
        if through or level <= 0:
            opt = pes = 0.0
        else:
            opt = max(0.0, v["q0"] - v["decrease"])
            pes = min(level, max(0.0, v["q0"] - at_px))
        bad = any(v["t0"] < b <= v["t1"] for b in breaks)
        done.append({"q0": v["q0"], "exact": exact, "optimistic": opt, "pessimistic": pes, "discarded": bad,
                     "side": s, "through": through})

    def on_state(book, ev):
        ts = int(ev["ts_exchange"])
        for v in [a for a in active if ts > a["t1"]]:
            active.remove(v)
            finish(v, book)
        for v in active:
            if int(ev["side"]) == v["side"] and int(ev["price"]) == v["price"]:
                new = _level_size(book, v["side"], v["price"])
                v["decrease"] += max(0.0, v["last"] - new)
                v["last"] = new
        if nxt[0] is None:
            nxt[0] = ts - ts % every + every
        if ts >= nxt[0]:
            nxt[0] = ts - ts % every + every
            bb, ba = book.best(1), book.best(-1)
            if bb is None or ba is None or bb >= ba:            # crossed/one-sided book: no sample (counted)
                skipped[0] += 1
                return
            for s in (1, -1):
                p = book.best(s)
                if p is None:
                    continue
                ahead = [(oid, book.orders[oid].size) for oid in book.levels[s][p]]
                q0 = float(sum(x[1] for x in ahead))
                if q0 > min_ahead:
                    active.append({"t0": ts, "t1": ts + HORIZON_NS, "side": s, "price": p, "ahead": ahead, "q0": q0,
                                   "decrease": 0.0, "last": q0})

    book, stats = M.reconstruct(tee(), on_state=on_state)
    kept = [d for d in done if not d["discarded"]]
    if not kept:
        return {"H15": None, "samples": 0, "discarded": len(done), "stats": stats.to_dict()}
    width = [(d["pessimistic"] - d["optimistic"]) / d["q0"] for d in kept]
    inside = [d["optimistic"] - 1e-9 <= d["exact"] <= d["pessimistic"] + 1e-9 for d in kept]
    return {"H15": float(np.mean(width)), "samples": len(kept), "discarded": len(done) - len(kept),
            "containment_rate": float(np.mean(inside)), "skipped_crossed": skipped[0],
            "above_pessimistic": int(sum(d["exact"] > d["pessimistic"] + 1e-9 for d in kept)),
            "below_optimistic": int(sum(d["exact"] < d["optimistic"] - 1e-9 for d in kept)), "mean_exact_fraction": float(np.mean([d["exact"] / d["q0"]
                                                                                              for d in kept])),
            "stats": stats.to_dict(),
            "label": "hypothetical back-of-queue orders; MBO position exact between reconciliations, L2 bounds only"}


__all__ = ["h15", "day_events", "capture_segments"]

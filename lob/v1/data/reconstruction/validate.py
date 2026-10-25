"""Independent reconstruction validation against recorded snapshots (M29).

* **MBP (Bybit)**: every venue snapshot after the first is an independent reference. ``l2_snapshot_check`` replays
  the deltas up to each snapshot and compares the reconstructed book just before it with the snapshot content
  (levels matched, size errors, missing/extra prices within the snapshot depth).
* **MBO (Bitstamp)**: the reconstructed order book aggregated to price levels is compared with the venue's
  ``order_book`` channel (top-100 MBP) at its timestamps (``mbo_vs_mbp``), and with later REST censuses
  (``mbo.reconcile``, reported by ``mbo.reconstruct``).

Both are external references produced by the venue, not by CleoLOB, so agreement is evidence about the
reconstruction itself.
"""
from __future__ import annotations

import numpy as np

from ..schema import EventKind
from .l2 import DenseBook, apply, messages


def l2_snapshot_check(chunks, *, depth: int = 50) -> dict:
    book = DenseBook()
    seen_first = False
    checks = []
    for ts, _seq, snap, m in messages(chunks):
        if snap and seen_first:
            lv = m[m["kind"] == EventKind.LEVEL_SET]
            res = {}
            for side in (1, -1):
                s = lv[lv["side"] == side]
                ref = dict(zip(s["price"].tolist(), s["size"].tolist()))
                ref_p = sorted(ref, reverse=side > 0)[:depth]
                rp, rq = book.levels(side, depth)
                rec = dict(zip(rp.tolist(), rq.tolist()))
                matched = sum(1 for p in ref_p if abs(rec.get(p, 0.0) - ref[p]) < 1e-9)
                res[side] = {"levels": len(ref_p), "matched": matched,
                             "missing": sum(1 for p in ref_p if p not in rec),
                             "extra": sum(1 for p in rp.tolist() if p not in ref and
                                          ((p > ref_p[-1]) if side > 0 else (p < ref_p[-1])))
                             if ref_p else 0}
            checks.append({"ts": ts, "bid": res[1], "ask": res[-1]})
        apply(book, m, snap)
        seen_first = seen_first or snap
    tot = sum(c["bid"]["levels"] + c["ask"]["levels"] for c in checks)
    ok = sum(c["bid"]["matched"] + c["ask"]["matched"] for c in checks)
    return {"snapshots_checked": len(checks), "levels": tot, "matched": ok,
            "match_fraction": (ok / tot) if tot else None, "checks": checks[:50],
            "status": "OK" if checks else "NO_REFERENCE",
            "note": "a day with a single snapshot has no independent reference (status NO_REFERENCE)"}


def mbo_vs_mbp(book_at, references: list[dict], *, levels: int = 20) -> dict:
    """``book_at(ts)`` returns an MBOBook state at exchange time ts; references from ``bitstamp.book_snapshots``."""
    rows = []
    for ref in references:
        book = book_at(ref["ts"])
        if book is None:
            continue
        row = {}
        for side, key in ((1, "bids"), (-1, "asks")):
            mine = dict(book.depth(side, levels))
            theirs = dict(ref[key][:levels])
            prices = list(theirs)
            exact = sum(1 for p in prices if abs(mine.get(p, 0.0) - theirs[p]) < 1e-6)
            row[key] = {"levels": len(prices), "price_present": sum(p in mine for p in prices), "exact": exact}
        rows.append(row)
    lv = sum(r[k]["levels"] for r in rows for k in ("bids", "asks"))
    ex = sum(r[k]["exact"] for r in rows for k in ("bids", "asks"))
    pres = sum(r[k]["price_present"] for r in rows for k in ("bids", "asks"))
    return {"references": len(rows), "levels": lv, "exact_fraction": (ex / lv) if lv else None,
            "price_presence_fraction": (pres / lv) if lv else None}


def first_snapshot_levels(chunks) -> np.ndarray:
    for _ts, _seq, snap, m in messages(chunks):
        if snap:
            return m[m["kind"] == EventKind.LEVEL_SET]
    return np.zeros(0)


__all__ = ["l2_snapshot_check", "mbo_vs_mbp", "first_snapshot_levels"]

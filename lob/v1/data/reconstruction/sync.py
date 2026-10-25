"""Trade/book synchronization diagnostics for MBP sources (M25).

For every print the book state *before* the print (last book message with exchange time <= print time) is used to
classify the print price against the passive touch (``at_touch``, ``inside`` the spread, ``through`` the touch into
deeper levels, ``beyond_book`` when the price is not visible). The first later book message that reduces the
passive-side size at the print price gives the **book lag** of the print (how long until the book reflects it).
Prints whose depletion never appears within ``horizon_ms`` are ``unmatched``. These are diagnostics of how the
venue's trade and book streams line up, required before any fill or queue inference uses them together.
"""
from __future__ import annotations

import numpy as np

from .l2 import DenseBook, apply, messages


def diagnostics(book_chunks, trades: np.ndarray, *, horizon_ms: int = 1000, max_trades: int = 200_000) -> dict:
    tr = trades[np.argsort(trades["ts_exchange"], kind="stable")][:max_trades]
    book = DenseBook()
    cls = {"at_touch": 0, "inside": 0, "through": 0, "beyond_book": 0, "no_book": 0}
    lags: list[float] = []
    pending: list[tuple[int, int, int, float]] = []          # (trade ts, passive side, price, size before)
    unmatched = 0
    i = 0
    last_ts = None
    for ts, _seq, snap, m in messages(book_chunks):
        while i < len(tr) and tr["ts_exchange"][i] <= ts:
            t = tr[i]
            passive = -1 if t["side"] > 0 else 1
            best = book.best(passive) if last_ts is not None else None
            if best is None:
                cls["no_book"] += 1
            else:
                p = int(t["price"])
                diff = (p - best) if passive < 0 else (best - p)     # >0: deeper than the passive touch
                if diff == 0:
                    cls["at_touch"] += 1
                elif diff < 0:
                    cls["inside"] += 1
                elif book.size_at(passive, np.asarray([p]))[0] > 0:
                    cls["through"] += 1
                else:
                    cls["beyond_book"] += 1
                pending.append((int(t["ts_exchange"]), passive, p, float(book.size_at(passive, np.asarray([p]))[0])))
            i += 1
        apply(book, m, snap)
        last_ts = ts
        if pending:
            keep = []
            for (t0, side, p, before) in pending:
                now = float(book.size_at(side, np.asarray([p]))[0])
                if now < before - 1e-9 or (before > 0 and now == 0):
                    lags.append((ts - t0) / 1e6)
                elif ts - t0 > horizon_ms * 1_000_000:
                    unmatched += 1
                else:
                    keep.append((t0, side, p, before))
            pending = keep
        if i >= len(tr) and not pending:
            break
    unmatched += len(pending)
    n = sum(cls.values())
    lag = np.asarray(lags)
    return {"trades": int(n), "classification": cls,
            "fractions": {k: (v / n if n else None) for k, v in cls.items()},
            "matched": int(len(lag)), "unmatched": int(unmatched),
            "book_lag_ms": ({"p50": float(np.median(lag)), "p90": float(np.quantile(lag, 0.9)),
                             "p99": float(np.quantile(lag, 0.99)), "max": float(lag.max())} if len(lag) else None),
            "horizon_ms": horizon_ms}


__all__ = ["diagnostics"]

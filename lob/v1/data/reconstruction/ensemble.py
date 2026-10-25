"""Multiple plausible reconstructions for ambiguous observations (M28).

Aggregate L2 at a fixed sampling interval leaves three registered ambiguities:

1. **trade/cancel attribution** - a visible decrease at a price with prints in the same or previous bin can be
   explained by the prints (v0.9 rule: prints first) or by cancellations (prints elsewhere in the queue);
2. **within-bin ordering** - additions and decreases inside one bin have no observed order;
3. **hidden volume** - executed volume beyond the visible size may be hidden liquidity or merged updates.

``ensemble`` generates K reconstructions of the per-bin inferred event counts by sampling, per decrease, whether
the overlapping printed volume explains it (probability ``p_trade_first`` drawn per member from a registered prior)
and how many merged messages each visible change stands for (geometric with a registered mean), and returns the
member counts plus their spread. Downstream studies receive the ensemble, so reconstruction ambiguity becomes an
explicit uncertainty instead of a hidden choice.
"""
from __future__ import annotations

import numpy as np


def decrease_records(frames: dict, bins: slice | None = None) -> dict:
    """Per-bin visible increases/decreases at the touch with the overlapping passive prints (from day frames)."""
    sl = bins or slice(None)
    depth = np.asarray(frames["depth"][sl], float)
    vol = np.asarray(frames["volume"][sl], float)
    bid, ask = np.asarray(frames["bid_px"][sl]), np.asarray(frames["ask_px"][sl])
    valid = np.asarray(frames["valid"][sl], bool)
    ok = valid[1:] & valid[:-1]
    out = {}
    for side, px, col in ((0, bid, 1), (1, ask, 0)):
        same = (px[1:] == px[:-1]) & ok
        d = depth[1:, side, 0] - depth[:-1, side, 0]
        out[side] = {"increase": np.where(same, np.maximum(d, 0), 0.0), "decrease": np.where(same, np.maximum(-d, 0), 0.0),
                     "printed": np.where(same, vol[1:, col], 0.0)}
    return out


def ensemble(records: dict, *, members: int = 32, seed: int = 0, p_trade_first: tuple[float, float] = (0.6, 1.0),
             merge_mean: float = 1.5) -> dict:
    """K reconstructions of (adds, cancels) per bin and side; returns totals per member and their spread."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(members):
        p = rng.uniform(*p_trade_first)
        m = {}
        for side, r in records.items():
            dec, pr, inc = r["decrease"], r["printed"], r["increase"]
            attribute = rng.random(len(dec)) < p
            explained = np.where(attribute, np.minimum(pr, dec), 0.0)
            cancel_vol = dec - explained
            # merged-message multiplicity: each visible change stands for >= 1 messages
            mult_c = rng.geometric(1.0 / merge_mean, size=len(dec))
            mult_a = rng.geometric(1.0 / merge_mean, size=len(inc))
            m[side] = {"cancel_events": int(((cancel_vol > 1e-9) * mult_c).sum()),
                       "add_events": int(((inc > 1e-9) * mult_a).sum()),
                       "cancel_volume": float(cancel_vol.sum())}
        totals.append({"p_trade_first": p, **{f"side{s}": v for s, v in m.items()}})
    def spread(key, sub):
        x = np.asarray([t[key][sub] for t in totals], float)
        return {"mean": float(x.mean()), "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95)),
                "relative_width": float((np.quantile(x, 0.95) - np.quantile(x, 0.05)) / max(abs(x.mean()), 1e-9))}
    return {"members": totals, "spread": {f"side{s}:{k}": spread(f"side{s}", k)
                                          for s in records for k in ("cancel_events", "add_events", "cancel_volume")},
            "prior": {"p_trade_first": list(p_trade_first), "merge_mean": merge_mean}}


__all__ = ["decrease_records", "ensemble"]

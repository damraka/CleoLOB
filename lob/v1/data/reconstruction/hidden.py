"""Hidden/iceberg liquidity evidence and uncertainty propagation (M27).

Hidden liquidity is never observed directly on public feeds; v1 measures **lower-bound evidence** and propagates
an interval, never a point value:

* MBP (L2): a print at the passive touch whose size exceeds the visible touch size immediately before it, with the
  level still non-empty after it, implies at least ``size - visible`` unseen volume (``refill_evidence``);
* MBO (L3): a print whose maker order id is unknown to the reconstructed book, or whose size exceeds the known
  order's remaining size, implies unseen (hidden or not-yet-visible) volume.

``hidden_interval`` turns the evidence into the registered interval ``[0, h_max]`` of the hidden fraction (unseen
executed volume / executed volume at the touch, upper bound by a Wilson-style margin), which the bounded replay
grid uses instead of an arbitrary constant.
"""
from __future__ import annotations

import math

import numpy as np


def refill_evidence(touch_size_before: np.ndarray, print_size: np.ndarray, touch_size_after: np.ndarray) -> dict:
    before, size, after = (np.asarray(x, float) for x in (touch_size_before, print_size, touch_size_after))
    excess = np.maximum(size - before, 0.0)
    implied = (excess > 0) & (after > 0)
    return {"prints": int(len(size)), "prints_with_excess": int(implied.sum()),
            "unseen_volume_lower_bound": float(excess[implied].sum()), "executed_volume": float(size.sum())}


def l3_evidence(trade_maker_known: np.ndarray, trade_size: np.ndarray, known_remaining: np.ndarray) -> dict:
    known = np.asarray(trade_maker_known, bool)
    size = np.asarray(trade_size, float)
    rem = np.asarray(known_remaining, float)
    over = known & (size > rem + 1e-9)
    unseen = float(size[~known].sum() + (size[over] - rem[over]).sum())
    return {"prints": int(len(size)), "unknown_maker": int((~known).sum()), "oversized": int(over.sum()),
            "unseen_volume_lower_bound": unseen, "executed_volume": float(size.sum())}


def hidden_interval(evidence: dict, *, z: float = 1.959963984540054) -> dict:
    """[0, h_max]: h_max is an upper confidence bound on the unseen fraction of executed volume."""
    v = evidence["executed_volume"]
    if v <= 0:
        return {"lower": 0.0, "upper": 1.0, "point_lower_bound": None, "status": "NOT_IDENTIFIED"}
    p = min(evidence["unseen_volume_lower_bound"] / v, 1.0)
    n = max(evidence["prints"], 1)
    upper = min(1.0, (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n))
    return {"lower": 0.0, "upper": float(upper), "point_lower_bound": float(p), "status": "BOUNDED",
            "label": "unseen-volume lower bound with an upper confidence margin; never an identified hidden size"}


__all__ = ["refill_evidence", "l3_evidence", "hidden_interval"]

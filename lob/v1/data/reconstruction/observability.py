"""Queue-position observability classification and L2 queue bounds (M26).

``classify(capability)`` maps a source capability to what queue quantities are identifiable:

* ``exact_order_id`` (L3 between reconciliations): the size and number of orders ahead of a given order;
* ``inferred_position`` (L3 anchored on a census whose within-level order is assumed to be time priority);
* ``aggregate_level`` (L2): only the level size is observed, so the position of a hypothetical order that joined at
  the back is **partially identified**: ``l2_queue_bounds`` returns, after each observed level change, the interval
  of the size still ahead. Prints at the price always consume the front (they reduce ``ahead`` in both bounds);
  unexplained decreases (cancellations) reduce ``ahead`` fully in the optimistic bound (cancels in front) and not at
  all in the pessimistic bound (cancels behind); increases join behind and never change ``ahead``;
* ``none``: no queue quantity is identifiable.
"""
from __future__ import annotations

import numpy as np

from ..schema import Capability

IDENTIFIABLE = {
    "exact_order_id": ("orders_ahead", "size_ahead", "fill_time", "priority_loss"),
    "inferred_position": ("size_ahead_given_census_priority", "fill_time_given_census_priority"),
    "aggregate_level": ("size_ahead_interval", "clearance_time_interval"),
    "none": (),
}


def classify(cap: Capability | dict) -> dict:
    q = cap.queue_observability if isinstance(cap, Capability) else cap["queue_observability"]
    return {"queue_observability": q, "identifiable": list(IDENTIFIABLE[q]),
            "partial_identification": q == "aggregate_level", "exact": q == "exact_order_id"}


def l2_queue_bounds(initial_ahead: float, level_sizes: np.ndarray, prints: np.ndarray) -> dict:
    """Bounds on the size ahead of a back-of-queue order at one price.

    ``level_sizes[k]`` is the visible size after observation k, excluding the hypothetical order (k = 0 is the join),
    and ``prints[k]`` the passive volume printed at the price between observations k-1 and k. Returns the
    optimistic and pessimistic ``ahead`` paths and the first observation at which each bound reaches zero (the order
    would start to fill).
    """
    sizes = np.asarray(level_sizes, float)
    pr = np.asarray(prints, float)
    lo, hi = [float(initial_ahead)], [float(initial_ahead)]
    for k in range(1, len(sizes)):
        dec = max(sizes[k - 1] - sizes[k], 0.0)
        traded = min(pr[k], sizes[k - 1])
        cancel = max(dec - traded, 0.0)
        a_lo = max(lo[-1] - traded - cancel, 0.0)        # optimistic: every cancel was in front
        a_hi = max(hi[-1] - traded, 0.0)                 # pessimistic: every cancel was behind
        lo.append(a_lo)
        hi.append(a_hi)
    lo_a, hi_a = np.asarray(lo), np.asarray(hi)

    def first_zero(a):
        z = np.flatnonzero(a <= 1e-12)
        return int(z[0]) if len(z) else None

    return {"ahead_optimistic": lo_a, "ahead_pessimistic": hi_a, "fill_start_optimistic": first_zero(lo_a),
            "fill_start_pessimistic": first_zero(hi_a), "width": hi_a - lo_a}


__all__ = ["IDENTIFIABLE", "classify", "l2_queue_bounds"]

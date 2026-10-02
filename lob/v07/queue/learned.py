"""Cancellation queue-position distribution learned from genuine order-level data (workstream 79).

On the Bitstamp captures every resting order has a source identity, so the
position of each cancelled order inside its price level can be measured *if*
priority is price-time with the declared tracking rule (additions join the back;
size increases and reprices re-queue). The feed does not establish that rule,
so the learned distribution is ASSUMPTION_DEPENDENT on it.

The statistic is ``u = volume ahead of the cancelled order / level volume`` at
the cancellation (0 = front, 1 = back). Its empirical CDF ``F`` plugs into
``lob.v07.queue.models`` as the share of a cancellation that falls ahead of an
order with fraction ``a`` of the level in front of it. Uniform ``u`` gives the
pro-rata rule ``F(a) = a``.
"""
from __future__ import annotations

from collections import OrderedDict
from decimal import Decimal
from typing import Iterable

import numpy as np

from ..data.schema import Record

BINS = 10


def cancellation_positions(records: Iterable[Record]) -> dict:
    """Queue-position fractions of cancellations under tracked price-time priority."""
    levels: dict[tuple[str, Decimal], OrderedDict[str, Decimal]] = {}
    where: dict[str, tuple[str, Decimal]] = {}
    positions: list[float] = []
    skipped = {"unknown_order": 0, "single_order_level": 0, "partial": 0}
    for r in records:
        if r.kind == "order_add":
            key = (r.side, r.price)
            levels.setdefault(key, OrderedDict())[r.order_id] = r.amount
            where[r.order_id] = key
        elif r.kind in {"order_cancel", "order_execute", "order_modify"}:
            key = where.get(r.order_id)
            if key is None:
                skipped["unknown_order"] += 1
                continue
            queue = levels[key]
            if r.kind == "order_cancel":
                total = sum(queue.values())
                ahead = Decimal(0)
                for oid, qty in queue.items():
                    if oid == r.order_id:
                        break
                    ahead += qty
                if len(queue) <= 1:
                    skipped["single_order_level"] += 1
                elif total > 0:
                    positions.append(float(ahead / total))
                if r.amount is not None and r.amount < queue[r.order_id]:
                    skipped["partial"] += 1
                    queue[r.order_id] -= r.amount
                    continue
                queue.pop(r.order_id, None)
                where.pop(r.order_id, None)
            elif r.kind == "order_execute":
                remaining = queue[r.order_id] - (r.amount or Decimal(0))
                if remaining <= 0:
                    queue.pop(r.order_id, None)
                    where.pop(r.order_id, None)
                else:
                    queue[r.order_id] = remaining
            else:  # modify: new remaining quantity and/or price; increases and reprices re-queue
                new_price = r.price if r.price is not None else key[1]
                new_qty = r.amount if r.amount is not None else queue[r.order_id]
                if new_price != key[1] or new_qty > queue[r.order_id]:
                    queue.pop(r.order_id, None)
                    nkey = (key[0], new_price)
                    levels.setdefault(nkey, OrderedDict())[r.order_id] = new_qty
                    where[r.order_id] = nkey
                else:
                    queue[r.order_id] = new_qty
        elif r.kind == "gap":
            levels.clear()
            where.clear()
    u = np.asarray(positions)
    return {"n": int(len(u)), "skipped": skipped, "positions": u}


class EmpiricalCDF:
    """Piecewise-linear CDF on ``BINS`` equal bins of ``u`` (monotone, F(0)=0, F(1)=1)."""

    def __init__(self, u: np.ndarray, bins: int = BINS) -> None:
        if len(u) == 0:
            raise ValueError("no cancellation positions")
        counts, _ = np.histogram(np.clip(u, 0, 1), bins=bins, range=(0, 1))
        self.knots = np.linspace(0, 1, bins + 1)
        self.values = np.r_[0.0, np.cumsum(counts) / counts.sum()]

    def __call__(self, a: float) -> float:
        return float(np.interp(min(max(a, 0.0), 1.0), self.knots, self.values))

    def to_dict(self) -> dict:
        return {"knots": self.knots.tolist(), "cdf": self.values.tolist()}


def summary(u: np.ndarray, *, seed: int = 0, samples: int = 2000) -> dict:
    """Mean position with bootstrap interval and the KS distance from uniform (pro-rata)."""
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(u, len(u)).mean()) for _ in range(samples)]
    s = np.sort(u)
    ks = float(np.max(np.abs(np.arange(1, len(s) + 1) / len(s) - s))) if len(s) else None
    return {"n": int(len(u)), "mean": float(u.mean()), "ci95": [float(np.quantile(means, 0.025)),
                                                                 float(np.quantile(means, 0.975))],
            "ks_from_uniform": ks, "front_quintile_share": float(np.mean(u < 0.2)),
            "back_quintile_share": float(np.mean(u >= 0.8))}

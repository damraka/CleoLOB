"""Venue and instrument normalization (M17, M18): price/size grids, units and contract metadata.

``infer_grid`` reads the decimal precision of the first messages' prices and sizes (exact decimal arithmetic) and
returns the observed tick and lot. ``grid`` applies the registered rule (protocol amendment 2): the tick used is the
coarser of the registered and the observed price precision (a registered tick finer than the archive's precision
would make 'spread in ticks', 'visible levels' and queue features meaningless), and the lot is the finer of the two
size precisions (sizes are never rounded away). Any difference from the registry is a recorded known anomaly.
After normalization no price is off-grid and no size is negative (property-tested).
"""
from __future__ import annotations

from decimal import Decimal
import itertools
from typing import Iterable


def _exp(values: Iterable[str]) -> int:
    lo = 0
    for v in values:
        d = Decimal(v)
        if d != 0:
            lo = min(lo, d.normalize().as_tuple().exponent if d == d.to_integral_value() else d.as_tuple().exponent)
    return lo


def infer_grid(messages: Iterable[dict], n: int = 2000) -> dict:
    prices, sizes = [], []
    for m in itertools.islice(messages, n):
        for key in ("b", "a"):
            for p, q in m["data"].get(key, ()):
                prices.append(p)
                sizes.append(q)
    if not prices:
        raise ValueError("no book levels to infer a grid from")
    p_exp = _exp(prices)
    q_exp = _exp(s for s in sizes if Decimal(s) != 0)
    # strip trailing zeros: '30294.80' has precision 0.1 if every price ends in 0 at the second decimal
    decimals = max(0, -p_exp)
    while decimals > 0 and all(Decimal(p).quantize(Decimal(10) ** -(decimals - 1)) == Decimal(p) for p in prices):
        decimals -= 1
    q_decimals = max(0, -q_exp)
    while q_decimals > 0 and all(Decimal(s).quantize(Decimal(10) ** -(q_decimals - 1)) == Decimal(s)
                                 for s in sizes if Decimal(s) != 0):
        q_decimals -= 1
    return {"tick": float(Decimal(10) ** -decimals), "lot": float(Decimal(10) ** -q_decimals),
            "price_decimals": decimals, "size_decimals": q_decimals, "levels_seen": len(prices)}


def grid(registered_tick: float, registered_lot: float, observed: dict) -> dict:
    tick = max(float(registered_tick), observed["tick"])
    lot = min(float(registered_lot), observed["lot"])
    anomalies = []
    if abs(tick - registered_tick) > 1e-12:
        anomalies.append(f"observed price precision {observed['tick']:g} is coarser than the registered tick "
                         f"{registered_tick:g}; using {tick:g}")
    if abs(lot - registered_lot) > 1e-15:
        anomalies.append(f"observed size precision {observed['lot']:g} is finer than the registered lot "
                         f"{registered_lot:g}; using {lot:g}")
    return {"tick": tick, "lot": lot, "registered_tick": registered_tick, "registered_lot": registered_lot,
            "observed": observed, "anomalies": anomalies}

"""Exchange invariants (M39) checked after every command in fuzz tests and, optionally, in studies.

* the resting book is never crossed or locked;
* every level's total equals the sum of its orders' remaining quantities and is positive;
* level prices are sorted and indexed consistently; every live order is in exactly its level;
* queue sequence numbers increase within each level (FIFO);
* quantity conservation: for every taker fill there is an equal maker fill at the same price (report level).
"""
from __future__ import annotations

from .matching import BUY, SELL, MatchingEngine, Report


class InvariantError(AssertionError):
    pass


def check(engine: MatchingEngine) -> None:
    b, a = engine.best(BUY), engine.best(SELL)
    if b is not None and a is not None and b >= a:
        raise InvariantError(f"crossed or locked book {b} >= {a}")
    live = {o.oid for o in engine.orders.values() if o.status == "live"}
    seen = set()
    for side in (BUY, SELL):
        prices = engine._prices[side]
        if prices != sorted(prices) or set(prices) != set(engine.levels[side]):
            raise InvariantError("price index out of sync")
        for p, lv in engine.levels[side].items():
            if not lv.orders or lv.total <= 0:
                raise InvariantError(f"empty level {p}")
            if lv.total != sum(o.remaining for o in lv.orders.values()):
                raise InvariantError(f"level {p} total mismatch")
            seqs = [o.seq for o in lv.orders.values()]
            if seqs != sorted(seqs):
                raise InvariantError(f"level {p} not FIFO")
            for o in lv.orders.values():
                if o.status != "live" or o.price != p or o.side != side or o.remaining <= 0:
                    raise InvariantError(f"order {o.oid} inconsistent with its level")
                seen.add(o.oid)
    if seen != live:
        raise InvariantError("live orders and level membership differ")


def check_reports(reports: list[Report]) -> None:
    fills = [r for r in reports if r.kind == "fill"]
    makers = sorted((r.counter, r.oid, r.price, r.qty) for r in fills if r.liquidity == "maker")
    takers = sorted((r.oid, r.counter, r.price, r.qty) for r in fills if r.liquidity == "taker")
    if makers != takers:
        raise InvariantError("maker and taker fills do not pair")


__all__ = ["InvariantError", "check", "check_reports"]

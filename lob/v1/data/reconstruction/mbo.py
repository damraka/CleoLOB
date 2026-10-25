"""Order-id-aware MBO/L3 reconstruction (M22) with census reconciliation (part of M29).

Input: canonical events of an L3 source in receive order (Bitstamp prospective captures). Registered semantics:

* the first ``SNAPSHOT_BEGIN .. SNAPSHOT_END`` census anchors the book; within a price level the census order is
  taken as time priority (an *assumption*, recorded per order as ``priority = "census"``); orders created after the
  anchor have ``priority = "observed"``;
* ``ORDER_ADD`` appends an order at the back of its level; ``ORDER_MODIFY`` with a smaller size at the same price
  keeps priority (partial execution or reduction), with a larger size or a new price the order loses priority and
  moves to the back (venue-typical rule, recorded as an assumption); ``ORDER_DELETE`` removes it;
* events whose exchange time precedes the anchor census are already reflected in it and are skipped (counted);
* every later census is a reconciliation point: the reconstructed order set is compared with the census (missing,
  extra and size-mismatched orders) and the book is re-anchored, so drift never accumulates silently;
* a ``GAP`` event (capture disconnect) invalidates the book until the next census.

``queue_position(order_id)`` returns the size and number of orders ahead at the order's price: exact for observed-
priority orders between reconciliations, assumption-dependent for census-priority orders.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

from ..schema import EventKind


@dataclass
class Order:
    oid: int
    side: int
    price: int
    size: float
    priority: str           # "census" | "observed"


@dataclass
class MBOStats:
    events: int = 0
    adds: int = 0
    modifies: int = 0
    deletes: int = 0
    trades: int = 0
    trades_maker_known: int = 0
    unknown_modifies: int = 0
    unknown_deletes: int = 0
    duplicate_adds: int = 0
    priority_losses: int = 0
    pre_anchor_skipped: int = 0
    gaps: int = 0
    censuses: int = 0
    crossed_states: int = 0
    reconciliations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reconciliations"] = self.reconciliations[:200]
        return d


class MBOBook:
    def __init__(self) -> None:
        self.orders: dict[int, Order] = {}
        self.levels = {1: {}, -1: {}}            # side -> price -> OrderedDict[oid, None] (FIFO)

    def reset(self) -> None:
        self.orders.clear()
        self.levels = {1: {}, -1: {}}

    def _enqueue(self, o: Order) -> None:
        self.levels[o.side].setdefault(o.price, OrderedDict())[o.oid] = None

    def _dequeue(self, o: Order) -> None:
        q = self.levels[o.side].get(o.price)
        if q is not None:
            q.pop(o.oid, None)
            if not q:
                del self.levels[o.side][o.price]

    def add(self, oid: int, side: int, price: int, size: float, priority: str = "observed") -> bool:
        if oid in self.orders:
            return False
        o = Order(oid, side, price, size, priority)
        self.orders[oid] = o
        self._enqueue(o)
        return True

    def modify(self, oid: int, price: int, size: float) -> str:
        o = self.orders.get(oid)
        if o is None:
            return "unknown"
        if size <= 0:
            self.delete(oid)
            return "deleted"
        if price == o.price and size <= o.size:
            o.size = size
            return "kept"
        self._dequeue(o)
        o.price, o.size, o.priority = price, size, "observed"
        self._enqueue(o)
        return "lost_priority"

    def delete(self, oid: int) -> bool:
        o = self.orders.pop(oid, None)
        if o is None:
            return False
        self._dequeue(o)
        return True

    def best(self, side: int) -> int | None:
        lv = self.levels[side]
        if not lv:
            return None
        return max(lv) if side > 0 else min(lv)

    def depth(self, side: int, n: int) -> list[tuple[int, float]]:
        lv = self.levels[side]
        prices = sorted(lv, reverse=side > 0)[:n]
        return [(p, sum(self.orders[i].size for i in lv[p])) for p in prices]

    def queue_position(self, oid: int) -> dict | None:
        o = self.orders.get(oid)
        if o is None:
            return None
        ahead_n, ahead_q = 0, 0.0
        for i in self.levels[o.side][o.price]:
            if i == oid:
                break
            ahead_n += 1
            ahead_q += self.orders[i].size
        return {"price": o.price, "orders_ahead": ahead_n, "size_ahead": ahead_q, "priority": o.priority}


def _census(block: np.ndarray) -> dict[int, tuple[int, int, float]]:
    adds = block[block["kind"] == EventKind.ORDER_ADD]
    return {int(r["order_id"]): (int(r["side"]), int(r["price"]), float(r["size"])) for r in adds}


def reconcile(book: MBOBook, census: dict[int, tuple[int, int, float]]) -> dict:
    have = {oid: (o.side, o.price, o.size) for oid, o in book.orders.items()}
    missing = [k for k in census if k not in have]
    extra = [k for k in have if k not in census]
    mismatch = [k for k in census if k in have and have[k] != census[k]]
    n = max(len(census), 1)
    return {"census_orders": len(census), "reconstructed_orders": len(have), "missing": len(missing),
            "extra": len(extra), "mismatched": len(mismatch),
            "exact_fraction": 1.0 - (len(missing) + len(mismatch)) / n}


def reconstruct(chunks, *, on_state=None) -> tuple[MBOBook, MBOStats]:
    """Replay canonical L3 events; ``on_state(book, event)`` is called after every applied order event."""
    book, st = MBOBook(), MBOStats()
    anchor_ts, valid, in_census, block = None, False, False, []
    for ev in chunks:
        for r in ev:
            k = int(r["kind"])
            st.events += 1
            if k == EventKind.SNAPSHOT_BEGIN:
                in_census, block = True, []
                continue
            if in_census:
                if k == EventKind.SNAPSHOT_END:
                    in_census = False
                    census = _census(np.asarray(block, dtype=ev.dtype)) if block else {}
                    st.censuses += 1
                    if valid:
                        rec = reconcile(book, census)
                        rec["ts"] = int(r["ts_exchange"])
                        st.reconciliations.append(rec)
                    book.reset()
                    # census order within each price level is the assumed time priority
                    for oid, (side, price, size) in census.items():
                        book.add(oid, side, price, size, priority="census")
                    anchor_ts, valid = int(r["ts_exchange"]), True
                else:
                    block.append(r)
                continue
            if k == EventKind.GAP:
                st.gaps += 1
                valid = False
                continue
            if not valid:
                continue
            ts = int(r["ts_exchange"])
            if anchor_ts is not None and ts <= anchor_ts:
                st.pre_anchor_skipped += 1
                continue
            oid, side, price, size = int(r["order_id"]), int(r["side"]), int(r["price"]), float(r["size"])
            if k == EventKind.ORDER_ADD:
                st.adds += 1
                if not book.add(oid, side, price, size):
                    st.duplicate_adds += 1
            elif k == EventKind.ORDER_MODIFY:
                st.modifies += 1
                res = book.modify(oid, price, size)
                if res == "unknown":
                    st.unknown_modifies += 1
                elif res == "lost_priority":
                    st.priority_losses += 1
            elif k == EventKind.ORDER_DELETE:
                st.deletes += 1
                if not book.delete(oid):
                    st.unknown_deletes += 1
            elif k == EventKind.TRADE:
                st.trades += 1
                if oid in book.orders:
                    st.trades_maker_known += 1
            else:
                continue
            bb, ba = book.best(1), book.best(-1)
            if bb is not None and ba is not None and bb >= ba:
                st.crossed_states += 1
            if on_state is not None:
                on_state(book, r)
    return book, st


__all__ = ["Order", "MBOBook", "MBOStats", "reconstruct", "reconcile"]

"""Matching engine v2: deterministic order-level price-time priority (M30), order semantics (M32) and queue
behaviour (M34).

Prices are integer ticks and quantities integer lots; nothing is random, so identical command sequences give
identical reports and states (the journal and the reference engine rely on this).

Order semantics:

* ``limit`` (GTC / IOC / FOK, optional ``post_only``) and ``market`` orders; a market order walks at most
  ``rules.max_walk_levels`` levels and its unfilled remainder is cancelled (reported ``expire``);
* partial fills at each maker price, maker reports carry the remaining quantity, the taker is reported per fill;
* ``cancel`` of a live order removes it; cancelling a filled, cancelled or unknown order is rejected with reason
  ``too_late`` or ``unknown`` (the cancel race is therefore observable);
* ``replace`` changes price and/or quantity: a quantity decrease at the same price keeps queue priority (under
  ``modify_priority = decrease_only``), anything else re-queues at the back and may trade as a taker;
* self-trade prevention, post-only handling and allocation follow ``Rules``.

Queue behaviour: orders at a price are FIFO by entry sequence; ``queue_ahead(oid)`` returns the orders and lots in
front of an order; cancellations ahead advance it, additions behind never do.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import OrderedDict
from dataclasses import asdict, dataclass

from .rules import Rules

BUY, SELL = 1, -1
KINDS = ("limit", "market")
TIFS = ("GTC", "IOC", "FOK")


class EngineError(ValueError):
    pass


@dataclass
class Order:
    oid: int
    owner: str
    side: int
    qty: int
    kind: str = "limit"
    price: int | None = None
    tif: str = "GTC"
    post_only: bool = False
    remaining: int = 0
    seq: int = 0
    ts: int = 0
    status: str = "new"            # new | live | filled | cancelled | rejected | expired

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Report:
    kind: str                      # ack | fill | cancel | reject | expire | replace
    oid: int
    owner: str
    ts: int
    price: int | None = None
    qty: int = 0
    liquidity: str | None = None   # maker | taker (fills)
    counter: int | None = None     # counterparty order id (fills)
    remaining: int = 0
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Level:
    __slots__ = ("price", "orders", "total")

    def __init__(self, price: int) -> None:
        self.price = price
        self.orders: OrderedDict[int, Order] = OrderedDict()
        self.total = 0


@dataclass
class EngineStats:
    commands: int = 0
    fills: int = 0
    volume: int = 0
    self_trade_prevented: int = 0
    rejects: int = 0
    cancel_too_late: int = 0


class MatchingEngine:
    def __init__(self, rules: Rules | None = None) -> None:
        self.rules = rules or Rules()
        self.levels: dict[int, dict[int, Level]] = {BUY: {}, SELL: {}}
        self._prices: dict[int, list[int]] = {BUY: [], SELL: []}     # ascending; best bid is [-1], best ask [0]
        self.orders: dict[int, Order] = {}                            # every order ever seen (status kept)
        self._seq = 0
        self.stats = EngineStats()
        self.journal = None                                           # optional journal (M37)

    # ------------------------------------------------------------------------------------------ queries
    def best(self, side: int) -> int | None:
        p = self._prices[side]
        if not p:
            return None
        return p[-1] if side == BUY else p[0]

    def spread(self) -> int | None:
        b, a = self.best(BUY), self.best(SELL)
        return None if b is None or a is None else a - b

    def depth(self, side: int, n: int = 10) -> list[tuple[int, int]]:
        p = self._prices[side]
        prices = p[::-1][:n] if side == BUY else p[:n]
        return [(x, self.levels[side][x].total) for x in prices]

    def level_qty(self, side: int, price: int) -> int:
        lv = self.levels[side].get(price)
        return lv.total if lv else 0

    def queue_ahead(self, oid: int) -> dict | None:
        o = self.orders.get(oid)
        if o is None or o.status != "live":
            return None
        lv = self.levels[o.side][o.price]
        n, q = 0, 0
        for k, x in lv.orders.items():
            if k == oid:
                break
            n += 1
            q += x.remaining
        return {"orders_ahead": n, "lots_ahead": q, "price": o.price, "level_total": lv.total}

    def live_orders(self, owner: str | None = None) -> list[Order]:
        return [o for o in self.orders.values() if o.status == "live" and (owner is None or o.owner == owner)]

    # ------------------------------------------------------------------------------------------ internals
    def _add_level(self, side: int, price: int) -> Level:
        lv = Level(price)
        self.levels[side][price] = lv
        insort(self._prices[side], price)
        return lv

    def _remove_level(self, side: int, price: int) -> None:
        del self.levels[side][price]
        p = self._prices[side]
        i = bisect_left(p, price)
        del p[i]

    def _rest(self, o: Order) -> None:
        lv = self.levels[o.side].get(o.price) or self._add_level(o.side, o.price)
        self._seq += 1
        o.seq = self._seq
        o.status = "live"
        lv.orders[o.oid] = o
        lv.total += o.remaining

    def _unrest(self, o: Order) -> None:
        lv = self.levels[o.side][o.price]
        lv.orders.pop(o.oid)
        lv.total -= o.remaining
        if not lv.orders:
            self._remove_level(o.side, o.price)

    def _crosses(self, side: int, price: int | None) -> bool:
        opp = self.best(-side)
        if opp is None:
            return False
        if price is None:
            return True
        return price >= opp if side == BUY else price <= opp

    def _fillable(self, o: Order) -> int:
        """Quantity an order could fill now (FOK check); own orders do not count unless self trades are allowed."""
        total, levels = 0, 0
        opp = -o.side
        prices = self._prices[opp] if opp == SELL else self._prices[opp][::-1]
        for p in prices:
            if o.price is not None and ((o.side == BUY and p > o.price) or (o.side == SELL and p < o.price)):
                break
            levels += 1
            if levels > self.rules.max_walk_levels:
                break
            lv_orders = self.levels[opp][p].orders.values()
            if (self.rules.allocation == "pro_rata" and self.rules.self_trade == "cancel_taker"
                    and any(x.owner == o.owner for x in lv_orders)):
                return total
            for x in lv_orders:
                if x.owner == o.owner and self.rules.self_trade != "allow":
                    if self.rules.self_trade == "cancel_taker":
                        return total
                    continue
                total += x.remaining
                if total >= o.remaining:
                    return total
        return total

    def _allocate(self, eligible: list[Order], take: int) -> list[tuple[Order, int]]:
        """(maker, lots) pairs at one level among ``eligible`` makers (queue order) for ``take`` lots."""
        if not eligible:
            return []
        if self.rules.allocation == "price_time":
            out, left = [], take
            for x in eligible:
                if left <= 0:
                    break
                q = min(left, x.remaining)
                out.append((x, q))
                left -= q
            return out
        total = sum(x.remaining for x in eligible)
        take = min(take, total)
        alloc = {x.oid: min(x.remaining, max(self.rules.pro_rata_min, (x.remaining * take) // total))
                 for x in eligible}
        over = sum(alloc.values()) - take
        for x in reversed(eligible):          # trim the minimum-lot overshoot from the back of the queue
            if over <= 0:
                break
            cut = min(over, alloc[x.oid])
            alloc[x.oid] -= cut
            over -= cut
        left = take - sum(alloc.values())
        for x in eligible:                    # FIFO remainder
            if left <= 0:
                break
            extra = min(left, x.remaining - alloc[x.oid])
            alloc[x.oid] += extra
            left -= extra
        return [(x, alloc[x.oid]) for x in eligible if alloc[x.oid] > 0]

    def _cancel_taker(self, o: Order, ts: int) -> Report:
        self.stats.self_trade_prevented += 1
        rep = Report("cancel", o.oid, o.owner, ts, o.price, o.remaining, reason="self_trade")
        o.status, o.remaining = "cancelled", 0
        return rep

    def _match(self, o: Order, ts: int) -> list[Report]:
        """Self-trade rules per level: ``cancel_maker`` cancels the owner's resting orders at the level first;
        ``cancel_taker`` trades (price-time) with the orders ahead of the owner's first resting order, then cancels
        the taker remainder (pro-rata: any own order at the level cancels the taker before trading there)."""
        reps: list[Report] = []
        opp = -o.side
        walked = 0
        stp = self.rules.self_trade
        while o.remaining > 0:
            price = self.best(opp)
            if price is None or (o.price is not None and ((o.side == BUY and price > o.price) or
                                                           (o.side == SELL and price < o.price))):
                break
            walked += 1
            if walked > self.rules.max_walk_levels:
                break
            lv = self.levels[opp][price]
            own_present = any(x.owner == o.owner for x in lv.orders.values())
            stop_after = False
            if own_present and stp == "cancel_maker":
                for x in [x for x in lv.orders.values() if x.owner == o.owner]:
                    self.stats.self_trade_prevented += 1
                    reps.append(Report("cancel", x.oid, x.owner, ts, x.price, x.remaining, reason="self_trade"))
                    self._unrest(x)
                    x.status, x.remaining = "cancelled", 0
                continue
            if own_present and stp == "cancel_taker":
                if self.rules.allocation == "pro_rata":
                    reps.append(self._cancel_taker(o, ts))
                    return reps
                eligible = []
                for x in lv.orders.values():
                    if x.owner == o.owner:
                        break
                    eligible.append(x)
                stop_after = True
            else:
                eligible = list(lv.orders.values())
            for x, q in self._allocate(eligible, o.remaining):
                x.remaining -= q
                lv.total -= q
                o.remaining -= q
                self.stats.fills += 1
                self.stats.volume += q
                reps.append(Report("fill", x.oid, x.owner, ts, price, q, "maker", o.oid, x.remaining))
                reps.append(Report("fill", o.oid, o.owner, ts, price, q, "taker", x.oid, o.remaining))
                if x.remaining == 0:
                    x.status = "filled"
                    lv.orders.pop(x.oid)
            if not lv.orders:
                self._remove_level(opp, price)
            if stop_after:
                if o.remaining > 0:
                    reps.append(self._cancel_taker(o, ts))
                return reps
        return reps

    # ------------------------------------------------------------------------------------------ commands
    def submit(self, order: Order, ts: int = 0) -> list[Report]:
        self.stats.commands += 1
        if self.journal is not None:
            self.journal.record("submit", ts, order=order.to_dict())
        o = Order(**{k: v for k, v in order.to_dict().items() if k not in ("remaining", "seq", "status")})
        o.remaining, o.ts = o.qty, ts
        if o.oid in self.orders:
            return self._reject(o, ts, "duplicate_order_id")
        self.orders[o.oid] = o
        if o.side not in (BUY, SELL) or o.kind not in KINDS or o.tif not in TIFS:
            return self._reject(o, ts, "invalid_order")
        if not self.rules.valid_qty(o.qty):
            return self._reject(o, ts, "invalid_quantity")
        if o.kind == "limit" and o.price is None:
            return self._reject(o, ts, "limit_without_price")
        if o.kind == "market" and (o.price is not None or o.post_only):
            return self._reject(o, ts, "invalid_market_order")
        reps = [Report("ack", o.oid, o.owner, ts, o.price, o.qty, remaining=o.qty)]
        if o.post_only and self._crosses(o.side, o.price):
            if self.rules.post_only == "reject":
                return reps + self._reject(o, ts, "post_only_would_cross")[0:1]
            opp = self.best(-o.side)
            o.price = opp - 1 if o.side == BUY else opp + 1
            reps.append(Report("replace", o.oid, o.owner, ts, o.price, o.remaining, reason="post_only_slide"))
        if o.tif == "FOK" and self._fillable(o) < o.remaining:
            o.status = "expired"
            reps.append(Report("expire", o.oid, o.owner, ts, o.price, o.remaining, reason="fok_not_fillable"))
            o.remaining = 0
            return reps
        reps += self._match(o, ts)
        if o.remaining > 0 and o.status not in ("cancelled",):
            if o.kind == "market" or o.tif in ("IOC", "FOK"):
                reps.append(Report("expire", o.oid, o.owner, ts, o.price, o.remaining,
                                   reason="market_unfilled" if o.kind == "market" else "ioc_remainder"))
                o.status, o.remaining = "expired", 0
            else:
                self._rest(o)
        elif o.remaining == 0 and o.status not in ("cancelled", "expired"):
            o.status = "filled"
        return reps

    def _reject(self, o: Order, ts: int, reason: str) -> list[Report]:
        self.stats.rejects += 1
        o.status, o.remaining = "rejected", 0
        return [Report("reject", o.oid, o.owner, ts, o.price, o.qty, reason=reason)]

    def cancel(self, oid: int, ts: int = 0, owner: str | None = None) -> list[Report]:
        self.stats.commands += 1
        if self.journal is not None:
            self.journal.record("cancel", ts, oid=oid, owner=owner)
        o = self.orders.get(oid)
        if o is None:
            return [Report("reject", oid, owner or "?", ts, reason="unknown")]
        if owner is not None and o.owner != owner:
            return [Report("reject", oid, owner, ts, reason="not_owner")]
        if o.status != "live":
            self.stats.cancel_too_late += 1
            return [Report("reject", oid, o.owner, ts, reason="too_late")]
        q = o.remaining
        self._unrest(o)
        o.status, o.remaining = "cancelled", 0
        return [Report("cancel", oid, o.owner, ts, o.price, q)]

    def replace(self, oid: int, ts: int = 0, *, price: int | None = None, qty: int | None = None,
                owner: str | None = None) -> list[Report]:
        """New remaining quantity ``qty`` and/or new ``price``; see the module docstring for priority rules."""
        self.stats.commands += 1
        if self.journal is not None:
            self.journal.record("replace", ts, oid=oid, price=price, qty=qty, owner=owner)
        o = self.orders.get(oid)
        if o is None or o.status != "live" or (owner is not None and o.owner != owner):
            if o is not None and o.status != "live":
                self.stats.cancel_too_late += 1
            return [Report("reject", oid, owner or (o.owner if o else "?"), ts,
                           reason="too_late" if o is not None else "unknown")]
        new_price = o.price if price is None else price
        new_qty = o.remaining if qty is None else qty
        if new_qty <= 0:
            journal, self.journal = self.journal, None
            try:
                return self.cancel(oid, ts)
            finally:
                self.journal = journal
        if not self.rules.valid_qty(new_qty):
            return [Report("reject", oid, o.owner, ts, reason="invalid_quantity")]
        if (new_price == o.price and new_qty <= o.remaining and self.rules.modify_priority == "decrease_only"):
            lv = self.levels[o.side][o.price]
            lv.total -= o.remaining - new_qty
            o.remaining = new_qty
            return [Report("replace", oid, o.owner, ts, o.price, new_qty, remaining=new_qty, reason="kept_priority")]
        self._unrest(o)
        o.price, o.remaining = new_price, new_qty
        reps = [Report("replace", oid, o.owner, ts, new_price, new_qty, remaining=new_qty, reason="lost_priority")]
        o.status = "new"
        reps += self._match(o, ts)
        if o.remaining > 0 and o.status != "cancelled":
            self._rest(o)
        elif o.remaining == 0 and o.status != "cancelled":
            o.status = "filled"
        return reps

    # ------------------------------------------------------------------------------------------ state
    def state(self) -> dict:
        """Canonical state (for journal recovery and differential tests)."""
        return {"seq": self._seq,
                "levels": {str(s): [[p, [[o.oid, o.owner, o.remaining, o.seq] for o in self.levels[s][p].orders.values()]]
                                    for p in (self._prices[s][::-1] if s == BUY else self._prices[s])]
                           for s in (BUY, SELL)},
                "orders": {str(k): [v.status, v.remaining] for k, v in sorted(self.orders.items())}}


__all__ = ["BUY", "SELL", "Order", "Report", "MatchingEngine", "EngineError", "EngineStats", "KINDS", "TIFS"]

"""Aggregate-queue limit order book with explicit strategy queue positions (v0.8 market core).

The book is a model component, not a replacement for the frozen v0.5 matching
engine. Prices are integer ticks. Each side keeps ``levels`` contiguous price
levels measured from its own touch; ``bq[i]`` is the *total* visible size at
``bid_px - i`` and ``aq[i]`` at ``ask_px + i`` (background plus strategy), so a
snapshot is exactly what an aggregate-L2 observer would see. Empty levels are
allowed away from the touch; the touch is never empty.

Semantics (each covered by tests):

* market orders walk the opposite side from the touch; at each price the volume
  is consumed in FIFO order: background ahead of the first strategy order, that
  order, background between, the next order, ..., then background behind;
* a depleted touch moves the price away (``shift_out``) and a new deepest level
  is drawn from the registered refill sampler; when the touch improves
  (``shift_in``) the deepest level leaves the window unless a strategy order
  rests there, in which case the window grows (it shrinks back as the touch
  recedes), so strategy orders are never dropped;
* a background addition joins the back of its level (strategy positions are
  unchanged); an addition inside the spread creates a new touch;
* a background cancellation removes a contiguous block of whole background lots
  at a position chosen by ``cancel_position`` (``uniform``: the ground-truth default;
  ``front``/``back``: the extreme assumptions used for bounds), never strategy
  volume, and reduces the ``ahead`` of each strategy order by the overlap;
* ``hidden_fraction`` > 0 makes a fully consumed touch replenish once per market
  order by that fraction of what was consumed (a stylized hidden/iceberg
  sensitivity knob, never a claim about hidden liquidity in real data).

Only the strategy's own orders carry identity. Background liquidity is aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import numpy as np

BUY, SELL = 1, -1
CANCEL_POSITIONS = ("uniform", "front", "back")
MAX_WALK = 200


class BookError(RuntimeError):
    """An operation would break a book invariant."""


@dataclass
class OwnOrder:
    oid: int
    side: int                     # +1 resting bid (buy), -1 resting ask (sell)
    price: int
    qty: int
    remaining: int
    ahead: int                    # background lots in front of this order at its price (whole lots)
    entered_t: float
    status: str = "resting"       # resting | filled | cancelled
    fills: list = field(default_factory=list)   # (exchange time, price, qty)

    @property
    def filled(self) -> int:
        return self.qty - self.remaining


class AggBook:
    def __init__(self, *, refill: Callable[[int, np.random.Generator], int], rng: np.random.Generator,
                 levels: int = 40, mid_ticks: int = 4000, spread: int = 1, cancel_position: str = "uniform",
                 hidden_fraction: float = 0.0) -> None:
        if cancel_position not in CANCEL_POSITIONS:
            raise ValueError(f"cancel_position must be one of {CANCEL_POSITIONS}")
        if not 0.0 <= hidden_fraction <= 1.0:
            raise ValueError("hidden_fraction must lie in [0, 1]")
        if levels < 5 or spread < 1:
            raise ValueError("need at least 5 levels and a positive spread")
        self.refill, self.rng, self.levels = refill, rng, levels
        self.cancel_position, self.hidden_fraction = cancel_position, float(hidden_fraction)
        self.bid_px = int(mid_ticks) - spread // 2 - spread % 2
        self.ask_px = self.bid_px + spread
        self.bq = [max(1, int(refill(i, rng))) for i in range(levels)]
        self.aq = [max(1, int(refill(i, rng))) for i in range(levels)]
        self.own: dict[int, OwnOrder] = {}
        self._at: dict[tuple[int, int], list[int]] = {}
        self._next = 1
        self.tape: list[tuple[float, int, int, int, bool]] = []   # (t, price, qty, aggressor side, strategy taker)

    # ------------------------------------------------------------------ geometry
    def spread(self) -> int:
        return self.ask_px - self.bid_px

    def mid(self) -> float:
        return 0.5 * (self.bid_px + self.ask_px)

    def touch(self, side: int) -> int:
        return self.bid_px if side == BUY else self.ask_px

    def queues(self, side: int) -> list[int]:
        return self.bq if side == BUY else self.aq

    def index(self, side: int, price: int) -> int | None:
        i = self.bid_px - price if side == BUY else price - self.ask_px
        return i if 0 <= i < len(self.queues(side)) else None

    def price(self, side: int, level: int) -> int:
        return self.bid_px - level if side == BUY else self.ask_px + level

    def own_volume(self, side: int, price: int) -> int:
        return sum(self.own[o].remaining for o in self._at.get((side, price), ()))

    def snapshot(self, visible: int = 10) -> tuple[int, int, list[int], list[int]]:
        return self.bid_px, self.ask_px, self.bq[:visible], self.aq[:visible]

    # ------------------------------------------------------------------ shifts
    def _shift_out(self, side: int) -> None:
        q = self.queues(side)
        guard = 0
        while q[0] <= 0:
            if self._at.get((side, self.touch(side))):
                raise BookError("strategy volume at an empty touch")
            q.pop(0)
            if len(q) < self.levels:      # a window grown by _shift_in shrinks back to its registered size
                q.append(max(1, int(self.refill(self.levels - 1, self.rng))))
            if side == BUY:
                self.bid_px -= 1
            else:
                self.ask_px += 1
            guard += 1
            if guard > 10 * self.levels:
                raise BookError("touch could not be restored")

    def _shift_in(self, side: int, ticks: int) -> None:
        q = self.queues(side)
        for _ in range(ticks):
            dropped = self.price(side, len(q) - 1)
            if len(q) >= self.levels and not self._at.get((side, dropped)):
                q.pop()           # the deepest level leaves the window unless a strategy order rests there
            q.insert(0, 0)
            if side == BUY:
                self.bid_px += 1
            else:
                self.ask_px -= 1

    # ------------------------------------------------------------------ background flow
    def add(self, side: int, level: int, qty: int) -> int:
        """Background limit order; ``level`` < 0 improves the touch by ``-level`` ticks if the spread allows."""
        qty = int(qty)
        if qty <= 0:
            return 0
        if level < 0:
            room = self.spread() - 1
            ticks = min(-level, room)
            if ticks > 0:
                self._shift_in(side, ticks)
            level = 0
        level = min(level, len(self.queues(side)) - 1)
        self.queues(side)[level] += qty
        return qty

    def cancel(self, side: int, level: int, qty: int) -> int:
        """Remove up to ``qty`` background lots at ``level`` (never strategy volume)."""
        q = self.queues(side)
        level = min(max(level, 0), len(q) - 1)
        price = self.price(side, level)
        ids = self._at.get((side, price), ())
        own = sum(self.own[o].remaining for o in ids)
        background = q[level] - own
        c = min(int(qty), background)
        if c <= 0:
            return 0
        if ids:
            # block positions are whole lots, so every strategy order's queue position stays an integer
            if self.cancel_position == "uniform":
                start = int(self.rng.integers(0, background - c + 1)) if background > c else 0
            elif self.cancel_position == "front":
                start = 0
            else:
                start = background - c
            end = start + c
            for oid in ids:
                order = self.own[oid]
                overlap = max(0, min(order.ahead, end) - start) if start < order.ahead else 0
                order.ahead = max(0, order.ahead - overlap)
        q[level] -= c
        if level == 0 and q[0] <= 0:
            self._shift_out(side)
        return c

    # ------------------------------------------------------------------ aggressive flow
    def market(self, side: int, qty: int, t: float, *, strategy: bool = False,
               max_walk: int = MAX_WALK) -> tuple[int, float, list[tuple[int, int]]]:
        """Market order; ``side`` +1 buys from the asks. Returns (filled, notional in ticks, [(price, qty)])."""
        remaining = int(qty)
        filled, notional, fills = 0, 0.0, []
        opposite = -side
        walked = 0
        hidden_used: set[int] = set()
        while remaining > 0 and walked < max_walk:
            q = self.queues(opposite)
            price = self.touch(opposite)
            take = min(remaining, q[0])
            if take > 0:
                if strategy and self._at.get((opposite, price)):
                    raise BookError("self-trade: strategy market order against its own resting order")
                self._allocate(opposite, price, take, t)
                q[0] -= take
                remaining -= take
                filled += take
                notional += take * price
                fills.append((price, take))
                self.tape.append((t, price, take, side, strategy))
            if q[0] <= 0:
                if remaining > 0 and self.hidden_fraction > 0 and price not in hidden_used and take > 0:
                    hidden_used.add(price)
                    hidden = int(math.floor(self.hidden_fraction * take + 0.5))
                    if hidden > 0:
                        q[0] = hidden
                        continue
                self._shift_out(opposite)
                walked += 1
        return filled, notional, fills

    def _allocate(self, side: int, price: int, take: int, t: float) -> None:
        """FIFO allocation of ``take`` lots at ``price`` between background and strategy orders."""
        ids = self._at.get((side, price))
        if not ids:
            return
        rem = take
        consumed_bg = 0
        for oid in list(ids):
            order = self.own[oid]
            bg = min(rem, max(0, order.ahead - consumed_bg))
            consumed_bg += bg
            rem -= bg
            if rem <= 0:
                break
            mine = min(rem, order.remaining)
            if mine > 0:
                order.remaining -= mine
                order.fills.append((t, price, int(mine)))
                rem -= mine
            if order.remaining <= 0:
                order.remaining = 0
                order.status = "filled"
                ids.remove(oid)
            if rem <= 0:
                break
        consumed_bg += max(rem, 0)
        for oid in ids:
            order = self.own[oid]
            order.ahead = max(0, order.ahead - min(consumed_bg, order.ahead))
        if not ids:
            del self._at[(side, price)]

    # ------------------------------------------------------------------ strategy orders
    def submit_limit(self, side: int, qty: int, t: float, *, improve: int = 0, level: int = 0) -> int | None:
        """Strategy limit order at its own touch (``level``), or ``improve`` ticks inside the spread.

        The order joins the back of its level: ``ahead`` is the background volume already there plus nothing
        for later arrivals. Never crosses: a marketable request is placed at the touch instead.
        """
        qty = int(qty)
        if qty <= 0:
            return None
        if improve > 0:
            ticks = min(improve, self.spread() - 1)
            if ticks > 0:
                self._shift_in(side, ticks)
            level = 0
        q = self.queues(side)
        level = min(max(level, 0), len(q) - 1)
        price = self.price(side, level)
        own = self.own_volume(side, price)
        oid = self._next
        self._next += 1
        self.own[oid] = OwnOrder(oid, side, price, qty, qty, int(q[level] - own), t)
        self._at.setdefault((side, price), []).append(oid)
        q[level] += qty
        return oid

    def cancel_own(self, oid: int) -> int:
        order = self.own.get(oid)
        if order is None or order.status != "resting":
            return 0
        i = self.index(order.side, order.price)
        if i is None:
            raise BookError("resting strategy order outside the modelled window")
        q = self.queues(order.side)
        q[i] -= order.remaining
        cancelled = order.remaining
        order.status = "cancelled"
        ids = self._at[(order.side, order.price)]
        ids.remove(oid)
        if not ids:
            del self._at[(order.side, order.price)]
        if i == 0 and q[0] <= 0:
            self._shift_out(order.side)
        return cancelled

    def resting(self) -> list[OwnOrder]:
        return [o for o in self.own.values() if o.status == "resting"]

    # ------------------------------------------------------------------ read-only valuation
    def walk_cost(self, side: int, qty: int) -> tuple[float, int]:
        """Notional (ticks) and fillable quantity of a hypothetical market order on the modelled window."""
        q = self.queues(-side)
        remaining, notional, fillable = int(qty), 0.0, 0
        for i, size in enumerate(q):
            if remaining <= 0:
                break
            take = min(remaining, size)
            notional += take * self.price(-side, i)
            fillable += take
            remaining -= take
        return notional, fillable

    def check(self) -> None:
        if not self.bid_px < self.ask_px:
            raise BookError("crossed or locked book")
        for side in (BUY, SELL):
            q = self.queues(side)
            if q[0] <= 0 or min(q) < 0 or len(q) < self.levels:
                raise BookError("invalid queue sizes")
        for (side, price), ids in self._at.items():
            i = self.index(side, price)
            if i is None or self.queues(side)[i] < sum(self.own[o].remaining for o in ids):
                raise BookError("strategy volume exceeds its level")
            for oid in ids:
                o = self.own[oid]
                if o.status != "resting" or o.remaining <= 0 or o.ahead < 0 or o.ahead != int(o.ahead):
                    raise BookError("inconsistent strategy order")

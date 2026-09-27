"""Historical replay behind the ExchangeSimulator interface, with bounded passive fills (M8).

Every execution agent and learned policy runs unchanged: the environment, risk
limits, completion rule and settlement see an ``ExchangeSimulator``. Underneath:

* Displayed historical levels are held as identity-free ``HIST`` liquidity and
  reset to the recorded quantities at every historical update (no market impact
  of the hypothetical parent is assumed; consumed displayed depth refills only
  when the source reports it).
* Own marketable orders match against displayed levels only (no hidden size).
* Own resting orders receive passive fills only from ``lob.fill_bounds`` trackers
  driven by aggressor-signed historical prints and displayed-level updates, under
  one declared ``fill_mode`` (conservative or optimistic). No exact FIFO fill is
  claimed; each mode is a self-consistent path because later decisions depend on
  earlier fills.
* Simulator time maps to historical time by a frozen clock ratio, and quantities
  by a frozen lot scale, both fitted on development data only.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, replace
import math
from typing import List

import numpy as np

from .engine import ExchangeSimulator, Order, OrderBook, OrderStatus, OrderType, Side, SimConfig, Trade
from .fill_bounds import ChildOrder, L2FillTracker
from .historical_execution import L2_WITH_TRADES

FILL_MODES = ("conservative", "optimistic")
HIST = "HIST"


@dataclass
class HistoricalEpisode:
    """A replay window in historical seconds; levels in integer ticks, quantities in lots (float)."""

    start_s: float
    updates: list            # [(hist_time_s, {price: qty}, {price: qty})] complete-book snapshots, top N
    prints: list             # [(hist_time_s, price_ticks, qty_lots, aggressor 'BUY'/'SELL')]
    tick_size: float         # currency per tick
    clock_ratio: float       # historical seconds per simulated second
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.updates:
            raise ValueError("episode has no book updates")
        if not math.isfinite(self.clock_ratio) or self.clock_ratio <= 0:
            raise ValueError("clock ratio must be positive")


class HistoricalSimulator(ExchangeSimulator):
    """Replays one episode; all background flow is historical."""

    def __init__(self, cfg: SimConfig, episode: HistoricalEpisode, fill_mode: str) -> None:
        if fill_mode not in FILL_MODES:
            raise ValueError(f"fill_mode must be one of {FILL_MODES}")
        first_bids, first_asks = episode.updates[0][1], episode.updates[0][2]
        mid = (max(first_bids) + min(first_asks)) / 2 if first_bids and first_asks else 1000
        cfg = replace(cfg, tick_size=episode.tick_size, initial_mid_ticks=max(21, int(round(mid))), lot_size=1)
        self.cfg = cfg
        self.book = OrderBook(cfg.tick_size, cfg.lot_size, cfg.check_invariants)
        seeds = np.random.SeedSequence(cfg.seed).spawn(5)
        self.rng, self.rng_flow, self.rng_lat, self.rng_cancel, self.rng_resilience = [
            np.random.default_rng(s) for s in seeds]
        self.seed_manifest = {"global_seed": cfg.seed, "historical_episode": episode.label, "fill_mode": fill_mode}
        self.t = 0.0
        self._heap: list = []
        self._seq = self._oid = self._background_oid = 0
        self.orders: dict[int, Order] = {}
        self.events: list[dict] = []
        self.event_count = 0
        self._cancel_pending: set[int] = set()
        self.episode = episode
        self.fill_mode = fill_mode
        self._hist_orders: dict[tuple[Side, int], Order] = {}
        self._update_i = 0
        self._print_i = 0
        self._trackers: dict[int, tuple[L2FillTracker, float]] = {}
        self.book_valid = True
        self.stats = {"updates": 0, "prints": 0, "crossed_updates_skipped": 0, "bounded_fill_lots": 0.0,
                      "fractional_lots_unapplied": 0.0}
        self._sync(episode.updates[0][1], episode.updates[0][2])
        self.book.last_mid = float(self.book.mid())

    # ------------------------------------------------------------------ clock mapping
    def _hist_time(self, sim_t: float) -> float:
        return self.episode.start_s + sim_t * self.episode.clock_ratio

    def _sim_time(self, hist_t: float) -> float:
        return (hist_t - self.episode.start_s) / self.episode.clock_ratio

    # ------------------------------------------------------------------ displayed liquidity
    def _sync(self, bids: dict, asks: dict) -> None:
        if bids and asks and max(bids) >= min(asks):
            self.stats["crossed_updates_skipped"] += 1
            self.book_valid = False
            return
        self.book_valid = bool(bids) and bool(asks)
        # Displayed liquidity at or through one of our own resting prices would have met that
        # order; its interaction is represented only by the fill-bound trackers, so such levels
        # are withheld from the engine book instead of creating a crossed book.
        own = [o for o in self.book.orders.values() if o.owner != HIST]
        own_bid = max((o.price for o in own if o.side is Side.BUY), default=None)
        own_ask = min((o.price for o in own if o.side is Side.SELL), default=None)
        if own_bid is not None:
            withheld = [p for p in asks if p <= own_bid]
            asks = {p: q for p, q in asks.items() if p > own_bid}
            self.stats["levels_withheld_against_own_orders"] = self.stats.get(
                "levels_withheld_against_own_orders", 0) + len(withheld)
        if own_ask is not None:
            withheld = [p for p in bids if p >= own_ask]
            bids = {p: q for p, q in bids.items() if p < own_ask}
            self.stats["levels_withheld_against_own_orders"] = self.stats.get(
                "levels_withheld_against_own_orders", 0) + len(withheld)
        # Stale levels on both sides go first, so a multi-level move never leaves new levels
        # facing not-yet-removed stale ones (the final state of each update is unchanged).
        for side, levels in ((Side.BUY, bids), (Side.SELL, asks)):
            for key in [k for k in self._hist_orders if k[0] is side and k[1] not in levels]:
                order = self._hist_orders.pop(key)
                if order.order_id in self.book.orders:
                    self.book.cancel(order.order_id, self.t)
        for side, levels in ((Side.BUY, bids), (Side.SELL, asks)):
            for price, qty in levels.items():
                lots = int(qty)
                existing = self._hist_orders.get((side, price))
                if existing is not None and existing.order_id in self.book.orders:
                    if lots <= 0:
                        self.book.cancel(existing.order_id, self.t)
                        del self._hist_orders[(side, price)]
                    elif lots != existing.remaining:
                        _, _, volumes = self.book._levels(side)
                        volumes[price] += lots - existing.remaining
                        existing.qty = existing.filled_qty + lots
                        existing.remaining = lots
                    continue
                if lots > 0:
                    order = Order(self._next_oid(True), side, lots, OrderType.LIMIT, HIST, self.t, price=price)
                    order.validate(self.cfg.lot_size)
                    order.ts_active = self.t
                    order.transition(OrderStatus.ACKNOWLEDGED, self.t)
                    order.transition(OrderStatus.RESTING, self.t)
                    self.book._add_resting(order)
                    self._hist_orders[(side, price)] = order
        for oid, (tracker, _) in self._trackers.items():
            levels = bids if tracker.order.side == "BUY" else asks
            tracker.on_level(float(levels.get(tracker.order.price, 0.0)))

    # ------------------------------------------------------------------ bounded passive fills
    def _register(self, order: Order) -> None:
        if order.owner == HIST or order.order_id in self._trackers or order.order_id not in self.book.orders:
            return
        hist = self._hist_orders.get((order.side, order.price))
        displayed = float(hist.remaining) if hist is not None and hist.order_id in self.book.orders else 0.0
        child = ChildOrder(str(order.order_id), "BUY" if order.side is Side.BUY else "SELL", order.price,
                           float(order.remaining), 0, 1)
        self._trackers[order.order_id] = (L2FillTracker(child, displayed, L2_WITH_TRADES), 0.0)

    def _apply_print(self, price: int, qty: float, aggressor: str) -> None:
        taker_side = Side.BUY if aggressor == "BUY" else Side.SELL
        self.book.trades.append(Trade(self.t, price, max(1, int(round(qty))), taker_side, HIST, HIST, None, None))
        key = "conservative" if self.fill_mode == "conservative" else "optimistic"
        for oid in list(self._trackers):
            tracker, applied = self._trackers[oid]
            tracker.on_print(price, qty, aggressor)
            order = self.book.orders.get(oid)
            if order is None:
                continue
            bound = tracker.fill[key]
            lots = int(math.floor(bound + 1e-9)) - int(math.floor(applied + 1e-9))
            lots = min(lots, order.remaining)
            if lots > 0:
                self._fill_resting(order, lots, taker_side)
                self.stats["bounded_fill_lots"] += lots
            self._trackers[oid] = (tracker, bound)

    def _fill_resting(self, order: Order, lots: int, taker_side: Side) -> None:
        _, _, volumes = self.book._levels(order.side)
        order.remaining -= lots
        order._filled_qty += lots
        volumes[order.price] -= lots
        self.book.trades.append(Trade(self.t, order.price, lots, taker_side, HIST, order.owner, None, order.order_id))
        if order.remaining == 0:
            queue = self.book._levels(order.side)[0][order.price]
            queue.remove(order)
            if not queue:
                self.book._remove_level(order.side, order.price)
            del self.book.orders[order.order_id]
            order.transition(OrderStatus.FILLED, self.t)
            self._trackers.pop(order.order_id, None)
        elif order.status is not OrderStatus.CANCEL_PENDING:
            order.transition(OrderStatus.PARTIALLY_FILLED, self.t)

    # ------------------------------------------------------------------ event loop
    def step(self, dt: float) -> List[Trade]:
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        end = self.t + dt
        cursor = len(self.book.trades)
        episode = self.episode
        while True:
            next_heap = self._heap[0][0] if self._heap else math.inf
            next_update = self._sim_time(episode.updates[self._update_i + 1][0]) \
                if self._update_i + 1 < len(episode.updates) else math.inf
            next_print = self._sim_time(episode.prints[self._print_i][0]) \
                if self._print_i < len(episode.prints) else math.inf
            nxt = min(next_heap, next_update, next_print)
            if nxt > end:
                break
            self.t = max(self.t, nxt)
            self.book.expire(self.t)
            if next_print == nxt:
                _, price, qty, aggressor = episode.prints[self._print_i]
                self._print_i += 1
                self.stats["prints"] += 1
                self._apply_print(price, qty, aggressor)
            elif next_update == nxt:
                self._update_i += 1
                self.stats["updates"] += 1
                _, bids, asks = episode.updates[self._update_i]
                self._sync(bids, asks)
            else:
                import heapq
                now, sequence, kind, payload = heapq.heappop(self._heap)
                if kind == "order":
                    order = payload
                    if not order.is_terminal:
                        self.book.process(order, now)
                        if order.order_id in self._cancel_pending and not order.is_terminal:
                            order.transition(OrderStatus.CANCEL_PENDING, now)
                        self._register(order)
                elif kind == "cancel":
                    accepted = self.book.cancel(payload, now)
                    self._cancel_pending.discard(payload)
                    self._trackers.pop(payload, None)
                    order = self.orders.get(payload)
                    if order is not None and order.status is OrderStatus.CANCEL_PENDING and not accepted:
                        order.transition(OrderStatus.IN_FLIGHT, now)
                elif kind == "expire":
                    pass
                else:
                    raise RuntimeError(f"historical replay does not support event kind {kind}")
            self.event_count += 1
        self.t = end
        self.book.expire(end)
        return self.book.trades[cursor:]


def episode_from_arrays(times: np.ndarray, bids: list[dict], asks: list[dict], prints: list, *, start_s: float,
                        tick_size: float, clock_ratio: float, label: str) -> HistoricalEpisode:
    updates = [(float(t), b, a) for t, b, a in zip(times, bids, asks)]
    index = max(0, bisect_right([u[0] for u in updates], start_s) - 1)
    updates = updates[index:]
    return HistoricalEpisode(start_s, updates, [p for p in prints if p[0] > start_s], tick_size, clock_ratio, label)



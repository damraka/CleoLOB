"""Event-driven Limit Order Book engine.

Implements:
- Price-time priority matching (limit / market / cancel) on dict[price] -> FIFO deque,
  with sorted price ladders maintained via ``bisect`` (O(log n) inserts, O(1) best-of-book).
- Latency simulation: strategy orders reach the matching engine after a stochastic delay
  (base + exponential jitter) through a global event heap, so queue position is earned
  at *arrival* time, not submission time.
- Market impact & resilience: market orders consume real depth (slippage emerges from the
  book), and a replenishment process refills depleted top levels toward a target depth.
- Zero-intelligence background flow (Poisson limit/market/cancel arrivals) so the book
  has organic dynamics for agents to interact with.

All prices are integer ticks internally; multiply by ``SimConfig.tick_size`` for currency.
"""
from __future__ import annotations

import heapq
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class Side(Enum):
    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class Order:
    order_id: int
    side: Side
    qty: int
    otype: OrderType
    owner: str
    ts_submit: float
    price: Optional[int] = None      # ticks; None for market orders
    ts_active: float = 0.0           # when it actually hit the matching engine
    remaining: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        self.remaining = self.qty


@dataclass
class Trade:
    time: float
    price: int                       # ticks
    qty: int
    taker_side: Side
    taker_owner: str
    maker_owner: str


class OrderBook:
    """Price-time priority book: dict[price] -> FIFO deque + bisect-sorted price lists."""

    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size
        self.bids: Dict[int, Deque[Order]] = {}
        self.asks: Dict[int, Deque[Order]] = {}
        self.bid_prices: List[int] = []      # ascending; best bid = last element
        self.ask_prices: List[int] = []      # ascending; best ask = first element
        self.bid_vol: Dict[int, int] = {}
        self.ask_vol: Dict[int, int] = {}
        self.orders: Dict[int, Order] = {}   # resting, active orders only
        self.trades: List[Trade] = []
        self.last_mid: float = 0.0

    # ------------------------------------------------------------- queries
    def best_bid(self) -> Optional[int]:
        return self.bid_prices[-1] if self.bid_prices else None

    def best_ask(self) -> Optional[int]:
        return self.ask_prices[0] if self.ask_prices else None

    def mid(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None:
            self.last_mid = (bb + ba) / 2.0
        return self.last_mid

    def spread(self) -> Optional[int]:
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) if (bb is not None and ba is not None) else None

    def microprice(self) -> float:
        """Volume-weighted mid: (bb * ask_vol + ba * bid_vol) / (bid_vol + ask_vol)."""
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return self.mid()
        vb, va = self.bid_vol.get(bb, 0), self.ask_vol.get(ba, 0)
        if vb + va == 0:
            return (bb + ba) / 2.0
        return (bb * va + ba * vb) / (vb + va)

    def imbalance(self, levels: int = 5) -> float:
        bids, asks = self.depth(levels)
        vb = sum(v for _, v in bids)
        va = sum(v for _, v in asks)
        return 0.0 if (vb + va) == 0 else (vb - va) / (vb + va)

    def depth(self, levels: int = 10) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Top-N (price, volume) per side, best level first."""
        bids = [(p, self.bid_vol[p]) for p in reversed(self.bid_prices[-levels:])]
        asks = [(p, self.ask_vol[p]) for p in self.ask_prices[:levels]]
        return bids, asks

    def walk_cost(self, side: Side, qty: int) -> Tuple[float, int]:
        """Hypothetical VWAP (ticks) of crossing ``qty`` right now -> (vwap, fillable_qty).

        Used to price leftover inventory (slippage of an immediate forced liquidation).
        """
        if side is Side.SELL:
            ladder, vols = list(reversed(self.bid_prices)), self.bid_vol
        else:
            ladder, vols = list(self.ask_prices), self.ask_vol
        remaining, cash = qty, 0.0
        for p in ladder:
            take = min(remaining, vols[p])
            cash += take * p
            remaining -= take
            if remaining == 0:
                break
        filled = qty - remaining
        return (cash / filled if filled else self.mid()), filled

    # ----------------------------------------------------------- mutations
    def _levels(self, side: Side) -> Tuple[Dict[int, Deque[Order]], List[int], Dict[int, int]]:
        if side is Side.BUY:
            return self.bids, self.bid_prices, self.bid_vol
        return self.asks, self.ask_prices, self.ask_vol

    def _add_resting(self, order: Order) -> None:
        book, prices, vols = self._levels(order.side)
        p = int(order.price)  # type: ignore[arg-type]
        if p not in book:
            book[p] = deque()
            insort(prices, p)
            vols[p] = 0
        book[p].append(order)
        vols[p] += order.remaining
        self.orders[order.order_id] = order

    def _remove_level(self, side: Side, price: int) -> None:
        book, prices, vols = self._levels(side)
        del book[price]
        del vols[price]
        prices.pop(bisect_left(prices, price))

    def _match(self, taker: Order, now: float) -> List[Trade]:
        """Consume opposite-side liquidity in price-time priority. Slippage emerges here."""
        trades: List[Trade] = []
        if taker.side is Side.BUY:
            book, prices, vols = self.asks, self.ask_prices, self.ask_vol
        else:
            book, prices, vols = self.bids, self.bid_prices, self.bid_vol
        while taker.remaining > 0 and prices:
            best = prices[0] if taker.side is Side.BUY else prices[-1]
            if taker.otype is OrderType.LIMIT:
                if taker.side is Side.BUY and best > int(taker.price):   # type: ignore[arg-type]
                    break
                if taker.side is Side.SELL and best < int(taker.price):  # type: ignore[arg-type]
                    break
            queue = book[best]
            while queue and taker.remaining > 0:
                maker = queue[0]
                if not maker.active or maker.remaining == 0:
                    queue.popleft()
                    continue
                fill = min(taker.remaining, maker.remaining)
                maker.remaining -= fill
                taker.remaining -= fill
                vols[best] -= fill
                trades.append(Trade(now, best, fill, taker.side, taker.owner, maker.owner))
                if maker.remaining == 0:
                    maker.active = False
                    queue.popleft()
                    self.orders.pop(maker.order_id, None)
            if not queue:
                self._remove_level(taker.side.opposite, best)
        self.trades.extend(trades)
        if trades:
            self.last_mid = float(trades[-1].price)
        return trades

    def process(self, order: Order, now: float) -> List[Trade]:
        """Route an order that has *arrived* at the engine (post-latency)."""
        order.ts_active = now
        trades = self._match(order, now)
        if order.otype is OrderType.LIMIT and order.remaining > 0:
            self._add_resting(order)
        else:
            order.active = False  # market leftovers behave as IOC
        return trades

    def cancel(self, order_id: int) -> bool:
        order = self.orders.get(order_id)
        if order is None or not order.active:
            return False
        order.active = False
        book, prices, vols = self._levels(order.side)
        p = int(order.price)  # type: ignore[arg-type]
        vols[p] -= order.remaining
        try:
            book[p].remove(order)
        except ValueError:
            pass
        if p in book and not book[p]:
            self._remove_level(order.side, p)
        del self.orders[order_id]
        return True


@dataclass
class SimConfig:
    tick_size: float = 0.01
    initial_mid_ticks: int = 10_000          # $100.00
    seed: int = 42
    # Zero-intelligence background flow (per second, per side where applicable)
    limit_rate: float = 25.0
    market_rate: float = 6.0
    cancel_rate: float = 0.15                # per resting order per second (hazard rate)
    offset_p: float = 0.25                   # geometric distribution of limit-price offsets
    limit_qty_mean: float = 35.0
    market_qty_mean: float = 35.0
    # Resilience: replenishment toward target depth after liquidity shocks
    target_level_vol: int = 300
    resilience: float = 0.8                  # refill intensity scale
    resilience_levels: int = 5
    # Latency applied to strategy (non-ZI) messages
    latency_base: float = 0.005              # seconds
    latency_jitter: float = 0.005            # mean of exponential jitter


class ExchangeSimulator:
    """Global event loop: latency queue + background flow + resilience refill.

    Strategy orders are *scheduled*, not applied: ``submit``/``cancel`` push events onto a
    heap at ``now + latency`` and only take effect when the clock reaches them, so agents
    compete for queue position under realistic delays.
    """

    ZI = "ZI"

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.book = OrderBook(cfg.tick_size)
        self.rng = np.random.default_rng(cfg.seed)          # background flow stream
        self.rng_lat = np.random.default_rng(cfg.seed + 1)  # latency stream (kept separate)
        self.t: float = 0.0
        self._heap: List[Tuple[float, int, str, object]] = []
        self._seq = 0
        self._oid = 0
        self.book.last_mid = float(cfg.initial_mid_ticks)
        self._seed_book()

    # ------------------------------------------------------------- helpers
    def _next_oid(self) -> int:
        self._oid += 1
        return self._oid

    def _push(self, t: float, kind: str, payload: object) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, kind, payload))

    def _latency(self) -> float:
        return self.cfg.latency_base + float(self.rng_lat.exponential(self.cfg.latency_jitter))

    def _qty(self, mean: float) -> int:
        return max(1, int(self.rng.exponential(mean)))

    def _seed_book(self) -> None:
        mid = self.cfg.initial_mid_ticks
        for i in range(1, 21):
            for side, price in ((Side.BUY, mid - i), (Side.SELL, mid + i)):
                o = Order(self._next_oid(), side, self.cfg.target_level_vol,
                          OrderType.LIMIT, self.ZI, 0.0, price=price)
                self.book.process(o, 0.0)

    # ---------------------------------------------------------- public API
    def submit(self, side: Side, qty: int, owner: str,
               price_ticks: Optional[int] = None) -> int:
        """Submit a strategy order; it reaches the book after simulated latency."""
        otype = OrderType.LIMIT if price_ticks is not None else OrderType.MARKET
        order = Order(self._next_oid(), side, qty, otype, owner, self.t, price=price_ticks)
        self._push(self.t + self._latency(), "order", order)
        return order.order_id

    def cancel(self, order_id: int) -> None:
        """Cancel request — also subject to latency (fills can race the cancel)."""
        self._push(self.t + self._latency(), "cancel", order_id)

    def fills_for(self, owner: str, start_index: int) -> Tuple[List[Trade], int]:
        """Trades involving ``owner`` (as maker or taker) since ``start_index``."""
        new = [tr for tr in self.book.trades[start_index:]
               if owner in (tr.taker_owner, tr.maker_owner)]
        return new, len(self.book.trades)

    def step(self, dt: float) -> List[Trade]:
        """Advance the clock by ``dt``, processing every event inside the window."""
        t_end = self.t + dt
        self._gen_background(self.t, t_end)
        start = len(self.book.trades)
        while self._heap and self._heap[0][0] <= t_end:
            t, _, kind, payload = heapq.heappop(self._heap)
            self.t = t
            if kind == "order":
                self.book.process(payload, t)              # type: ignore[arg-type]
            elif kind == "cancel":
                self.book.cancel(int(payload))             # type: ignore[arg-type]
            elif kind == "zi_limit":
                self._zi_limit(payload, t)                 # type: ignore[arg-type]
            elif kind == "zi_refill":
                self._zi_limit(payload, t, refill=True)    # type: ignore[arg-type]
            elif kind == "zi_market":
                self._zi_market(payload, t)                # type: ignore[arg-type]
            elif kind == "zi_cancel":
                self._zi_cancel()
        self.t = t_end
        return self.book.trades[start:]

    # --------------------------------------------------- background market
    def _gen_background(self, t0: float, t1: float) -> None:
        dt = t1 - t0
        cfg = self.cfg
        for side in (Side.BUY, Side.SELL):
            for _ in range(int(self.rng.poisson(cfg.limit_rate * dt))):
                self._push(float(self.rng.uniform(t0, t1)), "zi_limit", side)
            for _ in range(int(self.rng.poisson(cfg.market_rate * dt))):
                self._push(float(self.rng.uniform(t0, t1)), "zi_market", side)
        # Per-order cancel hazard -> book size is mean-reverting (stationary depth).
        for _ in range(int(self.rng.poisson(cfg.cancel_rate * len(self.book.orders) * dt))):
            self._push(float(self.rng.uniform(t0, t1)), "zi_cancel", None)
        # Resilience: refill intensity grows with the deficit of the top levels,
        # so depth recovers gradually after a liquidity shock (book "toparlanması").
        bids, asks = self.book.depth(cfg.resilience_levels)
        full = cfg.resilience_levels * cfg.target_level_vol
        for side, levels in ((Side.BUY, bids), (Side.SELL, asks)):
            deficit = sum(max(0, cfg.target_level_vol - v) for _, v in levels)
            deficit += max(0, cfg.resilience_levels - len(levels)) * cfg.target_level_vol
            lam = cfg.resilience * (deficit / full) * cfg.resilience_levels * 10.0 * dt
            for _ in range(int(self.rng.poisson(lam))):
                self._push(float(self.rng.uniform(t0, t1)), "zi_refill", side)

    def _zi_limit(self, side: Side, now: float, refill: bool = False) -> None:
        bb, ba = self.book.best_bid(), self.book.best_ask()
        anchor = int(round(self.book.mid()))
        if refill:
            offset = 1 + int(self.rng.integers(0, self.cfg.resilience_levels))
        else:
            offset = int(self.rng.geometric(self.cfg.offset_p))
        if side is Side.BUY:
            ref = ba if ba is not None else anchor + 1
            price = ref - offset           # never crosses: at most 1 tick inside spread
        else:
            ref = bb if bb is not None else anchor - 1
            price = ref + offset
        mean = self.cfg.target_level_vol / 3 if refill else self.cfg.limit_qty_mean
        o = Order(self._next_oid(), side, self._qty(mean), OrderType.LIMIT,
                  self.ZI, now, price=max(1, price))
        self.book.process(o, now)

    def _zi_market(self, side: Side, now: float) -> None:
        o = Order(self._next_oid(), side, self._qty(self.cfg.market_qty_mean),
                  OrderType.MARKET, self.ZI, now)
        self.book.process(o, now)

    def _zi_cancel(self) -> None:
        draw = int(self.rng.integers(0, 2**31 - 1))
        zi_ids = [oid for oid, o in self.book.orders.items()
                  if o.owner == self.ZI and o.active]
        if zi_ids:
            self.book.cancel(zi_ids[draw % len(zi_ids)])


if __name__ == "__main__":
    # Quick smoke test: 10 simulated seconds of pure background flow.
    sim = ExchangeSimulator(SimConfig())
    for _ in range(100):
        sim.step(0.1)
    bids, asks = sim.book.depth(5)
    ts = sim.cfg.tick_size
    print(f"t={sim.t:.1f}s  trades={len(sim.book.trades)}  "
          f"mid={sim.book.mid() * ts:.2f}  spread={sim.book.spread()} ticks")
    print("bids:", [(round(p * ts, 2), v) for p, v in bids])
    print("asks:", [(round(p * ts, 2), v) for p, v in asks])

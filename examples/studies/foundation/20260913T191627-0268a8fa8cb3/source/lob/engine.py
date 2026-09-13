"""Deterministic integer-tick FIFO exchange with explicit order lifecycle.

Persistent event clocks make background flow independent of observation step size.
Shared seeds do not imply identical realized markets under different agent feedback.
"""
from __future__ import annotations

import heapq
import math
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


def _integer(value: object, name: str, minimum: Optional[int] = None) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _number(value: object, name: str, *, positive: bool = False) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


class Side(Enum):
    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTD = "GTD"


class OrderStatus(Enum):
    CREATED = "CREATED"
    SENT = "SENT"
    IN_FLIGHT = "IN_FLIGHT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESTING = "RESTING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


_TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
_TRANSITIONS = {
    OrderStatus.CREATED: {OrderStatus.SENT, OrderStatus.ACKNOWLEDGED, OrderStatus.REJECTED},
    OrderStatus.SENT: {OrderStatus.IN_FLIGHT, OrderStatus.REJECTED},
    OrderStatus.IN_FLIGHT: {OrderStatus.ACKNOWLEDGED, OrderStatus.CANCEL_PENDING, OrderStatus.REJECTED},
    OrderStatus.ACKNOWLEDGED: {OrderStatus.RESTING, OrderStatus.PARTIALLY_FILLED, *_TERMINAL},
    OrderStatus.RESTING: {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED,
                         OrderStatus.CANCEL_PENDING, OrderStatus.FILLED, OrderStatus.CANCELLED,
                         OrderStatus.EXPIRED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.ACKNOWLEDGED, OrderStatus.RESTING,
                                 OrderStatus.CANCEL_PENDING, OrderStatus.FILLED,
                                 OrderStatus.CANCELLED, OrderStatus.EXPIRED},
    OrderStatus.CANCEL_PENDING: {OrderStatus.ACKNOWLEDGED, OrderStatus.IN_FLIGHT,
                               OrderStatus.RESTING, OrderStatus.PARTIALLY_FILLED, *_TERMINAL},
}


@dataclass
class Order:
    order_id: int
    side: Side
    qty: int
    otype: OrderType
    owner: str
    ts_submit: float
    price: Optional[int] = None
    ts_active: float = 0.0
    remaining: int = 0
    active: bool = True
    time_in_force: TimeInForce = TimeInForce.GTC
    post_only: bool = False
    expires_at: Optional[float] = None
    status: OrderStatus = field(default=OrderStatus.CREATED, init=False)
    status_history: List[Tuple[float, OrderStatus]] = field(default_factory=list, init=False)
    reject_reason: Optional[str] = field(default=None, init=False)
    queue_seq: int = field(default=0, init=False)
    _filled_qty: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.validate()
        self.remaining = self.qty
        self.active = True
        if self.otype is OrderType.MARKET and self.time_in_force is TimeInForce.GTC:
            self.time_in_force = TimeInForce.IOC
        self.status_history.append((self.ts_submit, self.status))

    def validate(self, lot_size: int = 1) -> None:
        _integer(self.order_id, "order_id")
        _integer(self.qty, "qty", 1)
        if self.qty % lot_size:
            raise ValueError("qty must be a multiple of lot_size")
        if not isinstance(self.side, Side) or not isinstance(self.otype, OrderType):
            raise ValueError("side and otype must be valid enums")
        if not isinstance(self.time_in_force, TimeInForce):
            raise ValueError("time_in_force must be a TimeInForce enum")
        if not isinstance(self.owner, str) or not self.owner.strip():
            raise ValueError("owner must be a nonempty string")
        _number(self.ts_submit, "ts_submit")
        _number(self.ts_active, "ts_active")
        if self.otype is OrderType.LIMIT:
            _integer(self.price, "limit price", 1)
        elif self.price is not None:
            raise ValueError("market orders cannot specify a price")
        if not isinstance(self.post_only, bool):
            raise ValueError("post_only must be boolean")
        if self.post_only and (self.otype is not OrderType.LIMIT or
                               self.time_in_force in {TimeInForce.IOC, TimeInForce.FOK}):
            raise ValueError("post_only requires a GTC or GTD limit order")
        if self.time_in_force is TimeInForce.GTD:
            if self.otype is not OrderType.LIMIT or self.expires_at is None:
                raise ValueError("GTD requires a limit order and expires_at")
            _number(self.expires_at, "expires_at")
            if self.expires_at <= self.ts_submit:
                raise ValueError("expires_at must be after ts_submit")
        elif self.expires_at is not None:
            raise ValueError("expires_at requires GTD")

    @property
    def filled_qty(self) -> int:
        return self._filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def transition(self, status: OrderStatus, now: float) -> None:
        _number(now, "transition time")
        if self.status_history and now < self.status_history[-1][0]:
            raise ValueError("order transition time cannot go backwards")
        if status is self.status:
            return
        if status not in _TRANSITIONS.get(self.status, set()):
            raise ValueError(f"invalid order transition {self.status.value} -> {status.value}")
        self.status = status
        self.active = not self.is_terminal
        self.status_history.append((now, status))


@dataclass
class Trade:
    time: float
    price: int
    qty: int
    taker_side: Side
    taker_owner: str
    maker_owner: str
    taker_order_id: Optional[int] = None
    maker_order_id: Optional[int] = None


class OrderBook:
    """FIFO queues and aggregate depth; ``orders`` contains resting orders only.

    ``order_history`` retains all admissions. Direct callers supply event times;
    ExchangeSimulator owns a monotonic clock and schedules GTD expiry automatically.
    """

    def __init__(self, tick_size: float = 0.01, lot_size: int = 1,
                 check_invariants: bool = False) -> None:
        _number(tick_size, "tick_size", positive=True)
        _integer(lot_size, "lot_size", 1)
        self.tick_size, self.lot_size = tick_size, lot_size
        self.check_invariants = check_invariants
        self.bids: Dict[int, Deque[Order]] = {}
        self.asks: Dict[int, Deque[Order]] = {}
        self.bid_prices: List[int] = []
        self.ask_prices: List[int] = []
        self.bid_vol: Dict[int, int] = {}
        self.ask_vol: Dict[int, int] = {}
        self.orders: Dict[int, Order] = {}
        self.order_history: Dict[int, Order] = {}
        self.trades: List[Trade] = []
        self.last_mid = 0.0
        self._queue_seq = 0
        self._expirations: List[Tuple[float, int]] = []
        self.time = 0.0

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
        return ba - bb if bb is not None and ba is not None else None

    def microprice(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return self.mid()
        vb, va = self.bid_vol[bb], self.ask_vol[ba]
        return (bb * va + ba * vb) / (vb + va)

    def imbalance(self, levels: int = 5) -> float:
        bids, asks = self.depth(levels)
        vb, va = sum(v for _, v in bids), sum(v for _, v in asks)
        return (vb - va) / (vb + va) if vb + va else 0.0

    def depth(self, levels: int = 10) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        _integer(levels, "levels", 0)
        if levels == 0:
            return [], []
        return ([(p, self.bid_vol[p]) for p in reversed(self.bid_prices[-levels:])],
                [(p, self.ask_vol[p]) for p in self.ask_prices[:levels]])

    def walk_cost(self, side: Side, qty: int) -> Tuple[float, int]:
        """Hypothetical crossing VWAP in ticks and available fill quantity."""
        if not isinstance(side, Side):
            raise ValueError("side must be a Side enum")
        _integer(qty, "qty", 0)
        if side is Side.SELL:
            ladder, vols = reversed(self.bid_prices), self.bid_vol
        else:
            ladder, vols = iter(self.ask_prices), self.ask_vol
        remaining, cash = qty, 0
        for p in ladder:
            take = min(remaining, vols[p])
            cash += take * p
            remaining -= take
            if remaining == 0:
                break
        filled = qty - remaining
        return (cash / filled if filled else self.mid()), filled

    def queue_position(self, order_id: int) -> Optional[Tuple[int, int]]:
        """Return (orders ahead, units ahead); None when the order is not resting."""
        order = self.orders.get(order_id)
        if order is None:
            return None
        units = 0
        for index, queued in enumerate(self._levels(order.side)[0][order.price]):
            if queued is order:
                return index, units
            units += queued.remaining
        raise AssertionError("order index disagrees with its queue")

    def snapshot(self, levels: Optional[int] = None) -> dict:
        bids, asks = self.depth(levels if levels is not None else max(len(self.bids), len(self.asks)))
        return {"tick_size": self.tick_size, "lot_size": self.lot_size,
                "bids": bids, "asks": asks, "mid": self.mid(), "spread": self.spread()}

    def _levels(self, side: Side) -> Tuple[Dict[int, Deque[Order]], List[int], Dict[int, int]]:
        return (self.bids, self.bid_prices, self.bid_vol) if side is Side.BUY else (
            self.asks, self.ask_prices, self.ask_vol)

    def _check_time(self, now: Optional[float]) -> float:
        value = self.time if now is None else now
        _number(value, "exchange time")
        if value < self.time:
            raise ValueError("book event time cannot go backwards")
        return float(value)

    def _remove_level(self, side: Side, price: int) -> None:
        book, prices, vols = self._levels(side)
        del book[price]
        del vols[price]
        prices.pop(bisect_left(prices, price))

    def _add_resting(self, order: Order) -> None:
        book, prices, vols = self._levels(order.side)
        price = order.price
        if price not in book:
            book[price] = deque()
            insort(prices, price)
            vols[price] = 0
        self._queue_seq += 1
        order.queue_seq = self._queue_seq
        book[price].append(order)
        vols[price] += order.remaining
        self.orders[order.order_id] = order
        if order.expires_at is not None:
            heapq.heappush(self._expirations, (order.expires_at, order.order_id))

    def _detach(self, order: Order) -> None:
        book, _, vols = self._levels(order.side)
        book[order.price].remove(order)
        vols[order.price] -= order.remaining
        if not book[order.price]:
            self._remove_level(order.side, order.price)
        del self.orders[order.order_id]

    def expire(self, now: float) -> None:
        now = self._check_time(now)
        while self._expirations and self._expirations[0][0] <= now:
            expiry, oid = heapq.heappop(self._expirations)
            order = self.orders.get(oid)
            if order is not None and order.expires_at == expiry:
                self._detach(order)
                order.transition(OrderStatus.EXPIRED, max(self.time, expiry))
        self.time = now

    def _crosses(self, order: Order, price: int) -> bool:
        return order.otype is OrderType.MARKET or (
            price <= order.price if order.side is Side.BUY else price >= order.price)

    def _fillable(self, order: Order) -> int:
        _, ladder, volumes = self._levels(order.side.opposite)
        ordered = ladder if order.side is Side.BUY else reversed(ladder)
        total = 0
        for price in ordered:
            if not self._crosses(order, price):
                break
            total += volumes[price]
            if total >= order.remaining:
                break
        return total

    def _match(self, taker: Order, now: float) -> List[Trade]:
        trades = []
        book, prices, volumes = self._levels(taker.side.opposite)
        while taker.remaining and prices:
            best = prices[0] if taker.side is Side.BUY else prices[-1]
            if not self._crosses(taker, best):
                break
            queue = book[best]
            while queue and taker.remaining:
                maker = queue[0]
                quantity = min(maker.remaining, taker.remaining)
                maker.remaining -= quantity
                taker.remaining -= quantity
                maker._filled_qty += quantity
                taker._filled_qty += quantity
                volumes[best] -= quantity
                trades.append(Trade(now, best, quantity, taker.side, taker.owner, maker.owner,
                                    taker.order_id, maker.order_id))
                if not maker.remaining:
                    maker.transition(OrderStatus.FILLED, now)
                    queue.popleft()
                    self.orders.pop(maker.order_id)
                elif maker.status is not OrderStatus.CANCEL_PENDING:
                    maker.transition(OrderStatus.PARTIALLY_FILLED, now)
            if not queue:
                self._remove_level(taker.side.opposite, best)
        self.trades.extend(trades)
        if trades:
            self.last_mid = float(trades[-1].price)
        return trades

    def _route(self, order: Order, now: float) -> List[Trade]:
        if order.expires_at is not None and order.expires_at <= now:
            order.transition(OrderStatus.EXPIRED, now)
            return []
        fillable = self._fillable(order)
        if order.post_only and fillable:
            order.reject_reason = "post_only_would_cross"
            order.transition(OrderStatus.REJECTED, now)
            return []
        if order.time_in_force is TimeInForce.FOK and fillable < order.remaining:
            order.reject_reason = "fok_insufficient_depth"
            order.transition(OrderStatus.CANCELLED, now)
            return []
        trades = self._match(order, now)
        if not order.remaining:
            order.transition(OrderStatus.FILLED, now)
        elif order.otype is OrderType.MARKET or order.time_in_force in {TimeInForce.IOC, TimeInForce.FOK}:
            if order.filled_qty:
                order.transition(OrderStatus.PARTIALLY_FILLED, now)
            order.transition(OrderStatus.CANCELLED, now)
        else:
            order.transition(OrderStatus.PARTIALLY_FILLED if order.filled_qty else OrderStatus.RESTING, now)
            self._add_resting(order)
        return trades

    def process(self, order: Order, now: float) -> List[Trade]:
        now = self._check_time(now)
        order.validate(self.lot_size)
        if order.ts_submit > now:
            raise ValueError("order cannot arrive before submission")
        if order.order_id in self.order_history:
            raise ValueError("order_id has already been admitted")
        if order.status not in {OrderStatus.CREATED, OrderStatus.IN_FLIGHT, OrderStatus.CANCEL_PENDING}:
            raise ValueError("only newly arriving orders can be admitted")
        self.expire(now)
        order.ts_active = now
        order.transition(OrderStatus.ACKNOWLEDGED, now)
        self.order_history[order.order_id] = order
        trades = self._route(order, now)
        if self.check_invariants:
            self.assert_invariants()
        return trades

    def cancel(self, order_id: int, now: Optional[float] = None) -> bool:
        now = self._check_time(now)
        _integer(order_id, "order_id")
        self.expire(now)
        order = self.orders.get(order_id)
        if order is None:
            return False
        self._detach(order)
        order.transition(OrderStatus.CANCELLED, now)
        if self.check_invariants:
            self.assert_invariants()
        return True

    def modify(self, order_id: int, qty: int, price: Optional[int] = None,
               now: Optional[float] = None) -> bool:
        """Set total lifetime quantity (including fills); reduce at same price keeps FIFO.

        Increases and reprices lose queue priority. Quantity equal to already filled
        cancels leaves. Validation occurs before any mutation. Unknown IDs return False.
        """
        now = self._check_time(now)
        _integer(qty, "modified total qty", 0)
        if qty % self.lot_size:
            raise ValueError("modified qty must align to lot_size")
        if price is not None:
            _integer(price, "modified price", 1)
        order = self.orders.get(order_id)
        if order is None:
            return False
        if qty < order.filled_qty:
            raise ValueError("modified total qty cannot be smaller than already filled qty")
        new_price = order.price if price is None else price
        if order.post_only:
            touch = self.best_ask() if order.side is Side.BUY else self.best_bid()
            if touch is not None and (new_price >= touch if order.side is Side.BUY else new_price <= touch):
                raise ValueError("post-only modification would cross")
        self.expire(now)
        if order_id not in self.orders:
            return False
        if qty == order.filled_qty:
            return self.cancel(order_id, now)
        remaining = qty - order.filled_qty
        if new_price == order.price and remaining <= order.remaining:
            self._levels(order.side)[2][order.price] -= order.remaining - remaining
            order.qty, order.remaining = qty, remaining
        else:
            self._detach(order)
            order.qty, order.remaining, order.price = qty, remaining, new_price
            order.ts_active = now
            order.transition(OrderStatus.ACKNOWLEDGED, now)
            self._route(order, now)
        if self.check_invariants:
            self.assert_invariants()
        return True

    def assert_invariants(self) -> None:
        bb, ba = self.best_bid(), self.best_ask()
        assert bb is None or ba is None or bb < ba, "crossed or locked book"
        live = {}
        for side in Side:
            levels, prices, volumes = self._levels(side)
            assert prices == sorted(set(prices)), "price ladder not strictly sorted"
            assert set(prices) == set(levels) == set(volumes), "level indices disagree"
            for price, queue in levels.items():
                assert isinstance(price, Integral) and price > 0 and queue
                assert volumes[price] == sum(order.remaining for order in queue)
                sequences = [order.queue_seq for order in queue]
                assert sequences == sorted(set(sequences)), "FIFO queue order invalid"
                for order in queue:
                    assert order.order_id not in live
                    assert order.side is side and order.price == price
                    assert order.active and not order.is_terminal and order.remaining > 0
                    assert order.remaining % self.lot_size == 0
                    assert order.remaining + order.filled_qty == order.qty
                    live[order.order_id] = order
        assert live == self.orders, "order index disagrees with queues"


@dataclass(frozen=True)
class SimConfig:
    tick_size: float = 0.01
    initial_mid_ticks: int = 10_000
    seed: int = 42
    limit_rate: float = 25.0
    market_rate: float = 6.0
    cancel_rate: float = 0.15
    offset_p: float = 0.25
    limit_qty_mean: float = 35.0
    market_qty_mean: float = 35.0
    target_level_vol: int = 300
    resilience: float = 0.8
    resilience_levels: int = 5
    latency_base: float = 0.005
    latency_jitter: float = 0.005
    lot_size: int = 1
    market_seed: Optional[int] = None
    order_flow_seed: Optional[int] = None
    latency_seed: Optional[int] = None
    cancellation_seed: Optional[int] = None
    resilience_seed: Optional[int] = None
    check_invariants: bool = False
    record_events: bool = True
    max_events: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("seed", "market_seed", "order_flow_seed", "latency_seed", "cancellation_seed", "resilience_seed"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name, 0)
        for name in ("initial_mid_ticks", "lot_size", "target_level_vol", "resilience_levels", "max_events"):
            _integer(getattr(self, name), name, 1)
        if self.initial_mid_ticks <= 20 or self.target_level_vol % self.lot_size:
            raise ValueError("initial_mid_ticks must exceed 20 and target volume must align to lot_size")
        for name in ("tick_size", "limit_qty_mean", "market_qty_mean", "offset_p"):
            _number(getattr(self, name), name, positive=True)
        if self.offset_p > 1:
            raise ValueError("offset_p must be in (0, 1]")
        for name in ("limit_rate", "market_rate", "cancel_rate", "resilience", "latency_base", "latency_jitter"):
            _number(getattr(self, name), name)
        if not isinstance(self.check_invariants, bool) or not isinstance(self.record_events, bool):
            raise ValueError("check_invariants and record_events must be booleans")


class ExchangeSimulator:
    """Persistent event clocks, FIFO exchange, constant plus exponential latency.

    Same-time events are processed by enqueue sequence; GTD expiry is effective
    before matching at its timestamp. Background IDs are negative and strategy IDs
    positive. Cancellation lifetimes apply only to background orders. Resilience
    uses homogeneous proposals with state-dependent acceptance, so its intensity
    reacts to the book without depending on caller step partitions.
    """
    ZI = "ZI"

    def __init__(self, cfg: SimConfig) -> None:
        if not isinstance(cfg, SimConfig):
            raise TypeError("cfg must be SimConfig")
        self.cfg = cfg
        self.book = OrderBook(cfg.tick_size, cfg.lot_size, cfg.check_invariants)
        seeds = np.random.SeedSequence(cfg.seed).spawn(5)
        names = ("market_seed", "order_flow_seed", "latency_seed", "cancellation_seed", "resilience_seed")
        streams = [np.random.default_rng(getattr(cfg, name) if getattr(cfg, name) is not None else child)
                   for name, child in zip(names, seeds)]
        self.rng, self.rng_flow, self.rng_lat, self.rng_cancel, self.rng_resilience = streams
        self.seed_manifest = {"global_seed": cfg.seed, **{name: getattr(cfg, name) for name in names},
                              "derived_spawn_keys": {name: list(child.spawn_key) for name, child in zip(names, seeds)}}
        self.t = 0.0
        self._heap: list[tuple[float, int, str, object]] = []
        self._seq = self._oid = self._background_oid = 0
        self.orders: dict[int, Order] = {}
        self.events: list[dict] = []
        self.event_count = 0
        self._cancel_pending: set[int] = set()
        self.book.last_mid = float(cfg.initial_mid_ticks)
        self._seed_book()
        for side in Side:
            self._next_arrival("zi_limit", side, cfg.limit_rate)
            self._next_arrival("zi_market", side, cfg.market_rate)
            self._next_arrival("zi_refill", side, cfg.resilience * cfg.resilience_levels * 10)

    def _push(self, t: float, kind: str, payload: object) -> None:
        _number(t, "event time")
        if t < self.t:
            raise ValueError("cannot schedule into the past")
        self._seq += 1
        heapq.heappush(self._heap, (float(t), self._seq, kind, payload))

    def _next_oid(self, background: bool = False) -> int:
        if background:
            self._background_oid -= 1
            return self._background_oid
        self._oid += 1
        return self._oid

    def _latency(self) -> float:
        return self.cfg.latency_base + float(self.rng_lat.exponential(self.cfg.latency_jitter))

    def _next_arrival(self, kind: str, side: Side, rate: float) -> None:
        if rate:
            rng = self.rng_resilience if kind == "zi_refill" else self.rng_flow
            delay = float(rng.exponential(1 / rate))
            at = self.t + delay
            if at == self.t:
                at = math.nextafter(self.t, math.inf)
            self._push(at, kind, side)

    def _qty(self, mean: float, rng=None) -> int:
        draw = int((self.rng if rng is None else rng).exponential(mean))
        return max(self.cfg.lot_size, draw // self.cfg.lot_size * self.cfg.lot_size)

    def _admit_background(self, order: Order, now: float) -> None:
        self.orders[order.order_id] = order
        self.book.process(order, now)
        if order.order_id in self.book.orders and self.cfg.cancel_rate:
            self._push(now + float(self.rng_cancel.exponential(1 / self.cfg.cancel_rate)), "zi_cancel", order.order_id)

    def _seed_book(self) -> None:
        for offset in range(1, 21):
            for side in Side:
                order = Order(self._next_oid(True), side, self.cfg.target_level_vol, OrderType.LIMIT,
                              self.ZI, 0.0, price=self.cfg.initial_mid_ticks - side.value * offset)
                self._admit_background(order, 0.0)

    def submit(self, side: Side, qty: int, owner: str, price_ticks: Optional[int] = None, *,
               time_in_force: TimeInForce = TimeInForce.GTC, post_only: bool = False,
               expires_at: Optional[float] = None) -> int:
        if owner == self.ZI:
            raise ValueError("ZI is reserved for background orders")
        order = Order(self._oid + 1, side, qty, OrderType.LIMIT if price_ticks is not None else OrderType.MARKET,
                      owner, self.t, price=price_ticks, time_in_force=time_in_force,
                      post_only=post_only, expires_at=expires_at)
        order.validate(self.cfg.lot_size)
        self._next_oid()
        order.transition(OrderStatus.SENT, self.t)
        order.transition(OrderStatus.IN_FLIGHT, self.t)
        self.orders[order.order_id] = order
        self._push(self.t + self._latency(), "order", order)
        if expires_at is not None:
            self._push(expires_at, "expire", order.order_id)
        return order.order_id

    def cancel(self, order_id: int) -> bool:
        _integer(order_id, "order_id")
        order = self.orders.get(order_id)
        if order is None or order.is_terminal or order_id in self._cancel_pending:
            return False
        order.transition(OrderStatus.CANCEL_PENDING, self.t)
        self._cancel_pending.add(order_id)
        self._push(self.t + self._latency(), "cancel", order_id)
        return True

    def modify(self, order_id: int, qty: int, price_ticks: Optional[int] = None) -> bool:
        _integer(qty, "modified total qty", 0)
        if qty % self.cfg.lot_size:
            raise ValueError("modified qty must align to lot_size")
        if price_ticks is not None:
            _integer(price_ticks, "modified price", 1)
        order = self.orders.get(order_id)
        if order is None or order.is_terminal:
            return False
        self._push(self.t + self._latency(), "modify", (order_id, qty, price_ticks))
        return True

    def cancel_replace(self, order_id: int, qty: int, price_ticks: int) -> Optional[int]:
        """Atomic replacement on arrival, conditional on the original still resting.

        The new quantity is fresh leaves, not a new parent budget. Execution agents
        must continue to apply their own parent reservations when using this API.
        """
        old = self.orders.get(order_id)
        if old is None or old.is_terminal:
            return None
        replacement = Order(self._oid + 1, old.side, qty, OrderType.LIMIT, old.owner, self.t,
                            price=price_ticks, time_in_force=old.time_in_force,
                            post_only=old.post_only, expires_at=old.expires_at)
        replacement.validate(self.cfg.lot_size)
        self._next_oid()
        replacement.transition(OrderStatus.SENT, self.t)
        replacement.transition(OrderStatus.IN_FLIGHT, self.t)
        self.orders[replacement.order_id] = replacement
        self._push(self.t + self._latency(), "replace", (order_id, replacement))
        if replacement.expires_at is not None:
            self._push(replacement.expires_at, "expire", replacement.order_id)
        return replacement.order_id

    def fills_for(self, owner: str, start_index: int) -> Tuple[List[Trade], int]:
        _integer(start_index, "trade cursor", 0)
        if start_index > len(self.book.trades):
            raise ValueError("trade cursor exceeds tape length")
        return ([trade for trade in self.book.trades[start_index:]
                 if owner in (trade.maker_owner, trade.taker_owner)], len(self.book.trades))

    def _zi_limit(self, side: Side, now: float, refill: bool = False) -> None:
        rng = self.rng_resilience if refill else self.rng
        bb, ba = self.book.best_bid(), self.book.best_ask()
        anchor = int(round(self.book.mid()))
        reference = (ba if ba is not None else anchor + 1) if side is Side.BUY else (bb if bb is not None else anchor - 1)
        if refill:
            # Measure missing *price slots*, not just surviving levels: walking a
            # whole level must not make its deficit disappear from the model.
            volumes = self.book.bid_vol if side is Side.BUY else self.book.ask_vol
            deficit = sum(max(0, self.cfg.target_level_vol - volumes.get(reference - side.value * offset, 0))
                          for offset in range(1, self.cfg.resilience_levels + 1))
            probability = deficit / (self.cfg.resilience_levels * self.cfg.target_level_vol)
            if rng.random() >= probability:
                return
        offset = 1 + int(rng.integers(self.cfg.resilience_levels)) if refill else int(rng.geometric(self.cfg.offset_p))
        price = max(1, reference - side.value * offset)
        mean = self.cfg.target_level_vol / 3 if refill else self.cfg.limit_qty_mean
        order = Order(self._next_oid(True), side, self._qty(mean, rng), OrderType.LIMIT, self.ZI, now, price=price)
        self._admit_background(order, now)

    def _zi_market(self, side: Side, now: float) -> None:
        self._admit_background(Order(self._next_oid(True), side, self._qty(self.cfg.market_qty_mean),
                                     OrderType.MARKET, self.ZI, now), now)

    def step(self, dt: float) -> List[Trade]:
        _number(dt, "dt")
        end = self.t + dt
        if not math.isfinite(end):
            raise ValueError("clock overflow")
        cursor = len(self.book.trades)
        while self._heap and self._heap[0][0] <= end:
            if self.event_count >= self.cfg.max_events:
                raise RuntimeError(f"exchange max_events={self.cfg.max_events} reached")
            now, sequence, kind, payload = heapq.heappop(self._heap)
            self.t = now
            self.book.expire(now)
            accepted = True
            reason = None
            if kind == "order":
                order = payload
                if not order.is_terminal:
                    self.book.process(order, now)
                    if order.order_id in self._cancel_pending and not order.is_terminal:
                        order.transition(OrderStatus.CANCEL_PENDING, now)
            elif kind in {"cancel", "zi_cancel"}:
                accepted = self.book.cancel(payload, now)
                if kind == "cancel":
                    self._cancel_pending.discard(payload)
                    order = self.orders.get(payload)
                    if order is not None and order.status is OrderStatus.CANCEL_PENDING and not accepted:
                        order.transition(OrderStatus.IN_FLIGHT, now)
                if not accepted:
                    reason = "order_not_resting"
            elif kind == "modify":
                oid, qty, price = payload
                try:
                    accepted = self.book.modify(oid, qty, price, now)
                except ValueError as exc:
                    accepted, reason = False, str(exc)
                if not accepted and reason is None:
                    reason = "order_not_resting"
            elif kind == "replace":
                oid, replacement = payload
                if self.book.cancel(oid, now):
                    self.book.process(replacement, now)
                else:
                    replacement.reject_reason = "replace_original_not_resting"
                    replacement.transition(OrderStatus.REJECTED, now)
                    accepted, reason = False, replacement.reject_reason
            elif kind == "expire":
                # Book expiry was applied before dispatch. In-flight GTD remains
                # in-flight until arrival, where it is acknowledged then expired.
                pass
            elif kind == "zi_limit":
                self._zi_limit(payload, now)
                self._next_arrival(kind, payload, self.cfg.limit_rate)
            elif kind == "zi_market":
                self._zi_market(payload, now)
                self._next_arrival(kind, payload, self.cfg.market_rate)
            elif kind == "zi_refill":
                self._zi_limit(payload, now, refill=True)
                self._next_arrival(kind, payload, self.cfg.resilience * self.cfg.resilience_levels * 10)
            else:
                raise RuntimeError(f"unknown event kind {kind}")
            self.event_count += 1
            if self.cfg.record_events:
                self.events.append({"time": now, "sequence": sequence, "kind": kind,
                                    "accepted": accepted, "reason": reason})
            if self.cfg.check_invariants:
                self.book.assert_invariants()
        self.t = end
        self.book.expire(end)
        return self.book.trades[cursor:]

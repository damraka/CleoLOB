"""Event-driven reference engine for v0.7 generators (workstreams 6, 46).

``BundleSimulator`` subclasses the frozen v0.5 ``ExchangeSimulator`` with every
zero-intelligence clock switched off. Every ``BIN_S`` seconds a pluggable
``CountModel`` receives the *current* book state and the realized event counts
of the previous bin (aggressive events are read from the engine's own trade
tape, so a strategy's market orders count) and returns the number of events of
each mark for the next bin. Events are placed uniformly inside the bin and
realized against the live book when they fire:

* market buy/sell — a background market order of an empirical size;
* add bid/ask — a background limit order at an empirical tick offset from the
  same-side touch (negative offsets improve inside the spread, never crossing);
* cancel bid/ask — an empirical *fraction* of the background volume at the level
  nearest an empirical offset is removed (queue position chosen by
  ``cancel_position``), so level sizes decay proportionally as in the data.

Because events act on the live book through the unchanged matching engine, the
book always stays valid, strategy orders interact with generated flow, and
flow responds to the state a strategy creates (endogenous response through the
state features). This is a hypothesis about aggregate flow, never a claim about
the causal behaviour of real participants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import math
from typing import Any, List, Protocol

import numpy as np

from ...engine import ExchangeSimulator, Order, OrderStatus, OrderType, Side, SimConfig, Trade, _number
from .events import MARKS, book_state, spread_bucket

BIN_S = 0.1
MAX_PER_MARK = 40
CANCEL_POSITIONS = ("uniform", "front", "back")


class CountModel(Protocol):
    family: str

    def initial(self) -> Any: ...

    def sample(self, mstate: Any, features: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...

    def update(self, mstate: Any, realized: np.ndarray, features: np.ndarray) -> Any: ...


@dataclass
class EmpiricalTables:
    """Development-only samples: market sizes (lots); per add/cancel mark and spread bucket, joint
    (tick offset, size in lots or cancelled fraction) pairs."""

    sizes: dict
    offsets: dict

    def size(self, mark: str, rng: np.random.Generator) -> int:
        values = self.sizes[mark]
        return max(1, int(round(float(values[rng.integers(len(values))])))) if len(values) else 1

    def pair(self, mark: str, rng: np.random.Generator, spread_ticks: float) -> tuple[int, float]:
        """Joint (tick offset, size or cancel fraction) draw, conditional on the spread bucket."""
        rows = self.offsets.get(f"{mark}@{spread_bucket(spread_ticks)}")
        if rows is None or not len(rows):
            rows = np.vstack([v for k, v in self.offsets.items() if k.startswith(mark) and len(v)] or [[[0, 1.0]]])
        offset, value = rows[rng.integers(len(rows))]
        return int(offset), float(value)


class BundleSimulator(ExchangeSimulator):
    def __init__(self, cfg: SimConfig, model: CountModel, tables: EmpiricalTables, *,
                 cancel_position: str = "uniform", generator_seed: int = 0) -> None:
        if cancel_position not in CANCEL_POSITIONS:
            raise ValueError(f"cancel_position must be one of {CANCEL_POSITIONS}")
        if cfg.limit_rate or cfg.market_rate or cfg.cancel_rate or cfg.resilience:
            raise ValueError("BundleSimulator requires every zero-intelligence rate to be zero")
        super().__init__(cfg)
        self.model, self.tables, self.cancel_position = model, tables, cancel_position
        self.rng_gen = np.random.default_rng(np.random.SeedSequence([cfg.seed, 7_707_001, generator_seed]))
        self.mstate = model.initial()
        self.generated = np.zeros(len(MARKS), dtype=np.int64)
        self.realized_adds_cancels = np.zeros(4, dtype=np.int64)
        self.diagnostics = {"bins": 0, "cap_hits": 0, "cancel_skipped": 0, "events": np.zeros(len(MARKS), int)}
        self._trade_cursor = 0
        self._bin_generated = np.zeros(len(MARKS), dtype=np.int64)
        self._push(BIN_S, "bundle", None)

    # ------------------------------------------------------------------ state
    def features(self) -> np.ndarray:
        bids, asks = self.book.depth(5)
        if not bids or not asks:
            return np.asarray([10.0, 0.0, 0.0])
        bq = np.asarray([q for _, q in bids], float)
        aq = np.asarray([q for _, q in asks], float)
        return np.asarray(book_state(bids[0][0], asks[0][0], bq, aq, 1.0))

    def _realized(self) -> np.ndarray:
        trades = self.book.trades[self._trade_cursor:]
        self._trade_cursor = len(self.book.trades)
        buys = {t.taker_order_id for t in trades if t.taker_side is Side.BUY}
        sells = {t.taker_order_id for t in trades if t.taker_side is Side.SELL}
        realized = self._bin_generated.copy()
        realized[0], realized[1] = len(buys), len(sells)
        return realized

    def _bundle(self, now: float) -> None:
        realized = self._realized()
        features = self.features()
        self.mstate = self.model.update(self.mstate, realized, features)
        counts = np.minimum(np.asarray(self.model.sample(self.mstate, features, self.rng_gen), dtype=np.int64),
                            MAX_PER_MARK)
        self.diagnostics["cap_hits"] += int(np.sum(counts >= MAX_PER_MARK))
        self.diagnostics["bins"] += 1
        self._bin_generated = counts.copy()
        for mark, count in enumerate(counts):
            for _ in range(int(count)):
                at = now + float(self.rng_gen.uniform(0.0, BIN_S))
                self._push(max(at, math.nextafter(now, math.inf)), "gen", mark)
        self._push(now + BIN_S, "bundle", None)

    # ------------------------------------------------------------------ realization
    def _gen(self, mark: int, now: float) -> None:
        name = MARKS[mark]
        rng = self.rng_gen
        self.diagnostics["events"][mark] += 1
        if name.startswith("market"):
            side = Side.BUY if name == "market_buy" else Side.SELL
            order = Order(self._next_oid(True), side, self.tables.size(name, rng), OrderType.MARKET, self.ZI, now)
            self._admit_background(order, now)
            return
        bid = name.endswith("bid")
        side = Side.BUY if bid else Side.SELL
        bb, ba = self.book.best_bid(), self.book.best_ask()
        offset, value = self.tables.pair(name, rng, (ba - bb) if bb is not None and ba is not None else 10.0)
        if name.startswith("add"):
            anchor = int(round(self.book.mid()))
            touch = (bb if bb is not None else (ba - 1 if ba is not None else anchor - 1)) if bid else \
                (ba if ba is not None else (bb + 1 if bb is not None else anchor + 1))
            price = touch - offset if bid else touch + offset
            if bid and ba is not None:
                price = min(price, ba - 1)
            if not bid and bb is not None:
                price = max(price, bb + 1)
            order = Order(self._next_oid(True), side, max(1, int(round(value))), OrderType.LIMIT, self.ZI, now,
                          price=max(1, price))
            self._admit_background(order, now)
            return
        prices = self.book.bid_prices if bid else self.book.ask_prices
        if not prices:
            self.diagnostics["cancel_skipped"] += 1
            return
        touch = prices[-1] if bid else prices[0]
        target = touch - offset if bid else touch + offset
        level = min(prices, key=lambda p: (abs(p - target), p))
        queue = [o for o in (self.book.bids if bid else self.book.asks)[level] if o.owner == self.ZI]
        if not queue:
            self.diagnostics["cancel_skipped"] += 1
            return
        if self.cancel_position == "uniform":
            queue = [queue[i] for i in rng.permutation(len(queue))]
        elif self.cancel_position == "back":
            queue = queue[::-1]
        remaining = max(1, int(round(min(1.0, max(0.0, value)) * sum(o.remaining for o in queue))))
        for order in queue:
            if remaining <= 0:
                break
            if order.remaining <= remaining:
                remaining -= order.remaining
                self.book.cancel(order.order_id, now)
            else:
                self.book.modify(order.order_id, order.filled_qty + order.remaining - remaining, None, now)
                remaining = 0

    # ------------------------------------------------------------------ event loop
    def step(self, dt: float) -> List[Trade]:
        """The base event loop, reproduced exactly, plus ``bundle`` and ``gen`` events."""
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
            accepted, reason = True, None
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
                pass
            elif kind == "bundle":
                self._bundle(now)
            elif kind == "gen":
                self._gen(payload, now)
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


def bundle_config(base: dict, *, target_level_vol: int, max_events: int) -> SimConfig:
    """A ``SimConfig`` with the v0.6 tick geometry and every zero-intelligence clock off."""
    fields = {**base, "limit_rate": 0.0, "market_rate": 0.0, "cancel_rate": 0.0, "resilience": 0.0,
              "target_level_vol": int(target_level_vol), "max_events": int(max_events), "record_events": False,
              "check_invariants": False}
    return SimConfig(**{k: v for k, v in fields.items() if k in SimConfig.__dataclass_fields__})


@dataclass(frozen=True)
class GeneratorSpec:
    """Picklable world: family name, ``SimConfig`` fields, a count model and empirical tables."""

    family: str
    config: dict
    model: Any
    tables: Any
    cancel_position: str = "uniform"
    meta: dict = field(default_factory=dict)

    def build(self, seed: int) -> BundleSimulator:
        cfg = bundle_config({**self.config, "seed": seed}, target_level_vol=self.config["target_level_vol"],
                            max_events=self.config["max_events"])
        return BundleSimulator(cfg, self.model, self.tables, cancel_position=self.cancel_position)

    def capped(self, seconds: float, events_per_second: float) -> GeneratorSpec:
        return GeneratorSpec(self.family, {**self.config, "max_events": int(events_per_second * (seconds + 60))},
                             self.model, self.tables, self.cancel_position, self.meta)

    def describe(self) -> dict:
        return {"family": self.family, "config": self.config, "cancel_position": self.cancel_position,
                "model": self.model.describe() if hasattr(self.model, "describe") else type(self.model).__name__,
                "meta": self.meta}


def config_dict(cfg: SimConfig) -> dict:
    return asdict(cfg)

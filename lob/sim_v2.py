"""Calibration-v2 simulator family: bounded, opt-in extensions of the v0.4 ZI exchange.

``ExtendedSimulator`` subclasses ``ExchangeSimulator`` without changing the base
engine, so the v0.4 baseline stays bit-identical. Each extension has explicit,
bounded parameters and is exact under Poisson thinning or a pre-sampled path:

* ``empirical_sizes``: limit and market order sizes drawn by inverse-CDF
  interpolation from train-only quantile tables (lots).
* ``inside_spread_prob``: when the spread exceeds one tick, a limit arrival is
  placed strictly inside the spread with this probability (spread-conditioned).
* ``cancel_depth_exponent``: background cancellation hazard is multiplied by
  (level volume / target_level_vol)^gamma, capped at ``MAX_MULTIPLIER``.
* ``imbalance_beta``: a market arrival on side s is accepted with probability
  proportional to 1 + beta * (+/-) top-5 imbalance (bid-heavy books favour buys).
* ``regime_multiplier`` / ``regime_switch_rate`` / ``regime_high_share``: a
  two-state Markov activity regime scaling limit, market and cancel intensity.
  The path is pre-sampled from its own seed stream.
* ``hawkes_alpha`` / ``hawkes_decay``: self-exciting market orders (Ogata
  thinning); excitation is shared across sides.

State-dependent intensities use proposals at the maximum rate and acceptance
probabilities, so they react to the book without depending on step partitions.
These are hypotheses about flow, calibrated only on training data; they do not
identify causal behaviour of real participants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import math
from typing import List

import numpy as np

from .engine import ExchangeSimulator, Order, OrderStatus, OrderType, Side, SimConfig, Trade, _number

MAX_MULTIPLIER = 4.0
REGIME_HORIZON_SECONDS = 200_000.0


@dataclass(frozen=True)
class FlowExtensions:
    limit_size_quantiles: tuple[float, ...] = ()
    market_size_quantiles: tuple[float, ...] = ()
    inside_spread_prob: float = 0.0
    cancel_depth_exponent: float = 0.0
    imbalance_beta: float = 0.0
    regime_multiplier: float = 1.0
    regime_switch_rate: float = 0.0
    regime_high_share: float = 0.5
    hawkes_alpha: float = 0.0
    hawkes_decay: float = 1.0
    extension_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("limit_size_quantiles", "market_size_quantiles"):
            values = tuple(float(v) for v in getattr(self, name))
            object.__setattr__(self, name, values)
            if values and (len(values) < 2 or any(v <= 0 or not math.isfinite(v) for v in values)
                           or any(b < a for a, b in zip(values, values[1:]))):
                raise ValueError(f"{name} must be positive nondecreasing quantiles")
        checks = {"inside_spread_prob": (0, 1), "cancel_depth_exponent": (-3, 3), "imbalance_beta": (-1, 1),
                  "regime_multiplier": (1, 20), "regime_switch_rate": (0, 10), "regime_high_share": (0.01, 0.99),
                  "hawkes_alpha": (0, 50), "hawkes_decay": (0.01, 100)}
        for name, (low, high) in checks.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not low <= value <= high:
                raise ValueError(f"{name} must lie in [{low}, {high}]")
        if self.hawkes_alpha and self.hawkes_alpha >= self.hawkes_decay:
            raise ValueError("hawkes_alpha must be below hawkes_decay (stationary branching ratio < 1)")

    @property
    def active(self) -> list[str]:
        names = []
        if self.limit_size_quantiles or self.market_size_quantiles:
            names.append("empirical_sizes")
        if self.inside_spread_prob:
            names.append("spread_conditioned_arrivals")
        if self.cancel_depth_exponent:
            names.append("depth_conditioned_cancellation")
        if self.imbalance_beta:
            names.append("imbalance_conditioned_market_flow")
        if self.regime_switch_rate and self.regime_multiplier > 1:
            names.append("markov_activity_regime")
        if self.hawkes_alpha:
            names.append("self_exciting_market_orders")
        return names

    def to_dict(self) -> dict:
        return {**asdict(self), "active": self.active}


def _inverse_cdf(quantiles: tuple[float, ...], u: float) -> float:
    grid = np.linspace(0.0, 1.0, len(quantiles))
    return float(np.interp(u, grid, quantiles))


class ExtendedSimulator(ExchangeSimulator):
    """v0.4 exchange plus declared flow extensions; identical order lifecycle and matching."""

    def __init__(self, cfg: SimConfig, extensions: FlowExtensions | None = None) -> None:
        self.ext = extensions or FlowExtensions()
        self._rng_ext = np.random.default_rng(np.random.SeedSequence([cfg.seed, 7_700_001, self.ext.extension_seed]))
        self._regime_switches = self._sample_regime_path()
        # Precomputed once: the lookup is on every event (performance only; results identical).
        self._regime_times = np.asarray(self._regime_switches[1:], dtype=float)
        self._regime_initial_high = bool(self._regime_switches) and self._regime_switches[0] == 1.0
        self._hawkes_excitation = 0.0
        self._hawkes_time = 0.0
        super().__init__(cfg)
        if self.ext.hawkes_alpha:
            # Replace the homogeneous market clocks with one self-exciting clock.
            self._heap = [item for item in self._heap if item[2] != "zi_market"]
            heapq.heapify(self._heap)
            self._schedule_hawkes()

    # ------------------------------------------------------------------ regime
    def _sample_regime_path(self) -> list[float]:
        rate = self.ext.regime_switch_rate
        if not (rate and self.ext.regime_multiplier > 1):
            return []
        share = self.ext.regime_high_share
        # Exit rates chosen so the stationary high share equals ``share``; mean switch rate = rate.
        exit_high, exit_low = rate / (2 * share), rate / (2 * (1 - share))
        t, high, switches = 0.0, self._rng_ext.random() < share, []
        switches.append(1.0 if high else 0.0)  # initial state marker
        while t < REGIME_HORIZON_SECONDS and len(switches) < 1_000_000:
            t += float(self._rng_ext.exponential(1 / (exit_high if high else exit_low)))
            switches.append(t)
            high = not high
        return switches

    def regime_high(self, t: float) -> bool:
        if not self._regime_switches:
            return False
        changes = int(np.searchsorted(self._regime_times, t, side="right"))
        return self._regime_initial_high != bool(changes % 2)

    def _activity(self, t: float) -> float:
        return self.ext.regime_multiplier if self.regime_high(t) else 1.0

    def _max_activity(self) -> float:
        return self.ext.regime_multiplier if self._regime_switches else 1.0

    # ------------------------------------------------------------------ proposals
    def _next_arrival(self, kind: str, side: Side, rate: float) -> None:
        if kind == "zi_market" and self.ext.imbalance_beta:
            rate = rate * (1 + abs(self.ext.imbalance_beta))
        if kind in {"zi_limit", "zi_market"}:
            rate = rate * self._max_activity()
        super()._next_arrival(kind, side, rate)

    def _accept(self, probability: float) -> bool:
        return bool(self._rng_ext.random() < probability)

    def _imbalance(self) -> float:
        bids, asks = self.book.depth(5)
        b, a = sum(q for _, q in bids), sum(q for _, q in asks)
        return (b - a) / (b + a) if b + a else 0.0

    def _market_acceptance(self, side: Side, now: float) -> bool:
        probability = self._activity(now) / self._max_activity()
        if self.ext.imbalance_beta:
            sign = 1.0 if side is Side.BUY else -1.0
            probability *= max(0.0, 1 + self.ext.imbalance_beta * sign * self._imbalance()) / (
                1 + abs(self.ext.imbalance_beta))
        return self._accept(probability)

    # ------------------------------------------------------------------ order generation
    def _size(self, quantiles: tuple[float, ...], mean: float, rng) -> int:
        if quantiles:
            draw = _inverse_cdf(quantiles, float(self._rng_ext.random()))
            return max(self.cfg.lot_size, int(round(draw)) // self.cfg.lot_size * self.cfg.lot_size)
        return self._qty(mean, rng)

    def _zi_limit(self, side: Side, now: float, refill: bool = False) -> None:
        if refill:
            return super()._zi_limit(side, now, refill=True)
        if not self._accept(self._activity(now) / self._max_activity()):
            return
        bb, ba = self.book.best_bid(), self.book.best_ask()
        anchor = int(round(self.book.mid()))
        reference = (ba if ba is not None else anchor + 1) if side is Side.BUY else (bb if bb is not None else anchor - 1)
        price = None
        if self.ext.inside_spread_prob and bb is not None and ba is not None and ba - bb > 1:
            if self._accept(self.ext.inside_spread_prob):
                price = int(self._rng_ext.integers(bb + 1, ba))
        if price is None:
            offset = int(self.rng.geometric(self.cfg.offset_p))
            price = max(1, reference - side.value * offset)
        quantity = self._size(self.ext.limit_size_quantiles, self.cfg.limit_qty_mean, self.rng)
        order = Order(self._next_oid(True), side, quantity, OrderType.LIMIT, self.ZI, now, price=price)
        self._admit_background(order, now)

    def _zi_market(self, side: Side, now: float) -> None:
        if not self._market_acceptance(side, now):
            return
        quantity = self._size(self.ext.market_size_quantiles, self.cfg.market_qty_mean, self.rng)
        self._admit_background(Order(self._next_oid(True), side, quantity, OrderType.MARKET, self.ZI, now), now)
        if self.ext.hawkes_alpha:
            self._decay_hawkes(now)
            self._hawkes_excitation += self.ext.hawkes_alpha

    # ------------------------------------------------------------------ cancellation
    def _admit_background(self, order: Order, now: float) -> None:
        if not self.ext.cancel_depth_exponent:
            return super()._admit_background(order, now)
        self.orders[order.order_id] = order
        self.book.process(order, now)
        if order.order_id in self.book.orders and self.cfg.cancel_rate:
            self._schedule_cancel_proposal(order.order_id, now)

    def _schedule_cancel_proposal(self, order_id: int, now: float) -> None:
        rate = self.cfg.cancel_rate * MAX_MULTIPLIER * self._max_activity()
        self._push(now + float(self.rng_cancel.exponential(1 / rate)), "v2_cancel", order_id)

    def _cancel_multiplier(self, order_id: int, now: float) -> float:
        order = self.book.orders.get(order_id)
        if order is None:
            return 0.0
        volumes = self.book.bid_vol if order.side is Side.BUY else self.book.ask_vol
        level = max(1, volumes.get(order.price, 0))
        multiplier = min(MAX_MULTIPLIER, (level / self.cfg.target_level_vol) ** self.ext.cancel_depth_exponent)
        return multiplier * self._activity(now) / (MAX_MULTIPLIER * self._max_activity())

    # ------------------------------------------------------------------ hawkes
    def _decay_hawkes(self, now: float) -> None:
        self._hawkes_excitation *= math.exp(-self.ext.hawkes_decay * max(0.0, now - self._hawkes_time))
        self._hawkes_time = now

    def _schedule_hawkes(self) -> None:
        self._decay_hawkes(self.t)
        bound = 2 * self.cfg.market_rate * self._max_activity() * (1 + abs(self.ext.imbalance_beta)) \
            + self._hawkes_excitation
        self._push(self.t + float(self.rng_flow.exponential(1 / bound)), "v2_market", bound)

    def _hawkes_candidate(self, now: float, bound: float) -> None:
        self._decay_hawkes(now)
        base = 2 * self.cfg.market_rate * self._max_activity() * (1 + abs(self.ext.imbalance_beta))
        if self._accept((base + self._hawkes_excitation) / bound):
            if self._accept(base / (base + self._hawkes_excitation)):
                side = Side.BUY if self._rng_ext.random() < 0.5 else Side.SELL
                self._zi_market(side, now)  # base arrival: activity/imbalance thinning applies
            else:
                side = Side.BUY if self._rng_ext.random() < 0.5 else Side.SELL
                quantity = self._size(self.ext.market_size_quantiles, self.cfg.market_qty_mean, self.rng)
                self._admit_background(Order(self._next_oid(True), side, quantity, OrderType.MARKET, self.ZI, now), now)
                self._decay_hawkes(now)
                self._hawkes_excitation += self.ext.hawkes_alpha

    # ------------------------------------------------------------------ event loop
    def step(self, dt: float) -> List[Trade]:
        """The base event loop, reproduced exactly, plus the two extension kinds."""
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
            elif kind == "zi_limit":
                self._zi_limit(payload, now)
                self._next_arrival(kind, payload, self.cfg.limit_rate)
            elif kind == "zi_market":
                self._zi_market(payload, now)
                self._next_arrival(kind, payload, self.cfg.market_rate)
            elif kind == "zi_refill":
                self._zi_limit(payload, now, refill=True)
                self._next_arrival(kind, payload, self.cfg.resilience * self.cfg.resilience_levels * 10)
            elif kind == "v2_cancel":
                if payload in self.book.orders:
                    if self._accept(self._cancel_multiplier(payload, now)):
                        self.book.cancel(payload, now)
                    else:
                        self._schedule_cancel_proposal(payload, now)
            elif kind == "v2_market":
                self._hawkes_candidate(now, payload)
                self._schedule_hawkes()
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


@dataclass(frozen=True)
class SimulatorSpec:
    """A complete, hashable candidate: base config plus extensions."""

    config: dict
    extensions: dict = field(default_factory=dict)

    def build(self, seed: int) -> ExtendedSimulator | ExchangeSimulator:
        cfg = SimConfig(**{**self.config, "seed": seed, "record_events": False, "check_invariants": False})
        ext = FlowExtensions(**self.extensions)
        return ExtendedSimulator(cfg, ext) if ext.active else ExchangeSimulator(cfg)


__all__ = ["ExtendedSimulator", "FlowExtensions", "SimulatorSpec", "OrderStatus"]

"""Latency models, delayed market data and asynchronous action accounting (workstreams 51, 52).

The v0.5 engine already delays every strategy action by ``base + Exp(jitter)``
using its own latency stream. v0.7 makes the model pluggable without editing the
frozen engine: ``attach(sim, model)`` replaces the instance's ``_latency`` draw
(still from ``sim.rng_lat``, so runs stay seed-deterministic). ``DelayedFeed``
lets an agent observe the book only as of ``now - feed_delay``. ``race_report``
classifies asynchronous outcomes (cancels that lost the race to a fill, orders
still in flight). These are simulator semantics; nothing here claims real venue
latency, colocation or HFT behaviour.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math

import numpy as np

from ...engine import OrderStatus


@dataclass(frozen=True)
class LatencyModel:
    """``kind``: fixed | exponential | lognormal | spiky. Seconds."""

    kind: str = "exponential"
    base: float = 0.005
    scale: float = 0.005
    sigma: float = 0.5
    spike_probability: float = 0.0
    spike_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"fixed", "exponential", "lognormal", "spiky"}:
            raise ValueError(f"unknown latency model {self.kind!r}")
        if min(self.base, self.scale, self.sigma, self.spike_seconds) < 0 or not 0 <= self.spike_probability <= 1:
            raise ValueError("latency parameters must be nonnegative and probabilities in [0, 1]")

    def draw(self, rng: np.random.Generator) -> float:
        if self.kind == "fixed":
            value = self.base
        elif self.kind == "lognormal":
            value = self.base + (float(rng.lognormal(math.log(self.scale), self.sigma)) if self.scale > 0 else 0.0)
        else:
            value = self.base + (float(rng.exponential(self.scale)) if self.scale > 0 else 0.0)
            if self.kind == "spiky" and rng.random() < self.spike_probability:
                value += self.spike_seconds
        return value

    def quantiles(self, n: int = 20000, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        draws = np.asarray([self.draw(rng) for _ in range(n)])
        return {f"p{q}": float(np.quantile(draws, q / 100)) for q in (50, 90, 99)}


ZOO = {
    "engine_default": LatencyModel("exponential", 0.005, 0.005),
    "zero": LatencyModel("fixed", 0.0, 0.0),
    "slow_fixed": LatencyModel("fixed", 0.050, 0.0),
    "heavy_tail": LatencyModel("lognormal", 0.002, 0.010, 1.0),
    "spiky": LatencyModel("spiky", 0.005, 0.005, spike_probability=0.02, spike_seconds=0.5),
}


def attach(sim, model: LatencyModel):
    """Use ``model`` for every subsequent strategy action latency of ``sim`` (instance-level override)."""
    sim._latency = lambda: model.draw(sim.rng_lat)
    sim.latency_model = model
    return sim


class DelayedFeed:
    """Record top-of-book snapshots as the simulator advances; serve the one visible ``delay`` seconds ago."""

    def __init__(self, sim, *, delay: float, levels: int = 5) -> None:
        if delay < 0:
            raise ValueError("feed delay must be nonnegative")
        self.sim, self.delay, self.levels = sim, delay, levels
        self.times: list[float] = []
        self.snapshots: list[dict] = []
        self.record()

    def record(self) -> None:
        bids, asks = self.sim.book.depth(self.levels)
        self.times.append(self.sim.t)
        self.snapshots.append({"t": self.sim.t, "bids": bids, "asks": asks})

    def step(self, dt: float, *, record_every: float = 0.01):
        """Advance the simulator, recording snapshots at most every ``record_every`` seconds."""
        trades, end = [], self.sim.t + dt
        while self.sim.t < end - 1e-12:
            trades.extend(self.sim.step(min(record_every, end - self.sim.t)))
            self.record()
        return trades

    def observe(self) -> dict:
        index = bisect_right(self.times, self.sim.t - self.delay + 1e-12) - 1
        return self.snapshots[max(index, 0)]


def race_report(sim, owner: str) -> dict:
    """Asynchronous outcomes of one owner's orders: fills, cancels, cancels that lost to fills, still in flight."""
    counts = {"orders": 0, "filled": 0, "cancelled": 0, "cancel_lost_to_fill": 0, "in_flight": 0, "resting": 0}
    for order in sim.orders.values():
        if order.owner != owner:
            continue
        counts["orders"] += 1
        if order.status is OrderStatus.FILLED:
            counts["filled"] += 1
            counts["cancel_lost_to_fill"] += int(any(s is OrderStatus.CANCEL_PENDING for _, s in order.status_history))
        elif order.status is OrderStatus.CANCELLED:
            counts["cancelled"] += 1
        elif order.status is OrderStatus.IN_FLIGHT:
            counts["in_flight"] += 1
        elif not order.is_terminal:
            counts["resting"] += 1
    return counts

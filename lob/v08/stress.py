"""Structured stress worlds and bounded adversarial search (M28).

Each registered scenario is a transparent transformation of a plausible world;
the result is labelled ``STRESS`` and is never presented as empirical evidence.

Scenarios: liquidity drought, spread widening, cancellation burst, market-order
burst, volatility shock, latency spike, stale-data shock, resilience collapse,
asymmetric order flow, queue disadvantage, hidden-liquidity sensitivity, regime
transition and a combined stress.

``adversarial_search`` looks for difficult *plausible* worlds: it searches a
registered box of rate multipliers (and refill depth) with a fixed evaluation
budget, keeps only candidates whose simulated summaries stay within the
registered plausibility radius (Mahalanobis, from ``lob.v08.ood``) of the
plausible world set, and maximizes a target (e.g. a policy's regret).
Unbounded or implausible worlds are rejected by construction.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..v07.exchange.latency import LatencyModel
from .execution.simulator import LatencyProfile
from .generators.endogenous import EndogenousGenerator
from .generators.families import MixtureGenerator
from .generators.scaled import ScaledGenerator
from .worlds import World

SCENARIOS = ("liquidity_drought", "spread_widening", "cancellation_burst", "market_order_burst", "volatility_shock",
             "latency_spike", "stale_data_shock", "resilience_collapse", "asymmetric_order_flow", "queue_disadvantage",
             "hidden_liquidity_sensitivity", "regime_transition", "combined")
#               market_buy, market_sell, add_bid, add_ask, cancel_bid, cancel_ask
MULTIPLIERS = {
    "liquidity_drought": [1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
    "spread_widening": [1.0, 1.0, 0.7, 0.7, 1.4, 1.4],
    "cancellation_burst": [1.0, 1.0, 1.0, 1.0, 2.5, 2.5],
    "market_order_burst": [3.0, 3.0, 1.0, 1.0, 1.0, 1.0],
    "volatility_shock": [2.0, 2.0, 1.0, 1.0, 1.3, 1.3],
    "asymmetric_order_flow": [1.8, 0.6, 1.0, 1.0, 1.0, 1.0],
}


def _scaled(world: World, multipliers, world_id: str, **book_changes) -> World:
    book = world.book.replace(**book_changes) if book_changes else world.book
    g = ScaledGenerator.wrap(world.generator, multipliers, book)
    return replace(world, world_id=world_id, generator=g, book=book, label="STRESS")


def apply(world: World, scenario: str) -> World:
    wid = f"{world.world_id}+{scenario}"
    if scenario in MULTIPLIERS:
        extra = {}
        if scenario == "liquidity_drought":
            extra = {"refill_mean": max(1.0, world.book.refill_mean * 0.3)}
        if scenario == "spread_widening":
            extra = {"improve_prob": world.book.improve_prob * 0.2}
        if scenario == "volatility_shock":
            extra = {"refill_mean": max(1.0, world.book.refill_mean * 0.5)}
        return _scaled(world, MULTIPLIERS[scenario], wid, **extra)
    if scenario == "latency_spike":
        spike = LatencyModel("spiky", 0.005, 0.005, spike_probability=0.2, spike_seconds=0.5)
        return replace(world, world_id=wid, latency=LatencyProfile(send=spike, ack=spike, cancel=spike), label="STRESS")
    if scenario == "stale_data_shock":
        return replace(world, world_id=wid, latency=LatencyProfile(feed=LatencyModel("fixed", 1.5, 0.0)),
                       label="STRESS")
    if scenario == "resilience_collapse":
        g = EndogenousGenerator.wrap(world.generator, depletion=2.5, replenish=0.0, impact=1.0)
        book = world.book.replace(refill_mean=max(1.0, world.book.refill_mean * 0.6))
        return replace(world, world_id=wid, generator=EndogenousGenerator(g.params, book), book=book, label="STRESS")
    if scenario == "queue_disadvantage":
        book = world.book.replace(cancel_position="back")
        g = _rebook(world.generator, book)
        return replace(world, world_id=wid, generator=g, book=book, label="STRESS")
    if scenario == "hidden_liquidity_sensitivity":
        book = world.book.replace(hidden_fraction=0.5)
        return replace(world, world_id=wid, generator=_rebook(world.generator, book), book=book, label="STRESS")
    if scenario == "regime_transition":
        burst = ScaledGenerator.wrap(world.generator, MULTIPLIERS["market_order_burst"], world.book)
        g = MixtureGenerator.of([world.generator, burst], [0.5, 0.5], switch_prob=0.01, book=world.book)
        return replace(world, world_id=wid, generator=g, label="STRESS")
    if scenario == "combined":
        w = apply(world, "liquidity_drought")
        w = _scaled(w, MULTIPLIERS["market_order_burst"], wid)
        spike = LatencyModel("spiky", 0.005, 0.005, spike_probability=0.2, spike_seconds=0.5)
        return replace(w, latency=LatencyProfile(send=spike, ack=spike, cancel=spike))
    raise KeyError(f"unknown scenario {scenario!r}")


def _rebook(generator, book):
    from .generators.base import MarketGenerator
    d = generator.to_dict()
    d["book"] = book.to_dict()
    return MarketGenerator.from_dict(d)


def adversarial_search(base: World, objective, *, reference, radius: float, budget: int, seed: int,
                       box: tuple[float, float] = (0.5, 2.0), n_bins: int = 1200) -> dict:
    """Bounded random search over rate multipliers; ``objective(world) -> float`` (higher = more difficult).

    ``reference`` is an ``ood.Reference`` fitted on plausible-world summaries; candidates whose simulated
    summaries lie beyond ``radius`` (Mahalanobis) are rejected as implausible and never evaluated.
    """
    from .market.frame import summaries
    rng = np.random.default_rng(seed)
    lo, hi = np.log(box[0]), np.log(box[1])
    tried, accepted = [], []
    for k in range(budget):
        mult = np.exp(rng.uniform(lo, hi, 6)).tolist()
        refill = float(np.exp(rng.uniform(lo, hi)))
        world = _scaled(base, mult, f"{base.world_id}+adv{k:03d}",
                        refill_mean=max(1.0, base.book.refill_mean * refill))
        s = summaries(world.generator.simulate(n_bins, seed=seed * 1000 + k, book=world.book))
        dist = float(reference.mahalanobis(s)[0])
        record = {"k": k, "multipliers": mult, "refill_factor": refill, "plausibility_distance": dist,
                  "plausible": dist <= radius}
        if dist <= radius:
            record["objective"] = float(objective(world))
            accepted.append(record)
        tried.append(record)
    best = max(accepted, key=lambda r: r["objective"]) if accepted else None
    return {"budget": budget, "box": list(box), "radius": radius, "evaluated": len(accepted),
            "rejected_implausible": len(tried) - len(accepted), "best": best, "candidates": tried,
            "label": "SYNTHETIC_STRESS (bounded adversarial search within a plausibility radius)"}

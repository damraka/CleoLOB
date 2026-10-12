"""Cross-venue, cross-instrument and portfolio execution infrastructure (M31).

Research infrastructure only: nothing here is a production router and no
routing claim is made.

* ``Venue`` — a world with its own fees and latency profile;
* ``route`` routers: ``single:<i>``, ``equal_split``, ``liquidity_weighted`` and
  ``sor`` (smart order routing: each lot goes to the venue with the lowest
  marginal observed cost = walk price + taker fee + a latency penalty);
* ``multi_venue_episode`` — synthetic lockstep simulation of several venues for
  one parent order (TWAP schedule, routed market children);
* ``basket_episode`` — two instruments with a registered cross-impact
  coefficient: strategy flow in A enters B's endogenous response (requires an
  endogenous-response generator in B); schedules ``independent`` or
  ``sequenced`` (B first, then A);
* ``replay_routing`` — the same routers on time-aligned historical frames of
  several venues (market orders only, labelled HYPOTHETICAL, no-impact or
  persistent-consumption assumption).

When real cross-venue data is not usable the infrastructure still runs on
synthetic fixtures and the real-data result is NOT_AVAILABLE.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .execution.simulator import Fees, LatencyProfile
from .market.book import BUY
from .market.frame import BIN_S
from .market.sim import MarketSim

ROUTERS = ("single:0", "single:1", "equal_split", "liquidity_weighted", "sor")


@dataclass
class Venue:
    name: str
    world: object
    fees: Fees
    latency_s: float = 0.0


def _book_view(sim: MarketSim, side: int) -> tuple[int, list[int]]:
    b = sim.book
    return (b.ask_px, list(b.aq[:10])) if side == BUY else (b.bid_px, list(b.bq[:10]))


def _marginal_costs(touch: int, depth: list, side: int, qty: int, mid_ref: float, fee_bps: float,
                    latency_penalty_ticks: float) -> list[float]:
    """Per-lot marginal cost (ticks over the reference mid) of the next ``qty`` lots on a snapshot."""
    out, step = [], (1 if side == BUY else -1)
    for k, size in enumerate(depth):
        price = touch + step * k
        for _ in range(int(size)):
            out.append(side * (price - mid_ref) + price * fee_bps * 1e-4 + latency_penalty_ticks)
            if len(out) >= qty:
                return out
    while len(out) < qty:
        out.append(side * (touch + step * (len(depth) + 10) - mid_ref) + latency_penalty_ticks)
    return out


def route(router: str, qty: int, views: list[tuple[int, list]], venues: list[Venue], side: int,
          mid_ref: float, latency_penalty_per_s: float = 2.0) -> list[int]:
    n = len(venues)
    if qty <= 0:
        return [0] * n
    if router.startswith("single:"):
        i = int(router.split(":")[1])
        return [qty if j == i else 0 for j in range(n)]
    if router == "equal_split":
        base = [qty // n] * n
        for j in range(qty - sum(base)):
            base[j] += 1
        return base
    if router == "liquidity_weighted":
        depth = np.asarray([max(float(v[1][0]), 1e-9) for v in views])
        alloc = np.floor(qty * depth / depth.sum()).astype(int)
        alloc[int(np.argmax(depth))] += qty - alloc.sum()
        return alloc.tolist()
    if router == "sor":
        curves = [_marginal_costs(views[j][0], views[j][1], side, qty, mid_ref, venues[j].fees.taker_bps,
                                  latency_penalty_per_s * venues[j].latency_s) for j in range(n)]
        alloc, cursor = [0] * n, [0] * n
        for _ in range(qty):
            j = min(range(n), key=lambda v: (curves[v][cursor[v]] if cursor[v] < len(curves[v]) else math.inf, v))
            alloc[j] += 1
            cursor[j] += 1
        return alloc
    raise KeyError(f"unknown router {router!r}")


def multi_venue_episode(venues: list[Venue], router: str, *, qty: int, horizon_s: float = 120.0,
                        decision_dt_s: float = 5.0, warmup_s: float = 30.0, seed: int = 0, side: int = BUY) -> dict:
    sims = [MarketSim(v.world.generator, v.world.book, seed * 31 + j) for j, v in enumerate(venues)]
    for _ in range(int(round(warmup_s / BIN_S))):
        for s in sims:
            s.run_bin()
    mids = [s.book.mid() for s in sims]       # each venue is measured against its own arrival mid
    steps = int(round(horizon_s / decision_dt_s))
    per_bins = int(round(decision_dt_s / BIN_S))
    fills = [[] for _ in venues]
    done = 0
    for k in range(steps):
        target = int(math.ceil(qty * (k + 1) / steps))
        child = target - done
        views = [_book_view(s, side) for s in sims]
        rel = [(int(v[0] - mids[j]), v[1]) for j, v in enumerate(views)]
        alloc = route(router, child, rel, venues, side, 0.0)
        for j, (s, v) in enumerate(zip(sims, venues)):
            if alloc[j] <= 0:
                continue
            delay = v.latency_s
            own = np.zeros(6, dtype=np.int64)
            own[0 if side == BUY else 1] = 1
            box: list = []
            s.run_bin([(s.t + min(delay, BIN_S * 0.999), 0,
                        lambda s=s, a=alloc[j], box=box: box.append(s.book.market(side, a, s.t, strategy=True)))],
                      own_counts=own)
            for price, q in box[0][2]:
                fills[j].append((price - mids[j], q, mids[j]))
            done += box[0][0]
        for s in sims:
            ran = 1 if any(alloc) else 0
            for _ in range(per_bins - ran):
                s.run_bin()
    filled = sum(q for f in fills for _, q, _ in f)
    gross = sum(side * d * q / m for f in fills for d, q, m in f) / qty * 1e4
    fees = sum((d + m) * q * v.fees.taker_bps * 1e-4 / m for f, v in zip(fills, venues) for d, q, m in f) / qty * 1e4
    return {"router": router, "seed": seed, "filled": int(filled), "completion": bool(filled >= qty),
            "gross_cost_bps": float(gross), "fee_bps": float(fees), "economic_cost_bps": float(gross + fees),
            "venue_share": [float(sum(q for _, q, _ in f) / max(filled, 1)) for f in fills],
            "limitation": "venues evolve independently (no cross-venue price linkage); costs are per own arrival mid"}


def basket_episode(world_a, world_b, *, qty_a: int, qty_b: int, cross_impact: float, schedule: str = "independent",
                   horizon_s: float = 120.0, decision_dt_s: float = 5.0, warmup_s: float = 30.0, seed: int = 0) -> dict:
    """Buy both legs; strategy flow in A is fed to B's response with coefficient ``cross_impact``."""
    if schedule not in ("independent", "sequenced"):
        raise ValueError("schedule must be independent or sequenced")
    sa, sb = MarketSim(world_a.generator, world_a.book, seed * 31), MarketSim(world_b.generator, world_b.book, seed * 31 + 1)
    for _ in range(int(round(warmup_s / BIN_S))):
        sa.run_bin()
        sb.run_bin()
    ma, mb = sa.book.mid(), sb.book.mid()
    steps = int(round(horizon_s / decision_dt_s))
    per = int(round(decision_dt_s / BIN_S))
    cost = {"a": [0.0, 0], "b": [0.0, 0]}
    for k in range(steps):
        if schedule == "independent":
            ca = int(math.ceil(qty_a * (k + 1) / steps)) - cost["a"][1]
            cb = int(math.ceil(qty_b * (k + 1) / steps)) - cost["b"][1]
        else:
            half = steps // 2
            ca = int(math.ceil(qty_a * max(0, k + 1 - half) / (steps - half))) - cost["a"][1] if k >= half else 0
            cb = int(math.ceil(qty_b * min(k + 1, half) / half)) - cost["b"][1] if k < half else 0
        for key, sim, child, mid in (("a", sa, ca, ma), ("b", sb, cb, mb)):
            flow = None if key == "a" else cross_impact * np.asarray(sa.strategy_flow, float)
            if child > 0:
                box: list = []
                sim.run_bin([(sim.t + 1e-6, 0, lambda s=sim, c=child, box=box: box.append(
                    s.book.market(BUY, c, s.t, strategy=True)))], external_flow=flow)
                filled, notional, _ = box[0]
                cost[key][0] += notional - mid * filled
                cost[key][1] += filled
            else:
                sim.run_bin(external_flow=flow)
        for _ in range(per - 1):
            sa.run_bin()
            sb.run_bin(external_flow=cross_impact * np.asarray(sa.strategy_flow, float))
    total = (cost["a"][0] + cost["b"][0]) / (ma * qty_a + mb * qty_b) * 1e4
    return {"schedule": schedule, "cross_impact": cross_impact, "seed": seed,
            "cost_bps": float(total), "cost_a_bps": float(cost["a"][0] / (ma * qty_a) * 1e4),
            "cost_b_bps": float(cost["b"][0] / (mb * qty_b) * 1e4),
            "completion": bool(cost["a"][1] >= qty_a and cost["b"][1] >= qty_b),
            "label": "SYNTHETIC (stylized cross-impact; not a production basket model)"}


def replay_routing(frames: list, router: str, venues_fees: list[Fees], *, qty: int, start_bin: int,
                   horizon_s: float = 120.0, decision_dt_s: float = 5.0, side: int = BUY,
                   impact: str = "none") -> dict:
    """Historical, hypothetical routing of TWAP children across time-aligned frames (market orders only)."""
    from .counterfactual import _walk
    steps = int(round(horizon_s / decision_dt_s))
    per = int(round(decision_dt_s / BIN_S))
    mids = [0.5 * (f.bid_px[start_bin] + f.ask_px[start_bin]) for f in frames]
    consumed = [{} if impact == "persistent" else None for _ in frames]
    venues = [Venue(str(i), None, fee) for i, fee in enumerate(venues_fees)]
    cost, filled = 0.0, 0.0
    shares = [0.0] * len(frames)
    done = 0
    for k in range(steps):
        row = start_bin + k * per
        target = int(math.ceil(qty * (k + 1) / steps))
        child = target - done
        views = [((f.ask_px[row], f.depth[row, 1].tolist()) if side == BUY else (f.bid_px[row], f.depth[row, 0].tolist()))
                 for f in frames]
        # each venue's prices are normalized to its own arrival mid so venues with different price levels compare
        rel_views = [(int(v[0] - mids[j]), v[1]) for j, v in enumerate(views)]
        alloc = route(router, child, rel_views, venues, side, 0.0)
        for j, a in enumerate(alloc):
            if a <= 0:
                continue
            notional, got, _ = _walk(frames[j], row, side, a, consumed[j])
            cost += side * (notional - mids[j] * got) / mids[j] + notional * venues_fees[j].taker_bps * 1e-4 / mids[j]
            filled += got
            shares[j] += got
        done = target
    return {"router": router, "impact": impact, "label": "HYPOTHETICAL",
            "economic_cost_bps": float(cost / qty * 1e4), "filled": float(filled),
            "venue_share": [s / max(filled, 1e-9) for s in shares]}


__all__ = ["Venue", "ROUTERS", "route", "multi_venue_episode", "basket_episode", "replay_routing", "LatencyProfile"]

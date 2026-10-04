"""Cross-impact groundwork and portfolio execution (workstreams 49, 50).

``CoupledMarkets`` holds several instrument simulators on one clock. A coupling matrix c[i][j] gives the
probability that an aggressive order of the strategy in instrument i triggers a same-direction background
market order of the same size in instrument j (synthetic cross-impact). c = 0 is the zero-cross-impact
baseline: instruments evolve exactly as they would alone. No cross-impact law is claimed from data:
there is no consumed multi-instrument dataset that could validate one.

``PortfolioExecution`` runs simultaneous TWAP parents in several instruments under a shared capital limit
(gross notional of children submitted per step) and a shared risk limit (absolute net inventory summed
across instruments). Accounting: portfolio implementation shortfall equals the sum of per-instrument
shortfalls, and cash plus inventory value is conserved against the counterparties.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...engine import Side


class CoupledMarkets:
    def __init__(self, worlds: dict, coupling: np.ndarray, seed: int) -> None:
        self.names = list(worlds)
        self.sims = {n: w.build(seed + i) for i, (n, w) in enumerate(worlds.items())}
        self.coupling = np.asarray(coupling, float)
        self.rng = np.random.default_rng(seed + 991)

    def step(self, dt: float) -> None:
        for sim in self.sims.values():
            sim.step(dt)

    def submit(self, name: str, side: Side, qty: int, owner: str) -> int:
        oid = self.sims[name].submit(side, qty, owner)
        i = self.names.index(name)
        for j, other in enumerate(self.names):
            if j != i and self.coupling[i, j] > 0 and self.rng.random() < self.coupling[i, j]:
                self.sims[other].submit(side, qty, "COUPLED")
        return oid


@dataclass
class Parent:
    instrument: str
    side: Side
    quantity: int


class PortfolioExecution:
    def __init__(self, markets: CoupledMarkets, parents: list[Parent], *, horizon: float, slices: int,
                 capital_limit: float | None = None, risk_limit: int | None = None) -> None:
        self.markets, self.parents = markets, parents
        self.horizon, self.slices = horizon, slices
        self.capital_limit, self.risk_limit = capital_limit, risk_limit

    def run(self) -> dict:
        m = self.markets
        arrival = {p.instrument: m.sims[p.instrument].book.mid() for p in self.parents}
        sent = {p.instrument: 0 for p in self.parents}
        cursors = {n: len(s.book.trades) for n, s in m.sims.items()}
        tau = self.horizon / self.slices
        binding = {"capital": 0, "risk": 0}
        for j in range(self.slices):
            budget = self.capital_limit
            for p in self.parents:
                target = round(p.quantity * (j + 1) / self.slices)
                child = target - sent[p.instrument]
                price = m.sims[p.instrument].book.mid()
                if budget is not None and child * price > budget:
                    child = int(budget // price)
                    binding["capital"] += 1
                if self.risk_limit is not None:
                    net = sum(sent[q.instrument] * (1 if q.side is Side.BUY else -1) for q in self.parents)
                    room = self.risk_limit - abs(net)
                    if child > room:
                        child = max(0, room)
                        binding["risk"] += 1
                if child > 0:
                    m.submit(p.instrument, p.side, int(child), "PORTFOLIO")
                    sent[p.instrument] += int(child)
                    if budget is not None:
                        budget -= child * price
            m.step(tau)
        m.step(5.0)
        per = {}
        for p in self.parents:
            trades = [t for t in m.sims[p.instrument].book.trades[cursors[p.instrument]:] if t.taker_owner == "PORTFOLIO"]
            sign = 1 if p.side is Side.BUY else -1
            filled = sum(t.qty for t in trades)
            cost = sum(sign * (t.price - arrival[p.instrument]) * t.qty for t in trades)
            cash = -sum(sign * t.price * t.qty for t in trades)
            per[p.instrument] = {"filled": filled, "shortfall_ticks": cost, "cash": cash,
                                 "inventory": sign * filled, "arrival": arrival[p.instrument],
                                 "shortfall_bps": cost / max(filled, 1) / arrival[p.instrument] * 1e4 if filled else 0.0}
        total = sum(v["shortfall_ticks"] for v in per.values())
        identity = abs(total - sum(-(v["cash"] + v["inventory"] * v["arrival"]) for v in per.values()))
        return {"per_instrument": per, "portfolio_shortfall_ticks": total, "identity_error": identity,
                "limit_binding_steps": binding}

"""Baseline optimal execution (Almgren-Chriss) + shared performance reporting."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Union

from .engine import ExchangeSimulator, Side, Trade


@dataclass
class ExecutionReport:
    label: str
    side: Side
    target_qty: int
    filled_qty: int
    arrival_price: float       # currency
    vwap: float                # currency
    shortfall_bps: float       # + = cost vs arrival benchmark
    total_cost: float          # currency
    n_child_orders: int
    duration: float

    def as_dict(self) -> Dict[str, Union[str, int, float]]:
        return {
            "Strategy": self.label,
            "Filled / Target": f"{self.filled_qty:,} / {self.target_qty:,}",
            "Arrival Price": round(self.arrival_price, 4),
            "Exec VWAP": round(self.vwap, 4),
            "Impl. Shortfall (bps)": round(self.shortfall_bps, 2),
            "Total Cost ($)": round(self.total_cost, 2),
            "Child Orders": self.n_child_orders,
            "Duration (s)": round(self.duration, 1),
        }


def build_report(label: str, side: Side, target_qty: int, fills: List[Trade],
                 arrival_ticks: float, tick_size: float, n_children: int,
                 duration: float) -> ExecutionReport:
    """Implementation shortfall vs. the arrival mid-price benchmark."""
    filled = sum(tr.qty for tr in fills)
    vwap_ticks = (sum(tr.price * tr.qty for tr in fills) / filled) if filled else arrival_ticks
    sign = 1.0 if side is Side.SELL else -1.0  # selling below arrival = cost
    shortfall_bps = sign * (arrival_ticks - vwap_ticks) / arrival_ticks * 1e4
    total_cost = sign * (arrival_ticks - vwap_ticks) * filled * tick_size
    return ExecutionReport(label, side, target_qty, filled,
                           arrival_ticks * tick_size, vwap_ticks * tick_size,
                           shortfall_bps, total_cost, n_children, duration)


class AlmgrenChrissAgent:
    """Discretised Almgren-Chriss implementation-shortfall schedule.

    Optimal remaining inventory under linear temporary impact and risk aversion λ:

        x(t) = X · sinh(κ (T − t)) / sinh(κ T),   κ = sqrt(λ σ² / η)

    λ → 0 recovers a TWAP-like linear schedule; larger λ front-loads execution
    (sell fast, accept impact, reduce price risk). Each due slice is released as a
    market order, so realised slippage comes from the simulated book, not a model.
    """

    def __init__(self, total_qty: int, horizon: float, n_slices: int = 20,
                 side: Side = Side.SELL, risk_aversion: float = 1e-6,
                 temp_impact: float = 1.3e-3, sigma: float = 1.5,
                 start_time: float = 0.0) -> None:
        # With these defaults: λ=1e-7 -> κT≈0.8 (≈TWAP) … λ=1e-4 -> κT≈25 (front-loaded).
        self.total_qty = total_qty
        self.horizon = horizon
        self.side = side
        tau = horizon / n_slices
        kappa = math.sqrt(risk_aversion * sigma ** 2 / max(temp_impact, 1e-12))
        # Slice j (0-indexed) is released at the *start* of interval [j·τ, (j+1)·τ),
        # trading the book down to targets[j+1] — so the schedule completes within T.
        self.release_times: List[float] = [start_time + j * tau for j in range(n_slices)]
        kt = kappa * horizon
        if kt < 1e-6:  # numerically TWAP
            self.targets: List[int] = [round(total_qty * (1 - j / n_slices))
                                       for j in range(n_slices + 1)]
        else:
            self.targets = [round(total_qty * math.sinh(kappa * (horizon - j * tau))
                                  / math.sinh(kt)) for j in range(n_slices + 1)]
        self.targets[-1] = 0
        self._next = 0
        self.filled = 0
        self.child_orders = 0
        self._cursor = 0
        self.fills: List[Trade] = []

    def poll_fills(self, sim: ExchangeSimulator, owner: str) -> None:
        new, self._cursor = sim.fills_for(owner, self._cursor)
        for tr in new:
            self.fills.append(tr)
            self.filled += tr.qty

    def on_step(self, sim: ExchangeSimulator, owner: str) -> None:
        """Release every schedule slice that has come due at the current sim time."""
        self.poll_fills(sim, owner)
        while self._next < len(self.release_times) and sim.t >= self.release_times[self._next]:
            target = self.targets[self._next + 1]
            self._next += 1
            child = max(0, (self.total_qty - self.filled) - target)
            if child > 0:
                sim.submit(self.side, child, owner)  # market order
                self.child_orders += 1

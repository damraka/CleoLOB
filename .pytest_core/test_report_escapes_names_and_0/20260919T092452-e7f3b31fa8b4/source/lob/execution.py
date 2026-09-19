"""Execution agents (baseline zoo) and shared performance reporting.

Agent hierarchy
---------------
ExecutionAgent          fill accounting + on_step(sim, owner) interface
├── ScheduleAgent       releases pre-computed slices (targets[j] = remaining after slice j)
│   ├── TWAPAgent       linear schedule
│   ├── VWAPAgent       schedule proportional to a (forecast) volume profile
│   └── AlmgrenChrissAgent   sinh schedule from linear-impact + risk-aversion model
└── POVAgent            reactive: tracks a participation rate of observed market volume

All baselines execute with market orders, so realised slippage is produced by the
simulated book, not by an impact model. Every agent exposes the same fields the
runner/UI rely on: ``filled``, ``fills``, ``child_orders``, ``poll_fills``, ``on_step``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .accounting import FeeConfig, Ledger, fee_config
from .engine import ExchangeSimulator, Side, SimConfig, Trade
from .risk import ExecutionRisk, RiskConfig


# ------------------------------------------------------------------ reporting
@dataclass
class ExecutionReport:
    label: str
    side: Side
    target_qty: int
    filled_qty: int
    arrival_price: float       # currency
    vwap: float                # currency
    shortfall_bps: float       # + = cost vs arrival benchmark, filled shares only
    total_cost: float          # currency, filled shares only
    n_child_orders: int
    duration: float
    effective_shortfall_bps: Optional[float] = 0.0
    leftover_qty: int = 0
    gross_cost: float = 0.0
    total_fees: float = 0.0
    fee_bps: float = 0.0
    gross_effective_bps: Optional[float] = None
    net_effective_bps: Optional[float] = None
    hypothetical_liquidation_cost: float = 0.0
    hypothetical_liquidation_fees: float = 0.0
    fillable_leftover_qty: int = 0
    unpriced_leftover_qty: int = 0
    outstanding_qty: int = 0
    completion_penalty_bps: float = 0.0
    optimization_cost_bps: float = 0.0
    status: str = "VALID"
    invalid_reasons: tuple[str, ...] = ()
    accounting: Optional[Dict[str, Any]] = None
    decision_horizon: Optional[float] = None
    decision_duration: Optional[float] = None
    settlement_duration: float = 0.0
    total_duration: Optional[float] = None
    settlement_complete: Optional[bool] = None
    late_filled_qty: int = 0
    late_fees: float = 0.0
    settlement_pending_order_ids: tuple[int, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "Strategy": self.label,
            "Filled / Target": f"{self.filled_qty:,} / {self.target_qty:,}",
            "Arrival Price": round(self.arrival_price, 4),
            "Exec VWAP": round(self.vwap, 4),
            "Impl. Shortfall (bps)": round(self.shortfall_bps, 2),
            "Effective IS (bps)": None if self.effective_shortfall_bps is None else round(self.effective_shortfall_bps, 2),
            "Total Cost ($)": round(self.total_cost, 2),
            "Fees ($)": round(self.total_fees, 4),
            "Status": self.status,
            "Unpriced Leftover": self.unpriced_leftover_qty,
            "Child Orders": self.n_child_orders,
            "Duration (s)": round(self.duration, 1),
            "Settlement (s)": round(self.settlement_duration, 3),
            "Settlement Complete": self.settlement_complete,
            "Late Filled": self.late_filled_qty,
        }


def build_report(label: str, side: Side, target_qty: int, fills: List[Trade],
                 arrival_ticks: float, tick_size: float, n_children: int,
                 duration: float, leftover_vwap_ticks: Optional[float] = None, *,
                 leftover_fillable_qty: Optional[int] = None,
                 fees: FeeConfig | Mapping[str, Any] | None = None, owner: Optional[str] = None,
                 total_fees: Optional[float] = None, terminal_penalty_bps: float = 0.0,
                 strict_terminal: bool = False, outstanding_qty: int = 0,
                 accounting: Optional[Dict[str, Any]] = None,
                 risk_rejections: int = 0,
                 decision_horizon: Optional[float] = None,
                 settlement_duration: float = 0.0,
                 settlement_complete: Optional[bool] = None,
                 late_filled_qty: int = 0, late_fees: float = 0.0,
                 settlement_pending_order_ids: tuple[int, ...] = ()) -> ExecutionReport:
    """Actual fee-inclusive shortfall and a separate hypothetical terminal valuation.

    Partial visible depth is never extrapolated to unfillable shares. Strict callers
    receive INVALID/None economic metrics when any leftover lacks a known price.
    Legacy calls without a quote retain arrival-mark valuation, explicitly WARNING.
    The optimisation objective adds a disclosed noncompletion penalty; it is not PnL.
    """
    if not isinstance(side, Side) or isinstance(target_qty, bool) or not isinstance(target_qty, int) or target_qty <= 0 or arrival_ticks <= 0 or tick_size <= 0:
        raise ValueError("positive target, arrival, tick size and valid side required")
    if not all(math.isfinite(v) for v in (arrival_ticks, tick_size, duration, terminal_penalty_bps)) or terminal_penalty_bps < 0 or duration < 0:
        raise ValueError("report inputs must be finite and terminal penalty nonnegative")
    if (not math.isfinite(settlement_duration) or settlement_duration < 0
            or not math.isfinite(late_fees)
            or (decision_horizon is not None and (not math.isfinite(decision_horizon) or decision_horizon <= 0))
            or (settlement_complete is not None and not isinstance(settlement_complete, bool))):
        raise ValueError("invalid settlement report inputs")
    fee_schedule = fee_config(fees)
    if any(isinstance(tr.qty, bool) or not isinstance(tr.qty, int) or tr.qty <= 0 or not math.isfinite(tr.price) or tr.price <= 0 for tr in fills):
        raise ValueError("report fills must have positive integer quantities and finite positive prices")
    filled = sum(tr.qty for tr in fills)
    if filled > target_qty:
        raise ArithmeticError("parent order overfilled")
    if isinstance(late_filled_qty, bool) or not isinstance(late_filled_qty, int) or not 0 <= late_filled_qty <= filled:
        raise ValueError("late_filled_qty must be an integer between zero and filled quantity")
    if settlement_complete and (outstanding_qty or settlement_pending_order_ids):
        raise ValueError("complete settlement cannot contain outstanding orders")
    vwap_ticks = (sum(tr.price * tr.qty for tr in fills) / filled) if filled else arrival_ticks
    sign = 1.0 if side is Side.SELL else -1.0  # selling below arrival = cost
    gross_cost = sign * (arrival_ticks - vwap_ticks) * filled * tick_size
    if total_fees is None:
        total_fees = sum(fee_schedule.charge(tr.price * tick_size, tr.qty,
                                           maker=owner is not None and tr.maker_owner == owner) for tr in fills)
    if not math.isfinite(total_fees):
        raise ValueError("total_fees must be finite")
    total_cost = gross_cost + total_fees
    arrival_price = arrival_ticks * tick_size
    denominator = target_qty * arrival_price
    shortfall_bps = total_cost / (filled * arrival_price) * 1e4 if filled else 0.0
    leftover = max(0, target_qty - filled)
    reasons: list[str] = []
    if leftover_fillable_qty is None:
        fillable = leftover if leftover_vwap_ticks is not None or not strict_terminal else 0
    else:
        if isinstance(leftover_fillable_qty, bool) or not isinstance(leftover_fillable_qty, int) or not 0 <= leftover_fillable_qty <= leftover:
            raise ValueError("leftover_fillable_qty must be an integer between zero and remaining quantity")
        fillable = leftover_fillable_qty
    if leftover_vwap_ticks is not None and (not math.isfinite(leftover_vwap_ticks) or leftover_vwap_ticks <= 0):
        raise ValueError("leftover quote must be positive and finite")
    if leftover_vwap_ticks is None and strict_terminal:
        fillable = 0
    unsettled = settlement_complete is False or bool(outstanding_qty or settlement_pending_order_ids)
    if unsettled:
        # Outstanding orders can still fill. There is no final remainder yet,
        # so do not mark it as hypothetically liquidated or publish final IS.
        fillable = 0
        reasons.append("unsettled_orders")
    unpriced = leftover - fillable
    liq = arrival_ticks if leftover_vwap_ticks is None else leftover_vwap_ticks
    hypothetical_cost = sign * (arrival_ticks - liq) * fillable * tick_size
    hypothetical_fees = fee_schedule.charge(liq * tick_size, fillable)
    if unpriced and not unsettled:
        reasons.append("insufficient_terminal_depth")
    gross_eff = None if unpriced or unsettled else (gross_cost + hypothetical_cost) / denominator * 1e4
    net_eff = None if unpriced or unsettled else (total_cost + hypothetical_cost + hypothetical_fees) / denominator * 1e4
    penalty = terminal_penalty_bps * leftover / target_qty
    objective = (total_cost + hypothetical_cost + hypothetical_fees) / denominator * 1e4 + penalty
    status = "INVALID" if reasons else "WARNING" if outstanding_qty or risk_rejections or (leftover and leftover_vwap_ticks is None) else "VALID"
    return ExecutionReport(label, side, target_qty, filled, arrival_price, vwap_ticks * tick_size,
                           shortfall_bps, total_cost, n_children, duration, net_eff, leftover,
                           gross_cost, total_fees, total_fees / denominator * 1e4,
                           gross_eff, net_eff, hypothetical_cost, hypothetical_fees, fillable,
                           unpriced, outstanding_qty, penalty, objective, status, tuple(reasons), accounting,
                           decision_horizon, duration, settlement_duration, duration + settlement_duration,
                           settlement_complete, late_filled_qty, late_fees, settlement_pending_order_ids)


# ------------------------------------------------------------------ base class
class ExecutionAgent:
    """Common fill accounting. Subclasses implement ``on_step``."""

    label: str = "agent"

    def __init__(self, total_qty: int, horizon: float, side: Side = Side.SELL,
                 start_time: float = 0.0, *, fees: FeeConfig | Mapping[str, Any] | None = None,
                 risk: RiskConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(total_qty, bool) or not isinstance(total_qty, int) or total_qty <= 0:
            raise ValueError("total_qty must be a positive integer")
        if not math.isfinite(horizon) or horizon <= 0 or not math.isfinite(start_time) or start_time < 0:
            raise ValueError("horizon must be positive and start_time nonnegative, both finite")
        if not isinstance(side, Side):
            raise ValueError("side must be Side.BUY or Side.SELL")
        self.total_qty = int(total_qty)
        self.horizon = float(horizon)
        self.side = side
        self.start_time = float(start_time)
        self.filled = 0
        self.child_orders = 0
        self.fills: List[Trade] = []
        self._cursor = 0
        self.fees = fee_config(fees)
        self.risk = ExecutionRisk(total_qty, side, risk)
        self.ledger: Optional[Ledger] = None

    def _ensure_ledger(self, sim: ExchangeSimulator) -> Ledger:
        if self.ledger is None:
            self.ledger = Ledger(inventory=self.total_qty if self.side is Side.SELL else 0,
                                 average_cost=sim.book.mid() * sim.cfg.tick_size, fees=self.fees)
        return self.ledger

    def available(self, sim: ExchangeSimulator) -> int:
        return self.risk.available(sim, self.filled)

    @property
    def remaining(self) -> int:
        return max(0, self.total_qty - self.filled)

    def poll_fills(self, sim: ExchangeSimulator, owner: str) -> None:
        ledger = self._ensure_ledger(sim)
        new, self._cursor = sim.fills_for(owner, self._cursor)
        for tr in new:
            ledger.apply_trade(tr, owner, sim.cfg.tick_size)
            self.fills.append(tr)
            self.filled += tr.qty
        self.risk.available(sim, self.filled)
        self.risk.check(sim, ledger)

    def _send_market(self, sim: ExchangeSimulator, owner: str, qty: int) -> None:
        qty = min(int(qty), self.available(sim))
        if qty > 0:
            oid = self.risk.submit(sim, owner, qty, self.filled, self._ensure_ledger(sim))
            if oid is not None:
                self.child_orders += 1

    def on_step(self, sim: ExchangeSimulator, owner: str) -> None:  # pragma: no cover
        raise NotImplementedError


class ScheduleAgent(ExecutionAgent):
    """Releases slice j at the *start* of interval [j·τ, (j+1)·τ) so the schedule
    completes within T. ``targets[j]`` = inventory that should remain after slice j
    (targets[0] = Q, targets[-1] = 0)."""

    def __init__(self, total_qty: int, horizon: float, n_slices: int = 20,
                 side: Side = Side.SELL, start_time: float = 0.0, *,
                 fees: FeeConfig | Mapping[str, Any] | None = None,
                 risk: RiskConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__(total_qty, horizon, side, start_time, fees=fees, risk=risk)
        if isinstance(n_slices, bool) or not isinstance(n_slices, int) or n_slices <= 0:
            raise ValueError("n_slices must be a positive integer")
        self.n_slices = n_slices
        tau = self.horizon / self.n_slices
        self.release_times: List[float] = [start_time + j * tau for j in range(self.n_slices)]
        self.targets: List[int] = self._targets()
        self.targets[0] = self.total_qty
        self.targets[-1] = 0
        self._next = 0

    def _targets(self) -> List[int]:  # pragma: no cover
        raise NotImplementedError

    def on_step(self, sim: ExchangeSimulator, owner: str) -> None:
        self.poll_fills(sim, owner)
        while self._next < self.n_slices and sim.t + 1e-12 >= self.release_times[self._next]:
            target = self.targets[self._next + 1]
            self._next += 1
            self._send_market(sim, owner, self.available(sim) - target)


# ------------------------------------------------------------------ baselines
class TWAPAgent(ScheduleAgent):
    """Time-weighted: equal slices at equal intervals."""

    label = "TWAP"

    def _targets(self) -> List[int]:
        return [round(self.total_qty * (1 - j / self.n_slices)) for j in range(self.n_slices + 1)]


class VWAPAgent(ScheduleAgent):
    """Volume-weighted: slice sizes follow a forecast volume profile (one weight per slice).

    A flat profile reduces to TWAP. Use ``estimate_volume_profile`` to build the forecast
    from a "previous day" simulation, mirroring how desks use historical volume curves.
    """

    label = "VWAP"

    def __init__(self, total_qty: int, horizon: float, n_slices: int = 20,
                 side: Side = Side.SELL, start_time: float = 0.0,
                 volume_profile: Optional[Sequence[float]] = None, *,
                 fees: FeeConfig | Mapping[str, Any] | None = None,
                 risk: RiskConfig | Mapping[str, Any] | None = None) -> None:
        if volume_profile is not None and len(volume_profile) != n_slices:
            raise ValueError("volume_profile must have exactly n_slices entries")
        self.volume_profile = list(volume_profile) if volume_profile is not None else None
        if self.volume_profile is not None and (any(not math.isfinite(w) or w < 0 for w in self.volume_profile) or sum(self.volume_profile) <= 0):
            raise ValueError("volume_profile must contain finite nonnegative weights with positive total")
        super().__init__(total_qty, horizon, n_slices, side, start_time, fees=fees, risk=risk)

    def _targets(self) -> List[int]:
        w = self.volume_profile or [1.0] * self.n_slices
        total = sum(w) or 1.0
        cum, targets = 0.0, []
        for j in range(self.n_slices + 1):
            targets.append(round(self.total_qty * (1 - cum / total)))
            if j < self.n_slices:
                cum += max(0.0, w[j])
        return targets


class AlmgrenChrissAgent(ScheduleAgent):
    """Discretised Almgren-Chriss implementation-shortfall schedule.

        x(t) = X · sinh(κ (T − t)) / sinh(κ T),   κ = sqrt(λ σ² / η)

    λ → 0 recovers TWAP; larger λ front-loads execution (accept impact, reduce risk).
    With the defaults: λ=1e-7 → κT≈0.8 (≈TWAP) … λ=1e-4 → κT≈25 (front-loaded).
    """

    label = "Almgren-Chriss"

    def __init__(self, total_qty: int, horizon: float, n_slices: int = 20,
                 side: Side = Side.SELL, risk_aversion: float = 1e-6,
                 temp_impact: float = 1.3e-3, sigma: float = 1.5,
                 start_time: float = 0.0, *, fees: FeeConfig | Mapping[str, Any] | None = None,
                 risk: RiskConfig | Mapping[str, Any] | None = None) -> None:
        if not all(math.isfinite(v) for v in (risk_aversion, temp_impact, sigma)) or risk_aversion < 0 or temp_impact <= 0 or sigma < 0:
            raise ValueError("AC requires nonnegative risk aversion/sigma and positive impact, all finite")
        self.kappa = math.sqrt(risk_aversion / temp_impact) * sigma
        if not math.isfinite(self.kappa * horizon):
            raise ValueError("AC kappa times horizon must be finite")
        super().__init__(total_qty, horizon, n_slices, side, start_time, fees=fees, risk=risk)

    def _targets(self) -> List[int]:
        kt = self.kappa * self.horizon
        n = self.n_slices
        if kt < 1e-6:  # numerically TWAP
            return [round(self.total_qty * (1 - j / n)) for j in range(n + 1)]
        # sinh(a)/sinh(b) in a form that never exponentiates a positive number.
        return [round(self.total_qty * math.exp(-kt * j / n)
                      * (-math.expm1(-2 * kt * (1 - j / n))) / (-math.expm1(-2 * kt)))
                for j in range(n + 1)]


class POVAgent(ExecutionAgent):
    """Percentage-of-volume: keep own fills ≈ p / (1 − p) × observed market volume.

    Reactive to the market's actual activity. Because a fixed participation rate may not
    finish inside T, the agent switches to a linear catch-up over the final
    ``(1 − catchup_frac) · T`` and dumps any remainder at the last decision step.
    """

    label = "POV"

    def __init__(self, total_qty: int, horizon: float, side: Side = Side.SELL,
                 start_time: float = 0.0, participation: float = 0.3,
                 catchup_frac: float = 0.85, decision_dt: float = 0.5,
                 min_child: int = 10, *, fees: FeeConfig | Mapping[str, Any] | None = None,
                 risk: RiskConfig | Mapping[str, Any] | None = None) -> None:
        if not 0.0 < participation < 1.0:
            raise ValueError("participation must be in (0, 1)")
        if not math.isfinite(catchup_frac) or not 0 <= catchup_frac <= 1 or not math.isfinite(decision_dt) or decision_dt <= 0:
            raise ValueError("catchup_frac must be in [0, 1] and decision_dt positive")
        if isinstance(min_child, bool) or not isinstance(min_child, int) or min_child <= 0:
            raise ValueError("min_child must be a positive integer")
        super().__init__(total_qty, horizon, side, start_time, fees=fees, risk=risk)
        self.participation = participation
        self.catchup_frac = catchup_frac
        self.decision_dt = decision_dt
        self.min_child = min_child
        self.market_volume = 0
        self._mkt_cursor: Optional[int] = None

    def _update_market_volume(self, sim: ExchangeSimulator, owner: str) -> None:
        trades = sim.book.trades
        if self._mkt_cursor is None:
            self._mkt_cursor = len(trades)  # only count volume after we start
        for tr in trades[self._mkt_cursor:]:
            if owner not in (tr.taker_owner, tr.maker_owner):
                self.market_volume += tr.qty
        self._mkt_cursor = len(trades)

    def on_step(self, sim: ExchangeSimulator, owner: str) -> None:
        self.poll_fills(sim, owner)
        self._update_market_volume(sim, owner)
        if self.remaining == 0:
            return
        t_rel = sim.t - self.start_time
        p = self.participation
        target_filled = p / (1.0 - p) * self.market_volume
        child = int(target_filled - self.filled - self.risk.outstanding(sim))

        if t_rel >= self.catchup_frac * self.horizon:
            steps_left = max(1, int(round((self.horizon - t_rel) / self.decision_dt)))
            child = max(child, math.ceil(self.remaining / steps_left))
        if t_rel + self.decision_dt >= self.horizon - 1e-9:
            child = self.remaining                       # last decision: finish
        if child >= self.min_child or child >= self.remaining:
            self._send_market(sim, owner, child)


# ------------------------------------------------------------------ helpers
def estimate_volume_profile(cfg: SimConfig, horizon: float, n_slices: int,
                            warmup: float = 5.0, n_days: int = 5,
                            seed_offset: int = 1000) -> List[float]:
    """Traded volume per slice averaged over ``n_days`` 'previous days' (same config,
    shifted seeds), normalised to sum to 1.

    This is the synthetic analogue of the historical volume curve a VWAP algo would use.
    """
    tau = horizon / n_slices
    acc = [0.0] * n_slices
    for d in range(n_days):
        prev = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + seed_offset + d})
        sim = ExchangeSimulator(prev)
        for _ in range(int(warmup / 0.1)):
            sim.step(0.1)
        for j in range(n_slices):
            acc[j] += float(sum(tr.qty for tr in sim.step(tau)))
    total = sum(acc)
    return [v / total for v in acc] if total > 0 else [1.0 / n_slices] * n_slices


BASELINE_LABELS: Dict[str, str] = {
    "ac": AlmgrenChrissAgent.label,
    "twap": TWAPAgent.label,
    "vwap": VWAPAgent.label,
    "pov": POVAgent.label,
}

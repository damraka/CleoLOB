"""Opt-in urgency shared by every policy; orders remain subject to exchange risk."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class CompletionConstraint:
    enabled: bool = False
    urgency_fraction: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not math.isfinite(self.urgency_fraction) or not 0 <= self.urgency_fraction <= 1:
            raise ValueError("completion enabled must be boolean and urgency_fraction in [0, 1]")

    def active(self, elapsed: float, horizon: float, decision_dt: float) -> bool:
        return self.enabled and (elapsed >= self.urgency_fraction * horizon - 1e-12
                                 or elapsed + decision_dt >= horizon - 1e-12)


def completion_constraint(value: CompletionConstraint | Mapping[str, Any] | None) -> CompletionConstraint:
    return value if isinstance(value, CompletionConstraint) else CompletionConstraint(**dict(value or {}))


def urgency_order(sim: Any, risk: Any, ledger: Any, owner: str, filled: int,
                  *, elapsed: float, horizon: float, decision_dt: float) -> int | None:
    """Cancel outstanding children, then schedule available residual before horizon.

    Cancellation never releases reservations early. Zero liquidity, quantity limits,
    latency and kill switches can all prevent completion. No post-horizon action is
    submitted and no terminal valuation enters actual fills.
    """
    if elapsed >= horizon - 1e-12:
        return None
    risk.check(sim, ledger)
    risk.cancel_all(sim, reason="completion_urgency")
    available = risk.available(sim, filled)
    if not available:
        return None
    steps = max(1, math.ceil((horizon - elapsed) / decision_dt - 1e-12))
    lot = sim.cfg.lot_size
    quantity = min(available, math.ceil(available / steps / lot) * lot)
    return risk.submit(sim, owner, quantity, filled, ledger)


def execution_metrics(sim: Any, fills: list, *, owner: str, start_time: float,
                      first_trade: int, report: Any) -> dict:
    """Observable actual execution and explicitly hypothetical midpoint valuation."""
    market_volume = sum(t.qty for t in sim.book.trades[first_trade:]
                        if owner not in (t.maker_owner, t.taker_owner))
    completed = report.filled_qty == report.target_qty
    settled = report.settlement_complete is True
    sign = 1 if report.side.value == -1 else -1
    two_sided = sim.book.best_bid() is not None and sim.book.best_ask() is not None
    residual_mark_cost = (sign * (report.arrival_price - sim.book.mid() * sim.cfg.tick_size)
                          * report.leftover_qty) if two_sided else None
    return {
        "actual_filled_qty": report.filled_qty, "terminal_inventory": report.leftover_qty,
        "actual_completion": completed if settled else None,
        "time_to_completion": max(t.time - start_time for t in fills) if completed else None,
        "realized_fill_cost": report.total_cost,
        "realized_fill_cost_bps": report.total_cost / (report.target_qty * report.arrival_price) * 1e4,
        "realized_slippage_bps": report.gross_cost / (report.filled_qty * report.arrival_price) * 1e4
                                   if report.filled_qty else None,
        "market_volume": market_volume,
        "participation": report.filled_qty / (report.filled_qty + market_volume)
                         if report.filled_qty + market_volume else None,
        "hypothetical_residual_midpoint_cost": (0.0 if not report.leftover_qty else residual_mark_cost) if settled else None,
        "participation_definition": "own actual fills / (own fills + other trades), decision plus settlement interval",
        "impact_proxy_definition": "gross effective shortfall; mixes spread, timing and impact, not causal impact",
    }

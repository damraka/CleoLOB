"""Strict execution-mandate semantics (M5), identical for every policy and control.

A mandate is completed only by *actual* fills whose fill time is at or before the
decision-horizon end (inclusive boundary). Fills during post-horizon settlement
are reported separately as late fills; they can complete the parent order but
never the mandate. Hypothetical residual valuation never creates a fill and never
changes completion. Unresolved settlement withholds final completion but not the
within-horizon outcome, which is determined by fills observed by the horizon.
"""
from __future__ import annotations

import math
from typing import Iterable

BOUNDARY_EPS = 1e-9


def mandate_metrics(fills: Iterable[tuple[float, int]], *, target_qty: int, start_time: float, horizon: float,
                    settlement_complete: bool | None, total_fees: float, late_fees: float,
                    realized_fill_cost_bps: float | None, hypothetical_residual_cost_bps: float | None,
                    net_effective_bps: float | None, participation: float | None) -> dict:
    """``fills`` are (absolute fill time, quantity) of the parent's own actual fills."""
    if isinstance(target_qty, bool) or not isinstance(target_qty, int) or target_qty <= 0:
        raise ValueError("target_qty must be a positive integer")
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive and finite")
    end = start_time + horizon
    ordered = sorted((float(t), int(q)) for t, q in fills)
    if any(q <= 0 for _, q in ordered):
        raise ValueError("fill quantities must be positive")
    if sum(q for _, q in ordered) > target_qty:
        raise ArithmeticError("parent order overfilled")
    within = sum(q for t, q in ordered if t <= end + BOUNDARY_EPS)
    total = sum(q for _, q in ordered)
    completed_at = None
    cumulative = 0
    for t, q in ordered:
        cumulative += q
        if cumulative == target_qty:
            completed_at = t
            break
    within_complete = within == target_qty
    # Every unit actually filled: complete regardless of settlement bookkeeping. Otherwise the
    # outcome is known only when settlement finished; unresolved settlement withholds it.
    final_complete = True if total == target_qty else (False if settlement_complete else None)
    lateness = None
    if completed_at is not None:
        lateness = max(0.0, completed_at - end) if not within_complete else 0.0
    return {
        "mandate_definition": "actual fills with fill time <= decision horizon end (inclusive)",
        "within_horizon_completion": within_complete,
        "within_horizon_filled_qty": within,
        "fill_fraction_at_horizon": within / target_qty,
        "residual_inventory_at_horizon": target_qty - within,
        "post_horizon_filled_qty": total - within,
        "final_settlement_completion": final_complete,
        "settlement_only_completion": bool(final_complete) and not within_complete,
        "final_filled_qty": total,
        "final_fill_fraction": total / target_qty,
        "residual_inventory_after_settlement": target_qty - total,
        "time_to_completion": None if completed_at is None else completed_at - start_time,
        "lateness_seconds": lateness,
        "fees_total": total_fees, "fees_within_horizon": total_fees - late_fees, "fees_post_horizon": late_fees,
        "realized_fill_cost_bps": realized_fill_cost_bps,
        "hypothetical_residual_valuation_bps": hypothetical_residual_cost_bps,
        "completion_adjusted_cost_bps": net_effective_bps,
        "implementation_shortfall_bps": net_effective_bps,
        "participation": participation,
        "valuation_note": "hypothetical residual valuation is never an actual fill and never changes completion",
    }

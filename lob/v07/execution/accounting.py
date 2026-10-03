"""Accounting v3 and horizon semantics (workstreams 78, 79).

``statement`` builds a separated account statement for one parent order from the frozen v0.5 ``Ledger``
(moving-average cost, PnL identity enforced at every snapshot): cash, inventory, fees and rebates
(negative maker fees are reported as rebates, never netted silently), realized fill value, residual
inventory mark (valued, never liquidated), implementation shortfall against the arrival price, and the
within-horizon / post-horizon (settlement) split. ``HORIZON_FIELDS`` are the result fields every v0.7
execution row must keep separate; ``check_horizon_fields`` enforces it.
"""
from __future__ import annotations

from ...accounting import FeeConfig, Ledger
from ...engine import Side

HORIZON_FIELDS = ("mandate_within_horizon_filled_qty", "mandate_within_horizon_completion",
                  "mandate_residual_inventory_at_horizon", "mandate_post_horizon_filled_qty",
                  "mandate_final_settlement_completion")


def statement(fills: list[tuple[float, float, int, bool]], *, side: Side, target: int, arrival: float, mark: float,
              horizon_end: float, fees: FeeConfig) -> dict:
    """``fills``: (time, price, qty, maker). Prices in currency. Returns the separated statement."""
    ledger = Ledger(fees=fees)
    fee_total = rebate_total = 0.0
    within = post = 0
    value = 0.0
    for time, price, qty, maker in fills:
        fee = ledger.apply_fill(side, price, int(qty), maker=maker)
        fee_total += max(fee, 0.0)
        rebate_total += -min(fee, 0.0)
        value += price * qty
        if time <= horizon_end:
            within += qty
        else:
            post += qty
    snap = ledger.snapshot(mark)
    filled = within + post
    sign = side.value
    residual = target - filled
    shortfall = sign * (value - arrival * filled) + fee_total - rebate_total
    return {"cash": snap.cash, "inventory": snap.inventory, "fees": fee_total, "rebates": rebate_total,
            "realized_fill_value": value, "residual_inventory": residual,
            "residual_mark_value": residual * mark, "implementation_shortfall": shortfall,
            "within_horizon_filled": within, "post_horizon_filled": post,
            "within_horizon_complete": within >= target, "final_complete": filled >= target,
            "equity": snap.equity, "pnl": snap.total_pnl}


def check_horizon_fields(row: dict) -> None:
    missing = [f for f in HORIZON_FIELDS if f not in row]
    if missing:
        raise ValueError(f"execution row collapses horizon semantics; missing {missing}")

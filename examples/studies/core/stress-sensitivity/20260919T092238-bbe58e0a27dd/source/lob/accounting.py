"""Owner-specific cash, inventory, fees and average-cost PnL accounting.

Prices accepted by the ledger are currency. Trade prices remain integer ticks.
Fees are recognised when a fill occurs; maker rebates may be negative. Initial
inventory is endowed at its stated cost basis, never recorded as trading profit.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .engine import Side, Trade


@dataclass(frozen=True)
class FeeConfig:
    maker_bps: float = 0.0
    taker_bps: float = 0.0
    per_share: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.per_share < 0 or self.taker_bps < 0:
            raise ValueError("per_share and taker_bps must be nonnegative")

    def charge(self, price: float, qty: int, *, maker: bool = False) -> float:
        if not math.isfinite(price) or price < 0 or isinstance(qty, bool) or not isinstance(qty, int) or qty < 0:
            raise ValueError("fee price and integer quantity must be finite and nonnegative")
        return price * qty * (self.maker_bps if maker else self.taker_bps) / 1e4 + qty * self.per_share


def fee_config(value: FeeConfig | Mapping[str, Any] | None) -> FeeConfig:
    return value if isinstance(value, FeeConfig) else FeeConfig(**dict(value or {}))


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    inventory: int
    average_cost: float
    gross_realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    realized_pnl: float
    equity: float
    total_pnl: float


class Ledger:
    """A single owner's signed position using moving-average cost basis."""

    def __init__(self, *, inventory: int = 0, average_cost: float = 0.0,
                 cash: float = 0.0, fees: FeeConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(inventory, bool) or not isinstance(inventory, int):
            raise ValueError("inventory must be an integer")
        if not math.isfinite(average_cost) or average_cost < 0 or not math.isfinite(cash):
            raise ValueError("initial cash and cost basis must be finite; basis must be nonnegative")
        if inventory and average_cost <= 0:
            raise ValueError("nonzero initial inventory requires a positive cost basis")
        self.inventory = inventory
        self.average_cost = float(average_cost) if inventory else 0.0
        self.cash = float(cash)
        self.initial_equity = self.cash + inventory * self.average_cost
        self.fees = fee_config(fees)
        self.total_fees = 0.0
        self.gross_realized_pnl = 0.0
        self.filled_qty = 0

    def apply_fill(self, side: Side, price: float, qty: int, *, maker: bool = False) -> float:
        if not isinstance(side, Side) or isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            raise ValueError("fill side and positive integer quantity required")
        if not math.isfinite(price) or price <= 0:
            raise ValueError("fill price must be positive and finite")
        signed_qty = side.value * qty
        old = self.inventory
        updated = old + signed_qty
        if old == 0 or old * signed_qty > 0:
            self.average_cost = (abs(old) * self.average_cost + qty * price) / abs(updated)
        else:
            closing = min(abs(old), qty)
            self.gross_realized_pnl += closing * (price - self.average_cost) * (1 if old > 0 else -1)
            if updated == 0:
                self.average_cost = 0.0
            elif old * updated < 0:
                self.average_cost = price
        fee = self.fees.charge(price, qty, maker=maker)
        self.inventory = updated
        self.cash -= signed_qty * price + fee
        self.total_fees += fee
        self.filled_qty += qty
        return fee

    def apply_trade(self, trade: Trade, owner: str, tick_size: float) -> float:
        if tick_size <= 0 or not math.isfinite(tick_size):
            raise ValueError("tick_size must be positive and finite")
        taker, maker = trade.taker_owner == owner, trade.maker_owner == owner
        if taker == maker:
            raise ValueError("trade must identify owner on exactly one side")
        side = trade.taker_side if taker else trade.taker_side.opposite
        return self.apply_fill(side, trade.price * tick_size, trade.qty, maker=maker)

    def snapshot(self, mark: float) -> AccountSnapshot:
        if not math.isfinite(mark) or mark < 0:
            raise ValueError("mark must be finite and nonnegative")
        unrealized = self.inventory * (mark - self.average_cost)
        equity = self.cash + self.inventory * mark
        pnl = self.gross_realized_pnl + unrealized - self.total_fees
        if not math.isclose(equity - self.initial_equity, pnl, abs_tol=1e-7, rel_tol=1e-10):
            raise ArithmeticError("cash/inventory PnL identity failed")
        return AccountSnapshot(self.cash, self.inventory, self.average_cost,
                               self.gross_realized_pnl, unrealized, self.total_fees,
                               self.gross_realized_pnl - self.total_fees, equity, pnl)

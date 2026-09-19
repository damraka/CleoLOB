"""Pre-trade execution limits with reservations held through cancel latency.

The parent budget is unconditional: executed plus all nonterminal leaves can
never exceed the requested parent. Optional monetary limits are pretrade limits
at conservative visible prices; future marks can exceed them. Buy market orders
become capped IOC limits. Sell proceeds can exceed an estimate when a later fill
price improves: a sell limit cannot enforce an upper bound on sale proceeds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .accounting import Ledger
from .engine import ExchangeSimulator, Side, TimeInForce


@dataclass(frozen=True)
class RiskConfig:
    max_order_qty: int | None = None
    max_order_notional: float | None = None
    max_position: int | None = None
    max_gross_notional: float | None = None
    max_loss: float | None = None
    kill_switch: bool = False

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "kill_switch":
                if not isinstance(value, bool):
                    raise ValueError("kill_switch must be a boolean")
            elif value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
                if name in ("max_order_qty", "max_position") and not isinstance(value, int):
                    raise ValueError(f"{name} must be an integer")


def risk_config(value: RiskConfig | Mapping[str, Any] | None) -> RiskConfig:
    return value if isinstance(value, RiskConfig) else RiskConfig(**dict(value or {}))


class ExecutionRisk:
    def __init__(self, total_qty: int, side: Side, config: RiskConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(total_qty, bool) or not isinstance(total_qty, int) or total_qty <= 0 or not isinstance(side, Side):
            raise ValueError("positive integer total_qty and valid side required")
        self.total_qty, self.side = total_qty, side
        self.config = risk_config(config)
        self.order_ids: list[int] = []
        self.events: list[dict[str, Any]] = []
        self.halted = self.config.kill_switch
        self._reserved_prices: dict[int, float] = {}

    def outstanding(self, sim: ExchangeSimulator) -> int:
        return sum(sim.orders[oid].remaining for oid in self.order_ids if not sim.orders[oid].is_terminal)

    def available(self, sim: ExchangeSimulator, filled: int) -> int:
        if isinstance(filled, bool) or not isinstance(filled, int) or not 0 <= filled <= self.total_qty:
            raise ArithmeticError("filled quantity is outside the parent budget")
        available = self.total_qty - filled - self.outstanding(sim)
        if available < 0:
            raise ArithmeticError("parent execution budget exceeded")
        return available

    def cancel_all(self, sim: ExchangeSimulator, *, reason: str = "cancel") -> None:
        for oid in self.order_ids:
            order = sim.orders[oid]
            if not order.is_terminal and getattr(order.status, "value", "").lower() != "cancel_pending":
                sim.cancel(oid)
                self.events.append({"t": sim.t, "event": reason, "order_id": oid})

    def _reject(self, sim: ExchangeSimulator, reason: str, requested: int) -> None:
        self.events.append({"t": sim.t, "event": "rejected", "reason": reason, "requested_qty": requested})

    def check(self, sim: ExchangeSimulator, ledger: Ledger) -> bool:
        """Latch the loss limit and request cancellation, even without new orders."""
        mark = sim.book.mid() * sim.cfg.tick_size
        if not self.halted and self.config.max_loss is not None and ledger.snapshot(mark).total_pnl <= -self.config.max_loss:
            self.halted = True
            self.events.append({"t": sim.t, "event": "loss_limit", "pnl": ledger.snapshot(mark).total_pnl})
        if self.halted:
            self.cancel_all(sim, reason="kill_switch")
        return not self.halted

    def submit(self, sim: ExchangeSimulator, owner: str, qty: int, filled: int,
               ledger: Ledger, price_ticks: int | None = None) -> int | None:
        if isinstance(qty, bool) or not isinstance(qty, int) or qty < 0:
            raise ValueError("requested quantity must be a nonnegative integer")
        if price_ticks is not None and (isinstance(price_ticks, bool) or not isinstance(price_ticks, int) or price_ticks <= 0):
            raise ValueError("limit price must be positive integer ticks")
        requested = qty
        mark = sim.book.mid() * sim.cfg.tick_size
        if not self.check(sim, ledger):
            self._reject(sim, "kill_switch", requested)
            return None
        qty = min(qty, self.available(sim, filled))
        if self.config.max_order_qty is not None:
            qty = min(qty, self.config.max_order_qty)
        outstanding = self.outstanding(sim)
        projected = ledger.inventory + self.side.value * outstanding
        # Existing oversized inventory may always be reduced toward a limit.
        if self.config.max_position is not None:
            room = self.config.max_position - self.side.value * projected
            qty = min(qty, max(0, room))
        levels = sim.book.ask_prices if self.side is Side.BUY else sim.book.bid_prices
        if price_ticks is None:
            if not levels:
                self._reject(sim, "no_opposite_liquidity", requested)
                return None
            # Upper-bound currency notional even for a sell order.
            risk_ticks = max(levels)
        else:
            risk_ticks = price_ticks if self.side is Side.BUY else max(price_ticks, sim.book.best_bid() or price_ticks)
        risk_price = risk_ticks * sim.cfg.tick_size
        if self.config.max_order_notional is not None:
            qty = min(qty, int(self.config.max_order_notional / risk_price))
        if self.config.max_gross_notional is not None:
            exposure_price = max([risk_price, mark] + [self._reserved_prices[oid] for oid in self.order_ids
                                                      if not sim.orders[oid].is_terminal])
            # Worst position if all prior leaves fill. Orders reducing oversized
            # exposure are permitted, including a crossing only up to the limit.
            max_units = int(self.config.max_gross_notional / exposure_price)
            room = max_units - self.side.value * projected
            qty = min(qty, max(0, room))
        lot = getattr(sim.cfg, "lot_size", 1)
        qty = max(0, int(qty) // lot * lot)
        if qty <= 0:
            self._reject(sim, "execution_budget", requested)
            return None
        kwargs: dict[str, Any] = {}
        if price_ticks is None and self.side is Side.BUY and (
                self.config.max_order_notional is not None or self.config.max_gross_notional is not None):
            price_ticks = risk_ticks
            kwargs["time_in_force"] = TimeInForce.IOC
        oid = sim.submit(self.side, qty, owner, price_ticks=price_ticks, **kwargs)
        self.order_ids.append(oid)
        self._reserved_prices[oid] = risk_price
        self.events.append({"t": sim.t, "event": "submitted", "order_id": oid,
                            "requested_qty": requested, "approved_qty": qty})
        return oid

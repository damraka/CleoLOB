"""Exact quantity, tick, lot and fee accounting (M33).

Every fill is booked in integer ticks and lots; notional is ``price_ticks * lots`` in "tick-lot" units and is
converted to quote currency only for reporting (``tick * lot`` per unit). Fees are basis points of notional, kept as
exact fractions, with a maker rebate allowed (negative maker bps). Positions, cash, realized PnL (average-cost),
volume, fee totals and per-owner fill counts are maintained; ``mark`` values the open position at a mark price.
Cash plus mark-to-market is conserved across counterparties up to fees (tested).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .matching import BUY, Report


@dataclass(frozen=True)
class FeeSchedule:
    maker_bps: Fraction = Fraction(1, 1)        # 1 bp
    taker_bps: Fraction = Fraction(55, 10)      # 5.5 bp

    def fee(self, notional: int, liquidity: str) -> Fraction:
        bps = self.maker_bps if liquidity == "maker" else self.taker_bps
        return Fraction(notional) * Fraction(bps) / 10_000


@dataclass
class Account:
    owner: str
    position: int = 0                       # lots, + long
    cash: Fraction = Fraction(0)            # tick-lot units
    fees: Fraction = Fraction(0)
    volume: int = 0
    maker_volume: int = 0
    taker_volume: int = 0
    fills: int = 0
    avg_cost: Fraction = Fraction(0)        # ticks per lot of the open position
    realized: Fraction = Fraction(0)

    def book(self, side: int, price: int, qty: int, fee: Fraction, liquidity: str) -> None:
        signed = qty if side == BUY else -qty
        self.cash -= Fraction(price * signed) + fee
        self.fees += fee
        self.volume += qty
        self.fills += 1
        if liquidity == "maker":
            self.maker_volume += qty
        else:
            self.taker_volume += qty
        new = self.position + signed
        if self.position == 0 or (self.position > 0) == (signed > 0):
            total = abs(self.position) + qty
            self.avg_cost = (self.avg_cost * abs(self.position) + price * qty) / total
        else:
            closed = min(qty, abs(self.position))
            direction = 1 if self.position > 0 else -1
            self.realized += (Fraction(price) - self.avg_cost) * closed * direction
            if abs(signed) > abs(self.position):
                self.avg_cost = Fraction(price)
            elif new == 0:
                self.avg_cost = Fraction(0)
        self.position = new

    def mark(self, mark_price: int) -> Fraction:
        return self.cash + Fraction(self.position * mark_price)


@dataclass
class Accounting:
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    tick: float = 1.0
    lot: float = 1.0
    accounts: dict[str, Account] = field(default_factory=dict)
    side_of: dict[int, int] = field(default_factory=dict)          # order id -> side

    def account(self, owner: str) -> Account:
        return self.accounts.setdefault(owner, Account(owner))

    def register(self, oid: int, side: int) -> None:
        self.side_of[oid] = side

    def on_reports(self, reports: list[Report]) -> None:
        for r in reports:
            if r.kind != "fill":
                continue
            side = self.side_of[r.oid]
            notional = r.price * r.qty
            self.account(r.owner).book(side, r.price, r.qty, self.fees.fee(notional, r.liquidity), r.liquidity)

    def quote(self, units: Fraction) -> float:
        return float(units) * self.tick * self.lot

    def summary(self, mark_price: int | None = None) -> dict:
        out = {}
        for k, a in sorted(self.accounts.items()):
            out[k] = {"position_lots": a.position, "cash_quote": self.quote(a.cash), "fees_quote": self.quote(a.fees),
                      "volume_lots": a.volume, "maker_lots": a.maker_volume, "taker_lots": a.taker_volume,
                      "fills": a.fills, "realized_quote": self.quote(a.realized)}
            if mark_price is not None:
                out[k]["equity_quote"] = self.quote(a.mark(mark_price))
        return out


__all__ = ["FeeSchedule", "Account", "Accounting"]

"""Venue specification layer (workstream 3): declared exchange rules and order validation.

``ExchangeRules`` states what a venue's matching engine accepts: tick and lot
grids, minimum size, supported order types and time-in-force, post-only
behaviour, self-trade prevention and fees. Rules for real venues are declared
from public documentation and are *assumptions about* those venues, not
verified exchange truth; simulator rules are exact by construction. Orders are
validated before submission, and rules convert to an ``lob.engine.SimConfig``
fragment so simulated venues share one declaration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json

ORDER_TYPES = ("limit", "market")
TIME_IN_FORCE = ("GTC", "IOC", "FOK", "GTD")
STP_MODES = ("none", "cancel_taker", "cancel_maker", "cancel_both")


class VenueRuleError(ValueError):
    """An order violates the declared venue rules."""


@dataclass(frozen=True)
class ExchangeRules:
    venue: str
    instrument: str
    tick: Decimal
    lot: Decimal
    min_qty: Decimal
    order_types: tuple[str, ...] = ORDER_TYPES
    time_in_force: tuple[str, ...] = ("GTC", "IOC", "FOK")
    post_only: bool = True
    post_only_cross: str = "reject"           # "reject" | "slide"
    self_trade_prevention: str = "none"
    maker_fee_bps: float = 0.0                # mandate fee convention (v0.6), not a venue fee schedule
    taker_fee_bps: float = 1.0
    matching: str = "price-time priority (declared)"
    source: str = "declaration"
    verified: bool = False                    # True only for simulator venues

    def __post_init__(self) -> None:
        if self.tick <= 0 or self.lot <= 0 or self.min_qty <= 0:
            raise VenueRuleError("tick, lot and minimum size must be positive")
        if not set(self.order_types) <= set(ORDER_TYPES) or not set(self.time_in_force) <= set(TIME_IN_FORCE):
            raise VenueRuleError("unknown order type or time in force")
        if self.self_trade_prevention not in STP_MODES or self.post_only_cross not in {"reject", "slide"}:
            raise VenueRuleError("unknown self-trade prevention or post-only mode")

    def sha256(self) -> str:
        values = {k: (str(v) if isinstance(v, Decimal) else v) for k, v in asdict(self).items()}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def validate(self, *, side: str, qty: Decimal, price: Decimal | None, order_type: str = "limit",
                 tif: str = "GTC", post_only: bool = False, best_opposite: Decimal | None = None) -> dict:
        """Return the accepted order (possibly with a post-only slide) or raise ``VenueRuleError``."""
        if side not in {"buy", "sell"}:
            raise VenueRuleError("side must be buy or sell")
        if order_type not in self.order_types or tif not in self.time_in_force:
            raise VenueRuleError(f"{order_type}/{tif} not supported on {self.venue}")
        if qty < self.min_qty or qty % self.lot:
            raise VenueRuleError(f"quantity {qty} violates lot {self.lot} / minimum {self.min_qty}")
        if order_type == "limit":
            if price is None or price <= 0 or price % self.tick:
                raise VenueRuleError(f"price {price} is not on the {self.tick} tick grid")
        elif price is not None or post_only:
            raise VenueRuleError("market orders take no price and cannot be post-only")
        if post_only and not self.post_only:
            raise VenueRuleError("post-only is not supported")
        if post_only and best_opposite is not None:
            crosses = price >= best_opposite if side == "buy" else price <= best_opposite
            if crosses:
                if self.post_only_cross == "reject":
                    raise VenueRuleError("post-only order would cross")
                price = best_opposite - self.tick if side == "buy" else best_opposite + self.tick
        return {"side": side, "qty": qty, "price": price, "order_type": order_type, "tif": tif,
                "post_only": post_only}

    def fee(self, notional: float, *, maker: bool) -> float:
        return notional * (self.maker_fee_bps if maker else self.taker_fee_bps) / 1e4

    def sim_config(self) -> dict:
        """``SimConfig`` fields implied by the rules (tick in native units; lot = 1 simulator lot)."""
        return {"tick_size": float(self.tick), "lot_size": 1}


VENUE_RULES = {
    ("deribit", "ETH-PERPETUAL"): ExchangeRules("deribit", "ETH-PERPETUAL", Decimal("0.05"), Decimal("1"), Decimal("1"),
                                                source="public contract specification (declared, unverified)"),
    ("deribit", "BTC-PERPETUAL"): ExchangeRules("deribit", "BTC-PERPETUAL", Decimal("0.5"), Decimal("10"),
                                                Decimal("10"), source="public contract specification (declared, "
                                                                      "unverified)"),
    ("bitmex", "XBTUSD"): ExchangeRules("bitmex", "XBTUSD", Decimal("0.5"), Decimal("1"), Decimal("1"),
                                        source="public contract specification (declared, unverified)"),
    ("cleolob", "SIM"): ExchangeRules("cleolob", "SIM", Decimal("1"), Decimal("1"), Decimal("1"),
                                      time_in_force=("GTC", "IOC", "FOK", "GTD"), source="lob.engine (exact)",
                                      verified=True),
}


def rules(venue: str, instrument: str) -> ExchangeRules:
    try:
        return VENUE_RULES[(venue, instrument)]
    except KeyError as exc:
        raise VenueRuleError(f"no declared rules for {venue}/{instrument}") from exc

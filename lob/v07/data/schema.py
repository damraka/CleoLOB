"""Canonical v0.7 market-data schema: venue specifications, record types and capability levels.

Every adapter emits ``Record`` objects in exact decimal native units together
with the ``VenueSpec`` that interprets them. A record never carries a field its
source does not provide: aggregate L2 records have no order identity, and an
adapter must not synthesize one. ``capability_level`` decides which downstream
analyses may run (``lob.v07.data.capability``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import json
from typing import Any

from ...capabilities import CANONICAL_MBO_CONTRACT, L2_CONTRACT, Capability, CapabilityContract, DataLevel

SCHEMA_VERSION = "cleolob-v07-canonical-1"
RECORD_KINDS = ("book_snapshot", "book_delta", "trade", "order_add", "order_cancel", "order_modify",
                "order_execute", "gap")
L2_KINDS = frozenset({"book_snapshot", "book_delta", "trade", "gap"})
MBO_KINDS = frozenset(RECORD_KINDS)
SIDES = ("bid", "ask", "buy", "sell")


L2_TRADES_CONTRACT = CapabilityContract(DataLevel.L2, L2_CONTRACT.available | {Capability.TRADE_PRINTS},
                                        "aggregate L2 price levels plus trade prints; no order identity or FIFO")


class SchemaError(ValueError):
    """A record or venue specification violates the canonical schema."""


@dataclass(frozen=True)
class VenueSpec:
    """Static interpretation of one venue/instrument. Values are declarations, checked against data by quality."""

    venue: str
    instrument: str
    tick: Decimal
    amount_unit: str          # what a native ``amount`` counts (e.g. "USD contracts")
    contract_type: str        # "inverse_perpetual", "linear_perpetual", "spot"
    underlying: str
    capability_level: str     # "aggregate_l2" | "order_level_mbo"
    timestamp_unit: str = "microseconds since epoch (exchange and local capture)"
    fifo_established: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tick <= 0:
            raise SchemaError("tick must be positive")
        if self.capability_level not in {"aggregate_l2", "order_level_mbo"}:
            raise SchemaError(f"unknown capability level {self.capability_level!r}")

    @property
    def contract(self) -> CapabilityContract:
        if self.capability_level == "aggregate_l2":
            return L2_TRADES_CONTRACT
        if not self.fifo_established:
            # Order identities exist, but the feed does not establish priority: no FIFO-derived capability.
            return CANONICAL_MBO_CONTRACT.restrict(CANONICAL_MBO_CONTRACT.available - {
                Capability.EXACT_FIFO_POSITION, Capability.QUANTITY_AHEAD, Capability.SNAPSHOT_ORDERING})
        return CANONICAL_MBO_CONTRACT

    def to_dict(self) -> dict:
        out = {k: (str(v) if isinstance(v, Decimal) else v) for k, v in asdict(self).items()}
        out["sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
        return out


VENUES: dict[tuple[str, str], VenueSpec] = {
    ("deribit", "ETH-PERPETUAL"): VenueSpec("deribit", "ETH-PERPETUAL", Decimal("0.05"), "USD (contract face value)",
                                            "inverse_perpetual", "ETH", "aggregate_l2"),
    ("deribit", "BTC-PERPETUAL"): VenueSpec("deribit", "BTC-PERPETUAL", Decimal("0.5"), "USD (contract face value)",
                                            "inverse_perpetual", "BTC", "aggregate_l2"),
    ("bitmex", "XBTUSD"): VenueSpec("bitmex", "XBTUSD", Decimal("0.5"), "USD contracts (1 USD each)",
                                    "inverse_perpetual", "BTC", "aggregate_l2",
                                    notes="Tardis normalizes BitMEX orderBookL2 updates to price levels; BitMEX "
                                          "level ids are not exposed and are not order identities"),
    ("bitstamp", "btcusd"): VenueSpec("bitstamp", "btcusd", Decimal("1"), "BTC", "spot", "BTC", "order_level_mbo",
                                      notes="self-recorded public websocket capture (v0.5); not a vendor archive"),
}


def venue_spec(venue: str, instrument: str) -> VenueSpec:
    try:
        return VENUES[(venue, instrument)]
    except KeyError as exc:
        raise SchemaError(f"no registered venue specification for {venue}/{instrument}") from exc


@dataclass(frozen=True)
class Record:
    """One canonical record. ``order_id`` only for order-level sources; ``price``/``amount`` exact decimals."""

    kind: str
    timestamp_us: int
    local_timestamp_us: int
    side: str | None = None
    price: Decimal | None = None
    amount: Decimal | None = None
    order_id: str | None = None
    trade_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, level: str) -> Record:
        if self.kind not in RECORD_KINDS:
            raise SchemaError(f"unknown record kind {self.kind!r}")
        allowed = L2_KINDS if level == "aggregate_l2" else MBO_KINDS
        if self.kind not in allowed:
            raise SchemaError(f"{self.kind} records are not available from {level}")
        if level == "aggregate_l2" and self.order_id is not None:
            raise SchemaError("aggregate L2 records cannot carry an order identity")
        if self.kind.startswith("order_") and not self.order_id:
            raise SchemaError(f"{self.kind} requires an order id")
        if self.side is not None and self.side not in SIDES:
            raise SchemaError(f"unknown side {self.side!r}")
        if self.kind in {"book_snapshot", "book_delta"}:
            if self.side not in {"bid", "ask"} or self.price is None or self.amount is None:
                raise SchemaError("book records need side bid/ask, price and amount")
            if self.price <= 0 or self.amount < 0:
                raise SchemaError("book prices must be positive and amounts nonnegative")
        if self.kind == "trade":
            if self.side not in {"buy", "sell", None} or self.price is None or self.amount is None:
                raise SchemaError("trades need price, amount and an aggressor side (buy/sell) or None")
            if self.price <= 0 or self.amount <= 0:
                raise SchemaError("trade price and amount must be positive")
        if self.timestamp_us < 0 or self.local_timestamp_us < 0:
            raise SchemaError("timestamps must be nonnegative")
        return self

    def canonical(self) -> str:
        values = {k: (str(v) if isinstance(v, Decimal) else v) for k, v in asdict(self).items()}
        return json.dumps(values, sort_keys=True, separators=(",", ":"))


def digest(records) -> str:
    """Order-sensitive digest of canonical records (deterministic replay identity)."""
    h = hashlib.sha256()
    for record in records:
        h.update(record.canonical().encode())
        h.update(b"\n")
    return h.hexdigest()

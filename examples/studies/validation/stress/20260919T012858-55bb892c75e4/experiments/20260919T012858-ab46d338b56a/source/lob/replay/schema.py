"""Strict, venue-neutral order-level event schema.

Prices are positive integer ticks and timestamps are nonnegative integer
nanoseconds. An adapter must resolve venue-specific meanings before constructing
these events; coercing floating point prices or timestamps is intentionally unsafe.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class EventType(str, Enum):
    ADD = "ADD"
    CANCEL = "CANCEL"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    EXECUTE = "EXECUTE"
    TRADE = "TRADE"
    SNAPSHOT = "SNAPSHOT"
    HALT = "HALT"
    RESUME = "RESUME"


class ReplaySide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class EventSchemaError(ValueError):
    """A malformed canonical event, with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    # Strings support CSV while rejecting decimals, exponents and implicit rounding.
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        try:
            value = int(value)
        except ValueError as exc:
            raise EventSchemaError("INVALID_INTEGER", f"{name} is too large") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EventSchemaError("INVALID_INTEGER", f"{name} must be an integer >= {minimum}")
    if value > 2**63 - 1:
        raise EventSchemaError("INVALID_INTEGER", f"{name} exceeds signed 64-bit range")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EventSchemaError("MISSING_ID", f"{name} must be a nonempty, unpadded string")
    if len(value) > 128 or any(ord(char) < 32 for char in value):
        raise EventSchemaError("INVALID_ID", f"{name} has invalid length or control characters")
    return value


def _side(value: Any) -> ReplaySide:
    try:
        return ReplaySide(value)
    except (TypeError, ValueError) as exc:
        raise EventSchemaError("INVALID_SIDE", "side must be BUY or SELL") from exc


@dataclass(frozen=True)
class SnapshotOrder:
    order_id: str
    side: ReplaySide
    price_ticks: int
    quantity: int

    def __post_init__(self) -> None:
        _identifier(self.order_id, "order_id")
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "price_ticks", _integer(self.price_ticks, "price_ticks"))
        object.__setattr__(self, "quantity", _integer(self.quantity, "quantity"))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SnapshotOrder":
        if not isinstance(row, Mapping) or set(row) != {"order_id", "side", "price_ticks", "quantity"}:
            raise EventSchemaError("INVALID_SNAPSHOT", "each snapshot order requires order_id, side, price_ticks, quantity")
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"order_id": self.order_id, "side": self.side.value,
                "price_ticks": self.price_ticks, "quantity": self.quantity}


@dataclass(frozen=True)
class MarketEvent:
    timestamp_ns: int
    sequence: int
    event_type: EventType
    symbol: str
    venue: str
    order_id: str | None = None
    side: ReplaySide | None = None
    price_ticks: int | None = None
    quantity: int | None = None
    orders: tuple[SnapshotOrder, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _integer(self.timestamp_ns, "timestamp_ns", minimum=0))
        object.__setattr__(self, "sequence", _integer(self.sequence, "sequence", minimum=0))
        try:
            object.__setattr__(self, "event_type", EventType(self.event_type))
        except (TypeError, ValueError) as exc:
            raise EventSchemaError("INVALID_EVENT_TYPE", "unknown canonical event_type") from exc
        _identifier(self.symbol, "symbol")
        _identifier(self.venue, "venue")
        if self.order_id is not None:
            _identifier(self.order_id, "order_id")
        if self.side is not None:
            object.__setattr__(self, "side", _side(self.side))
        if self.price_ticks is not None:
            object.__setattr__(self, "price_ticks", _integer(self.price_ticks, "price_ticks"))
        if self.quantity is not None:
            object.__setattr__(self, "quantity", _integer(self.quantity, "quantity"))
        if not isinstance(self.orders, (tuple, list)) or any(not isinstance(o, SnapshotOrder) for o in self.orders):
            raise EventSchemaError("INVALID_SNAPSHOT", "orders must contain SnapshotOrder values")
        if len(self.orders) > 100_000:
            raise EventSchemaError("RESOURCE_LIMIT", "snapshot exceeds 100000 orders")
        object.__setattr__(self, "orders", tuple(self.orders))

        kind = self.event_type
        order_events = {EventType.ADD, EventType.CANCEL, EventType.DELETE, EventType.MODIFY, EventType.EXECUTE}
        if kind in order_events and self.order_id is None:
            raise EventSchemaError("MISSING_ID", f"{kind.value} requires order_id")
        if kind == EventType.ADD and (self.side is None or self.price_ticks is None or self.quantity is None):
            raise EventSchemaError("MISSING_FIELD", "ADD requires side, price_ticks and quantity")
        if kind in {EventType.CANCEL, EventType.EXECUTE, EventType.TRADE} and self.quantity is None:
            raise EventSchemaError("MISSING_FIELD", f"{kind.value} requires quantity")
        if kind == EventType.TRADE and self.price_ticks is None:
            raise EventSchemaError("MISSING_FIELD", "TRADE requires price_ticks")
        if kind == EventType.DELETE and self.quantity is not None:
            raise EventSchemaError("UNEXPECTED_FIELD", "DELETE removes the whole order; omit quantity")
        if kind == EventType.MODIFY and self.quantity is None and self.price_ticks is None:
            raise EventSchemaError("MISSING_FIELD", "MODIFY requires new quantity or price_ticks")
        if kind not in order_events and self.order_id is not None:
            raise EventSchemaError("UNEXPECTED_FIELD", f"{kind.value} must not specify order_id")
        if kind in {EventType.HALT, EventType.RESUME, EventType.SNAPSHOT}:
            if any(v is not None for v in (self.side, self.price_ticks, self.quantity)):
                raise EventSchemaError("UNEXPECTED_FIELD", f"{kind.value} must not specify side, price_ticks or quantity")
        if kind != EventType.SNAPSHOT and self.orders:
            raise EventSchemaError("UNEXPECTED_FIELD", "only SNAPSHOT may carry orders")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "MarketEvent":
        if not isinstance(row, Mapping):
            raise EventSchemaError("CORRUPT_ROW", "each event must be an object")
        fields = {"timestamp_ns", "sequence", "event_type", "symbol", "venue", "order_id",
                  "side", "price_ticks", "quantity", "orders"}
        if set(row) - fields:
            raise EventSchemaError("UNKNOWN_FIELD", f"unknown event fields: {sorted(str(k) for k in set(row) - fields)}")
        clean = {key: value for key, value in row.items() if value is not None and value != ""}
        missing = {"timestamp_ns", "sequence", "event_type", "symbol", "venue"} - set(clean)
        if missing:
            raise EventSchemaError("MISSING_FIELD", f"missing required fields: {sorted(missing)}")
        if "orders" in clean:
            if clean["event_type"] != EventType.SNAPSHOT:
                raise EventSchemaError("UNEXPECTED_FIELD", "only SNAPSHOT may carry orders")
            if not isinstance(clean["orders"], list) or len(clean["orders"]) > 100_000:
                raise EventSchemaError("INVALID_SNAPSHOT", "orders must be a list of at most 100000 order objects")
            clean["orders"] = tuple(SnapshotOrder.from_mapping(item) for item in clean["orders"])
        elif clean["event_type"] == EventType.SNAPSHOT:
            raise EventSchemaError("MISSING_FIELD", "SNAPSHOT requires an explicit orders list, which may be empty")
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"timestamp_ns": self.timestamp_ns, "sequence": self.sequence,
                                  "event_type": self.event_type.value, "symbol": self.symbol, "venue": self.venue}
        for name in ("order_id", "side", "price_ticks", "quantity"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value.value if isinstance(value, ReplaySide) else value
        if self.event_type == EventType.SNAPSHOT:
            result["orders"] = [order.to_dict() for order in self.orders]
        return result

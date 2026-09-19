"""Exact reconstruction of a canonical historical order-level feed.

This is deliberately separate from the speculative matching engine. An EXECUTE
decrements the recorded maker ID, including a maker away from the best quote;
it never searches for a different maker or creates an unobserved execution.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .schema import EventType, MarketEvent, ReplaySide, SnapshotOrder


class ReconstructionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RestingOrder:
    order_id: str
    side: ReplaySide
    price_ticks: int
    quantity: int
    priority: int


@dataclass(frozen=True)
class HistoricalExecution:
    timestamp_ns: int
    sequence: int
    order_id: str
    side: ReplaySide
    price_ticks: int
    quantity: int

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp_ns": self.timestamp_ns, "sequence": self.sequence,
                "order_id": self.order_id, "side": self.side.value,
                "price_ticks": self.price_ticks, "quantity": self.quantity}


class HistoricalBook:
    """One instrument/venue, uncrossed continuous book, and exact order IDs.

    Queue priority is explicit: same-price size reductions retain position;
    price changes and size increases lose it. Adapters for venues with different
    amendment rules must translate their records to DELETE/ADD or snapshots.
    All rejected transitions leave book state unchanged.
    """

    def __init__(self) -> None:
        self._orders: dict[str, RestingOrder] = {}
        self._levels: dict[ReplaySide, dict[int, OrderedDict[str, None]]] = {
            ReplaySide.BUY: {}, ReplaySide.SELL: {}}
        self._seen_ids: set[str] = set()
        self._priority = 0
        self._last_event: MarketEvent | None = None
        self.symbol: str | None = None
        self.venue: str | None = None
        self.halted = False
        self.executions: list[HistoricalExecution] = []
        self.prints: list[MarketEvent] = []

    @property
    def orders(self) -> dict[str, RestingOrder]:
        """A copy prevents consumers from silently corrupting reconstruction."""
        return dict(self._orders)

    def best_bid(self) -> int | None:
        return max(self._levels[ReplaySide.BUY], default=None)

    def best_ask(self) -> int | None:
        return min(self._levels[ReplaySide.SELL], default=None)

    def depth(self, levels: int | None = None) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        if levels is not None and (isinstance(levels, bool) or not isinstance(levels, int) or levels < 0):
            raise ValueError("levels must be a nonnegative integer or None")
        result = []
        for side in (ReplaySide.BUY, ReplaySide.SELL):
            prices = sorted(self._levels[side], reverse=side == ReplaySide.BUY)
            result.append([(p, sum(self._orders[oid].quantity for oid in self._levels[side][p]))
                           for p in (prices if levels is None else prices[:levels])])
        return result[0], result[1]

    def queue(self, side: ReplaySide | str, price_ticks: int) -> tuple[str, ...]:
        return tuple(self._levels[ReplaySide(side)].get(price_ticks, {}))

    def _check_cross(self, side: ReplaySide, price: int) -> None:
        opposite = self.best_ask() if side == ReplaySide.BUY else self.best_bid()
        if opposite is not None and ((side == ReplaySide.BUY and price >= opposite)
                                     or (side == ReplaySide.SELL and price <= opposite)):
            raise ReconstructionError("CROSSED_BOOK", "event would lock or cross the continuous book")

    def _add(self, order_id: str, side: ReplaySide, price: int, quantity: int) -> None:
        self._priority += 1
        self._orders[order_id] = RestingOrder(order_id, side, price, quantity, self._priority)
        self._levels[side].setdefault(price, OrderedDict())[order_id] = None
        self._seen_ids.add(order_id)

    def _remove(self, order: RestingOrder) -> None:
        level = self._levels[order.side][order.price_ticks]
        del level[order.order_id]
        if not level:
            del self._levels[order.side][order.price_ticks]
        del self._orders[order.order_id]

    def _replace_quantity(self, order: RestingOrder, quantity: int) -> None:
        if quantity == 0:
            self._remove(order)
        else:
            self._orders[order.order_id] = RestingOrder(
                order.order_id, order.side, order.price_ticks, quantity, order.priority)

    def _load_snapshot(self, orders: tuple[SnapshotOrder, ...]) -> None:
        replacement = HistoricalBook()
        for order in orders:
            if order.order_id in replacement._seen_ids:
                raise ReconstructionError("DUPLICATE_ORDER_ID", f"duplicate snapshot order {order.order_id!r}")
            replacement._check_cross(order.side, order.price_ticks)
            replacement._add(order.order_id, order.side, order.price_ticks, order.quantity)
        self._orders = replacement._orders
        self._levels = replacement._levels
        self._priority = replacement._priority
        self._seen_ids.update(replacement._seen_ids)

    def apply(self, event: MarketEvent) -> None:
        """Apply exactly one event atomically, rejecting impossible transitions."""
        if not isinstance(event, MarketEvent):
            raise TypeError("apply requires a MarketEvent")
        last = self._last_event
        if last is not None:
            if (event.symbol, event.venue) != (self.symbol, self.venue):
                raise ReconstructionError("MIXED_STREAM", "one reconstruction requires one symbol and venue")
            if event.sequence <= last.sequence:
                raise ReconstructionError("DUPLICATE_SEQUENCE" if event.sequence == last.sequence else "SEQUENCE_REVERSAL",
                                          "sequence must be strictly increasing")
            if event.sequence != last.sequence + 1:
                raise ReconstructionError("SEQUENCE_GAP", f"expected sequence {last.sequence + 1}, got {event.sequence}")
            if event.timestamp_ns < last.timestamp_ns:
                raise ReconstructionError("TIMESTAMP_REVERSAL", "timestamp_ns must not decrease")

        kind = event.event_type
        if kind == EventType.SNAPSHOT:
            self._load_snapshot(event.orders)
        elif kind in {EventType.HALT, EventType.RESUME}:
            if self.halted == (kind == EventType.HALT):
                raise ReconstructionError("INVALID_HALT_TRANSITION", f"cannot {kind.value} in the current trading state")
            self.halted = kind == EventType.HALT
        elif kind == EventType.TRADE:
            # Prints may report auctions, off-book trades or delayed transactions.
            self.prints.append(event)
        else:
            if self.halted and kind in {EventType.ADD, EventType.MODIFY, EventType.EXECUTE}:
                raise ReconstructionError("EVENT_DURING_HALT", f"{kind.value} is disallowed during a halt")
            if kind == EventType.ADD:
                if event.order_id in self._seen_ids:
                    raise ReconstructionError("DUPLICATE_ORDER_ID", f"order ID {event.order_id!r} has already been used")
                assert event.side is not None and event.price_ticks is not None and event.quantity is not None
                self._check_cross(event.side, event.price_ticks)
                self._add(event.order_id, event.side, event.price_ticks, event.quantity)  # type: ignore[arg-type]
            else:
                order = self._orders.get(event.order_id)  # type: ignore[arg-type]
                if order is None:
                    raise ReconstructionError("UNKNOWN_ORDER_ID", f"no resting order {event.order_id!r}")
                if event.side is not None and event.side != order.side:
                    raise ReconstructionError("SIDE_MISMATCH", "event side does not match resting order")
                if kind != EventType.MODIFY and event.price_ticks is not None and event.price_ticks != order.price_ticks:
                    raise ReconstructionError("PRICE_MISMATCH", "event price does not match resting order")
                if kind == EventType.DELETE:
                    self._remove(order)
                elif kind in {EventType.CANCEL, EventType.EXECUTE}:
                    assert event.quantity is not None
                    if event.quantity > order.quantity:
                        raise ReconstructionError("EXCESS_QUANTITY", "event quantity exceeds resting quantity")
                    self._replace_quantity(order, order.quantity - event.quantity)
                    if kind == EventType.EXECUTE:
                        self.executions.append(HistoricalExecution(event.timestamp_ns, event.sequence, order.order_id,
                                                                    order.side, order.price_ticks, event.quantity))
                elif kind == EventType.MODIFY:
                    price = order.price_ticks if event.price_ticks is None else event.price_ticks
                    quantity = order.quantity if event.quantity is None else event.quantity
                    self._check_cross(order.side, price)
                    if price != order.price_ticks or quantity > order.quantity:
                        self._remove(order)
                        self._add(order.order_id, order.side, price, quantity)
                    else:
                        self._replace_quantity(order, quantity)
        self.symbol, self.venue = event.symbol, event.venue
        self._last_event = event

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe current state; all prices remain integer ticks."""
        bids, asks = self.depth()
        orders = sorted(self._orders.values(), key=lambda order: order.priority)
        return {"symbol": self.symbol, "venue": self.venue, "halted": self.halted,
                "timestamp_ns": None if self._last_event is None else self._last_event.timestamp_ns,
                "sequence": None if self._last_event is None else self._last_event.sequence,
                "bids": [[p, q] for p, q in bids], "asks": [[p, q] for p, q in asks],
                "orders": [{"order_id": o.order_id, "side": o.side.value, "price_ticks": o.price_ticks,
                            "quantity": o.quantity, "priority": o.priority} for o in orders]}

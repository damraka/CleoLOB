"""Source-native order-lifecycle replay that classifies anomalies instead of repairing them.

``lob.replay.book.HistoricalBook`` is deliberately fail-closed: the first impossible
transition stops reconstruction. Genuine venue captures contain censoring, boundary
ambiguity and occasional unknown identities, so this module replays the same
identity semantics while *counting* each anomaly with a stable code. Anomalous
events are never applied as guesses: an unknown cancel does not remove liquidity
that was never observed, an excess execution is not clipped into a fill, and a
duplicate ADD does not overwrite the live order.

Evidence levels stay explicit. Priority order is tracked, but FIFO position is
only exposed when the declared semantics establish source priority.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

KINDS = ("ADD", "MODIFY", "CANCEL", "EXECUTE", "TRADE", "SNAPSHOT", "RESET", "GAP")
SIDES = ("BUY", "SELL")
PRIORITY_RULES = ("reduction_keeps_increase_or_reprice_resets", "any_modify_resets")
SNAPSHOT_MODES = ("reset", "check")
ANOMALIES = (
    "DUPLICATE_ORDER_ID_LIVE", "DUPLICATE_ORDER_ID_REUSED", "UNKNOWN_MODIFY", "UNKNOWN_CANCEL",
    "UNKNOWN_EXECUTE", "MODIFY_AFTER_COMPLETION", "CANCEL_AFTER_COMPLETION",
    "EXECUTE_AFTER_COMPLETION", "EXCESS_CANCEL", "EXCESS_EXECUTION", "SIDE_MISMATCH",
    "EXECUTION_PRICE_MISMATCH", "DUPLICATE_SOURCE_SEQUENCE", "SOURCE_SEQUENCE_GAP",
    "SOURCE_SEQUENCE_REVERSAL", "EXCHANGE_TIMESTAMP_INVERSION", "RECEIVE_TIMESTAMP_INVERSION",
    "CROSSED_BOOK_AFTER_EVENT", "EVENT_BEFORE_INITIAL_CENSUS", "EVENT_AFTER_GAP",
    "LEFT_CENSORED_UNKNOWN_ID",
)
# Codes whose occurrence is explained by declared censoring rather than contradiction.
CENSORING_CODES = frozenset({"LEFT_CENSORED_UNKNOWN_ID", "EVENT_AFTER_GAP", "EVENT_BEFORE_INITIAL_CENSUS"})
# Clock-order observations: reported, but not contradictions of an order's lifecycle.
ORDERING_CODES = frozenset({"EXCHANGE_TIMESTAMP_INVERSION", "RECEIVE_TIMESTAMP_INVERSION"})
CROSSING_RULES = ("anomaly", "transient_aggressor")


class LifecycleInputError(ValueError):
    """The normalized stream itself is malformed; replay outcome is INVALID."""


@dataclass(frozen=True)
class CensusOrder:
    order_id: str
    side: str
    price: int
    quantity: int


@dataclass(frozen=True)
class LifecycleEvent:
    """Normalized event in exact integer ticks and lots.

    ``quantity`` means: ADD resting quantity; MODIFY new remaining quantity (or
    None for price-only); CANCEL quantity removed (None removes the remainder);
    EXECUTE executed maker quantity; TRADE printed quantity. ``sequence`` is the
    adapter's contiguous normalized order; ``source_sequence`` is the venue's own
    sequence when one exists.
    """

    sequence: int
    kind: str
    exchange_ts_ns: int | None = None
    receive_ts_ns: int | None = None
    source_sequence: int | None = None
    order_id: str | None = None
    side: str | None = None
    price: int | None = None
    quantity: int | None = None
    orders: tuple[CensusOrder, ...] = ()
    snapshot_mode: str = "reset"
    maker_order_id: str | None = None
    taker_order_id: str | None = None
    aggressor_side: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise LifecycleInputError(f"unknown lifecycle kind {self.kind!r}")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise LifecycleInputError("sequence must be a nonnegative integer")
        if self.kind in {"ADD", "MODIFY", "CANCEL", "EXECUTE"} and not self.order_id:
            raise LifecycleInputError(f"{self.kind} requires order_id")
        if self.kind == "ADD" and (self.side not in SIDES or not _positive(self.price) or not _positive(self.quantity)):
            raise LifecycleInputError("ADD requires side, positive price and positive quantity")
        if self.kind in {"EXECUTE", "TRADE"} and not _positive(self.quantity):
            raise LifecycleInputError(f"{self.kind} requires positive quantity")
        if self.kind == "MODIFY" and self.quantity is None and self.price is None:
            raise LifecycleInputError("MODIFY requires quantity or price")
        if self.kind == "MODIFY" and self.quantity is not None and (isinstance(self.quantity, bool) or self.quantity < 0):
            raise LifecycleInputError("MODIFY quantity must be nonnegative")
        if self.kind == "CANCEL" and self.quantity is not None and not _positive(self.quantity):
            raise LifecycleInputError("CANCEL quantity must be positive when present")
        if self.kind == "SNAPSHOT" and self.snapshot_mode not in SNAPSHOT_MODES:
            raise LifecycleInputError("snapshot_mode must be reset or check")
        if self.kind != "SNAPSHOT" and self.orders:
            raise LifecycleInputError("only SNAPSHOT carries orders")


def _positive(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class LifecycleSemantics:
    """Declared source semantics; declarations are obligations, not evidence."""

    priority_rule: str = "reduction_keeps_increase_or_reprice_resets"
    source_sequence_contiguous: bool = False
    census_complete: bool = True
    fifo_established: bool = False
    # "transient_aggressor": an ADD priced through the opposite best is an incoming
    # aggressive order. It is held outside the book; its executions are taker fills;
    # any remainder joins the book (with fresh priority) once it no longer crosses.
    crossing_add: str = "anomaly"

    def __post_init__(self) -> None:
        if self.priority_rule not in PRIORITY_RULES:
            raise ValueError("unknown priority rule")
        if self.crossing_add not in CROSSING_RULES:
            raise ValueError("unknown crossing rule")

    def to_dict(self) -> dict:
        return {"priority_rule": self.priority_rule,
                "source_sequence_contiguous": self.source_sequence_contiguous,
                "census_complete": self.census_complete, "fifo_established": self.fifo_established,
                "crossing_add": self.crossing_add}


@dataclass
class _Order:
    order_id: str
    side: str
    price: int
    quantity: int
    priority: int
    birth_ns: int | None
    initial_quantity: int
    executed: int = 0
    cancelled: int = 0


@dataclass
class Execution:
    sequence: int
    exchange_ts_ns: int | None
    order_id: str
    side: str
    price: int
    quantity: int
    remaining_after: int
    maker_birth_ns: int | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LifecycleReport:
    events: int = 0
    kinds: Counter = field(default_factory=Counter)
    anomalies: Counter = field(default_factory=Counter)
    anomaly_examples: dict = field(default_factory=dict)
    partial_executions: int = 0
    full_executions: int = 0
    partial_cancels: int = 0
    full_cancels: int = 0
    modifies_priority_preserved: int = 0
    modifies_priority_reset: int = 0
    modify_to_zero: int = 0
    left_censored_orders: int = 0
    right_censored_orders: int = 0
    censored_by_reset: int = 0
    censored_by_gap: int = 0
    observed_lifetimes_ns: list = field(default_factory=list)
    snapshot_checks: list = field(default_factory=list)
    trades: int = 0
    trades_with_maker_execution: int = 0
    trades_without_maker_execution: int = 0
    maker_executions_without_trade: int = 0
    transient_aggressors: int = 0
    aggressors_rested: int = 0
    taker_fill_events: int = 0
    aggressor_price_reports: int = 0
    maker_level_linkage: dict = field(default_factory=dict)

    def unexplained(self) -> dict:
        """Lifecycle contradictions not explained by declared censoring (clock order excluded)."""
        return {code: n for code, n in sorted(self.anomalies.items())
                if code not in CENSORING_CODES and code not in ORDERING_CODES and n}

    def ordering_observations(self) -> dict:
        return {code: n for code, n in sorted(self.anomalies.items()) if code in ORDERING_CODES and n}

    def to_dict(self) -> dict:
        lifetimes = sorted(self.observed_lifetimes_ns)
        def q(p: float):
            return lifetimes[min(len(lifetimes) - 1, int(p * len(lifetimes)))] if lifetimes else None
        return {"events": self.events, "kinds": dict(sorted(self.kinds.items())),
                "anomalies": dict(sorted(self.anomalies.items())),
                "unexplained_anomalies": self.unexplained(),
                "ordering_observations": self.ordering_observations(),
                "transient_aggressors": self.transient_aggressors, "aggressors_rested": self.aggressors_rested,
                "taker_fill_events": self.taker_fill_events, "aggressor_price_reports": self.aggressor_price_reports,
                "maker_executions_without_trade": self.maker_executions_without_trade,
                "maker_level_linkage": self.maker_level_linkage,
                "anomaly_examples": self.anomaly_examples,
                "partial_executions": self.partial_executions, "full_executions": self.full_executions,
                "partial_cancels": self.partial_cancels, "full_cancels": self.full_cancels,
                "modifies_priority_preserved": self.modifies_priority_preserved,
                "modifies_priority_reset": self.modifies_priority_reset, "modify_to_zero": self.modify_to_zero,
                "left_censored_orders": self.left_censored_orders,
                "right_censored_orders": self.right_censored_orders,
                "censored_by_reset": self.censored_by_reset, "censored_by_gap": self.censored_by_gap,
                "observed_complete_lifetimes": len(lifetimes),
                "lifetime_ns_quantiles": {"p10": q(.1), "p50": q(.5), "p90": q(.9)},
                "snapshot_checks": self.snapshot_checks, "trades": self.trades,
                "trades_with_maker_execution": self.trades_with_maker_execution,
                "trades_without_maker_execution": self.trades_without_maker_execution}


class OrderLifecycleBook:
    """Deterministic identity book with explicit, counted anomaly handling."""

    def __init__(self, semantics: LifecycleSemantics | None = None, *, max_orders: int = 2_000_000,
                 keep_executions: bool = True) -> None:
        self.semantics = semantics or LifecycleSemantics()
        self.max_orders = max_orders
        self.keep_executions = keep_executions
        self._orders: dict[str, _Order] = {}
        self._levels: dict[str, dict[int, OrderedDict[str, None]]] = {"BUY": {}, "SELL": {}}
        self._terminal: dict[str, str] = {}
        self._priority = 0
        self._initialized = False
        self._gap = False
        self._last_sequence: int | None = None
        self._last_source_sequence: int | None = None
        self._last_exchange_ns: int | None = None
        self._last_receive_ns: int | None = None
        # Linkage is order-independent within an identical (maker, exchange time) key.
        self._links: dict[tuple[str, int | None], list[int]] = {}  # executed, printed, print count
        self._aggressors: dict[str, _Order] = {}
        self.executions: list[Execution] = []
        self.report = LifecycleReport()
        self._digest = hashlib.sha256()

    # ------------------------------------------------------------------ queries
    @property
    def initialized(self) -> bool:
        return self._initialized and not self._gap

    def best(self, side: str) -> int | None:
        levels = self._levels[side]
        if not levels:
            return None
        return max(levels) if side == "BUY" else min(levels)

    def aggregate(self, depth: int | None = None) -> dict[str, list[list[int]]]:
        result = {}
        for side, key in (("BUY", "bids"), ("SELL", "asks")):
            prices = sorted(self._levels[side], reverse=side == "BUY")
            prices = prices if depth is None else prices[:depth]
            result[key] = [[p, sum(self._orders[o].quantity for o in self._levels[side][p])] for p in prices]
        return result

    def level_queue(self, side: str, price: int) -> tuple[str, ...]:
        """Tracked priority order; meaningful as FIFO only when semantics establish it."""
        if not self.semantics.fifo_established:
            raise PermissionError("exact FIFO position is not established by the declared source semantics")
        return tuple(self._levels[side].get(price, ()))

    def tracked_level_order(self, side: str, price: int) -> tuple[str, ...]:
        """Order implied by the declared priority rule, for comparison with a source census."""
        return tuple(self._levels[side].get(price, ()))

    def order(self, order_id: str) -> dict | None:
        order = self._orders.get(order_id)
        return None if order is None else {"order_id": order.order_id, "side": order.side, "price": order.price,
                                           "quantity": order.quantity, "birth_ns": order.birth_ns}

    def live_orders(self) -> dict[str, dict]:
        return {oid: self.order(oid) for oid in self._orders}

    # ------------------------------------------------------------------ mutation helpers
    def _anomaly(self, code: str, event: LifecycleEvent) -> None:
        self.report.anomalies[code] += 1
        examples = self.report.anomaly_examples.setdefault(code, [])
        if len(examples) < 3:
            examples.append({"sequence": event.sequence, "order_id": event.order_id,
                             "exchange_ts_ns": event.exchange_ts_ns})

    def _insert(self, order_id: str, side: str, price: int, quantity: int, birth: int | None) -> None:
        if len(self._orders) >= self.max_orders:
            raise LifecycleInputError("lifecycle book exceeds max_orders")
        self._priority += 1
        self._orders[order_id] = _Order(order_id, side, price, quantity, self._priority, birth, quantity)
        self._levels[side].setdefault(price, OrderedDict())[order_id] = None

    def _unlink(self, order: _Order) -> None:
        level = self._levels[order.side][order.price]
        del level[order.order_id]
        if not level:
            del self._levels[order.side][order.price]

    def _terminate(self, order: _Order, reason: str, ts: int | None) -> None:
        self._unlink(order)
        del self._orders[order.order_id]
        self._terminal[order.order_id] = reason
        if reason in {"FILLED", "CANCELLED"} and order.birth_ns is not None and ts is not None:
            self.report.observed_lifetimes_ns.append(ts - order.birth_ns)

    def _censor_all(self, reason: str) -> int:
        count = len(self._orders) + len(self._aggressors)
        for order in list(self._orders.values()):
            self._unlink(order)
            self._terminal[order.order_id] = reason
        for order_id in self._aggressors:
            self._terminal[order_id] = reason
        self._orders.clear()
        self._aggressors.clear()
        return count

    def _crosses(self, side: str, price: int) -> bool:
        opposite = self.best("SELL" if side == "BUY" else "BUY")
        return opposite is not None and (price >= opposite if side == "BUY" else price <= opposite)

    def _aggressor_event(self, event: LifecycleEvent) -> None:
        """Events on an incoming aggressive order: its executions are taker fills, not maker fills."""
        order = self._aggressors[event.order_id]
        if event.kind == "ADD":
            self._anomaly("DUPLICATE_ORDER_ID_LIVE", event)
            return
        if event.kind == "EXECUTE":
            if event.quantity > order.quantity:
                self._anomaly("EXCESS_EXECUTION", event)
                return
            self.report.taker_fill_events += 1
            order.quantity -= event.quantity
        elif event.kind == "CANCEL":
            amount = order.quantity if event.quantity is None else event.quantity
            if amount > order.quantity:
                self._anomaly("EXCESS_CANCEL", event)
                return
            order.quantity -= amount
        elif event.kind == "MODIFY":
            # While sweeping, a venue may report execution prices on the aggressor's own
            # updates; its limit price, which decides when a remainder may rest, is unchanged.
            if event.price is not None and event.price != order.price:
                self.report.aggressor_price_reports += 1
            order.quantity = order.quantity if event.quantity is None else event.quantity
        if order.quantity == 0:
            del self._aggressors[event.order_id]
            self._terminal[event.order_id] = "FILLED" if event.kind == "EXECUTE" else "CANCELLED"

    def _rest_aggressors(self, event: LifecycleEvent) -> None:
        """A remainder that no longer crosses joins the book with priority from this moment."""
        for order_id in list(self._aggressors):
            order = self._aggressors[order_id]
            if not self._crosses(order.side, order.price):
                del self._aggressors[order_id]
                self.report.aggressors_rested += 1
                self._insert(order_id, order.side, order.price, order.quantity, event.exchange_ts_ns)

    def _lookup(self, event: LifecycleEvent, code: str) -> _Order | None:
        order = self._orders.get(event.order_id)
        if order is not None:
            return order
        if event.order_id in self._terminal and self._terminal[event.order_id] in {"FILLED", "CANCELLED"}:
            self._anomaly(f"{code}_AFTER_COMPLETION", event)
        elif self._gap:
            self._anomaly("EVENT_AFTER_GAP", event)
        elif not self.semantics.census_complete or self._terminal.get(event.order_id, "").startswith("CENSORED"):
            self._anomaly("LEFT_CENSORED_UNKNOWN_ID", event)
        else:
            self._anomaly(f"UNKNOWN_{code}", event)
        return None

    # ------------------------------------------------------------------ apply
    def apply(self, event: LifecycleEvent) -> None:
        if not isinstance(event, LifecycleEvent):
            raise TypeError("apply requires a LifecycleEvent")
        if self._last_sequence is not None and event.sequence != self._last_sequence + 1:
            raise LifecycleInputError("normalized sequence must be contiguous; adapter defect")
        self._last_sequence = event.sequence
        self.report.events += 1
        self.report.kinds[event.kind] += 1
        self._check_clocks(event)
        kind = event.kind
        if kind == "SNAPSHOT":
            self._snapshot(event)
        elif kind == "RESET":
            self.report.censored_by_reset += self._censor_all("CENSORED_RESET")
            self._initialized, self._gap = True, False
        elif kind == "GAP":
            self.report.censored_by_gap += self._censor_all("CENSORED_GAP")
            self._gap = True
        elif kind == "TRADE":
            self._trade(event)
        elif not self._initialized:
            self._anomaly("EVENT_BEFORE_INITIAL_CENSUS", event)
        elif event.order_id in self._aggressors:
            self._aggressor_event(event)
        elif kind == "ADD":
            self._add(event)
        elif kind == "MODIFY":
            self._modify(event)
        elif kind == "CANCEL":
            self._cancel(event)
        elif kind == "EXECUTE":
            self._execute(event)
        if self._aggressors:
            self._rest_aggressors(event)
        if self._initialized and not self._gap:
            bid, ask = self.best("BUY"), self.best("SELL")
            if bid is not None and ask is not None and bid >= ask:
                self._anomaly("CROSSED_BOOK_AFTER_EVENT", event)
        self._digest.update(f"{event.sequence}:{kind}:{event.order_id}:{event.price}:{event.quantity}:"
                            f"{len(self._orders)}:{self.best('BUY')}:{self.best('SELL')}|".encode())

    def _check_clocks(self, event: LifecycleEvent) -> None:
        if event.source_sequence is not None:
            last = self._last_source_sequence
            if last is not None:
                if event.source_sequence == last:
                    self._anomaly("DUPLICATE_SOURCE_SEQUENCE", event)
                elif event.source_sequence < last:
                    self._anomaly("SOURCE_SEQUENCE_REVERSAL", event)
                elif self.semantics.source_sequence_contiguous and event.source_sequence != last + 1:
                    self._anomaly("SOURCE_SEQUENCE_GAP", event)
            self._last_source_sequence = event.source_sequence
        if event.exchange_ts_ns is not None:
            if self._last_exchange_ns is not None and event.exchange_ts_ns < self._last_exchange_ns:
                self._anomaly("EXCHANGE_TIMESTAMP_INVERSION", event)
            self._last_exchange_ns = max(event.exchange_ts_ns, self._last_exchange_ns or event.exchange_ts_ns)
        if event.receive_ts_ns is not None:
            if self._last_receive_ns is not None and event.receive_ts_ns < self._last_receive_ns:
                self._anomaly("RECEIVE_TIMESTAMP_INVERSION", event)
            self._last_receive_ns = event.receive_ts_ns

    def _snapshot(self, event: LifecycleEvent) -> None:
        census = {o.order_id: o for o in event.orders}
        if len(census) != len(event.orders):
            raise LifecycleInputError("snapshot contains duplicate order identities")
        if self._initialized and not self._gap:
            self.report.snapshot_checks.append(self._compare_census(event))
        if event.snapshot_mode == "check" and self._initialized and not self._gap:
            return
        self.report.censored_by_reset += self._censor_all("CENSORED_RESET") if self._initialized else 0
        for order in event.orders:
            if order.side not in SIDES or not _positive(order.price) or not _positive(order.quantity):
                raise LifecycleInputError("snapshot orders require side, positive price and quantity")
            self._terminal.pop(order.order_id, None)
            self._insert(order.order_id, order.side, order.price, order.quantity, None)
        self.report.left_censored_orders += len(event.orders)
        self._initialized, self._gap = True, False

    def _compare_census(self, event: LifecycleEvent) -> dict:
        census = {o.order_id: o for o in event.orders}
        live = self._orders
        common = census.keys() & live.keys()
        exact = sum(1 for oid in common if (census[oid].side, census[oid].price, census[oid].quantity)
                    == (live[oid].side, live[oid].price, live[oid].quantity))
        # Priority agreement: within each price level of the census, compare the census
        # listing order of common identities with the tracked priority order.
        concordant = discordant = 0
        by_level: dict[tuple[str, int], list[str]] = {}
        for order in event.orders:
            if order.order_id in common:
                by_level.setdefault((order.side, order.price), []).append(order.order_id)
        for (side, price), listed in by_level.items():
            tracked = [oid for oid in self._levels[side].get(price, ()) if oid in census]
            if [oid for oid in tracked if oid in listed] and len(listed) > 1:
                rank = {oid: i for i, oid in enumerate(tracked)}
                ordered = [oid for oid in listed if oid in rank]
                for a, b in zip(ordered, ordered[1:]):
                    concordant += rank[a] < rank[b]
                    discordant += rank[a] > rank[b]
        return {"sequence": event.sequence, "exchange_ts_ns": event.exchange_ts_ns,
                "census_orders": len(census), "replayed_orders": len(live), "common": len(common),
                "exact_order_matches": exact, "missing_from_replay": len(census.keys() - live.keys()),
                "extra_in_replay": len(live.keys() - census.keys()),
                "quantity_or_price_mismatch": len(common) - exact,
                "adjacent_priority_concordant": concordant, "adjacent_priority_discordant": discordant,
                "order_agreement": exact / max(1, len(census.keys() | live.keys()))}

    def _add(self, event: LifecycleEvent) -> None:
        if event.order_id in self._orders:
            self._anomaly("DUPLICATE_ORDER_ID_LIVE", event)
            return
        if event.order_id in self._terminal and self._terminal[event.order_id] in {"FILLED", "CANCELLED"}:
            self._anomaly("DUPLICATE_ORDER_ID_REUSED", event)
            return
        self._terminal.pop(event.order_id, None)
        if self.semantics.crossing_add == "transient_aggressor" and self._crosses(event.side, event.price):
            self.report.transient_aggressors += 1
            self._aggressors[event.order_id] = _Order(event.order_id, event.side, event.price, event.quantity,
                                                      0, event.exchange_ts_ns, event.quantity)
            return
        self._insert(event.order_id, event.side, event.price, event.quantity, event.exchange_ts_ns)

    def _modify(self, event: LifecycleEvent) -> None:
        order = self._lookup(event, "MODIFY")
        if order is None:
            return
        if event.side is not None and event.side != order.side:
            self._anomaly("SIDE_MISMATCH", event)
            return
        price = order.price if event.price is None else event.price
        quantity = order.quantity if event.quantity is None else event.quantity
        if quantity == 0:
            self.report.modify_to_zero += 1
            order.cancelled += order.quantity
            self._terminate(order, "CANCELLED", event.exchange_ts_ns)
            return
        reset = (self.semantics.priority_rule == "any_modify_resets"
                 or price != order.price or quantity > order.quantity)
        if reset:
            self.report.modifies_priority_reset += 1
            self._unlink(order)
            self._priority += 1
            order.price, order.priority = price, self._priority
            self._levels[order.side].setdefault(price, OrderedDict())[order.order_id] = None
        else:
            self.report.modifies_priority_preserved += 1
            order.cancelled += order.quantity - quantity
        order.quantity = quantity

    def _cancel(self, event: LifecycleEvent) -> None:
        order = self._lookup(event, "CANCEL")
        if order is None:
            return
        if event.side is not None and event.side != order.side:
            self._anomaly("SIDE_MISMATCH", event)
            return
        amount = order.quantity if event.quantity is None else event.quantity
        if amount > order.quantity:
            self._anomaly("EXCESS_CANCEL", event)
            return
        order.cancelled += amount
        order.quantity -= amount
        if order.quantity == 0:
            self.report.full_cancels += 1
            self._terminate(order, "CANCELLED", event.exchange_ts_ns)
        else:
            self.report.partial_cancels += 1

    def _execute(self, event: LifecycleEvent) -> None:
        order = self._lookup(event, "EXECUTE")
        if order is None:
            return
        if event.side is not None and event.side != order.side:
            self._anomaly("SIDE_MISMATCH", event)
            return
        if event.price is not None and event.price != order.price:
            self._anomaly("EXECUTION_PRICE_MISMATCH", event)
        if event.quantity > order.quantity:
            self._anomaly("EXCESS_EXECUTION", event)
            return
        order.executed += event.quantity
        order.quantity -= event.quantity
        link = self._links.setdefault((order.order_id, event.exchange_ts_ns), [0, 0, 0])
        link[0] += event.quantity
        if self.keep_executions:
            self.executions.append(Execution(event.sequence, event.exchange_ts_ns, order.order_id, order.side,
                                             order.price, event.quantity, order.quantity, order.birth_ns))
        if order.quantity == 0:
            self.report.full_executions += 1
            self._terminate(order, "FILLED", event.exchange_ts_ns)
        else:
            self.report.partial_executions += 1

    def _trade(self, event: LifecycleEvent) -> None:
        self.report.trades += 1
        if event.maker_order_id is None:
            return
        link = self._links.setdefault((event.maker_order_id, event.exchange_ts_ns), [0, 0, 0])
        link[1] += event.quantity
        link[2] += 1

    def _link_trades(self) -> None:
        """Order-independent linkage: per (maker, exchange time), print and execution totals must agree."""
        linked = unlinked = orphan = 0
        makers: dict[str, list[int]] = {}
        for (maker, _ts), (executed, printed, _count) in self._links.items():
            total = makers.setdefault(maker, [0, 0])
            total[0] += executed
            total[1] += printed
        printed_makers = [t for t in makers.values() if t[1]]
        self.report.maker_level_linkage = {
            "makers_with_prints": len(printed_makers),
            "makers_prints_equal_executions": sum(t[0] == t[1] for t in printed_makers),
            "makers_executions_without_prints": sum(1 for t in makers.values() if t[0] and not t[1])}
        for executed, printed, prints in self._links.values():
            if prints and executed == printed:
                linked += prints
            elif prints:
                unlinked += prints
            if executed and not prints:
                orphan += 1
        self.report.trades_with_maker_execution = linked
        self.report.trades_without_maker_execution = unlinked
        self.report.maker_executions_without_trade = orphan

    def finish(self) -> dict:
        """Right-censor live orders at the file boundary and return a deterministic summary."""
        self._link_trades()
        self.report.right_censored_orders = len(self._orders) + len(self._aggressors)
        state = {"live_orders": sorted((o.order_id, o.side, o.price, o.quantity) for o in self._orders.values()),
                 "report": self.report.to_dict()}
        digest = hashlib.sha256(self._digest.digest() + json.dumps(state, sort_keys=True).encode()).hexdigest()
        return {"report": self.report.to_dict(), "final_state_sha256": digest,
                "live_orders": len(self._orders), "semantics": self.semantics.to_dict()}


def replay(events: Iterable[LifecycleEvent], semantics: LifecycleSemantics | None = None, *,
           references: Iterable[dict] | None = None, depth: int = 10) -> dict:
    """Replay once, comparing aggregated state with references at their positions.

    Each reference is ``{"after_sequence": int, "bids": [[p, q]...], "asks": ...}``:
    the adapter decides (and declares) which normalized events precede it.
    """
    book = OrderLifecycleBook(semantics)
    refs = sorted(references or (), key=lambda r: r["after_sequence"])
    comparison = ReferenceComparison(depth)
    index = 0
    for event in events:
        while index < len(refs) and refs[index]["after_sequence"] < event.sequence:
            comparison.compare(book, refs[index])
            index += 1
        book.apply(event)
    while index < len(refs):
        comparison.compare(book, refs[index])
        index += 1
    summary = book.finish()
    summary["reference_comparison"] = comparison.to_dict()
    return summary


class ReferenceComparison:
    """Exact top-N aggregate comparison, with mismatch types counted separately."""

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.compared = self.exact = self.not_initialized = 0
        self.price_level_mismatch = self.quantity_mismatch = 0
        self.first_mismatches: list[dict] = []

    def compare(self, book: OrderLifecycleBook, reference: dict) -> None:
        if not book.initialized:
            self.not_initialized += 1
            return
        self.compared += 1
        state = book.aggregate(self.depth)
        expected = {"bids": [list(x) for x in reference["bids"][: self.depth]],
                    "asks": [list(x) for x in reference["asks"][: self.depth]]}
        if state == expected:
            self.exact += 1
            return
        prices_same = all([p for p, _ in state[s]] == [p for p, _ in expected[s]] for s in ("bids", "asks"))
        if prices_same:
            self.quantity_mismatch += 1
        else:
            self.price_level_mismatch += 1
        if len(self.first_mismatches) < 5:
            self.first_mismatches.append({"after_sequence": reference["after_sequence"],
                                          "replayed": {k: v[:3] for k, v in state.items()},
                                          "reference": {k: v[:3] for k, v in expected.items()}})

    def to_dict(self) -> dict:
        return {"depth": self.depth, "compared": self.compared, "exact": self.exact,
                "exact_fraction": self.exact / self.compared if self.compared else None,
                "price_level_mismatch": self.price_level_mismatch,
                "quantity_mismatch": self.quantity_mismatch, "not_initialized": self.not_initialized,
                "first_mismatches": self.first_mismatches}

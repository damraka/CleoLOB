"""Bounded MBO research replay, reusing the canonical order-ID book.

An adapter must supply actual identities and source-guaranteed snapshot FIFO.
This module observes recorded orders; it never inserts hypothetical agent fills.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .capabilities import CANONICAL_MBO_CONTRACT, Capability, CapabilityContract, CapabilityError, DataLevel
from .replay.book import HistoricalBook, ReconstructionError
from .replay.io import _json
from .replay.schema import EventSchemaError, MarketEvent, ReplaySide, SnapshotOrder, _integer


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class MBOEvent:
    """Canonical integer-tick event with an optional independent exchange clock.

    ``timestamp_ns`` is the nondecreasing capture/replay clock; exchange time
    may regress and is retained without sorting. RESET is an explicit empty
    census. EXECUTE identifies a maker; TRADE remains an unlinked print.
    """

    timestamp_ns: int
    sequence: int
    event_type: str
    symbol: str
    venue: str
    order_id: str | None = None
    side: ReplaySide | None = None
    price_ticks: int | None = None
    quantity: int | None = None
    orders: tuple[SnapshotOrder, ...] = ()
    exchange_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        reset = self.event_type == "RESET"
        if reset and self.orders:
            raise EventSchemaError("UNEXPECTED_FIELD", "RESET must not carry orders")
        canonical = MarketEvent(
            self.timestamp_ns, self.sequence, "SNAPSHOT" if reset else self.event_type,
            self.symbol, self.venue, self.order_id, self.side, self.price_ticks,
            self.quantity, self.orders,
        )
        for name in ("timestamp_ns", "sequence", "side", "price_ticks", "quantity", "orders"):
            object.__setattr__(self, name, getattr(canonical, name))
        object.__setattr__(self, "event_type", "RESET" if reset else canonical.event_type.value)
        if self.exchange_timestamp_ns is not None:
            object.__setattr__(self, "exchange_timestamp_ns", _integer(
                self.exchange_timestamp_ns, "exchange_timestamp_ns", minimum=0))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> MBOEvent:
        if not isinstance(row, Mapping):
            raise EventSchemaError("CORRUPT_ROW", "each MBO event must be an object")
        clean = dict(row)
        exchange_timestamp = clean.pop("exchange_timestamp_ns", None)
        reset = clean.get("event_type") == "RESET"
        if reset:
            if "orders" in clean:
                raise EventSchemaError("UNEXPECTED_FIELD", "RESET must omit orders")
            clean.update(event_type="SNAPSHOT", orders=[])
        event = MarketEvent.from_mapping(clean)
        values = event.to_dict()
        values["orders"] = event.orders
        if reset:
            values["event_type"] = "RESET"
        return cls(**values, exchange_timestamp_ns=exchange_timestamp)

    def canonical(self) -> MarketEvent:
        return MarketEvent(
            self.timestamp_ns, self.sequence,
            "SNAPSHOT" if self.event_type == "RESET" else self.event_type,
            self.symbol, self.venue, self.order_id, self.side, self.price_ticks,
            self.quantity, self.orders,
        )

    def to_dict(self) -> dict[str, Any]:
        result = self.canonical().to_dict()
        if self.event_type == "RESET":
            result.update(event_type="RESET")
            result.pop("orders")
        if self.exchange_timestamp_ns is not None:
            result["exchange_timestamp_ns"] = self.exchange_timestamp_ns
        return result


@dataclass(frozen=True)
class _UnorderedIdentity:
    order_id: str
    side: ReplaySide
    price_ticks: int
    quantity: int

    @property
    def priority(self) -> int:
        raise CapabilityError("exact_fifo_position", "source order priority is unavailable without FIFO capability")


@dataclass
class _Watch:
    order_id: str
    entry_timestamp_ns: int
    submission_timestamp_ns: int | None
    initial_position: int
    initial_quantity: int
    filled_quantity: int = 0
    cancelled_quantity: int = 0
    cancelled_ahead_quantity: int = 0
    first_fill_timestamp_ns: int | None = None
    initial_quantity_executed_timestamp_ns: int | None = None
    full_fill_timestamp_ns: int | None = None
    priority_reset_timestamp_ns: int | None = None
    priority_reset_reason: str | None = None
    terminal_timestamp_ns: int | None = None
    terminal_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class MBOBook:
    """Auditable price-time queues with explicitly bounded observation history.

    Same-price size reductions keep priority; increases/repricing lose it.
    EXECUTE always follows the recorded ID, including executions behind another
    visible order. This is reconstruction, not a synthetic FIFO matching rule.
    Snapshot list order must encode actual source priority, never L2 level order.
    """

    def __init__(self, *, max_events: int = 1_000_000, max_orders: int = 100_000,
                 max_total_order_records: int = 1_000_000, max_watches: int = 1000,
                 max_observations: int = 100_000,
                 capability_contract: CapabilityContract = CANONICAL_MBO_CONTRACT):
        if capability_contract.level != DataLevel.MBO:
            raise ValueError("MBOBook requires an order-level capability contract")
        capability_contract.require(Capability.ORDER_IDENTITY, Capability.AGGREGATE_DEPTH)
        self._capability_contract = capability_contract
        self._limits = {name: _positive(value, name) for name, value in {
            "max_events": max_events, "max_orders": max_orders,
            "max_total_order_records": max_total_order_records,
            "max_watches": max_watches, "max_observations": max_observations,
        }.items()}
        self._book = HistoricalBook()
        self._last: MBOEvent | None = None
        self._births: dict[str, int | None] = {}
        self._watches: dict[str, _Watch] = {}
        self._events = 0
        self._order_records = 0
        self._observations = 0
        self.exchange_clock_regressions = 0
        self._last_exchange_timestamp_ns: int | None = None

    @property
    def capability_contract(self) -> CapabilityContract:
        return self._capability_contract

    def require_capability(self, name: Capability | str) -> None:
        self._capability_contract.require(name)

    @property
    def orders(self):
        orders = self._book.orders
        if self.capability_contract.supports(Capability.EXACT_FIFO_POSITION):
            return orders
        return {oid: _UnorderedIdentity(o.order_id, o.side, o.price_ticks, o.quantity)
                for oid, o in orders.items()}

    def queue(self, side: ReplaySide | str, price_ticks: int) -> tuple[str, ...]:
        self.require_capability(Capability.EXACT_FIFO_POSITION)
        return self._book.queue(side, _integer(price_ticks, "price_ticks"))

    def aggregate_l2(self, depth: int | None = None) -> dict[str, list[list[int]]]:
        """Deterministic exact tick/lot aggregation; no float conversion."""
        bids, asks = self._book.depth(depth)
        return {"bids": [[p, q] for p, q in bids], "asks": [[p, q] for p, q in asks]}

    def snapshot(self) -> dict[str, Any]:
        state = self._book.snapshot()
        if not self.capability_contract.supports(Capability.EXACT_FIFO_POSITION):
            # Stable identity ordering does not masquerade as source FIFO.
            state["orders"] = sorted(state["orders"], key=lambda order: order["order_id"])
            for order in state["orders"]:
                order.pop("priority")
        state.update(evidence_level="MBO_SOURCE_IDENTITIES",
                     exchange_timestamp_ns=None if self._last is None else self._last.exchange_timestamp_ns)
        return state

    def queue_metrics(self, order_id: str) -> dict[str, Any]:
        """Current same-price queue, conditional on complete source FIFO data."""
        self.require_capability(Capability.QUANTITY_AHEAD)
        orders = self._book.orders
        if order_id not in orders:
            raise ReconstructionError("UNKNOWN_ORDER_ID", f"no resting order {order_id!r}")
        order = orders[order_id]
        queue = self._book.queue(order.side, order.price_ticks)
        position = queue.index(order_id)
        birth = self._births[order_id]
        assert self._last is not None
        return {
            "order_id": order_id, "timestamp_ns": self._last.timestamp_ns,
            "sequence": self._last.sequence, "side": order.side.value,
            "price_ticks": order.price_ticks, "quantity": order.quantity,
            "position": position, "orders_ahead": position,
            "quantity_ahead": sum(orders[oid].quantity for oid in queue[:position]),
            "quantity_behind": sum(orders[oid].quantity for oid in queue[position + 1:]),
            "age_ns": None if birth is None else self._last.timestamp_ns - birth,
            "age_is_left_censored": birth is None,
        }

    def watch(self, order_id: str) -> dict[str, Any]:
        """Observe a recorded order prospectively; never create an order.

        Register immediately after ADD to measure position at submission. Later
        registrations have a distinct observation origin and remain ineligible
        for the submission-cohort probability estimator.
        """
        self.require_capability(Capability.OBSERVED_ORDER_FILL)
        if order_id in self._watches:
            raise ValueError("order_id is already watched; each ID has one observation origin")
        if len(self._watches) >= self._limits["max_watches"]:
            raise ValueError("MBO research exceeds max_watches")
        if self._observations >= self._limits["max_observations"]:
            raise ValueError("MBO research exceeds max_observations")
        current = self.queue_metrics(order_id)
        assert self._last is not None
        at_submission = self._last.event_type == "ADD" and self._last.order_id == order_id
        watch = _Watch(order_id, self._last.timestamp_ns,
                       self._last.timestamp_ns if at_submission else None,
                       current["position"], current["quantity"])
        self._watches[order_id] = watch
        self._record(watch)
        return self.order_research(order_id)

    def _record(self, watch: _Watch) -> None:
        assert self._last is not None
        if watch.terminal_reason:
            current = {"order_id": watch.order_id, "timestamp_ns": self._last.timestamp_ns,
                       "sequence": self._last.sequence, "position": None,
                       # A replacement census ends this trajectory without
                       # establishing whether its last remainder traded or
                       # cancelled. Zero would fabricate that observation.
                       "quantity": None if watch.terminal_reason == "CENSORED_CENSUS" else 0,
                       "quantity_ahead": None, "quantity_behind": None, "age_ns": None}
        else:
            current = self.queue_metrics(watch.order_id)
        current.update(
            filled_quantity=watch.filled_quantity,
            cancelled_quantity=watch.cancelled_quantity,
            cancelled_ahead_quantity=watch.cancelled_ahead_quantity,
            queue_movement=(
                None
                if current["position"] is None
                or watch.priority_reset_timestamp_ns is not None
                else watch.initial_position - current["position"]
            ),
            priority_reset=watch.priority_reset_timestamp_ns is not None,
            priority_reset_timestamp_ns=watch.priority_reset_timestamp_ns,
            priority_reset_reason=watch.priority_reset_reason,
            terminal_reason=watch.terminal_reason,
        )
        watch.history.append(current)
        self._observations += 1

    def order_research(self, order_id: str) -> dict[str, Any]:
        """Copy of one observed trajectory; time-to-fill is never counterfactual."""
        watch = self._watches[order_id]
        origin = watch.submission_timestamp_ns
        return {
            "order_id": order_id, "entry_timestamp_ns": watch.entry_timestamp_ns,
            "observed_from_submission": origin is not None,
            "position_at_submission": watch.initial_position if origin is not None else None,
            "time_to_first_fill_ns": (
                None
                if origin is None or watch.first_fill_timestamp_ns is None
                else watch.first_fill_timestamp_ns - origin
            ),
            # Legacy-style execution threshold: cumulative recorded EXECUTE
            # quantity has reached the originally observed submitted quantity.
            # This is NOT necessarily a terminal fill after amendments.
            "time_to_initial_quantity_executed_ns": (
                None
                if origin is None
                or watch.initial_quantity_executed_timestamp_ns is None
                else watch.initial_quantity_executed_timestamp_ns - origin
            ),
            # Full fill means the observed identity actually leaves the book
            # through EXECUTE, with no prior explicit CANCEL on that identity.
            "time_to_full_fill_ns": (
                None
                if origin is None or watch.full_fill_timestamp_ns is None
                else watch.full_fill_timestamp_ns - origin
            ),
            "initial_quantity": watch.initial_quantity,
            "filled_quantity": watch.filled_quantity,
            "cancelled_quantity": watch.cancelled_quantity,
            "cancelled_ahead_quantity": watch.cancelled_ahead_quantity,
            "priority_reset": watch.priority_reset_timestamp_ns is not None,
            "priority_reset_timestamp_ns": watch.priority_reset_timestamp_ns,
            "priority_reset_reason": watch.priority_reset_reason,
            "terminal_reason": watch.terminal_reason,
            "terminal_timestamp_ns": watch.terminal_timestamp_ns,
            "history": [dict(item) for item in watch.history],
        }

    def apply(self, event: MBOEvent) -> None:
        if not isinstance(event, MBOEvent):
            raise TypeError("apply requires an MBOEvent")
        if event.event_type == "EXECUTE":
            self.require_capability(Capability.OBSERVED_ORDER_FILL)
        if self._last is None and event.event_type not in {"SNAPSHOT", "RESET"}:
            raise ReconstructionError("INITIAL_SNAPSHOT_REQUIRED", "MBO replay requires an initial SNAPSHOT or RESET")
        if self._events >= self._limits["max_events"]:
            raise ValueError("MBO replay exceeds max_events")
        census = event.event_type in {"SNAPSHOT", "RESET"}
        records = len(event.orders) if census else int(event.event_type == "ADD")
        if self._order_records + records > self._limits["max_total_order_records"]:
            raise ValueError("MBO replay exceeds max_total_order_records")
        old = self._book.orders
        new_count = len(event.orders) if census else len(old) + int(event.event_type == "ADD")
        if new_count > self._limits["max_orders"]:
            raise ValueError("MBO book exceeds max_orders")
        active = [watch for watch in self._watches.values() if watch.terminal_reason is None]
        if self._observations + len(active) > self._limits["max_observations"]:
            raise ValueError("MBO research exceeds max_observations")
        ahead = {}
        for watch in active:
            order = old[watch.order_id]
            queue = self._book.queue(order.side, order.price_ticks)
            ahead[watch.order_id] = set(queue[:queue.index(watch.order_id)])
        # The reference engine validates before mutation. No research state is
        # modified until this call succeeds, including on sequence rejection.
        self._book.apply(event.canonical())
        self._last = event
        self._events += 1
        self._order_records += records
        exchange_timestamp = event.exchange_timestamp_ns
        if exchange_timestamp is not None:
            if self._last_exchange_timestamp_ns is not None:
                self.exchange_clock_regressions += int(exchange_timestamp < self._last_exchange_timestamp_ns)
            self._last_exchange_timestamp_ns = exchange_timestamp
        if census:
            self._births = {order.order_id: None for order in event.orders}
        elif event.event_type == "ADD":
            self._births[event.order_id] = event.timestamp_ns
        new = self._book.orders
        if event.order_id is not None and event.order_id not in new:
            self._births.pop(event.order_id, None)
        for watch in active:
            if census:
                watch.terminal_reason = "CENSORED_CENSUS"
            else:
                target = old.get(event.order_id)
                if target is not None and event.order_id in ahead[watch.order_id]:
                    if event.event_type == "CANCEL":
                        watch.cancelled_ahead_quantity += event.quantity
                    elif event.event_type == "DELETE":
                        watch.cancelled_ahead_quantity += target.quantity
                    elif event.event_type == "MODIFY" and event.price_ticks in (None, target.price_ticks):
                        watch.cancelled_ahead_quantity += max(0, target.quantity - new[event.order_id].quantity)
                if event.order_id == watch.order_id:
                    # Same-price size reductions retain priority in the
                    # canonical book. Repricing or increasing visible size
                    # loses priority, so submission-position movement is no
                    # longer comparable after that amendment.
                    if (
                        event.event_type == "MODIFY"
                        and target is not None
                        and watch.order_id in new
                    ):
                        replacement = new[watch.order_id]
                        repriced = replacement.price_ticks != target.price_ticks
                        size_increased = replacement.quantity > target.quantity
                        if (
                            (repriced or size_increased)
                            and watch.priority_reset_timestamp_ns is None
                        ):
                            watch.priority_reset_timestamp_ns = event.timestamp_ns
                            watch.priority_reset_reason = (
                                "REPRICE" if repriced else "SIZE_INCREASE"
                            )

                    if event.event_type == "CANCEL":
                        watch.cancelled_quantity += event.quantity

                    if event.event_type == "EXECUTE":
                        watch.filled_quantity += event.quantity

                        if watch.first_fill_timestamp_ns is None:
                            watch.first_fill_timestamp_ns = event.timestamp_ns

                        if (
                            watch.filled_quantity >= watch.initial_quantity
                            and watch.initial_quantity_executed_timestamp_ns is None
                        ):
                            watch.initial_quantity_executed_timestamp_ns = (
                                event.timestamp_ns
                            )

                    if watch.order_id not in new:
                        if event.event_type == "EXECUTE":
                            watch.terminal_reason = "FILLED"

                            # A true observed full fill requires terminal
                            # removal by EXECUTE and no explicit cancellation
                            # of this watched identity beforehand. Same-ID
                            # MODIFY amendments are tracked separately.
                            if (
                                watch.cancelled_quantity == 0
                                and watch.full_fill_timestamp_ns is None
                            ):
                                watch.full_fill_timestamp_ns = (
                                    event.timestamp_ns
                                )
                        else:
                            watch.terminal_reason = "CANCELLED"
            if watch.terminal_reason:
                watch.terminal_timestamp_ns = event.timestamp_ns
            self._record(watch)

    def cohort_summary(self, horizon_ns: int) -> dict[str, Any]:
        """Descriptive fixed-horizon outcomes, with censoring explicitly counted.

        Denominator: watches registered at ADD whose entire horizon is observed
        or whose order terminates before it. A census before the horizon censors
        that observation. ``full_fill_frequency`` requires observed terminal
        removal by EXECUTE without a prior explicit CANCEL on that identity.
        Quantity amendments through MODIFY remain visible separately from that
        estimand. No independent-censoring or causal assumption is made; this
        complete-case frequency is not a population fill-probability model.
        """
        self.require_capability(Capability.OBSERVED_ORDER_FILL)
        horizon_ns = _integer(horizon_ns, "horizon_ns", minimum=0)
        eligible = [w for w in self._watches.values() if w.submission_timestamp_ns is not None]
        evaluated = []
        censored = 0
        last_time = None if self._last is None else self._last.timestamp_ns
        for watch in eligible:
            deadline = watch.entry_timestamp_ns + horizon_ns
            terminal = watch.terminal_timestamp_ns
            if watch.terminal_reason == "CENSORED_CENSUS" and terminal <= deadline:
                censored += 1
                continue
            if last_time < deadline and (terminal is None or terminal > deadline):
                censored += 1
                continue
            evaluated.append((watch, deadline))
        n = len(evaluated)
        any_fill = sum(w.first_fill_timestamp_ns is not None and w.first_fill_timestamp_ns <= t
                       for w, t in evaluated)
        full_fill = sum(w.full_fill_timestamp_ns is not None and w.full_fill_timestamp_ns <= t
                        for w, t in evaluated)
        resting = sum(w.terminal_timestamp_ns is None or w.terminal_timestamp_ns > t for w, t in evaluated)
        return {
            "estimand": "observed_submission_cohort_complete_case_frequency",
            "horizon_ns": horizon_ns, "submission_cohort_size": len(eligible),
            "non_submission_watches": len(self._watches) - len(eligible),
            "evaluated_orders": n, "censored_orders": censored,
            "any_passive_fill_frequency": any_fill / n if n else None,
            "full_fill_frequency": full_fill / n if n else None,
            "queue_survival_frequency": resting / n if n else None,
            "counterfactual_fill_probability": None,
            "limitation": "Recorded maker executions only; selective watching and censoring can bias frequencies.",
        }

    def summary(self) -> dict[str, Any]:
        return {"events": self._events, "order_records": self._order_records,
                "watched_orders": len(self._watches), "queue_observations": self._observations,
                "execution_count": len(self._book.executions),
                "executed_quantity": sum(item.quantity for item in self._book.executions),
                "trade_print_count": len(self._book.prints),
                "exchange_clock_regressions": self.exchange_clock_regressions,
                "book": self.snapshot(), "counterfactual_fills_available": False,
                "hidden_liquidity_available": False}


class _HashReader:
    def __init__(self, raw, maximum: int):
        self.raw = raw
        self.maximum = maximum
        self.count = 0
        self.digest = hashlib.sha256()

    def _record(self, data: bytes) -> bytes:
        self.count += len(data)
        if self.count > self.maximum:
            raise ValueError("MBO source exceeds max_file_bytes")
        self.digest.update(data)
        return data

    def read(self, size: int = -1) -> bytes:
        return self._record(self.raw.read(size))

    def readline(self, size: int = -1) -> bytes:
        return self._record(self.raw.readline(size))


class MBOReplay:
    """One-pass bounded JSONL adapter; full validity requires exhausting input.

    Yields successfully applied events so consumers can register watches after
    ADD. A rejected suffix leaves an explicitly incomplete validated prefix.
    SHA-256 values are exposed only after successful EOF/trailer verification.
    """

    def __init__(self, path: str | Path, *, max_file_bytes: int = 128 * 1024**2,
                 max_expanded_bytes: int = 256 * 1024**2, max_line_bytes: int = 2 * 1024**2,
                 max_events: int = 1_000_000, **book_limits):
        self.path = Path(path)
        self._limits = {name: _positive(value, name) for name, value in {
            "max_file_bytes": max_file_bytes, "max_expanded_bytes": max_expanded_bytes,
            "max_line_bytes": max_line_bytes, "max_events": max_events,
        }.items()}
        self.book = MBOBook(max_events=max_events, **book_limits)
        self._started = False
        self._stats = {"complete": False, "events": 0, "expanded_bytes": 0,
                       "source_sha256": None, "canonical_sha256": None}

    @property
    def capability_contract(self) -> CapabilityContract:
        return self.book.capability_contract

    def require_capability(self, name: Capability | str) -> None:
        self.capability_contract.require(name)

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def __iter__(self) -> Iterator[MBOEvent]:
        if self._started:
            raise ValueError("MBOReplay is one-pass; construct a new instance")
        self._started = True
        return self._iterate()

    def _iterate(self) -> Iterator[MBOEvent]:
        canonical_digest = hashlib.sha256()
        try:
            if not self.path.is_file():
                raise ValueError("MBO source must be a regular file")
            suffixes = self.path.suffixes
            compressed = bool(suffixes and suffixes[-1] == ".gz")
            format_suffix = suffixes[-2] if compressed and len(suffixes) >= 2 else self.path.suffix
            if format_suffix not in {".jsonl", ".ndjson"}:
                raise ValueError("MBO adapter supports JSONL/NDJSON, optionally gzip compressed")
            if self.path.stat().st_size > self._limits["max_file_bytes"]:
                raise ValueError("MBO source exceeds max_file_bytes")
            with ExitStack() as stack:
                raw = stack.enter_context(self.path.open("rb"))
                hashed = _HashReader(raw, self._limits["max_file_bytes"])
                source = stack.enter_context(gzip.GzipFile(fileobj=hashed, mode="rb")) if compressed else hashed
                line_number = 0
                while True:
                    line = source.readline(self._limits["max_line_bytes"] + 1)
                    if not line:
                        break
                    line_number += 1
                    self._stats["expanded_bytes"] += len(line)
                    if len(line) > self._limits["max_line_bytes"]:
                        raise ValueError(f"MBO line {line_number} exceeds max_line_bytes")
                    if self._stats["expanded_bytes"] > self._limits["max_expanded_bytes"]:
                        raise ValueError("MBO source exceeds max_expanded_bytes")
                    if line_number > self._limits["max_events"]:
                        raise ValueError("MBO source exceeds max_events")
                    try:
                        event = MBOEvent.from_mapping(_json(line.decode("utf-8")))
                        self.book.apply(event)
                    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
                        raise ValueError(f"Invalid MBO line {line_number}: {exc}") from exc
                    canonical_digest.update(json.dumps(event.to_dict(), sort_keys=True,
                                                       separators=(",", ":")).encode("utf-8") + b"\n")
                    self._stats["events"] += 1
                    yield event
                if line_number == 0:
                    raise ValueError("MBO source is empty")
                self._stats.update(complete=True, source_sha256=hashed.digest.hexdigest(),
                                   canonical_sha256=canonical_digest.hexdigest())
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError(f"Cannot fully read MBO source: {exc}") from exc

    def run(self) -> int:
        return sum(1 for _ in self)

    def summary(self) -> dict[str, Any]:
        return {"mode": "mbo_observed_order_reconstruction", **self.stats, "research": self.book.summary()}

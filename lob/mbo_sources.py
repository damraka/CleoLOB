"""Source-native order-level adapters with declared, testable semantics.

Each adapter converts a venue's native records into ``lob.order_lifecycle``
events in exact integer ticks/lots, plus independently published reference
books, without inventing identities or repairing gaps. Every interpretation
rule is part of an immutable ``SourceContract`` whose hash is recorded with
results, so a rule cannot change silently between development and validation.

Implemented sources:

* Bitstamp public websocket capture (``live_orders``, ``live_trades``,
  ``order_book``) plus REST ``order_book?group=2`` censuses. Genuine venue
  order-level messages, recorded live by ``tools/capture_bitstamp.py``.
* LOBSTER message/orderbook files (NASDAQ ITCH-derived academic format). The
  contract and adapter are complete; no LOBSTER data is available here.
"""
from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from .order_lifecycle import (CensusOrder, LifecycleEvent, LifecycleInputError, LifecycleSemantics,
                              OrderLifecycleBook, ReferenceComparison)

ALIGNMENTS = ("arrival", "exchange_ms_strict", "exchange_ms_inclusive")


@dataclass(frozen=True)
class SourceContract:
    """Declared source semantics; a declaration is an obligation, not proof."""

    source: str
    venue: str
    instrument: str
    tick_size: str
    lot_size: str
    identity: str
    sequence: str
    timestamps: str
    snapshot: str
    priority: str
    execution: str
    hidden_liquidity: str
    fifo_established: bool
    reference_alignment: str
    adapter_version: str

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {**asdict(self), "contract_sha256": self.sha256()}


def exact_units(text: Any, unit: str, name: str) -> int:
    """Convert a decimal string to integer units, refusing any rounding."""
    try:
        value = Decimal(str(text))
        step = Decimal(unit)
    except InvalidOperation as exc:
        raise LifecycleInputError(f"{name} is not a decimal: {text!r}") from exc
    units = value / step
    if not value.is_finite() or units != units.to_integral_value():
        raise LifecycleInputError(f"{name} {text!r} is not a whole multiple of {unit}")
    return int(units)


# ----------------------------------------------------------------------------- Bitstamp

def bitstamp_contract(pair: str = "btcusd", alignment: str = "exchange_ms_strict") -> SourceContract:
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}")
    return SourceContract(
        source="Bitstamp public websocket v2 + REST order_book group=2 (self-recorded capture)",
        venue="bitstamp", instrument=pair, tick_size="0.01", lot_size="0.00000001",
        identity="Source order id_str; never generated. Trades carry buy_order_id and sell_order_id.",
        sequence="Order channel: event_id/pre_event_id chain; a break is a GAP. Normalized order is "
                 "websocket arrival order on one connection; trades and book references carry no sequence.",
        timestamps="Order events: microtimestamp truncated to milliseconds by the venue. Order-book "
                   "references and REST censuses: microseconds. Local receive clock recorded separately.",
        snapshot="REST group=2 lists every resting order with its ID (complete census). The first census "
                 "resets the book; later censuses are compared without repairing the replay.",
        priority="Venue price-time priority is NOT established by the feed; tracked priority uses "
                 "reduction-keeps / increase-or-reprice-resets only for comparison with census order.",
        execution="order_changed: the reduction of tracked remaining quantity is an execution at the order's "
                  "limit price (amount_traded and price fields are counted cross-checks). order_deleted: zero "
                  "remaining means the remainder executed; positive remaining means it was cancelled. "
                  "Trade prints are cross-checked against maker executions, never used to create them. "
                  "An order created through the opposite best is an incoming aggressor (transient; its "
                  "executions are taker fills). Zero-price orders are market orders and never rest.",
        hidden_liquidity="Not observable; no hidden quantity is inferred.",
        fifo_established=False, reference_alignment=alignment, adapter_version="bitstamp-capture-4")


def _read_capture(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


class BitstampCaptureAdapter:
    """Two-phase, offline normalization of one bounded capture file.

    Order events are ordered by websocket arrival. The first REST census becomes
    the initial SNAPSHOT at its exchange microtimestamp T0; order events whose
    millisecond precedes T0's millisecond are reflected in the census and dropped;
    events in the same millisecond are applied only if their effect is not already
    reflected (counted as boundary cases). Later censuses and order_book references
    are placed at the declared alignment. A reconnect request, disconnect or
    capture error becomes an explicit GAP.
    """

    def __init__(self, path: str | Path, *, alignment: str = "exchange_ms_strict", pair: str = "btcusd"):
        self.path = Path(path)
        self.contract = bitstamp_contract(pair, alignment)
        self.alignment = alignment
        self.pair = pair
        self.stats: dict[str, Any] = {}
        self._built: tuple[list[LifecycleEvent], list[dict]] | None = None

    def _reset_stats(self) -> None:
        self.stats = {"ws_messages": 0, "order_messages": 0, "trade_messages": 0,
                      "book_messages": 0, "rest_censuses": 0, "dropped_pre_census": 0,
                      "census_boundary_applied": 0, "census_boundary_skipped": 0,
                      "unparsed_messages": 0, "gap_markers": 0, "capture_status": None,
                      "order_chain_links": 0, "order_chain_breaks": 0, "market_order_events_skipped": 0,
                      "delete_traded_field_mismatch": 0, "delete_remaining_mismatch": 0,
                      "changed_price_reports": 0, "changed_traded_field_unexplained": 0,
                      "changed_size_increase": 0, "changed_without_quantity_change": 0}

    def _price(self, text: str) -> int:
        return exact_units(text, self.contract.tick_size, "price")

    def _qty(self, text: str) -> int:
        return exact_units(text, self.contract.lot_size, "amount")

    def _records(self) -> list[tuple[str, int, int, Any]]:
        """(kind, exchange_us, receive_ns, payload) in arrival order."""
        records = []
        last_event_id = None
        for row in _read_capture(self.path):
            kind = row.get("kind")
            if kind == "ws":
                self.stats["ws_messages"] += 1
                try:
                    message = json.loads(row["raw"])
                except ValueError:
                    self.stats["unparsed_messages"] += 1
                    continue
                channel, event, data = message.get("channel", ""), message.get("event"), message.get("data") or {}
                if event == "bts:request_reconnect":
                    records.append(("gap", None, row["recv_ns"], "request_reconnect"))
                elif channel.startswith("live_orders_") and event in {"order_created", "order_changed", "order_deleted"}:
                    self.stats["order_messages"] += 1
                    # The order channel carries a source chain: pre_event_id names the previous
                    # event. A break proves lost or reordered messages and becomes a GAP.
                    event_id, previous = message.get("event_id"), message.get("pre_event_id")
                    if last_event_id is not None and event_id is not None:
                        if previous == last_event_id:
                            self.stats["order_chain_links"] += 1
                        else:
                            self.stats["order_chain_breaks"] += 1
                            records.append(("gap", None, row["recv_ns"], "order event chain break"))
                    last_event_id = event_id if event_id is not None else last_event_id
                    records.append((event, int(data["microtimestamp"]), row["recv_ns"], data))
                elif channel.startswith("live_trades_") and event == "trade":
                    self.stats["trade_messages"] += 1
                    records.append(("trade", int(data["microtimestamp"]), row["recv_ns"], data))
                elif channel.startswith("order_book_") and event == "data":
                    self.stats["book_messages"] += 1
                    records.append(("book", int(data["microtimestamp"]), row["recv_ns"], data))
            elif kind == "rest_snapshot":
                self.stats["rest_censuses"] += 1
                body = json.loads(row["body"])
                records.append(("census", int(body["microtimestamp"]), row["recv_ns"], body))
            elif kind == "capture_end":
                self.stats["capture_status"] = row.get("status")
                if row.get("status") != "COMPLETE":
                    records.append(("gap", None, row["recv_ns"], row.get("reason")))
        return records

    def _census_orders(self, body: dict) -> tuple[CensusOrder, ...]:
        orders = []
        for side, key in (("BUY", "bids"), ("SELL", "asks")):
            for price, amount, order_id in body[key]:
                orders.append(CensusOrder(str(order_id), side, self._price(price), self._qty(amount)))
        return tuple(orders)

    def build(self) -> tuple[list[LifecycleEvent], list[dict]]:
        if self._built is not None:
            return self._built
        self._reset_stats()
        self._market_orders: set[str] = set()
        records = self._records()
        census_positions = [i for i, r in enumerate(records) if r[0] == "census"]
        events: list[LifecycleEvent] = []
        references: list[dict] = []
        state: dict[str, dict] = {}  # adapter-local last known amounts per identity
        if not census_positions:
            self._built = (events, references)
            return self._built
        first = census_positions[0]
        t0 = records[first][1]
        census = self._census_orders(records[first][3])
        census_ids = {o.order_id: o for o in census}
        pending: list[tuple[int, dict]] = []  # later censuses/references awaiting placement (exchange_us, item)

        def emit(**fields) -> None:
            events.append(LifecycleEvent(sequence=len(events), **fields))

        emit(kind="SNAPSHOT", exchange_ts_ns=t0 * 1000, receive_ts_ns=records[first][2], orders=census,
             snapshot_mode="reset")
        for order in census:
            state[order.order_id] = {"amount": order.quantity, "traded": None, "price": order.price,
                                     "side": order.side}

        ordered = [r for i, r in enumerate(records) if i != first]
        t0_ms = t0 // 1000
        for kind, ts_us, recv_ns, data in ordered:
            if kind == "gap":
                self.stats["gap_markers"] += 1
                emit(kind="GAP", receive_ts_ns=recv_ns)
                continue
            if kind in {"census", "book"}:
                pending.append((ts_us, {"kind": kind, "data": data, "recv_ns": recv_ns}))
                continue
            ms = ts_us // 1000
            if ms < t0_ms:
                self.stats["dropped_pre_census"] += 1
                continue
            boundary = ms == t0_ms
            self._flush(pending, ts_us, events, references, emit)
            self._order_event(kind, ts_us, recv_ns, data, state, emit, boundary, census_ids)
        self._flush(pending, None, events, references, emit)
        self._built = (events, references)
        return self._built

    def _flush(self, pending, next_ts_us, events, references, emit) -> None:
        """Place censuses/references whose exchange time precedes the next order event."""
        keep = []
        for ts_us, item in pending:
            ready = next_ts_us is None or self._precedes(ts_us, next_ts_us, item["kind"])
            if not ready:
                keep.append((ts_us, item))
                continue
            if item["kind"] == "census":
                emit(kind="SNAPSHOT", exchange_ts_ns=ts_us * 1000, receive_ts_ns=item["recv_ns"],
                     orders=self._census_orders(item["data"]), snapshot_mode="check")
            else:
                data = item["data"]
                references.append({"after_sequence": len(events) - 1, "exchange_ts_ns": ts_us * 1000,
                                   "bids": [[self._price(p), self._qty(q)] for p, q in data["bids"]],
                                   "asks": [[self._price(p), self._qty(q)] for p, q in data["asks"]]})
        pending[:] = keep

    def _precedes(self, reference_us: int, order_us: int, kind: str) -> bool:
        if self.alignment == "arrival":
            return True
        order_ms = order_us // 1000
        reference_ms = reference_us // 1000
        if self.alignment == "exchange_ms_strict":
            # An order event in the reference's own millisecond is treated as later.
            return order_ms >= reference_ms
        return order_ms > reference_ms  # inclusive: same-millisecond events precede the reference

    def _order_event(self, kind, ts_us, recv_ns, data, state, emit, boundary, census_ids) -> None:
        oid = str(data["id_str"]) if kind != "trade" else None
        ts = ts_us * 1000
        if kind != "trade" and (oid in self._market_orders or (kind == "order_created" and data["price_str"] in {"0", "0.00"})):
            # A zero-price order is a market order: it never rests; its fills appear on the makers.
            self._market_orders.add(oid)
            self.stats["market_order_events_skipped"] += 1
            return
        if kind == "trade":
            maker = data["sell_order_id"] if int(data["type"]) == 0 else data["buy_order_id"]
            taker = data["buy_order_id"] if int(data["type"]) == 0 else data["sell_order_id"]
            emit(kind="TRADE", exchange_ts_ns=ts, receive_ts_ns=recv_ns, price=self._price(data["price_str"]),
                 quantity=self._qty(data["amount_str"]), maker_order_id=str(maker), taker_order_id=str(taker),
                 aggressor_side="BUY" if int(data["type"]) == 0 else "SELL")
            return
        side = "BUY" if int(data["order_type"]) == 0 else "SELL"
        price = self._price(data["price_str"])
        amount = self._qty(data["amount_str"])
        traded = self._qty(data.get("amount_traded") or "0")
        known = state.get(oid)
        if boundary:
            reflected = ((kind == "order_created" and oid in census_ids)
                         or (kind == "order_deleted" and oid not in census_ids and known is None)
                         or (kind == "order_changed" and known is not None and known["traded"] is None
                             and known["amount"] == amount and known["price"] == price))
            if reflected:
                self.stats["census_boundary_skipped"] += 1
                return
            self.stats["census_boundary_applied"] += 1
        if kind == "order_created":
            emit(kind="ADD", exchange_ts_ns=ts, receive_ts_ns=recv_ns, order_id=oid, side=side, price=price,
                 quantity=amount)
            state[oid] = {"amount": amount, "traded": traded, "price": price, "side": side}
            return
        common = {"exchange_ts_ns": ts, "receive_ts_ns": recv_ns, "order_id": oid, "side": side}
        if known is None:
            # Unknown identity: report it truthfully; the lifecycle replay classifies it.
            if kind == "order_deleted":
                emit(kind="EXECUTE", price=price, quantity=traded, **common) if amount == 0 and traded else \
                    emit(kind="CANCEL", **common)
            else:
                emit(kind="MODIFY", price=price, quantity=amount, **common)
            return
        remaining = known["amount"]
        if kind == "order_deleted":
            # Deletion semantics (from the development capture): zero remaining amount means the
            # remainder executed and amount_traded reports that final execution; a positive
            # remaining amount means the remainder was cancelled. Tracked remaining quantity is
            # authoritative; disagreeing source fields are counted, never used to repair state.
            if amount == 0:
                if traded != remaining:
                    self.stats["delete_traded_field_mismatch"] += 1
                emit(kind="EXECUTE", price=known["price"], quantity=remaining, **common)
            else:
                if amount != remaining:
                    self.stats["delete_remaining_mismatch"] += 1
                emit(kind="CANCEL", **common)
            state.pop(oid, None)
            return
        # order_changed (from the development capture): every observed change reduced the
        # remaining amount through an execution; amount_traded is sometimes cumulative and
        # sometimes per event, and the price field reports execution prices while an
        # aggressor sweeps. The tracked remaining reduction is therefore the executed
        # quantity at the order's own limit price; source fields are counted cross-checks.
        reduction = remaining - amount
        previous_traded = known["traded"]
        if price != known["price"]:
            self.stats["changed_price_reports"] += 1
        if reduction > 0:
            consistent = traded == reduction or (previous_traded is not None and traded == previous_traded + reduction)
            if not consistent:
                self.stats["changed_traded_field_unexplained"] += 1
            emit(kind="EXECUTE", price=known["price"], quantity=reduction, **common)
        elif reduction < 0:
            self.stats["changed_size_increase"] += 1
            emit(kind="MODIFY", quantity=amount, **common)
        else:
            self.stats["changed_without_quantity_change"] += 1
        known.update(amount=amount, traded=traded)


# ----------------------------------------------------------------------------- LOBSTER

LOBSTER_TYPES = {1: "submission", 2: "partial_cancellation", 3: "deletion", 4: "visible_execution",
                 5: "hidden_execution", 6: "cross_trade", 7: "trading_halt"}


def lobster_contract(ticker: str, tick_size: str = "0.0001") -> SourceContract:
    return SourceContract(
        source="LOBSTER message and orderbook files (NASDAQ TotalView-ITCH derived)", venue="nasdaq",
        instrument=ticker, tick_size=tick_size, lot_size="1",
        identity="Order reference number from ITCH; 0 or reused references are anomalies.",
        sequence="Message file row order; the orderbook file row i is the book after message i.",
        timestamps="Seconds after midnight with nanosecond decimals; no receive clock.",
        snapshot="No initial census: orders submitted before the window are left-censored.",
        priority="Price-time priority is the venue rule, but truncated files cannot establish queue "
                 "position for left-censored orders; FIFO is not claimed.",
        execution="Type 4 decrements the identified visible order; type 5 hidden and type 6 cross "
                  "trades are prints that never touch the visible book.",
        hidden_liquidity="Type 5 reveals that hidden executions occurred; hidden resting size is unknown.",
        fifo_established=False, reference_alignment="orderbook row after its message",
        adapter_version="lobster-1")


class LobsterAdapter:
    """LOBSTER rows: time, type, order_id, size, price, direction (1 buy / -1 sell)."""

    def __init__(self, message_path: str | Path, orderbook_path: str | Path | None, *, ticker: str,
                 levels: int, tick_size: str = "0.0001"):
        self.message_path, self.orderbook_path = Path(message_path), orderbook_path and Path(orderbook_path)
        self.contract = lobster_contract(ticker, tick_size)
        self.levels = levels
        self.stats: dict[str, Any] = {}

    def build(self) -> tuple[list[LifecycleEvent], list[dict]]:
        self.stats = {"messages": 0, "types": {}}
        events: list[LifecycleEvent] = [LifecycleEvent(sequence=0, kind="RESET")]
        references = []
        books = None
        if self.orderbook_path is not None:
            books = self.orderbook_path.read_text(encoding="utf-8").splitlines()
        for row_index, line in enumerate(self.message_path.read_text(encoding="utf-8").splitlines()):
            time_s, kind, order_id, size, price, direction = [x.strip() for x in line.split(",")[:6]]
            code = int(kind)
            self.stats["messages"] += 1
            self.stats["types"][LOBSTER_TYPES.get(code, "unknown")] = self.stats["types"].get(
                LOBSTER_TYPES.get(code, "unknown"), 0) + 1
            ts = exact_units(time_s, "0.000000001", "time")
            side = "BUY" if int(direction) == 1 else "SELL"
            fields = {"exchange_ts_ns": ts, "source_sequence": row_index + 1}
            price_ticks, quantity = int(price), int(size)
            if code == 1:
                events.append(LifecycleEvent(len(events), "ADD", order_id=order_id, side=side,
                                             price=price_ticks, quantity=quantity, **fields))
            elif code == 2:
                events.append(LifecycleEvent(len(events), "CANCEL", order_id=order_id, side=side,
                                             quantity=quantity, **fields))
            elif code == 3:
                events.append(LifecycleEvent(len(events), "CANCEL", order_id=order_id, side=side, **fields))
            elif code == 4:
                events.append(LifecycleEvent(len(events), "EXECUTE", order_id=order_id, side=side,
                                             price=price_ticks, quantity=quantity, **fields))
            elif code in (5, 6):
                events.append(LifecycleEvent(len(events), "TRADE", price=price_ticks, quantity=quantity,
                                             aggressor_side="SELL" if side == "BUY" else "BUY", **fields))
            elif code == 7:
                events.append(LifecycleEvent(len(events), "GAP", **fields))
            else:
                raise LifecycleInputError(f"unknown LOBSTER event type {code}")
            if books is not None:
                values = [int(x) for x in books[row_index].split(",")]
                asks = [[values[i], values[i + 1]] for i in range(0, 4 * self.levels, 4) if values[i] > 0
                        and values[i] != 9999999999]
                bids = [[values[i + 2], values[i + 3]] for i in range(0, 4 * self.levels, 4) if values[i + 2] > 0
                        and values[i + 2] != -9999999999]
                references.append({"after_sequence": len(events) - 1, "bids": bids, "asks": asks})
        return events, references


# ----------------------------------------------------------------------------- validation


def validate_source(adapter, semantics: LifecycleSemantics, *, depth: int = 10) -> dict:
    """Replay twice for determinism and compare with independent references.

    Lifecycle anomalies are counted, not repaired. Reference agreement is exact
    top-``depth`` equality; missing references leave agreement unavailable.
    """
    runs = []
    for _ in range(2):
        events, references = adapter.build()
        book = OrderLifecycleBook(semantics)
        comparison = ReferenceComparison(depth)
        refs = sorted(references, key=lambda r: r["after_sequence"])
        index = 0
        for event in events:
            while index < len(refs) and refs[index]["after_sequence"] < event.sequence:
                comparison.compare(book, refs[index])
                index += 1
            book.apply(event)
        for reference in refs[index:]:
            comparison.compare(book, reference)
        summary = book.finish()
        summary["reference_comparison"] = comparison.to_dict()
        runs.append((summary, book))
        if hasattr(adapter, "_built"):
            adapter._built = None  # rebuild from source bytes for the determinism check
    first, book = runs[0]
    deterministic = first["final_state_sha256"] == runs[1][0]["final_state_sha256"]
    return {"contract": adapter.contract.to_dict(), "adapter_stats": dict(adapter.stats),
            "semantics": semantics.to_dict(), "deterministic_replay": deterministic,
            "replay_sha256": [r[0]["final_state_sha256"] for r in runs], **first,
            "executions": len(book.executions)}


def m1_gate(result: dict) -> dict:
    """The frozen M1 gate (configs/v05/protocol.json success_gates.M1)."""
    agreement = result["reference_comparison"]["exact_fraction"]
    unexplained = result["report"]["unexplained_anomalies"]
    if agreement is None:
        status = "NOT_AVAILABLE"
    elif not result["deterministic_replay"] or unexplained or agreement < 0.90:
        status = "FAILED"
    elif agreement >= 0.99:
        status = "ESTABLISHED"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "E1_source_aggregate_agreement": agreement,
            "deterministic_replay": result["deterministic_replay"], "unexplained_anomalies": unexplained,
            "gate": "ESTABLISHED: deterministic AND zero unexplained anomalies AND E1 >= 0.99; "
                    "INCONCLUSIVE: deterministic AND 0.90 <= E1 < 0.99 (and no unexplained anomalies); "
                    "FAILED otherwise"}


def tolerant_reference_agreement(events: list[LifecycleEvent], references: list[dict],
                                 semantics: LifecycleSemantics, *, window_ms: int = 100, depth: int = 10) -> dict:
    """SECONDARY diagnostic, not the frozen E1: does any replayed state in
    [reference time - window, reference time] exactly equal the reference top-N?

    It measures whether mismatches are timing ambiguity (a lagged or coarsely
    timestamped reference) rather than a wrong reconstructed state.
    """
    window = window_ms * 1_000_000
    targets = [((tuple(map(tuple, r["bids"][:depth])), tuple(map(tuple, r["asks"][:depth]))),
                r["exchange_ts_ns"], r["after_sequence"]) for r in references if r.get("exchange_ts_ns") is not None]
    book = OrderLifecycleBook(semantics)
    recent: list[tuple[int, tuple]] = []
    by_position: dict[int, list] = {}
    for target in targets:
        by_position.setdefault(target[2], []).append(target)
    matched = compared = 0
    horizon_needed = sorted(t[1] for t in targets)
    for event in events:
        book.apply(event)
        ts = event.exchange_ts_ns
        if book.initialized and ts is not None:
            index = bisect.bisect_left(horizon_needed, ts)
            if index < len(horizon_needed) and horizon_needed[index] - ts <= window:
                state = book.aggregate(depth)
                recent.append((ts, (tuple(map(tuple, state["bids"])), tuple(map(tuple, state["asks"])))))
        for target, ref_ts, _ in by_position.pop(event.sequence, []):
            compared += 1
            recent = [(t, s) for t, s in recent if ref_ts - t <= window]
            if any(s == target for t, s in recent if t <= ref_ts):
                matched += 1
    return {"definition": f"exact top-{depth} match with any replayed state within {window_ms} ms before the "
                          "reference timestamp; SECONDARY diagnostic, not the frozen E1",
            "compared": compared, "matched": matched, "fraction": matched / compared if compared else None}

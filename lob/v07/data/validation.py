"""Stream-level market-data validation, serialization and checksums (workstream 1).

``validate_stream(records, spec, date=...)`` scans canonical records once and
returns quality flags with counts and examples; it never repairs data. Flags:

* ``timestamp_reversal`` (local capture time decreases), ``exchange_clock_regression``;
* ``duplicate_event`` (identical consecutive records; repeated trade ids);
* ``missing_sequence`` (gaps in source sequence numbers, when present);
* ``crossed_book`` / ``locked_book`` (best bid above / equal to best ask after a
  local-timestamp group, aggregate L2 only);
* ``negative_value`` (schema violations: negative price or size);
* ``impossible_cancel`` / ``unknown_order_id`` / ``duplicate_order_id`` (order-level only);
* ``gap`` (local capture gaps longer than ``gap_s``) and explicit ``gap`` records;
* ``timestamp_resolution_loss`` (fraction of timestamps on a whole-millisecond grid);
* ``corrupt_interval`` (runs of crossed/locked groups longer than ``corrupt_s``);
* ``missing_channel`` (no book or no trade records);
* ``timezone_error`` (exchange timestamps outside the declared UTC date +- 1 h).

Each flag has a severity (``FAIL`` or ``WARN``); the stream status is the worst.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from typing import Iterable

from .schema import Record, SchemaError, VenueSpec

SEVERITY = {"timestamp_reversal": "FAIL", "exchange_clock_regression": "WARN", "duplicate_event": "WARN",
            "missing_sequence": "WARN", "crossed_book": "WARN", "locked_book": "WARN", "negative_value": "FAIL",
            "impossible_cancel": "FAIL", "unknown_order_id": "WARN", "duplicate_order_id": "FAIL", "gap": "WARN",
            "timestamp_resolution_loss": "WARN", "corrupt_interval": "FAIL", "missing_channel": "WARN",
            "timezone_error": "FAIL"}


def to_json(record: Record) -> str:
    return record.canonical()


def from_json(text: str) -> Record:
    values = json.loads(text)
    for key in ("price", "amount"):
        if values.get(key) is not None:
            values[key] = Decimal(values[key])
    return Record(**values)


def checksum(records: Iterable[Record]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(to_json(r).encode())
        h.update(b"\n")
    return h.hexdigest()


def validate_stream(records: Iterable[Record], spec: VenueSpec, *, date: str | None = None, gap_s: float = 60.0,
                    corrupt_s: float = 5.0, examples: int = 3) -> dict:
    level = spec.capability_level
    counts = {k: 0 for k in SEVERITY}
    found: dict[str, list] = {k: [] for k in SEVERITY}

    def flag(name: str, detail) -> None:
        counts[name] += 1
        if len(found[name]) < examples:
            found[name].append(detail)
    day_lo = day_hi = None
    if date:
        start = datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp() * 1e6
        day_lo, day_hi = start - 3600e6, start + 25 * 3600e6
    last_local = None
    last_exchange: dict[str, int] = {}
    previous: Record | None = None
    trade_ids: set = set()
    last_sequence = None
    bids: dict[Decimal, Decimal] = {}
    asks: dict[Decimal, Decimal] = {}
    live: dict[str, Decimal] = {}
    group_local = None
    bad_since = None
    n = book = trades = whole_ms = 0

    def close_group(at):
        nonlocal bad_since
        if not bids or not asks:
            return
        best_bid, best_ask = max(bids), min(asks)
        bad = best_bid >= best_ask
        if best_bid > best_ask:
            flag("crossed_book", at)
        elif best_bid == best_ask:
            flag("locked_book", at)
        if bad and bad_since is None:
            bad_since = at
        elif not bad and bad_since is not None:
            if (at - bad_since) / 1e6 > corrupt_s:
                flag("corrupt_interval", [bad_since, at])
            bad_since = None

    for r in records:
        n += 1
        try:
            r.validate(level)
        except SchemaError as exc:
            flag("negative_value", str(exc))
            continue
        if last_local is not None and r.local_timestamp_us < last_local:
            flag("timestamp_reversal", r.local_timestamp_us)
        if last_local is not None and (r.local_timestamp_us - last_local) / 1e6 > gap_s:
            flag("gap", [last_local, r.local_timestamp_us])
        channel = "trade" if r.kind == "trade" else "book"
        if channel in last_exchange and r.timestamp_us < last_exchange[channel]:
            flag("exchange_clock_regression", r.timestamp_us)
        if previous is not None and to_json(previous) == to_json(r):
            flag("duplicate_event", r.local_timestamp_us)
        if day_lo is not None and r.timestamp_us and not (day_lo <= r.timestamp_us <= day_hi):
            flag("timezone_error", r.timestamp_us)
        whole_ms += int(r.timestamp_us % 1000 == 0)
        sequence = r.extra.get("sequence") if r.extra else None
        if sequence is not None:
            if last_sequence is not None and sequence != last_sequence + 1:
                flag("missing_sequence", [last_sequence, sequence])
            last_sequence = sequence
        if r.kind == "gap":
            flag("gap", r.local_timestamp_us)
        elif r.kind == "trade":
            trades += 1
            if r.trade_id:
                if r.trade_id in trade_ids:
                    flag("duplicate_event", f"trade {r.trade_id}")
                trade_ids.add(r.trade_id)
        elif r.kind in {"book_snapshot", "book_delta"}:
            book += 1
            if group_local is not None and r.local_timestamp_us != group_local:
                close_group(group_local)
            group_local = r.local_timestamp_us
            side = bids if r.side == "bid" else asks
            if r.amount == 0:
                side.pop(r.price, None)
            else:
                side[r.price] = r.amount
        elif r.kind == "order_add":
            book += 1
            if r.order_id in live and not r.extra.get("census"):
                flag("duplicate_order_id", r.order_id)
            live[r.order_id] = r.amount
        elif r.kind in {"order_cancel", "order_execute", "order_modify"}:
            book += 1
            if r.order_id not in live:
                flag("unknown_order_id", r.order_id)
            elif r.kind == "order_cancel" and r.amount is not None and r.amount > live[r.order_id]:
                flag("impossible_cancel", r.order_id)
            elif r.kind == "order_cancel":
                live.pop(r.order_id, None)
            elif r.kind == "order_execute":
                remaining = live[r.order_id] - (r.amount or 0)
                if remaining <= 0:
                    live.pop(r.order_id, None)
                else:
                    live[r.order_id] = remaining
        last_local = r.local_timestamp_us if last_local is None else max(last_local, r.local_timestamp_us)
        last_exchange[channel] = r.timestamp_us
        previous = r
    if group_local is not None:
        close_group(group_local)
    if book == 0:
        flag("missing_channel", "book")
    if trades == 0:
        flag("missing_channel", "trades")
    resolution = whole_ms / n if n else 0.0
    if n and resolution > 0.99:
        flag("timestamp_resolution_loss", f"{resolution:.4f} of timestamps on a whole-millisecond grid")
    severities = [SEVERITY[k] for k, v in counts.items() if v]
    status = "FAIL" if "FAIL" in severities else "WARN" if severities else "PASS"
    return {"status": status, "records": n, "book_records": book, "trade_records": trades,
            "whole_millisecond_fraction": resolution, "flags": {k: v for k, v in counts.items() if v},
            "examples": {k: v for k, v in found.items() if v}, "severity": SEVERITY, "venue": asdict(spec) | {
                "tick": str(spec.tick)}}

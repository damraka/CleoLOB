"""Bounded, strict CSV and JSONL adapters for user-provided canonical events."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from .schema import EventSchemaError, MarketEvent
from .validation import (DEFAULT_MAX_EVENTS, DataValidationError, QualityIssue,
                         QualityReport, check_event_limit, validate_events)

DEFAULT_MAX_SOURCE_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_ROW_BYTES = 2 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EventSchemaError("DUPLICATE_FIELD", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise EventSchemaError("INVALID_NUMBER", f"non-finite JSON number {value}")


def _json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _read_file(path: str | Path, *, max_events: int, max_source_bytes: int,
               max_row_bytes: int) -> tuple[tuple[MarketEvent, ...], QualityReport]:
    check_event_limit(max_events)
    check_event_limit(max_source_bytes)
    check_event_limit(max_row_bytes)
    source = Path(path).expanduser().resolve()
    report = QualityReport(source=str(source))
    digest = hashlib.sha256()
    events: list[MarketEvent] = []
    row_number = 0

    def lines() -> Iterator[str]:
        nonlocal row_number
        total = 0
        with source.open("rb") as handle:
            while True:
                raw = handle.readline(max_row_bytes + 1)
                if not raw:
                    break
                row_number += 1
                total += len(raw)
                if total > max_source_bytes:
                    raise EventSchemaError("RESOURCE_LIMIT", f"source exceeds max_source_bytes={max_source_bytes}")
                if len(raw) > max_row_bytes:
                    raise EventSchemaError("RESOURCE_LIMIT", f"row exceeds max_row_bytes={max_row_bytes}")
                digest.update(raw)
                yield raw.decode("utf-8-sig" if row_number == 1 else "utf-8")

    def append(row: Any) -> None:
        if len(events) >= max_events:
            raise EventSchemaError("RESOURCE_LIMIT", f"dataset exceeds max_events={max_events}")
        events.append(MarketEvent.from_mapping(row))

    try:
        if source.suffix.lower() not in {".csv", ".jsonl", ".ndjson"}:
            raise EventSchemaError("UNSUPPORTED_FORMAT", "supported canonical formats are .csv, .jsonl and .ndjson")
        if not source.is_file():
            raise EventSchemaError("SOURCE_ERROR", "source is not a regular file")
        if source.stat().st_size > max_source_bytes:
            raise EventSchemaError("RESOURCE_LIMIT", f"source exceeds max_source_bytes={max_source_bytes}")
        if source.suffix.lower() == ".csv":
            reader = csv.DictReader(lines(), strict=True)
            names = reader.fieldnames
            if names is None:
                raise EventSchemaError("EMPTY_DATASET", "CSV is empty")
            if len(set(names)) != len(names):
                raise EventSchemaError("DUPLICATE_FIELD", "duplicate CSV column names")
            required = {"timestamp_ns", "sequence", "event_type", "symbol", "venue"}
            if not required.issubset(names):
                raise EventSchemaError("MISSING_FIELD", f"CSV header missing required fields: {sorted(required - set(names))}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise EventSchemaError("CORRUPT_ROW", "CSV row width differs from header")
                if row.get("orders"):
                    row["orders"] = _json(row["orders"])
                append(row)
        else:
            for line in lines():
                if not line.strip():
                    raise EventSchemaError("CORRUPT_ROW", "blank JSONL records are not allowed")
                append(_json(line))
        report = validate_events(events, max_events=max_events)
        report.source = str(source)
        report.source_sha256 = digest.hexdigest()
    except (EventSchemaError, OSError, UnicodeError, json.JSONDecodeError, csv.Error, RecursionError, ValueError) as exc:
        code = exc.code if isinstance(exc, EventSchemaError) else "CORRUPT_ROW"
        if isinstance(exc, OSError):
            code = "SOURCE_ERROR"
        report.event_count = len(events)
        report.issues.append(QualityIssue(code, str(exc), row=row_number or None))
        # A partial digest is not a source hash; never label it as one.
        raise DataValidationError(report) from exc
    if not report.valid:
        raise DataValidationError(report)
    return tuple(events), report


def load_events(path: str | Path, *, max_events: int = DEFAULT_MAX_EVENTS,
                max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
                max_row_bytes: int = DEFAULT_MAX_ROW_BYTES) -> tuple[MarketEvent, ...]:
    """Load and fully validate a canonical feed, or raise DataValidationError.

    Resource limits reject the entire input, rather than returning a truncated
    prefix that could be mistaken for a valid full dataset.
    """
    return _read_file(path, max_events=max_events, max_source_bytes=max_source_bytes,
                      max_row_bytes=max_row_bytes)[0]


def validate_file(path: str | Path, *, max_events: int = DEFAULT_MAX_EVENTS,
                   max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
                   max_row_bytes: int = DEFAULT_MAX_ROW_BYTES) -> QualityReport:
    """Return a quality report for valid or invalid input; never repair it."""
    try:
        return _read_file(path, max_events=max_events, max_source_bytes=max_source_bytes,
                          max_row_bytes=max_row_bytes)[1]
    except DataValidationError as exc:
        return exc.report

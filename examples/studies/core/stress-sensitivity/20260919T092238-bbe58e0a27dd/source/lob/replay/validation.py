"""Deterministic quality reports and fail-closed replay preflight checks."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .book import HistoricalBook, ReconstructionError
from .schema import MarketEvent

DEFAULT_MAX_EVENTS = 1_000_000


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str
    row: int | None = None
    sequence: int | None = None
    severity: str = "ERROR"

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "code": self.code, "message": self.message,
                "row": self.row, "sequence": self.sequence}


@dataclass
class QualityReport:
    event_count: int = 0
    checked_events: int = 0
    issues: list[QualityIssue] = field(default_factory=list)
    event_types: dict[str, int] = field(default_factory=dict)
    canonical_sha256: str | None = None
    source: str | None = None
    source_sha256: str | None = None
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "ERROR" for issue in self.issues)

    @property
    def status(self) -> str:
        return "INVALID" if not self.valid else ("WARNING" if self.issues else "VALID")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "valid": self.valid, "event_count": self.event_count,
                "checked_events": self.checked_events, "event_types": dict(sorted(self.event_types.items())),
                "issues": [issue.to_dict() for issue in self.issues], "canonical_sha256": self.canonical_sha256,
                "source": self.source, "source_sha256": self.source_sha256,
                "first_timestamp_ns": self.first_timestamp_ns, "last_timestamp_ns": self.last_timestamp_ns}


class DataValidationError(ValueError):
    """Critical dataset failures, always accompanied by a serializable report."""

    def __init__(self, report: QualityReport):
        self.report = report
        message = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues[:3])
        super().__init__(message or "historical data validation failed")


def check_event_limit(max_events: int) -> None:
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
        raise ValueError("max_events must be a positive integer")


def bounded_events(events: Iterable[MarketEvent], max_events: int) -> tuple[MarketEvent, ...]:
    """Consume at most max_events + 1 values; exceeding the limit is an error."""
    check_event_limit(max_events)
    output: list[MarketEvent] = []
    for index, event in enumerate(events, start=1):
        if index > max_events:
            raise DataValidationError(QualityReport(
                event_count=index, issues=[QualityIssue("RESOURCE_LIMIT", f"dataset exceeds max_events={max_events}", row=index)]))
        if not isinstance(event, MarketEvent):
            raise DataValidationError(QualityReport(
                event_count=index, issues=[QualityIssue("INVALID_EVENT", "expected canonical MarketEvent", row=index)]))
        output.append(event)
    return tuple(output)


def canonical_digest(events: Iterable[MarketEvent]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_events(events: Iterable[MarketEvent], *, max_events: int = DEFAULT_MAX_EVENTS) -> QualityReport:
    """Validate input order without sorting, repairing, or inventing messages.

    Stop state validation at the first critical transition: subsequent lifecycle
    errors would be ambiguous. ``checked_events`` discloses the accepted prefix.
    A nonzero first sequence is valid; later gaps are always critical.
    """
    try:
        materialized = bounded_events(events, max_events)
    except DataValidationError as exc:
        return exc.report
    report = QualityReport(event_count=len(materialized), canonical_sha256=canonical_digest(materialized))
    report.event_types = dict(Counter(event.event_type.value for event in materialized))
    if not materialized:
        report.issues.append(QualityIssue("EMPTY_DATASET", "dataset must contain at least one event"))
        return report
    report.first_timestamp_ns = materialized[0].timestamp_ns
    report.last_timestamp_ns = materialized[-1].timestamp_ns
    book = HistoricalBook()
    for row, event in enumerate(materialized, start=1):
        try:
            book.apply(event)
        except ReconstructionError as exc:
            report.issues.append(QualityIssue(exc.code, str(exc), row=row, sequence=event.sequence))
            break
        report.checked_events += 1
    return report

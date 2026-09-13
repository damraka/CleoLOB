"""Deterministic, validated event-step historical replay."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .book import HistoricalBook
from .io import DEFAULT_MAX_ROW_BYTES, DEFAULT_MAX_SOURCE_BYTES, _read_file
from .schema import MarketEvent
from .validation import (DEFAULT_MAX_EVENTS, DataValidationError,
                         bounded_events, check_event_limit, validate_events)


class HistoricalReplay:
    """Preflight the complete stream before exposing any historical book state.

    Replay uses feed timestamps, independent of wall time. Explicit step() works
    while paused; pause blocks run()/advance_to(). reset() deterministically
    rewinds. This interface intentionally does not submit counterfactual orders.
    """

    def __init__(self, events: Iterable[MarketEvent], *, max_events: int = DEFAULT_MAX_EVENTS):
        self._events = bounded_events(events, max_events)
        self.quality = validate_events(self._events, max_events=max_events)
        if not self.quality.valid:
            raise DataValidationError(self.quality)
        self.book = HistoricalBook()
        self.position = 0
        self.paused = False

    @property
    def complete(self) -> bool:
        return self.position == len(self._events)

    @property
    def timestamp_ns(self) -> int | None:
        return None if self.position == 0 else self._events[self.position - 1].timestamp_ns

    def step(self) -> MarketEvent | None:
        """Apply one event; return None at EOF. Manual stepping ignores pause."""
        if self.complete:
            return None
        event = self._events[self.position]
        self.book.apply(event)
        self.position += 1
        return event

    def run(self, *, max_events: int | None = None) -> int:
        """Advance to EOF or an explicit batch limit; return processed count."""
        if max_events is not None:
            check_event_limit(max_events)
        processed = 0
        while not self.paused and not self.complete and (max_events is None or processed < max_events):
            self.step()
            processed += 1
        return processed

    def advance_to(self, timestamp_ns: int) -> int:
        """Apply every event at or before a feed timestamp, including ties."""
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a nonnegative integer")
        if self.timestamp_ns is not None and timestamp_ns < self.timestamp_ns:
            raise ValueError("cannot reverse replay time; use reset()")
        processed = 0
        while not self.paused and not self.complete and self._events[self.position].timestamp_ns <= timestamp_ns:
            self.step()
            processed += 1
        return processed

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def reset(self) -> None:
        self.book = HistoricalBook()
        self.position = 0
        self.paused = False

    def summary(self) -> dict[str, Any]:
        return {"mode": "historical_reconstruction", "event_count": len(self._events),
                "processed_events": self.position, "complete": self.complete, "paused": self.paused,
                "canonical_sha256": self.quality.canonical_sha256, "quality": self.quality.to_dict(),
                "book": self.book.snapshot(), "execution_count": len(self.book.executions),
                "executed_quantity": sum(execution.quantity for execution in self.book.executions),
                "trade_print_count": len(self.book.prints),
                "trade_print_quantity": sum(event.quantity for event in self.book.prints),
                "assumptions": ["One symbol and venue per stream; contiguous sequences and nondecreasing nanosecond timestamps.",
                                "Order-level continuous book; locked/crossed books and ID reuse require adapter handling.",
                                "EXECUTE decrements the specified resting ID; TRADE is a separate non-mutating print.",
                                "Size reductions retain priority; increases and repricing lose priority.",
                                "Historical executions and trade prints are separate feeds and must not be summed as volume.",
                                "No counterfactual agent fills, market impact, or wall-clock pacing are modeled."]}


def replay_file(path: str | Path, *, max_events: int = DEFAULT_MAX_EVENTS,
                max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
                max_row_bytes: int = DEFAULT_MAX_ROW_BYTES) -> dict[str, Any]:
    """Validate a whole canonical file, replay it, and return a JSON-safe report."""
    events, report = _read_file(path, max_events=max_events, max_source_bytes=max_source_bytes,
                                max_row_bytes=max_row_bytes)
    replay = HistoricalReplay(events, max_events=max_events)
    replay.quality = report
    replay.run()
    return replay.summary()

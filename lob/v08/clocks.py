"""Clock and event-time normalization (M4).

Nine clocks are kept apart: wall-clock, event time (event count), exchange
timestamp, receive (capture) timestamp, strategy observation, send,
acknowledgement, cancel request and cancel confirmation. ``normalize`` converts
any supported unit to integer nanoseconds deterministically; ``validate`` checks
the causal order each clock must respect (receive after exchange up to a
registered skew, observation after receive, send after observation, ack after
send, cancel confirmation after request) and per-stream monotonicity, and
reports violations instead of repairing them. ``event_time`` and
``volume_time`` give clocks for event-time and volume-time studies that are
independent of wall-clock sampling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLOCKS = ("wall", "event", "exchange", "receive", "observation", "send", "ack", "cancel_request", "cancel_confirm")
UNITS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
ORDER = (("exchange", "receive"), ("receive", "observation"), ("observation", "send"), ("send", "ack"),
         ("cancel_request", "cancel_confirm"))


def normalize(values, unit: str) -> np.ndarray:
    """Integer nanoseconds; float inputs are rounded to the nearest nanosecond (deterministic)."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {sorted(UNITS)}")
    a = np.asarray(values)
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)):
            raise ValueError("timestamps must be finite")
        return np.rint(a * UNITS[unit]).astype(np.int64)
    return a.astype(np.int64) * UNITS[unit]


@dataclass
class ClockSet:
    """Per-event clocks in integer nanoseconds (missing clocks absent)."""

    clocks: dict = field(default_factory=dict)

    def add(self, name: str, values, unit: str = "ns") -> ClockSet:
        if name not in CLOCKS:
            raise ValueError(f"unknown clock {name!r}")
        arr = normalize(values, unit)
        if self.clocks and len(arr) != len(next(iter(self.clocks.values()))):
            raise ValueError("all clocks must have one value per event")
        self.clocks[name] = arr
        return self

    def __len__(self) -> int:
        return len(next(iter(self.clocks.values()))) if self.clocks else 0


def validate(cs: ClockSet, *, receive_skew_ns: int = 50_000_000, monotonic: tuple[str, ...] = ("exchange", "receive")
             ) -> dict:
    issues = {}
    for a, b in ORDER:
        if a in cs.clocks and b in cs.clocks:
            tolerance = receive_skew_ns if (a, b) == ("exchange", "receive") else 0
            bad = np.nonzero(cs.clocks[b] < cs.clocks[a] - tolerance)[0]
            issues[f"{b}_before_{a}"] = {"count": int(len(bad)), "examples": bad[:5].tolist()}
    for name in monotonic:
        if name in cs.clocks:
            bad = np.nonzero(np.diff(cs.clocks[name]) < 0)[0] + 1
            issues[f"{name}_regression"] = {"count": int(len(bad)), "examples": bad[:5].tolist()}
    if "exchange" in cs.clocks and "receive" in cs.clocks:
        lag = (cs.clocks["receive"] - cs.clocks["exchange"]) / 1e6
        issues["receive_lag_ms"] = {"p50": float(np.median(lag)), "p99": float(np.quantile(lag, 0.99)),
                                    "min": float(lag.min())}
    violations = sum(v.get("count", 0) for v in issues.values())
    return {"events": len(cs), "clocks": sorted(cs.clocks), "issues": issues, "violations": int(violations),
            "status": "PASS" if violations == 0 else "FAIL", "repairs": "none"}


def event_time(n: int) -> np.ndarray:
    """Event-time clock: the event index (independent of wall-clock gaps)."""
    return np.arange(int(n), dtype=np.int64)


def volume_time(volumes, bucket: float) -> np.ndarray:
    """Volume-time clock: bucket index of cumulative traded volume."""
    if bucket <= 0:
        raise ValueError("bucket must be positive")
    return (np.cumsum(np.asarray(volumes, float)) // bucket).astype(np.int64)


def resample_wall(times_ns: np.ndarray, values: np.ndarray, step_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Last value at or before each wall-clock grid point (causal; no look-ahead)."""
    times_ns = np.asarray(times_ns, np.int64)
    grid = np.arange(times_ns[0], times_ns[-1] + 1, step_ns, dtype=np.int64)
    idx = np.searchsorted(times_ns, grid, side="right") - 1
    return grid, np.asarray(values)[idx]


def resample_events(values: np.ndarray, every: int) -> np.ndarray:
    """Every ``every``-th event (event-time sampling)."""
    return np.asarray(values)[:: int(every)]

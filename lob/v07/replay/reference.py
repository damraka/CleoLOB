"""Replay v3 integrity: agreement with independent reference snapshots, counterfactual separation, corrupt intervals.

* ``snapshot_agreement`` replays incremental L2 (``lob.replay.l2.L2Replay``) and, at
  each reference ``book_snapshot_5`` row (Tardis, captured independently from the
  same feed), compares the reconstructed top 5 levels per side at the same local
  timestamp: exact agreement, price agreement and absolute size error.
* ``CounterfactualChild`` keeps hypothetical child-order state strictly separate
  from observed book state: the observed book is never mutated, the child only
  records bounded fills from ``lob.v07.queue.models`` under a declared model.
* ``exclude_corrupt`` drops episodes that overlap corrupt intervals reported by
  stream validation (the corrupt-interval policy).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal
import gzip
from pathlib import Path

from ...replay.l2 import L2Replay
from ..data.tape import READER_LIMITS
from ..queue.models import MODELS, LevelEvent, fill


def _reference_rows(path: Path, levels: int):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            asks = [(Decimal(row[f"asks[{i}].price"]), Decimal(row[f"asks[{i}].amount"])) for i in range(levels)
                    if row.get(f"asks[{i}].price")]
            bids = [(Decimal(row[f"bids[{i}].price"]), Decimal(row[f"bids[{i}].amount"])) for i in range(levels)
                    if row.get(f"bids[{i}].price")]
            yield int(row["local_timestamp"]), int(row["timestamp"]), bids, asks


def snapshot_agreement(l2: Path, snapshots: Path, *, levels: int = 5, every: int = 50, limit: int | None = None) -> dict:
    """Compare every ``every``-th reference snapshot with the reconstruction at the same exchange timestamp."""
    reference = _reference_rows(snapshots, levels)
    replay = iter(L2Replay(l2, depth=levels, **READER_LIMITS))
    compared = exact = prices = 0
    size_error = 0.0
    state = None
    pending = next(reference, None)
    k = 0
    for current in replay:
        while pending is not None and pending[1] < current.timestamp_us:
            if state is not None and k % every == 0:
                _, _, bids, asks = pending
                ours_b, ours_a = list(state.bids[:levels]), list(state.asks[:levels])
                compared += 1
                exact += int(ours_b == bids and ours_a == asks)
                prices += int([p for p, _ in ours_b] == [p for p, _ in bids] and [p for p, _ in ours_a] == [p for p, _ in asks])
                for (p1, q1), (p2, q2) in zip(ours_b + ours_a, bids + asks):
                    if p1 == p2:
                        size_error += float(abs(q1 - q2))
            k += 1
            pending = next(reference, None)
            if limit and compared >= limit:
                pending = None
        state = current
        if pending is None:
            break
    return {"compared": compared, "exact_agreement": exact / compared if compared else None,
            "price_agreement": prices / compared if compared else None,
            "mean_abs_size_error_same_price": size_error / max(compared, 1), "levels": levels, "every": every,
            "rule": "reference snapshot vs reconstruction just before the first update with a later exchange timestamp"}


@dataclass
class CounterfactualChild:
    """A hypothetical passive child order; observed book state is read, never written."""

    side: int                     # +1 resting bid, -1 resting ask
    price: float
    qty: float
    model: str
    events: list = field(default_factory=list)
    level_at_join: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown queue model {self.model!r}")

    def observe(self, event: LevelEvent) -> None:
        self.events.append(event)

    def outcome(self) -> dict:
        result = fill(self.model, self.events, qty=self.qty, level_at_join=self.level_at_join)
        return {**result, "hypothetical": True, "observed_state_modified": False}


def exclude_corrupt(episodes: list[tuple[float, float]], intervals: list[tuple[float, float]]) -> dict:
    """Corrupt-interval policy: an episode [start, stop) overlapping any corrupt interval is excluded (not repaired)."""
    kept, dropped = [], []
    for start, stop in episodes:
        if any(a < stop and start < b for a, b in intervals):
            dropped.append((start, stop))
        else:
            kept.append((start, stop))
    return {"kept": kept, "excluded": dropped, "policy": "exclude overlapping episodes; never repair"}

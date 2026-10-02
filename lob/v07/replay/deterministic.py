"""Deterministic canonical-record replay with book hashes and checkpoints (workstream 4).

``BookReplay`` applies canonical aggregate-L2 records (``lob.v07.data.schema``)
to a price-level book. After every local-timestamp group it extends a hash chain
over the top ``levels`` of both sides, so two replays agree if and only if their
visible book sequences agree. ``checkpoint()`` captures the full state (book,
record index, chain head) as canonical JSON; ``BookReplay.resume`` continues
from it and must reproduce the same chain as an uninterrupted replay.
Nothing is repaired: a crossed book is counted, not fixed.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from typing import Iterable

from ..data.schema import Record

CHECKPOINT_SCHEMA = "cleolob-v07-replay-checkpoint-1"


class BookReplay:
    def __init__(self, *, levels: int = 10) -> None:
        self.levels = levels
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.index = 0
        self.groups = 0
        self.crossed_groups = 0
        self.chain = "0" * 64
        self._group_local: int | None = None
        self._in_snapshot = False

    def top(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        bids = sorted(self.bids.items(), key=lambda kv: -kv[0])[: self.levels]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[: self.levels]
        return [(str(p), str(q)) for p, q in bids], [(str(p), str(q)) for p, q in asks]

    def book_hash(self) -> str:
        bids, asks = self.top()
        return hashlib.sha256(json.dumps([bids, asks], separators=(",", ":")).encode()).hexdigest()

    def _close_group(self) -> None:
        if self._group_local is None:
            return
        self.groups += 1
        if self.bids and self.asks and max(self.bids) >= min(self.asks):
            self.crossed_groups += 1
        self.chain = hashlib.sha256(f"{self.chain}|{self._group_local}|{self.book_hash()}".encode()).hexdigest()

    def apply(self, record: Record) -> None:
        if record.kind not in {"book_snapshot", "book_delta"}:
            self.index += 1
            return
        if self._group_local is not None and record.local_timestamp_us != self._group_local:
            self._close_group()
        if record.kind == "book_snapshot" and not self._in_snapshot:
            self.bids.clear()
            self.asks.clear()
        self._in_snapshot = record.kind == "book_snapshot"
        side = self.bids if record.side == "bid" else self.asks
        if record.amount == 0:
            side.pop(record.price, None)
        else:
            side[record.price] = record.amount
        self._group_local = record.local_timestamp_us
        self.index += 1

    def run(self, records: Iterable[Record], *, stop: int | None = None) -> BookReplay:
        for record in records:
            if stop is not None and self.index >= stop:
                break
            self.apply(record)
        return self

    def finish(self) -> str:
        self._close_group()
        self._group_local = None
        return self.chain

    def checkpoint(self) -> str:
        state = {"schema": CHECKPOINT_SCHEMA, "levels": self.levels, "index": self.index, "groups": self.groups,
                 "crossed_groups": self.crossed_groups, "chain": self.chain, "group_local": self._group_local,
                 "in_snapshot": self._in_snapshot,
                 "bids": sorted([str(p), str(q)] for p, q in self.bids.items()),
                 "asks": sorted([str(p), str(q)] for p, q in self.asks.items())}
        text = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return json.dumps({"state": state, "sha256": hashlib.sha256(text.encode()).hexdigest()}, sort_keys=True)

    @classmethod
    def resume(cls, checkpoint: str, records: Iterable[Record]) -> BookReplay:
        """Restore from ``checkpoint`` and skip the first ``index`` records of ``records`` (same source)."""
        stored = json.loads(checkpoint)
        state = stored["state"]
        text = json.dumps(state, sort_keys=True, separators=(",", ":"))
        if state.get("schema") != CHECKPOINT_SCHEMA or hashlib.sha256(text.encode()).hexdigest() != stored["sha256"]:
            raise ValueError("checkpoint schema or integrity mismatch")
        replay = cls(levels=state["levels"])
        replay.bids = {Decimal(p): Decimal(q) for p, q in state["bids"]}
        replay.asks = {Decimal(p): Decimal(q) for p, q in state["asks"]}
        replay.index, replay.groups, replay.chain = state["index"], state["groups"], state["chain"]
        replay.crossed_groups, replay._group_local = state["crossed_groups"], state["group_local"]
        replay._in_snapshot = state["in_snapshot"]
        iterator = iter(records)
        for _ in range(state["index"]):
            next(iterator)
        for record in iterator:
            replay.apply(record)
        return replay

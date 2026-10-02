"""Tardis normalized CSV adapters (Deribit and BitMEX): incremental L2 levels and trades.

Both venues use Tardis's normalized ``incremental_book_L2`` and ``trades`` CSV
schemas, so one reader serves both; the ``VenueSpec`` carries venue semantics.
Rows are parsed as exact decimals and merged by local capture time (book rows
before trades at equal times, stable within each file). Nothing is repaired:
crossed books, gaps and clock regressions are reported by ``lob.v07.data.quality``.
"""
from __future__ import annotations

import csv
from decimal import Decimal
import gzip
import heapq
from pathlib import Path
from typing import Iterator

from ...replay.l2 import exact_decimal, timestamp_us
from ..data.schema import Record, SchemaError, venue_spec
from .base import Adapter

L2_COLUMNS = ("exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount")
TRADE_COLUMNS = ("exchange", "symbol", "timestamp", "local_timestamp", "id", "side", "price", "amount")
_SNAPSHOT = {"true": True, "false": False}


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else \
        path.open("r", encoding="utf-8", newline="")


class TardisAdapter(Adapter):
    def __init__(self, venue: str, instrument: str, *, l2: str | Path | None, trades: str | Path | None) -> None:
        super().__init__()
        self.spec = venue_spec(venue, instrument)
        if self.spec.capability_level != "aggregate_l2":
            raise SchemaError("Tardis normalized files are aggregate L2")
        self.l2, self.trades = (Path(l2) if l2 else None), (Path(trades) if trades else None)

    def _check_header(self, reader: csv.DictReader, expected: tuple[str, ...], name: str) -> None:
        missing = [c for c in expected if c not in (reader.fieldnames or [])]
        if missing:
            raise SchemaError(f"{name}: missing columns {missing}")

    def _book(self) -> Iterator[tuple[int, int, int, Record]]:
        with _open(self.l2) as stream:
            reader = csv.DictReader(stream)
            self._check_header(reader, L2_COLUMNS, "incremental_book_L2")
            for i, row in enumerate(reader):
                self._identity(row)
                snapshot = _SNAPSHOT.get(row["is_snapshot"])
                if snapshot is None or row["side"] not in {"bid", "ask"}:
                    raise SchemaError(f"incremental_book_L2 row {i}: invalid is_snapshot or side")
                local = timestamp_us(row["local_timestamp"], "local_timestamp")
                yield local, 0, i, Record("book_snapshot" if snapshot else "book_delta",
                                          timestamp_us(row["timestamp"], "timestamp"), local, row["side"],
                                          exact_decimal(row["price"], "price"),
                                          exact_decimal(row["amount"], "amount", allow_zero=True))

    def _trades(self) -> Iterator[tuple[int, int, int, Record]]:
        with _open(self.trades) as stream:
            reader = csv.DictReader(stream)
            self._check_header(reader, TRADE_COLUMNS, "trades")
            for i, row in enumerate(reader):
                self._identity(row)
                side = row["side"].lower()
                local = timestamp_us(row["local_timestamp"], "local_timestamp")
                yield local, 1, i, Record("trade", timestamp_us(row["timestamp"], "timestamp"), local,
                                          side if side in {"buy", "sell"} else None,
                                          exact_decimal(row["price"], "price"), exact_decimal(row["amount"], "amount"),
                                          trade_id=row["id"] or None)

    def _identity(self, row: dict) -> None:
        if row["exchange"] != self.spec.venue or row["symbol"] != self.spec.instrument:
            raise SchemaError(f"row belongs to {row['exchange']}/{row['symbol']}, not "
                              f"{self.spec.venue}/{self.spec.instrument}")

    def records(self) -> Iterator[Record]:
        streams = [s() for s, p in ((self._book, self.l2), (self._trades, self.trades)) if p is not None]
        counts = {"book_snapshot": 0, "book_delta": 0, "trade": 0}
        for _, _, _, record in heapq.merge(*streams, key=lambda item: item[:3]):
            counts[record.kind] += 1
            yield record
        self.stats = {"records": counts, "complete": True}


def tick_consistency(adapter: TardisAdapter, *, limit: int = 500_000) -> dict:
    """Fraction of book prices on the declared tick grid (a declaration check, not a repair)."""
    tick = adapter.spec.tick
    checked = off = 0
    for record in adapter.records():
        if record.kind in {"book_snapshot", "book_delta"}:
            checked += 1
            off += int(record.price % tick != Decimal(0))
            if checked >= limit:
                break
    return {"checked": checked, "off_grid": off, "off_grid_fraction": off / checked if checked else None}

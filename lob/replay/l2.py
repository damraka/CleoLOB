"""Bounded, exact aggregate-book reconstruction for Tardis incremental L2 CSV.

Rows retain capture order. A local-timestamp group is applied atomically before
its state is exposed, and a contiguous snapshot block resets the book only once.
This is market-by-price data: it provides neither order IDs nor queue position.
Crossed levels are retained and reported, never silently repaired.

Schema: https://docs.tardis.dev/downloadable-csv-files/data-types
Semantics: https://docs.tardis.dev/faq/order-books
"""
from __future__ import annotations

import csv
import gzip
import re
import zlib
from bisect import bisect_left, insort
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Iterator

from ..capabilities import CapabilityContract, CapabilityError, L2_CONTRACT


L2_COLUMNS = (
    "exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot",
    "side", "price", "amount",
)
_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z", re.ASCII)
Level = tuple[Decimal, Decimal]


L2_CAPABILITIES = MappingProxyType({
    "aggregate_depth": True,
    "order_identity": False,
    "exact_fifo_position": False,
    "quantity_ahead": False,
    "observed_order_fill": False,
    "counterfactual_passive_fill": False,
    "hidden_order_quantity": False,
})


def require_l2_capability(name: str) -> None:
    """Fail explicitly when a consumer requests evidence absent from L2."""
    # Keep the v0.3 public error wording and mapping while the formal contract
    # supports additional price-level observations used by new consumers.
    try:
        L2_CONTRACT.require(name)
    except ValueError as exc:
        if "Unknown capability" in str(exc):
            raise CapabilityError(name, f"Unknown L2 capability {name!r}") from exc
        raise CapabilityError(name, f"{name} is unavailable from aggregate L2: no individual order identity or FIFO") from exc


def exact_decimal(text: str, name: str, *, allow_zero: bool = False) -> Decimal:
    """Parse bounded nonnegative decimal text without binary float conversion."""
    if not isinstance(text, str) or len(text) > 80 or not _DECIMAL.fullmatch(text):
        raise ValueError(f"{name} must be unpadded, finite decimal text")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{name} is not a decimal") from exc
    if not value.is_finite() or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    if len(value.as_tuple().digits) > 38 or value.as_tuple().exponent < -30 or value.adjusted() > 30:
        raise ValueError(f"{name} exceeds supported decimal precision/range")
    return value


def timestamp_us(text: str, name: str) -> int:
    """Parse a nonnegative signed-64-bit microsecond timestamp exactly."""
    if not text or len(text) > 19 or not text.isascii() or not text.isdecimal():
        raise ValueError(f"{name} must be integer microseconds")
    value = int(text)
    if value > 2**63 - 1:
        raise ValueError(f"{name} exceeds signed 64-bit range")
    return value


@dataclass(frozen=True)
class L2State:
    timestamp_us: int
    local_timestamp_us: int
    exchange: str
    symbol: str
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    is_snapshot: bool
    bid_levels: int
    ask_levels: int
    rows_in_group: int

    @property
    def capability_contract(self) -> CapabilityContract:
        return L2_CONTRACT

    @property
    def capabilities(self) -> dict[str, bool]:
        return dict(L2_CAPABILITIES)

    def require_capability(self, name: str) -> None:
        require_l2_capability(name)

    @property
    def crossed(self) -> bool:
        return bool(self.bids and self.asks and self.bids[0][0] >= self.asks[0][0])

    @property
    def one_sided(self) -> bool:
        return bool(self.bids) != bool(self.asks)

    @property
    def empty(self) -> bool:
        return not self.bids and not self.asks


@dataclass(frozen=True)
class _Row:
    exchange: str
    symbol: str
    timestamp: int
    local_timestamp: int
    is_snapshot: bool
    side: str
    price: Decimal
    amount: Decimal


class L2Replay:
    """One-pass replay of one exchange/instrument, with bounded resource use.

    Iterate to exhaustion before treating ``stats['complete']`` as true. A
    consumer stopping early has only an unvalidated prefix, including when a
    gzip file has an unread corrupt trailer. Limits reject rather than truncate.
    ``depth`` limits exported states; all observed levels remain in the book.
    Counts of crossed/one-sided/empty books describe completed local-timestamp
    groups after the first snapshot. No source sequence numbers exist here, so
    timestamp checks cannot prove that upstream messages were not lost.
    """

    def __init__(
        self, path: str | Path, *, depth: int = 5,
        max_file_bytes: int = 256 * 1024**2,
        max_expanded_bytes: int = 2 * 1024**3,
        max_rows: int = 20_000_000,
        max_levels: int = 100_000,
        max_group_rows: int = 100_000,
        max_line_bytes: int = 64 * 1024,
    ):
        self.path = Path(path)
        for name, value in {
            "depth": depth, "max_file_bytes": max_file_bytes,
            "max_expanded_bytes": max_expanded_bytes, "max_rows": max_rows,
            "max_levels": max_levels, "max_group_rows": max_group_rows,
            "max_line_bytes": max_line_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if depth > max_levels:
            raise ValueError("depth cannot exceed max_levels")
        self.depth = depth
        self.max_file_bytes = max_file_bytes
        self.max_expanded_bytes = max_expanded_bytes
        self.max_rows = max_rows
        self.max_levels = max_levels
        self.max_group_rows = max_group_rows
        self.max_line_bytes = max_line_bytes
        self._started = False
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._bid_prices: list[Decimal] = []
        self._ask_prices: list[Decimal] = []
        self._stats = {
            "rows": 0, "groups": 0, "states": 0, "snapshots": 0,
            "presnapshot_rows": 0, "crossed_groups": 0,
            "one_sided_groups": 0, "empty_groups": 0,
            "missing_level_deletes": 0, "exchange_clock_regressions": 0,
            "max_local_gap_us": 0, "max_bid_levels": 0, "max_ask_levels": 0,
            "max_levels_observed": 0, "first_timestamp_us": None,
            "last_timestamp_us": None, "first_local_timestamp_us": None,
            "last_local_timestamp_us": None, "file_bytes": 0,
            "expanded_bytes": 0, "exchange": None, "symbol": None,
            "complete": False,
        }

    @property
    def capability_contract(self) -> CapabilityContract:
        return L2_CONTRACT

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    @property
    def capabilities(self) -> dict[str, bool]:
        return dict(L2_CAPABILITIES)

    def require_capability(self, name: str) -> None:
        require_l2_capability(name)

    def summary(self) -> dict:
        return self.stats

    def _rows(self) -> Iterator[_Row]:
        try:
            size = self.path.stat().st_size
            self._stats["file_bytes"] = size
            if size > self.max_file_bytes:
                raise ValueError("L2 source exceeds max_file_bytes")
            opener = gzip.open if self.path.name.endswith(".gz") else open
            with opener(self.path, "rb") as source:
                header = None
                physical_line = 0
                while True:
                    line = source.readline(self.max_line_bytes + 1)
                    if not line:
                        break
                    physical_line += 1
                    self._stats["expanded_bytes"] += len(line)
                    if len(line) > self.max_line_bytes:
                        raise ValueError(f"L2 line {physical_line} exceeds max_line_bytes")
                    if self._stats["expanded_bytes"] > self.max_expanded_bytes:
                        raise ValueError("L2 source exceeds max_expanded_bytes")
                    # CSV has one physical line per row. Multiline records are
                    # rejected to keep the per-record resource limit explicit.
                    try:
                        fields = next(csv.reader([line.decode("utf-8")], strict=True))
                    except (UnicodeError, csv.Error) as exc:
                        raise ValueError(f"Invalid UTF-8/CSV at L2 line {physical_line}") from exc
                    if header is None:
                        if len(fields) != len(L2_COLUMNS) or set(fields) != set(L2_COLUMNS):
                            raise ValueError("L2 CSV header must contain exactly the eight schema columns")
                        header = fields
                        continue
                    if physical_line - 1 > self.max_rows:
                        raise ValueError("L2 source exceeds max_rows")
                    if len(fields) != len(header):
                        raise ValueError(f"L2 line {physical_line} has wrong column count")
                    row = dict(zip(header, fields))
                    try:
                        for name in ("exchange", "symbol"):
                            value = row[name]
                            if not value or value != value.strip() or len(value) > 128 or any(
                                ord(char) < 32 for char in value
                            ):
                                raise ValueError(f"{name} must be a bounded, nonempty identifier")
                        if row["is_snapshot"] not in ("true", "false"):
                            raise ValueError("is_snapshot must be true or false")
                        if row["side"] not in ("bid", "ask"):
                            raise ValueError("side must be bid or ask")
                        yield _Row(
                            exchange=row["exchange"], symbol=row["symbol"],
                            timestamp=timestamp_us(row["timestamp"], "timestamp"),
                            local_timestamp=timestamp_us(row["local_timestamp"], "local_timestamp"),
                            is_snapshot=row["is_snapshot"] == "true", side=row["side"],
                            price=exact_decimal(row["price"], "price"),
                            amount=exact_decimal(row["amount"], "amount", allow_zero=True),
                        )
                    except ValueError as exc:
                        raise ValueError(f"Invalid L2 line {physical_line}: {exc}") from exc
                if header is None:
                    raise ValueError("L2 source is empty")
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError(f"Cannot fully read L2 source: {exc}") from exc

    def _update(self, row: _Row) -> None:
        levels, prices = (self._bids, self._bid_prices) if row.side == "bid" else (
            self._asks, self._ask_prices
        )
        if row.amount == 0:
            if row.price in levels:
                del levels[row.price]
                prices.pop(bisect_left(prices, row.price))
            else:
                self._stats["missing_level_deletes"] += 1
        else:
            if row.price not in levels:
                if len(self._bids) + len(self._asks) >= self.max_levels:
                    raise ValueError("L2 book exceeds max_levels")
                insort(prices, row.price)
            levels[row.price] = row.amount
        self._stats["max_bid_levels"] = max(self._stats["max_bid_levels"], len(self._bids))
        self._stats["max_ask_levels"] = max(self._stats["max_ask_levels"], len(self._asks))
        self._stats["max_levels_observed"] = max(
            self._stats["max_levels_observed"], len(self._bids) + len(self._asks)
        )

    def _state(self, last: _Row, group_rows: int, snapshot: bool) -> L2State:
        state = L2State(
            timestamp_us=last.timestamp, local_timestamp_us=last.local_timestamp,
            exchange=last.exchange, symbol=last.symbol,
            bids=tuple((price, self._bids[price]) for price in reversed(self._bid_prices[-self.depth:])),
            asks=tuple((price, self._asks[price]) for price in self._ask_prices[:self.depth]),
            is_snapshot=snapshot, bid_levels=len(self._bids), ask_levels=len(self._asks),
            rows_in_group=group_rows,
        )
        self._stats["states"] += 1
        self._stats["crossed_groups"] += int(state.crossed)
        self._stats["one_sided_groups"] += int(state.one_sided)
        self._stats["empty_groups"] += int(state.empty)
        return state

    def __iter__(self) -> Iterator[L2State]:
        if self._started:
            raise ValueError("L2Replay is one-pass; construct a new instance to replay again")
        self._started = True
        return self._iterate()

    def _iterate(self) -> Iterator[L2State]:
        last = None
        group_rows = 0
        group_snapshot = False
        initialized = False
        last_was_snapshot = False
        for row in self._rows():
            if last is not None:
                if (row.exchange, row.symbol) != (last.exchange, last.symbol):
                    raise ValueError("L2 source must contain one exchange and symbol")
                if row.local_timestamp < last.local_timestamp:
                    raise ValueError("L2 local_timestamp regressed; capture order cannot be repaired")
                if row.local_timestamp != last.local_timestamp:
                    self._stats["groups"] += 1
                    if initialized:
                        yield self._state(last, group_rows, group_snapshot)
                    group_rows = 0
                    group_snapshot = False
                if row.timestamp < last.timestamp:
                    self._stats["exchange_clock_regressions"] += 1
                self._stats["max_local_gap_us"] = max(
                    self._stats["max_local_gap_us"], row.local_timestamp - last.local_timestamp
                )
            else:
                self._stats.update(
                    first_timestamp_us=row.timestamp,
                    first_local_timestamp_us=row.local_timestamp,
                    exchange=row.exchange, symbol=row.symbol,
                )
            group_rows += 1
            if group_rows > self.max_group_rows:
                raise ValueError("L2 local_timestamp group exceeds max_group_rows")
            self._stats["rows"] += 1
            self._stats["last_timestamp_us"] = row.timestamp
            self._stats["last_local_timestamp_us"] = row.local_timestamp
            if row.is_snapshot and not last_was_snapshot:
                self._bids.clear()
                self._asks.clear()
                self._bid_prices.clear()
                self._ask_prices.clear()
                self._stats["snapshots"] += 1
                initialized = True
            group_snapshot = group_snapshot or row.is_snapshot
            if initialized:
                self._update(row)
            else:
                self._stats["presnapshot_rows"] += 1
            last_was_snapshot = row.is_snapshot
            last = row
        if not initialized:
            raise ValueError("L2 source has no initial snapshot")
        if last is not None:
            self._stats["groups"] += 1
            yield self._state(last, group_rows, group_snapshot)
        self._stats["complete"] = True

"""Canonical market-event schema for heterogeneous venues (M20).

Every connector converts native records into chunks of one structured numpy dtype, ``EVENT_DTYPE``, so
reconstruction, replay, feature extraction and columnar storage never depend on a venue format:

==============  =======  ===================================================================================
field           dtype    meaning
==============  =======  ===================================================================================
ts_exchange     int64    exchange timestamp, ns since the epoch (venue resolution recorded in the capability)
ts_receive      int64    local receive timestamp, ns (``NO_TS`` when the source has none, e.g. archives)
seq             int64    venue sequence / update id (``-1`` if none)
kind            uint8    ``EventKind``
side            int8     +1 bid / buyer-initiated, -1 ask / seller-initiated, 0 none
price           int64    integer ticks on the dataset grid
size            float64  lots on the dataset grid (absolute level size for LEVEL_SET; order size for orders)
order_id        int64    venue order id (L3 sources only; ``-1`` otherwise)
aux             int64    trade id, maker order id or other source-specific integer (``-1`` if none)
==============  =======  ===================================================================================

``Capability`` declares what a source can and cannot identify. It is attached to every dataset, propagated into
every derived artifact and checked by the claim-permission layer: an L2 source can never yield an L3 claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum

import numpy as np

NO_TS = np.iinfo(np.int64).min


class EventKind(IntEnum):
    SNAPSHOT_BEGIN = 1      # the following LEVEL_SET / ORDER_ADD events form a full snapshot
    SNAPSHOT_END = 2
    LEVEL_SET = 3           # MBP: absolute size at (side, price); 0 removes the level
    ORDER_ADD = 4           # MBO: new resting order (order_id, side, price, size)
    ORDER_MODIFY = 5        # MBO: size and/or price change of an existing order
    ORDER_DELETE = 6        # MBO: order removed (cancel or full fill; the trade stream says which)
    TRADE = 7               # print: side = aggressor side, aux = trade id
    QUOTE = 8               # L1 best bid/ask update (two events: side +1 and -1)
    DEPTH_BAND = 9          # aggregated depth within a percentage band (Binance bookDepth): price = band (bp)
    GAP = 10                # explicit gap marker: the stream is known to be incomplete from here


EVENT_DTYPE = np.dtype([("ts_exchange", "<i8"), ("ts_receive", "<i8"), ("seq", "<i8"), ("kind", "u1"),
                        ("side", "i1"), ("price", "<i8"), ("size", "<f8"), ("order_id", "<i8"), ("aux", "<i8")])

QUEUE_OBSERVABILITY = ("none", "aggregate_level", "inferred_position", "exact_order_id")


@dataclass(frozen=True)
class Capability:
    """What a source can identify; every field is a declared obligation, verified by quality checks where possible."""
    book: str                         # "none" | "l1" | "depth_bands" | "l2" | "l3"
    depth_levels: int | None          # visible price levels per side (None = unbounded / full)
    order_ids: bool
    trades: bool
    trade_side: bool
    trade_ids: bool
    exchange_ts_resolution_ns: int
    receive_ts: bool
    sequence_numbers: bool
    snapshots: bool
    queue_observability: str
    cancel_visibility: str            # "explicit" | "inferred_from_depth_minus_trades" | "none"
    hidden_liquidity: str = "unobservable"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.book not in ("none", "l1", "depth_bands", "l2", "l3"):
            raise ValueError(f"unknown book capability {self.book}")
        if self.queue_observability not in QUEUE_OBSERVABILITY:
            raise ValueError(f"unknown queue observability {self.queue_observability}")
        if self.book == "l3" and not self.order_ids:
            raise ValueError("an L3 capability requires order ids")
        if self.queue_observability == "exact_order_id" and self.book != "l3":
            raise ValueError("exact queue observability requires L3 order-level data")

    def to_dict(self) -> dict:
        return asdict(self)

    def permits(self, need: str) -> bool:
        """Whether a claim needing ``need`` is identifiable from this source."""
        rules = {
            "l1": self.book in ("l1", "l2", "l3"),
            "l2": self.book in ("l2", "l3"),
            "l3": self.book == "l3",
            "queue_position": self.queue_observability == "exact_order_id",
            "queue_clearance": self.book in ("l2", "l3") and self.trades,
            "trade_side": self.trades and self.trade_side,
            "explicit_cancels": self.cancel_visibility == "explicit",
            "hidden_liquidity": self.hidden_liquidity == "observable",
            "latency": self.receive_ts,
        }
        if need not in rules:
            raise KeyError(f"unknown requirement {need}")
        return rules[need]


@dataclass(frozen=True)
class Instrument:
    venue: str                        # e.g. "bybit"
    segment: str                      # e.g. "linear", "spot", "um", "exchange"
    symbol: str
    tick: float
    lot: float
    kind: str                         # "linear_perpetual" | "spot" | ...
    quote: str = "USDT"

    @property
    def stream(self) -> str:
        return f"{self.venue}-{self.segment}:{self.symbol}"


def empty(n: int = 0) -> np.ndarray:
    out = np.zeros(n, dtype=EVENT_DTYPE)
    out["ts_receive"] = NO_TS
    out["seq"] = -1
    out["order_id"] = -1
    out["aux"] = -1
    return out


def concat(chunks) -> np.ndarray:
    chunks = [c for c in chunks if len(c)]
    return np.concatenate(chunks) if chunks else empty(0)


def validate_events(ev: np.ndarray) -> dict:
    """Structural checks of a canonical event array (never repairs)."""
    if ev.dtype != EVENT_DTYPE:
        raise TypeError("not a canonical event array")
    kinds = {int(k) for k in np.unique(ev["kind"])} if len(ev) else set()
    unknown = kinds - {int(k) for k in EventKind}
    ts = ev["ts_exchange"]
    return {"events": int(len(ev)), "unknown_kinds": sorted(unknown),
            "ts_backwards": int((np.diff(ts) < 0).sum()) if len(ts) > 1 else 0,
            "negative_sizes": int((ev["size"] < 0).sum()), "nonfinite_sizes": int((~np.isfinite(ev["size"])).sum()),
            "by_kind": {EventKind(k).name: int((ev["kind"] == k).sum()) for k in sorted(kinds - unknown)}}


__all__ = ["EventKind", "EVENT_DTYPE", "NO_TS", "Capability", "Instrument", "QUEUE_OBSERVABILITY", "empty", "concat",
           "validate_events"]

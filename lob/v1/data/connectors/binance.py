"""Binance public-archive connector (M12): USD-M futures ``trades``, ``aggTrades``, ``bookDepth`` and spot trades.

Capability labels are deliberately narrow:

* ``bookDepth`` is **not** an order book: each row is the cumulative quantity and notional within a percentage
  band around the mid (``percentage`` in {-5, ..., -1, 1, ..., 5} and sub-percent bands where present), sampled at
  coarse intervals. It is stored as ``DEPTH_BAND`` events (``price`` = band in basis points) and its capability is
  ``depth_bands``; no queue, level or spread quantity may be derived from it;
* ``bookTicker`` (L1) stopped on 2024-03-30 for USD-M and is therefore unavailable for fresh periods;
* trades carry the aggressor side (``is_buyer_maker`` true = seller-initiated) and trade ids.

Archives are verified against the published ``.CHECKSUM`` files at acquisition. Timestamps are milliseconds for
USD-M; spot files switched to microseconds in 2025, which is detected from the magnitude and recorded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
import zipfile

import numpy as np
import pandas as pd

from ..schema import Capability, EventKind, Instrument, empty
from .base import Licence, Source, register

BASE = "https://data.binance.vision/data"
UM = BASE + "/futures/um/daily/{kind}/{sym}/{sym}-{kind}-{day}.zip"
SPOT = BASE + "/spot/daily/{kind}/{sym}/{sym}-{kind}-{day}.zip"
BOOK_TICKER_UNTIL = "2024-03-30"

INSTRUMENTS = {
    ("um", "BTCUSDT"): (0.1, 0.001, "linear_perpetual"),
    ("um", "ETHUSDT"): (0.01, 0.001, "linear_perpetual"),
    ("um", "SOLUSDT"): (0.01, 0.01, "linear_perpetual"),
    ("spot", "BTCUSDT"): (0.01, 0.00001, "spot"),
    ("spot", "ETHUSDT"): (0.01, 0.0001, "spot"),
}
TRADE_COLUMNS = {"um": ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"],
                 "spot": ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]}
DEPTH_COLUMNS = ["timestamp", "percentage", "depth", "notional"]


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"expected one member in {Path(path).name}")
        with archive.open(names[0]) as handle:
            first = handle.readline().decode("ascii", "replace")
        has_header = not first.split(",")[0].strip().lstrip("-").replace(".", "").isdigit()
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=0 if has_header else None, names=None if has_header else columns)
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {missing}")
    return frame


def _ns(t: np.ndarray) -> tuple[np.ndarray, str]:
    t = np.asarray(t, np.int64)
    if len(t) and np.median(t) > 1e15:
        return t * 1_000, "us"
    return t * 1_000_000, "ms"


def trade_events(path: Path, segment: str, instrument: Instrument) -> tuple[np.ndarray, dict]:
    f = _read_csv(path, TRADE_COLUMNS[segment][:6])
    maker = f["is_buyer_maker"]
    if maker.dtype != bool:
        maker = maker.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if maker.isna().any():
        raise ValueError("unparseable is_buyer_maker values")
    out = empty(len(f))
    out["ts_exchange"], unit = _ns(f["time"].to_numpy())
    out["kind"] = EventKind.TRADE
    out["side"] = np.where(maker.to_numpy(bool), -1, 1)
    px = f["price"].to_numpy(float) / instrument.tick
    out["price"] = np.round(px).astype(np.int64)
    out["size"] = f["qty"].to_numpy(float) / instrument.lot
    out["aux"] = f["id"].to_numpy(np.int64)
    meta = {"time_unit": unit, "off_tick": int((np.abs(px - np.round(px)) > 1e-6).sum()),
            "id_gaps": int((np.diff(out["aux"]) != 1).sum()) if len(out) > 1 else 0}
    return out, meta


def depth_band_events(path: Path) -> tuple[np.ndarray, dict]:
    f = _read_csv(path, DEPTH_COLUMNS)
    ts = pd.to_datetime(f["timestamp"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    out = empty(len(f))
    out["ts_exchange"] = ts
    out["kind"] = EventKind.DEPTH_BAND
    pct = f["percentage"].to_numpy(float)
    out["side"] = np.where(pct < 0, 1, -1)              # negative bands are below the mid (bids)
    out["price"] = np.round(pct * 100).astype(np.int64)  # band in basis points
    out["size"] = f["depth"].to_numpy(float)
    snapshots = np.unique(ts)
    meta = {"snapshots": int(len(snapshots)),
            "median_interval_s": float(np.median(np.diff(snapshots)) / 1e9) if len(snapshots) > 1 else None,
            "bands_bp": sorted({int(x) for x in np.round(pct * 100)})}
    return out, meta


class BinanceConnector:
    name = "binance"

    def instrument(self, segment: str, symbol: str, date: str) -> Instrument:
        tick, lot, kind = INSTRUMENTS[(segment, symbol)]
        return Instrument("binance", segment, symbol, tick, lot, kind)

    def sources(self, segment: str, symbol: str, date: str) -> list[Source]:
        tpl = UM if segment == "um" else SPOT
        out = [Source("trades", tpl.format(kind="trades", sym=symbol, day=date), f"binance-{segment}-trades",
                      tpl.format(kind="trades", sym=symbol, day=date) + ".CHECKSUM")]
        if segment == "um":
            out.append(Source("depth", UM.format(kind="bookDepth", sym=symbol, day=date), "binance-um-bookdepth",
                              UM.format(kind="bookDepth", sym=symbol, day=date) + ".CHECKSUM", optional=True))
        return out

    def capability(self, segment: str, symbol: str, date: str) -> Capability:
        return Capability(book="depth_bands" if segment == "um" else "none", depth_levels=None, order_ids=False,
                          trades=True, trade_side=True, trade_ids=True, exchange_ts_resolution_ns=1_000_000,
                          receive_ts=False, sequence_numbers=False, snapshots=False, queue_observability="none",
                          cancel_visibility="none",
                          notes=("bookDepth = cumulative depth within percentage bands; not an order book",
                                 f"bookTicker (L1) unavailable after {BOOK_TICKER_UNTIL}"))

    def licence(self) -> Licence:
        return Licence("Binance public data (data.binance.vision)", "https://github.com/binance/binance-public-data",
                       "2026-10-25", "research use under Binance terms; raw data never redistributed",
                       redistribution=False, commercial=False)

    def events(self, files: dict[str, Path], instrument: Instrument, *, chunk: int = 200_000) -> Iterator[np.ndarray]:
        if "depth" in files:
            ev, _ = depth_band_events(files["depth"])
            for i in range(0, len(ev), chunk):
                yield ev[i:i + chunk]

    def trades(self, files: dict[str, Path], instrument: Instrument) -> np.ndarray:
        return trade_events(files["trades"], instrument.segment, instrument)[0]


CONNECTOR = register(BinanceConnector())

__all__ = ["BinanceConnector", "CONNECTOR", "trade_events", "depth_band_events", "INSTRUMENTS", "BOOK_TICKER_UNTIL"]

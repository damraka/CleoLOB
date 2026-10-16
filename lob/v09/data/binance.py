"""Binance USD-M public-archive adapters (M17): ``bookTicker`` (L1 every change) and ``trades``.

bookTicker CSV: ``update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time,
event_time`` (ms; a header row may be absent in older files). trades CSV: ``id, price, qty, quote_qty, time,
is_buyer_maker`` (``is_buyer_maker`` true means the seller was the aggressor). Only L1 is observed: deeper levels are
CAPABILITY_MISSING and are never imputed.
"""
from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

BOOK_COLUMNS = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time",
                "event_time"]
TRADE_COLUMNS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"expected one member in {Path(path).name}")
        with archive.open(names[0]) as handle:
            first = handle.readline().decode("ascii", "replace")
        has_header = not first.split(",")[0].strip().lstrip("-").isdigit()
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=0 if has_header else None, names=None if has_header else columns)
    frame.columns = [c.strip() for c in frame.columns]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {missing}")
    return frame[columns]


def read_book_ticker(path: Path) -> dict[str, np.ndarray]:
    f = _read(path, BOOK_COLUMNS)
    return {"u": f["update_id"].to_numpy(np.int64), "bid": f["best_bid_price"].to_numpy(float),
            "bid_q": f["best_bid_qty"].to_numpy(float), "ask": f["best_ask_price"].to_numpy(float),
            "ask_q": f["best_ask_qty"].to_numpy(float), "t": f["transaction_time"].to_numpy(np.int64),
            "event_t": f["event_time"].to_numpy(np.int64)}


def read_trades(path: Path) -> dict[str, np.ndarray]:
    f = _read(path, TRADE_COLUMNS)
    maker = f["is_buyer_maker"]
    if maker.dtype != bool:
        maker = maker.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if maker.isna().any():
        raise ValueError("unparseable is_buyer_maker values")
    return {"id": f["id"].to_numpy(np.int64), "t": f["time"].to_numpy(np.int64) / 1000.0,
            "price": f["price"].to_numpy(float), "size": f["qty"].to_numpy(float),
            "side": np.where(maker.to_numpy(bool), -1, 1).astype(np.int8)}

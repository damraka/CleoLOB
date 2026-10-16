"""Bybit public-archive adapters (M17): ``ob500``/``ob200`` snapshot+delta order books and public trades.

Order-book archive: a zip holding one JSON object per line,
``{"topic", "type": "snapshot"|"delta", "ts": ms, "data": {"s", "b": [[px, sz]...], "a": [...], "u", "seq"}}``.
Sizes are absolute level sizes (``0`` removes the level). ``u`` increments by one between consecutive deltas;
a snapshot resets the book (``u`` may jump). ``ts`` is the exchange system time in milliseconds.

Trades archive: gzip CSV ``timestamp`` (seconds, 0.1 ms resolution), ``symbol``, ``side`` (taker side, ``Buy`` =
buyer-initiated), ``size``, ``price``, ``tickDirection``, ``trdMatchID``, ... for the linear segment; the spot segment
uses ``id``, ``timestamp`` (ms), ``price``, ``volume``, ``side``. Prices and sizes are parsed from their decimal
strings and mapped to integer ticks / lots on the registered (verified) grid by ``normalize``.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path
from typing import Iterator
import zipfile

import numpy as np


def book_messages(path: Path) -> Iterator[dict]:
    """Yield raw messages in file order (never re-sorted: ordering problems are measured, not hidden)."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"expected one member in {Path(path).name}, found {len(names)}")
        with archive.open(names[0]) as handle:
            for line in io.BufferedReader(handle, buffer_size=1 << 20):
                if line.strip():
                    yield json.loads(line)


def read_trades(path: Path, segment: str = "linear") -> dict[str, np.ndarray]:
    """Trades as arrays: t (s since epoch), price, size, side (+1 buyer-initiated, -1 seller-initiated), id."""
    t, px, sz, side, ids = [], [], [], [], []
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if segment == "linear":
                t.append(float(row["timestamp"]))
                sz.append(float(row["size"]))
                ids.append(row["trdMatchID"])
            else:
                t.append(float(row["timestamp"]) / 1000.0)
                sz.append(float(row["volume"]))
                ids.append(row["id"])
            px.append(float(row["price"]))
            s = row["side"].strip().lower()
            side.append(1 if s == "buy" else -1 if s == "sell" else 0)
    return {"t": np.asarray(t, float), "price": np.asarray(px, float), "size": np.asarray(sz, float),
            "side": np.asarray(side, np.int8), "id": np.asarray(ids, dtype=object)}

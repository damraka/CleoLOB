"""Adapter development kit: synthetic fixture generator and mapping template (workstream 85).

``write_tardis`` writes a deterministic synthetic Tardis-format L2 + trades pair (no real market data) for
adapter, replay and tape tests; ``crossed_every`` plants crossed books for validator tests.

Mapping template for a new venue adapter (subclass ``lob.v07.adapters.base.Adapter``):

1. declare a ``VenueSpec`` (tick, amount unit, contract type, capability level, FIFO established or not);
2. map each native row to one canonical ``Record`` kind without inventing fields the source lacks
   (no order ids on aggregate L2; no aggressor side when the source has none);
3. keep source order; never repair crossed books or gaps (validation reports them);
4. run ``conformance(adapter)`` and ``validate_stream(adapter.records(), spec)`` on real and fixture files;
5. add the venue to ``lob.v07.exchange.venue.VENUE_RULES`` as DECLARED (unverified) rules.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def write_tardis(directory: Path, *, venue: str = "deribit", symbol: str = "ETH-PERPETUAL", date: str = "2020-04-01",
                 seconds: float = 600.0, tick: float = 0.05, seed: int = 7, levels: int = 12,
                 crossed_every: int | None = None) -> dict[str, Path]:
    """Snapshot plus random level updates every ~50 ms and trades; returns {'l2': path, 'trades': path}."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    l2 = directory / f"{venue}_incremental_book_L2_{date}_{symbol}.csv.gz"
    trades = directory / f"{venue}_trades_{date}_{symbol}.csv.gz"
    t0 = 1_585_699_200_000_000
    mid_ticks = 4000
    rows = ["exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"]
    trade_rows = ["exchange,symbol,timestamp,local_timestamp,id,side,price,amount"]
    book = {"bid": {}, "ask": {}}
    for k in range(levels):
        for side, sign in (("bid", -1), ("ask", 1)):
            price = mid_ticks + sign * (k + 1)
            book[side][price] = int(rng.integers(1, 50)) * 10
            rows.append(f"{venue},{symbol},{t0},{t0 + 150},true,{side},{price * tick:.2f},{book[side][price]}")
    t, n, trade_id = t0, 0, 0
    pending = None
    while t < t0 + seconds * 1e6:
        t += int(rng.integers(10_000, 90_000))
        n += 1
        side = "bid" if rng.random() < 0.5 else "ask"
        sign = -1 if side == "bid" else 1
        price = mid_ticks + sign * int(rng.integers(1, levels + 1))
        amount = 0 if (price in book[side] and rng.random() < 0.2) else int(rng.integers(1, 50)) * 10
        if crossed_every and n % crossed_every == 0:
            price, amount = mid_ticks + (5 if side == "bid" else -5), 10
        book[side][price] = amount
        rows.append(f"{venue},{symbol},{t},{t + 200},false,{side},{price * tick:.2f},{amount}")
        if amount == 0:
            book[side].pop(price, None)
        if pending is not None:
            rows.append(f"{venue},{symbol},{t + 1},{t + 201},false,{pending[0]},{pending[1] * tick:.2f},0")
            book[pending[0]].pop(pending[1], None)
            pending = None
        if crossed_every and n % crossed_every == 0:
            pending = (side, price)
        if rng.random() < 0.15:
            trade_id += 1
            aggressor = "buy" if rng.random() < 0.5 else "sell"
            px = (mid_ticks + (1 if aggressor == "buy" else -1)) * tick
            for _ in range(int(rng.integers(1, 3))):
                trade_rows.append(f"{venue},{symbol},{t},{t + 300},{trade_id},{aggressor},{px:.2f},"
                                  f"{int(rng.integers(1, 20)) * 10}")
                trade_id += 1
    with gzip.open(l2, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(rows) + "\n")
    with gzip.open(trades, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(trade_rows) + "\n")
    return {"l2": l2, "trades": trades}

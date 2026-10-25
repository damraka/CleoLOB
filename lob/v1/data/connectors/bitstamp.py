"""Bitstamp order-level (MBO/L3) connector for prospective captures (M14).

Input: the segments written by ``bitstamp_capture`` (gzip JSON lines: ``ws`` records with the exact message text and
local receive clocks, ``rest_snapshot`` census bodies). Output, in canonical events:

* ``live_orders`` -> ``ORDER_ADD`` (``order_created``), ``ORDER_MODIFY`` (``order_changed``: new remaining amount,
  possibly after a partial execution), ``ORDER_DELETE`` (``order_deleted``: cancelled or fully executed - the trade
  stream says which), with the venue order id, side (``order_type`` 0 = buy/bid), price and remaining size;
* ``live_trades`` -> ``TRADE`` with side = aggressor (``type`` 0 = buy), ``aux`` = trade id and ``order_id`` = the
  maker (resting) order id;
* REST census -> ``SNAPSHOT_BEGIN``, one ``ORDER_ADD`` per resting order, ``SNAPSHOT_END`` (the reconstruction anchor);
* ``order_book`` channel (top 100 aggregated levels) -> separate MBP reference snapshots for validation (M29).

Every event carries the exchange ``microtimestamp`` (us -> ns) and the local receive time; nothing is reordered.
Only Bitstamp's own definitions are used; message gaps (``GAP`` segments) are emitted as ``GAP`` events.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import NO_TS, Capability, EventKind, Instrument, empty
from .base import Licence, Source, register

INSTRUMENTS = {"btcusd": (0.01, 1e-8, "spot"), "ethusd": (0.01, 1e-8, "spot")}


def _ticks(price_str: str, tick: float) -> int:
    return int(round(float(price_str) / tick))


def _lots(amount_str: str, lot: float) -> float:
    return float(round(float(amount_str) / lot))


def segment_records(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def mbo_events(records, instrument: Instrument, *, chunk: int = 200_000) -> Iterator[np.ndarray]:
    tick, lot = instrument.tick, instrument.lot
    buf = empty(chunk)
    n = 0

    def put(ts, recv, kind, side, price, size, oid=-1, aux=-1, seq=-1):
        nonlocal n
        r = buf[n]
        r["ts_exchange"], r["ts_receive"], r["seq"], r["kind"] = ts, recv, seq, kind
        r["side"], r["price"], r["size"], r["order_id"], r["aux"] = side, price, size, oid, aux
        n += 1

    for rec in records:
        if n > chunk - 10_000:
            yield buf[:n].copy()
            n = 0
        recv = int(rec.get("recv_ns", NO_TS))
        kind = rec["kind"]
        if kind == "capture_end" and rec.get("status") != "COMPLETE":
            put(recv, recv, EventKind.GAP, 0, 0, 0.0)
            continue
        if kind == "rest_snapshot":
            body = json.loads(rec["body"])
            ts = int(body["microtimestamp"]) * 1000
            orders = [(1, o) for o in body["bids"]] + [(-1, o) for o in body["asks"]]
            if n + len(orders) + 2 > chunk:
                yield buf[:n].copy()
                n = 0
                if len(orders) + 2 > chunk:
                    raise ValueError("census larger than the chunk size")
            put(ts, recv, EventKind.SNAPSHOT_BEGIN, 0, 0, 0.0)
            for side, (px, amt, oid) in orders:
                put(ts, recv, EventKind.ORDER_ADD, side, _ticks(px, tick), _lots(amt, lot), int(oid))
            put(ts, recv, EventKind.SNAPSHOT_END, 0, 0, 0.0)
            continue
        if kind != "ws":
            continue
        msg = json.loads(rec["raw"])
        event, channel, d = msg.get("event"), msg.get("channel", ""), msg.get("data") or {}
        if event == "bts:request_reconnect":
            put(recv, recv, EventKind.GAP, 0, 0, 0.0)
        elif channel.startswith("live_orders_") and event in ("order_created", "order_changed", "order_deleted"):
            k = {"order_created": EventKind.ORDER_ADD, "order_changed": EventKind.ORDER_MODIFY,
                 "order_deleted": EventKind.ORDER_DELETE}[event]
            side = 1 if int(d["order_type"]) == 0 else -1
            size = 0.0 if k == EventKind.ORDER_DELETE else _lots(d["amount_str"], lot)
            put(int(d["microtimestamp"]) * 1000, recv, k, side, _ticks(d["price_str"], tick), size, int(d["id"]))
        elif channel.startswith("live_trades_") and event == "trade":
            side = 1 if int(d["type"]) == 0 else -1
            maker = int(d["sell_order_id"] if side == 1 else d["buy_order_id"])
            put(int(d["microtimestamp"]) * 1000, recv, EventKind.TRADE, side, _ticks(d["price_str"], tick),
                _lots(d["amount_str"], lot), maker, int(d["id"]))
    if n:
        yield buf[:n].copy()


def book_snapshots(records, instrument: Instrument, *, levels: int = 100) -> list[dict]:
    """``order_book`` channel snapshots (top aggregated levels) as reference books for validation."""
    out = []
    for rec in records:
        if rec["kind"] != "ws":
            continue
        msg = json.loads(rec["raw"])
        if not msg.get("channel", "").startswith("order_book_") or msg.get("event") != "data":
            continue
        d = msg["data"]
        out.append({"ts": int(d["microtimestamp"]) * 1000, "recv": int(rec["recv_ns"]),
                    "bids": [(_ticks(p, instrument.tick), _lots(a, instrument.lot)) for p, a in d["bids"][:levels]],
                    "asks": [(_ticks(p, instrument.tick), _lots(a, instrument.lot)) for p, a in d["asks"][:levels]]})
    return out


class BitstampConnector:
    name = "bitstamp"

    def instrument(self, segment: str, symbol: str, date: str) -> Instrument:
        tick, lot, kind = INSTRUMENTS[symbol]
        return Instrument("bitstamp", segment, symbol, tick, lot, kind, quote="USD")

    def sources(self, segment: str, symbol: str, date: str) -> list[Source]:
        return [Source("capture", f"capture://bitstamp/{symbol}/{date}", "cleolob-v1-bitstamp-capture-1")]

    def capability(self, segment: str, symbol: str, date: str) -> Capability:
        return Capability(book="l3", depth_levels=None, order_ids=True, trades=True, trade_side=True, trade_ids=True,
                          exchange_ts_resolution_ns=1_000, receive_ts=True, sequence_numbers=False, snapshots=True,
                          queue_observability="exact_order_id", cancel_visibility="explicit",
                          notes=("prospective public websocket capture with REST censuses",
                                 "queue priority inferred from order ids and event order within a price level; "
                                 "hidden/iceberg orders are not visible"))

    def licence(self) -> Licence:
        return Licence("Bitstamp public market data", "https://www.bitstamp.net/api/", "2026-10-25",
                       "non-commercial research; raw capture never redistributed", redistribution=False,
                       commercial=False)

    def events(self, files: dict[str, Path], instrument: Instrument, *, chunk: int = 200_000) -> Iterator[np.ndarray]:
        for seg in sorted(files):
            yield from mbo_events(segment_records(files[seg]), instrument, chunk=chunk)


CONNECTOR = register(BitstampConnector())

__all__ = ["BitstampConnector", "CONNECTOR", "mbo_events", "book_snapshots", "segment_records", "INSTRUMENTS"]

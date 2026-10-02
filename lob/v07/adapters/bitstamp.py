"""Bitstamp order-level adapter: the frozen v0.5 capture adapter mapped onto the v0.7 canonical schema.

The v0.5 ``BitstampCaptureAdapter`` (unchanged, contract ``bitstamp-capture-4``)
normalizes the self-recorded capture; this module only re-expresses its events
as canonical records in native decimals. FIFO priority is not established by
the feed (``fifo_established=False``), so queue-position results on this data
are ``ASSUMPTION_DEPENDENT`` on the declared priority rule. The captures were
consumed in v0.5 and are development/retrospective only in v0.7.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Iterator

from ...mbo_sources import BitstampCaptureAdapter
from ..data.schema import Record, venue_spec
from .base import Adapter

_KIND = {"ADD": "order_add", "MODIFY": "order_modify", "CANCEL": "order_cancel", "EXECUTE": "order_execute",
         "TRADE": "trade", "GAP": "gap", "RESET": "gap"}
_SIDE = {"BUY": "bid", "SELL": "ask"}


class BitstampAdapter(Adapter):
    def __init__(self, capture: str | Path, *, pair: str = "btcusd") -> None:
        super().__init__()
        self.spec = venue_spec("bitstamp", pair)
        self.source = BitstampCaptureAdapter(capture, pair=pair)
        contract = self.source.contract
        self.tick, self.lot = Decimal(contract.tick_size), Decimal(contract.lot_size)
        self.contract_sha256 = contract.sha256()

    def records(self) -> Iterator[Record]:
        events, _ = self.source.build()
        self.stats = {**self.source.stats, "events": len(events), "contract_sha256": self.contract_sha256}
        for event in events:
            ts = (event.exchange_ts_ns or 0) // 1000
            local = (event.receive_ts_ns or event.exchange_ts_ns or 0) // 1000
            if event.kind == "SNAPSHOT":
                if event.snapshot_mode != "reset":
                    # Later censuses are validation references placed at exchange time, not book state changes.
                    self.stats["census_checks_skipped"] = self.stats.get("census_checks_skipped", 0) + 1
                    continue
                for order in event.orders:
                    yield Record("order_add", ts, local, _SIDE[order.side], order.price * self.tick,
                                 order.quantity * self.lot, order_id=order.order_id, extra={"census": True})
                continue
            price = event.price * self.tick if event.price is not None else None
            amount = event.quantity * self.lot if event.quantity is not None else None
            if event.kind == "TRADE":
                side = {"BUY": "buy", "SELL": "sell"}.get(event.aggressor_side or "")
                yield Record("trade", ts, local, side, price, amount,
                             extra={"maker": event.maker_order_id, "taker": event.taker_order_id})
                continue
            side = _SIDE.get(event.side or "")
            yield Record(_KIND[event.kind], ts, local, side, price, amount, order_id=event.order_id,
                         extra={"source_sequence": event.source_sequence} if event.source_sequence else {})

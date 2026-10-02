"""v0.7 stream validation: corrupted fixtures, round-trip serialization, deterministic checksums."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D

import pytest

from lob.v07.adapters.tardis import TardisAdapter
from lob.v07.data import validation as v
from lob.v07.data.schema import Record, venue_spec
from tests.v07_fixtures import write_tardis

ETH = venue_spec("deribit", "ETH-PERPETUAL")
MBO = venue_spec("bitstamp", "btcusd")
T0 = 1_604_188_800_000_000   # 2020-11-01T00:00:00Z


def _book(t, side, price, amount, kind="book_delta"):
    return Record(kind, t, t + 100, side, D(price), D(amount))


def _clean():
    return [_book(T0, "bid", "100", "5", "book_snapshot"), _book(T0, "ask", "101", "5", "book_snapshot"),
            Record("trade", T0 + 1000, T0 + 1100, "buy", D("101"), D("1"), trade_id="t1"),
            _book(T0 + 2000, "ask", "101", "4")]


def test_clean_stream_passes_and_round_trips() -> None:
    records = _clean()
    report = v.validate_stream(records, ETH, date="2020-11-01")
    assert report["status"] in {"PASS", "WARN"} and "crossed_book" not in report["flags"]
    assert [v.from_json(v.to_json(r)) for r in records] == records
    assert v.checksum(records) == v.checksum([v.from_json(v.to_json(r)) for r in records])
    assert v.checksum(records) != v.checksum(records[::-1])


@pytest.mark.parametrize("mutate,flag", [
    (lambda r: r + [_book(T0 + 3000, "bid", "102", "1")], "crossed_book"),
    (lambda r: r + [_book(T0 + 3000, "bid", "101", "1")], "locked_book"),
    (lambda r: r[:3] + [replace(r[3], local_timestamp_us=T0 - 10)], "timestamp_reversal"),
    (lambda r: r + [r[-1]], "duplicate_event"),
    (lambda r: r + [Record("trade", T0 + 5000, T0 + 5100, "sell", D("100"), D("1"), trade_id="t1")], "duplicate_event"),
    (lambda r: r + [_book(T0 + 120_000_000, "ask", "101", "3")], "gap"),
    (lambda r: [replace(x, timestamp_us=x.timestamp_us - 3 * 86400 * 10**6) for x in r], "timezone_error"),
    (lambda r: [x for x in r if x.kind != "trade"], "missing_channel"),
    (lambda r: r + [Record("book_delta", T0 + 4000, T0 + 4100, "bid", D("-1"), D("1"))], "negative_value"),
    (lambda r: [replace(x, timestamp_us=(x.timestamp_us // 1000) * 1000) for x in r], "timestamp_resolution_loss"),
])
def test_corrupted_fixtures_are_flagged(mutate, flag) -> None:
    report = v.validate_stream(mutate(_clean()), ETH, date="2020-11-01")
    assert flag in report["flags"], report


def test_corrupt_interval_and_sequence_gaps() -> None:
    records = _clean() + [_book(T0 + 3000, "bid", "102", "1"), _book(T0 + 9_000_000, "bid", "102", "0")]
    assert "corrupt_interval" in v.validate_stream(records, ETH)["flags"]
    seq = [replace(r, extra={"sequence": s}) for r, s in zip(_clean(), (1, 2, 4, 5))]
    assert v.validate_stream(seq, ETH)["flags"]["missing_sequence"] == 1


def test_order_level_checks() -> None:
    def o(kind, oid, amount="1"):
        return Record(kind, T0, T0, "bid", D("100"), D(amount) if amount else None, order_id=oid)
    records = [o("order_add", "a"), o("order_add", "a"), o("order_cancel", "zz"), o("order_add", "b", "1"),
               o("order_cancel", "b", "5"), Record("trade", T0, T0, "buy", D("100"), D("1"))]
    flags = v.validate_stream(records, MBO)["flags"]
    assert flags["duplicate_order_id"] == 1 and flags["unknown_order_id"] == 1 and flags["impossible_cancel"] == 1


def test_synthetic_tardis_file_validates(tmp_path) -> None:
    files = write_tardis(tmp_path, seconds=120.0)
    adapter = TardisAdapter("deribit", "ETH-PERPETUAL", l2=files["l2"], trades=files["trades"])
    report = v.validate_stream(adapter.records(), ETH, date="2020-04-01")
    assert "timestamp_reversal" not in report["flags"] and report["book_records"] > 100

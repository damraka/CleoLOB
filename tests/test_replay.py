"""Canonical data validation and exact historical reconstruction regression tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from lob.replay import (DataValidationError, EventSchemaError, HistoricalBook,
                        HistoricalReplay, MarketEvent, ReconstructionError,
                        SnapshotOrder, load_events, replay_file, validate_events,
                        validate_file)


def event(sequence=1, kind="ADD", **kwargs):
    fields = {"timestamp_ns": sequence * 100, "sequence": sequence,
              "event_type": kind, "symbol": "DEMO", "venue": "TEST"}
    if kind == "ADD":
        fields.update(order_id=f"o{sequence}", side="BUY", price_ticks=99, quantity=10)
    fields.update(kwargs)
    return MarketEvent(**fields)


def write_jsonl(tmp_path, events, name="events.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in events), encoding="utf-8")
    return path


def test_exact_execution_targets_recorded_id_and_never_runs_matching():
    events = [event(1), event(2, side="SELL", price_ticks=101),
              event(3, side="SELL", price_ticks=102),
              event(4, "EXECUTE", order_id="o3", quantity=4)]
    replay = HistoricalReplay(events)
    assert replay.run() == 4
    assert replay.book.orders["o2"].quantity == 10  # better quote deliberately untouched
    assert replay.book.orders["o3"].quantity == 6
    assert replay.book.depth() == ([(99, 10)], [(101, 10), (102, 6)])
    assert replay.book.executions[0].order_id == "o3"
    assert replay.book.executions[0].price_ticks == 102
    assert replay.book.executions[0].side.value == "SELL"


def test_partial_cancel_full_execution_delete_remove_levels():
    replay = HistoricalReplay([event(1), event(2),
                               event(3, "CANCEL", order_id="o1", quantity=3),
                               event(4, "EXECUTE", order_id="o1", quantity=7),
                               event(5, "DELETE", order_id="o2")])
    replay.run()
    assert replay.book.depth() == ([], [])
    assert replay.book.orders == {}
    assert replay.summary()["executed_quantity"] == 7


def test_cancel_entire_remaining_quantity_is_valid():
    replay = HistoricalReplay([event(), event(2, "CANCEL", order_id="o1", quantity=10)])
    replay.run()
    assert replay.book.orders == {}


def test_modification_quantity_is_new_remaining_and_queue_rules_are_explicit():
    replay = HistoricalReplay([event(1), event(2),
                               event(3, "MODIFY", order_id="o1", quantity=7),
                               event(4, "MODIFY", order_id="o1", quantity=8),
                               event(5, "MODIFY", order_id="o2", price_ticks=98)])
    replay.run(max_events=3)
    assert replay.book.queue("BUY", 99) == ("o1", "o2")
    assert replay.book.orders["o1"].quantity == 7
    replay.step()
    assert replay.book.queue("BUY", 99) == ("o2", "o1")
    replay.step()
    assert replay.book.queue("BUY", 98) == ("o2",)
    assert replay.book.queue("BUY", 99) == ("o1",)
    assert replay.book.orders["o2"].quantity == 10


def test_snapshot_replaces_state_preserves_list_queue_order_and_execution_history():
    orders = (SnapshotOrder("snap-b", "BUY", 98, 5), SnapshotOrder("snap-a", "BUY", 98, 9),
              SnapshotOrder("snap-c", "SELL", 102, 3))
    replay = HistoricalReplay([event(1), event(2, "EXECUTE", order_id="o1", quantity=1),
                               event(3, "SNAPSHOT", orders=orders),
                               event(4, "CANCEL", order_id="snap-a", quantity=4)])
    replay.run()
    assert "o1" not in replay.book.orders
    assert replay.book.queue("BUY", 98) == ("snap-b", "snap-a")
    assert replay.book.depth() == ([(98, 10)], [(102, 3)])
    assert len(replay.book.executions) == 1


def test_trade_print_does_not_mutate_depth_or_count_as_execution():
    replay = HistoricalReplay([event(1), event(2, "TRADE", price_ticks=120, quantity=200)])
    replay.run()
    assert replay.book.depth() == ([(99, 10)], [])
    summary = replay.summary()
    assert summary["trade_print_quantity"] == 200
    assert summary["trade_print_count"] == 1
    assert summary["executed_quantity"] == 0


def test_halt_allows_cancels_and_prints_until_resume():
    replay = HistoricalReplay([event(1), event(2, "HALT"),
                               event(3, "CANCEL", order_id="o1", quantity=1),
                               event(4, "TRADE", price_ticks=105, quantity=20),
                               event(5, "RESUME"), event(6)])
    replay.run()
    assert replay.book.halted is False
    assert replay.book.depth() == ([(99, 19)], [])


@pytest.mark.parametrize(("events", "code"), [
    ([], "EMPTY_DATASET"),
    ([event(), event(2, "EXECUTE", order_id="missing", quantity=1)], "UNKNOWN_ORDER_ID"),
    ([event(), event(2, order_id="o1")], "DUPLICATE_ORDER_ID"),
    ([event(), event(2, "DELETE", order_id="o1"), event(3, order_id="o1")], "DUPLICATE_ORDER_ID"),
    ([event(), event(2, "CANCEL", order_id="o1", quantity=11)], "EXCESS_QUANTITY"),
    ([event(), event(2, "EXECUTE", order_id="o1", quantity=11)], "EXCESS_QUANTITY"),
    ([event(), event(2, "EXECUTE", order_id="o1", side="SELL", quantity=1)], "SIDE_MISMATCH"),
    ([event(), event(2, "EXECUTE", order_id="o1", price_ticks=98, quantity=1)], "PRICE_MISMATCH"),
    ([event(), event(2, "MODIFY", order_id="o1", side="SELL", quantity=1)], "SIDE_MISMATCH"),
    ([event(), event(2, side="SELL", price_ticks=99)], "CROSSED_BOOK"),
    ([event(), event(2, side="SELL", price_ticks=98)], "CROSSED_BOOK"),
    ([event(), event(2, timestamp_ns=99)], "TIMESTAMP_REVERSAL"),
    ([event(), event(1)], "DUPLICATE_SEQUENCE"),
    ([event(4), event(3)], "SEQUENCE_REVERSAL"),
    ([event(), event(3)], "SEQUENCE_GAP"),
    ([event(), event(2, symbol="OTHER")], "MIXED_STREAM"),
    ([event(), event(2, venue="OTHER")], "MIXED_STREAM"),
    ([event(1, "RESUME")], "INVALID_HALT_TRANSITION"),
    ([event(1, "HALT"), event(2, "HALT")], "INVALID_HALT_TRANSITION"),
    ([event(1, "HALT"), event(2)], "EVENT_DURING_HALT"),
    ([event(), event(2, "HALT"), event(3, "EXECUTE", order_id="o1", quantity=1)], "EVENT_DURING_HALT"),
])
def test_invalid_streams_block_all_replay(events, code):
    report = validate_events(events)
    assert not report.valid
    assert report.status == "INVALID"
    assert report.issues[0].code == code
    with pytest.raises(DataValidationError) as caught:
        HistoricalReplay(events)
    assert caught.value.report.to_dict() == report.to_dict()


def test_direct_book_rejects_crossing_amend_atomically():
    book = HistoricalBook()
    book.apply(event())
    book.apply(event(2, side="SELL", price_ticks=101))
    before = book.snapshot()
    with pytest.raises(ReconstructionError, match="lock or cross"):
        book.apply(event(3, "MODIFY", order_id="o1", price_ticks=101))
    assert book.snapshot() == before
    book.apply(event(3, "MODIFY", order_id="o1", price_ticks=100))
    assert book.best_bid() == 100  # rejected transition did not advance sequence


@pytest.mark.parametrize("orders", [
    (SnapshotOrder("a", "BUY", 100, 1), SnapshotOrder("a", "BUY", 99, 1)),
    (SnapshotOrder("a", "BUY", 100, 1), SnapshotOrder("b", "SELL", 99, 1)),
])
def test_invalid_snapshot_is_atomic(orders):
    book = HistoricalBook()
    book.apply(event())
    before = book.snapshot()
    with pytest.raises(ReconstructionError):
        book.apply(event(2, "SNAPSHOT", orders=orders))
    assert book.snapshot() == before


@pytest.mark.parametrize(("field", "value"), [
    ("timestamp_ns", -1), ("timestamp_ns", 1.0), ("timestamp_ns", "NaN"),
    ("sequence", True), ("sequence", -1), ("sequence", 2**63),
    ("price_ticks", 0), ("price_ticks", 1.5), ("price_ticks", "1e2"),
    ("quantity", 0), ("quantity", -1), ("quantity", False), ("quantity", float("inf")),
    ("side", "BID"), ("event_type", "UNKNOWN"), ("order_id", ""),
    ("symbol", "PADDED "), ("venue", "bad\nvenue"),
])
def test_strict_schema_rejects_ambiguous_units_and_identifiers(field, value):
    row = event().to_dict()
    row[field] = value
    with pytest.raises(EventSchemaError):
        MarketEvent.from_mapping(row)


@pytest.mark.parametrize("row", [
    {**event().to_dict(), "order_id": None},
    {**event(2, "DELETE", order_id="o1").to_dict(), "quantity": 1},
    {**event(2, "TRADE", price_ticks=100, quantity=1).to_dict(), "order_id": "o1"},
    {**event().to_dict(), "orders": []},
    {**event().to_dict(), "unknown": "value"},
    {k: v for k, v in event(1, "SNAPSHOT").to_dict().items() if k != "orders"},
    {**event(1, "SNAPSHOT").to_dict(), "orders": [{}]},
])
def test_schema_requires_event_specific_fields(row):
    with pytest.raises(EventSchemaError):
        MarketEvent.from_mapping(row)


def test_tied_timestamps_sequence_order_pause_reset_and_chunk_invariance():
    events = [event(10, timestamp_ns=10), event(11, timestamp_ns=10),
              event(12, "EXECUTE", timestamp_ns=20, order_id="o10", quantity=2)]
    replay = HistoricalReplay(events)
    assert replay.advance_to(10) == 2
    replay.pause()
    assert replay.run() == 0
    assert replay.advance_to(30) == 0
    assert replay.step() == events[2]
    assert replay.complete
    assert replay.step() is None
    replay.resume()
    before = replay.summary()
    replay.reset()
    assert replay.timestamp_ns is None
    assert replay.book.orders == {}
    assert replay.run() == 3
    assert replay.summary() == before
    other = HistoricalReplay(iter(events))
    other.run(max_events=1)
    other.run(max_events=2)
    assert other.summary() == before
    with pytest.raises(ValueError, match="reverse"):
        replay.advance_to(19)


def test_returned_order_mapping_cannot_mutate_book():
    replay = HistoricalReplay([event()])
    replay.step()
    replay.book.orders.clear()
    assert len(replay.book.orders) == 1


def test_load_jsonl_snapshot_source_hash_and_replay(tmp_path):
    events = [event(1, "SNAPSHOT", orders=(SnapshotOrder("s", "BUY", 100, 10),)),
              event(2, "EXECUTE", order_id="s", quantity=3)]
    path = write_jsonl(tmp_path, events)
    assert load_events(path) == tuple(events)
    report = validate_file(path)
    assert report.valid
    assert report.checked_events == 2
    assert report.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = replay_file(path)
    assert summary["complete"]
    assert summary["book"]["bids"] == [[100, 7]]
    assert summary["quality"] == report.to_dict()


def test_csv_and_jsonl_have_identical_canonical_hashes(tmp_path):
    events = [event(1), event(2, "CANCEL", order_id="o1", quantity=1)]
    jsonl = write_jsonl(tmp_path, events)
    path = tmp_path / "events.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0].to_dict()))
        writer.writeheader()
        writer.writerows(item.to_dict() for item in events)
    assert load_events(path) == tuple(events)
    assert validate_file(path).canonical_sha256 == validate_file(jsonl).canonical_sha256
    assert validate_file(path).source_sha256 != validate_file(jsonl).source_sha256


@pytest.mark.parametrize(("content", "suffix", "code"), [
    ("{broken}\n", ".jsonl", "CORRUPT_ROW"),
    ("[]\n", ".jsonl", "CORRUPT_ROW"),
    ("\n", ".jsonl", "CORRUPT_ROW"),
    ("", ".jsonl", "EMPTY_DATASET"),
    ('{"timestamp_ns":1,"timestamp_ns":2}\n', ".jsonl", "DUPLICATE_FIELD"),
    ('{"quantity":NaN}\n', ".jsonl", "INVALID_NUMBER"),
    ("timestamp_ns,sequence,event_type,symbol,venue,sequence\n", ".csv", "DUPLICATE_FIELD"),
    ("timestamp_ns,sequence,event_type,symbol\n", ".csv", "MISSING_FIELD"),
    ("timestamp_ns,sequence,event_type,symbol,venue\n1,1,HALT,DEMO,TEST,extra\n", ".csv", "CORRUPT_ROW"),
    ("timestamp_ns,sequence,event_type,symbol,venue\n1,1,HALT\n", ".csv", "CORRUPT_ROW"),
    ("", ".parquet", "UNSUPPORTED_FORMAT"),
])
def test_corrupt_files_produce_invalid_reports(tmp_path, content, suffix, code):
    path = tmp_path / f"bad{suffix}"
    path.write_text(content, encoding="utf-8")
    report = validate_file(path)
    assert not report.valid
    assert report.issues[0].code == code
    with pytest.raises(DataValidationError):
        replay_file(path)


def test_bad_utf8_and_missing_file_produce_quality_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\n")
    assert validate_file(path).issues[0].code == "CORRUPT_ROW"
    assert validate_file(tmp_path / "missing.jsonl").issues[0].code == "SOURCE_ERROR"


def test_file_limits_fail_instead_of_silently_truncating(tmp_path):
    path = write_jsonl(tmp_path, [event(1), event(2)])
    for kwargs in ({"max_events": 1}, {"max_source_bytes": 8}, {"max_row_bytes": 8}):
        report = validate_file(path, **kwargs)
        assert not report.valid
        assert report.issues[0].code == "RESOURCE_LIMIT"
        assert report.source_sha256 is None
        with pytest.raises(DataValidationError):
            load_events(path, **kwargs)


def test_event_generator_consumption_is_bounded():
    consumed = []

    def infinite():
        for index in range(1, 1000):
            consumed.append(index)
            yield event(index)

    report = validate_events(infinite(), max_events=2)
    assert consumed == [1, 2, 3]
    assert report.issues[0].code == "RESOURCE_LIMIT"


def test_invalid_late_event_blocks_file_before_replay(tmp_path):
    path = write_jsonl(tmp_path, [event(1), event(2, "CANCEL", order_id="missing", quantity=1)])
    report = validate_file(path)
    assert not report.valid
    assert report.event_count == 2
    assert report.checked_events == 1
    assert report.issues[0].row == 2
    assert report.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(DataValidationError):
        load_events(path)


def test_fixture_is_valid_and_replay_is_deterministic():
    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "canonical-events.jsonl"
    first = replay_file(path)
    second = replay_file(path)
    assert first == second
    assert first["event_count"] == 12
    assert first["execution_count"] == 2
    assert first["executed_quantity"] == 7

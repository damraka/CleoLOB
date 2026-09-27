"""M1: source-native lifecycle validation. All fixtures here are synthetic."""
from __future__ import annotations

import gzip
import json

import pytest

from lob.mbo_sources import (BitstampCaptureAdapter, LobsterAdapter, bitstamp_contract, exact_units,
                             m1_gate, validate_source)
from lob.order_lifecycle import (CensusOrder, LifecycleEvent as E, LifecycleInputError, LifecycleSemantics,
                                 OrderLifecycleBook, replay)


def book(events, **semantics):
    b = OrderLifecycleBook(LifecycleSemantics(**semantics))
    for i, (kind, fields) in enumerate(events):
        b.apply(E(sequence=i, kind=kind, **fields))
    return b


CENSUS = ("SNAPSHOT", {"orders": (CensusOrder("a", "BUY", 99, 10), CensusOrder("b", "BUY", 99, 5),
                                  CensusOrder("c", "SELL", 101, 7)), "exchange_ts_ns": 0})


def test_partial_and_full_execution_and_lifetimes():
    b = book([("RESET", {}), ("ADD", {"order_id": "x", "side": "BUY", "price": 99, "quantity": 10, "exchange_ts_ns": 5}),
              ("EXECUTE", {"order_id": "x", "quantity": 4, "exchange_ts_ns": 7}),
              ("EXECUTE", {"order_id": "x", "quantity": 6, "exchange_ts_ns": 9})])
    report = b.finish()["report"]
    assert report["partial_executions"] == 1 and report["full_executions"] == 1
    assert report["lifetime_ns_quantiles"]["p50"] == 4 and not report["unexplained_anomalies"]
    assert [e.remaining_after for e in b.executions] == [6, 0]


def test_duplicate_ids_are_counted_and_not_applied():
    b = book([CENSUS, ("ADD", {"order_id": "a", "side": "SELL", "price": 105, "quantity": 1}),
              ("CANCEL", {"order_id": "c"}), ("ADD", {"order_id": "c", "side": "SELL", "price": 102, "quantity": 3})])
    anomalies = b.finish()["report"]["anomalies"]
    assert anomalies["DUPLICATE_ORDER_ID_LIVE"] == 1 and anomalies["DUPLICATE_ORDER_ID_REUSED"] == 1
    assert b.order("a")["side"] == "BUY" and b.order("c") is None


@pytest.mark.parametrize("kind,code", [("MODIFY", "UNKNOWN_MODIFY"), ("CANCEL", "UNKNOWN_CANCEL"),
                                       ("EXECUTE", "UNKNOWN_EXECUTE")])
def test_unknown_identity_with_complete_census_is_unexplained(kind, code):
    fields = {"order_id": "zz", "quantity": 1}
    b = book([CENSUS, (kind, fields)])
    report = b.finish()["report"]
    assert report["unexplained_anomalies"] == {code: 1}
    assert b.aggregate() == {"bids": [[99, 15]], "asks": [[101, 7]]}  # nothing guessed


def test_unknown_identity_without_complete_census_is_left_censoring():
    b = book([("RESET", {}), ("CANCEL", {"order_id": "old", "quantity": 3})], census_complete=False)
    report = b.finish()["report"]
    assert report["anomalies"] == {"LEFT_CENSORED_UNKNOWN_ID": 1} and not report["unexplained_anomalies"]


@pytest.mark.parametrize("kind,code", [("MODIFY", "MODIFY_AFTER_COMPLETION"), ("CANCEL", "CANCEL_AFTER_COMPLETION"),
                                       ("EXECUTE", "EXECUTE_AFTER_COMPLETION")])
def test_events_after_completion(kind, code):
    b = book([CENSUS, ("EXECUTE", {"order_id": "c", "quantity": 7}), (kind, {"order_id": "c", "quantity": 1})])
    assert b.finish()["report"]["unexplained_anomalies"] == {code: 1}


def test_excess_quantities_are_not_clipped():
    b = book([CENSUS, ("EXECUTE", {"order_id": "b", "quantity": 6}), ("CANCEL", {"order_id": "b", "quantity": 9})])
    report = b.finish()["report"]
    assert report["anomalies"] == {"EXCESS_EXECUTION": 1, "EXCESS_CANCEL": 1}
    assert b.order("b")["quantity"] == 5


def test_priority_preserved_on_reduction_and_reset_on_increase_or_reprice():
    b = book([CENSUS, ("ADD", {"order_id": "d", "side": "BUY", "price": 99, "quantity": 2}),
              ("MODIFY", {"order_id": "a", "quantity": 8})])
    assert b.tracked_level_order("BUY", 99) == ("a", "b", "d")
    b.apply(E(sequence=3, kind="MODIFY", order_id="a", quantity=12))
    assert b.tracked_level_order("BUY", 99) == ("b", "d", "a")
    b.apply(E(sequence=4, kind="MODIFY", order_id="b", price=98))
    assert b.tracked_level_order("BUY", 99) == ("d", "a") and b.tracked_level_order("BUY", 98) == ("b",)
    report = b.finish()["report"]
    assert report["modifies_priority_preserved"] == 1 and report["modifies_priority_reset"] == 2


def test_fifo_position_refused_unless_established():
    b = book([CENSUS])
    with pytest.raises(PermissionError):
        b.level_queue("BUY", 99)
    assert book([CENSUS], fifo_established=True).level_queue("BUY", 99) == ("a", "b")


def test_sequence_and_timestamp_anomalies():
    b = book([("RESET", {"source_sequence": 10, "exchange_ts_ns": 100, "receive_ts_ns": 5}),
              ("ADD", {"order_id": "x", "side": "BUY", "price": 1, "quantity": 1, "source_sequence": 10,
                       "exchange_ts_ns": 90, "receive_ts_ns": 4}),
              ("ADD", {"order_id": "y", "side": "BUY", "price": 1, "quantity": 1, "source_sequence": 13}),
              ("ADD", {"order_id": "z", "side": "BUY", "price": 1, "quantity": 1, "source_sequence": 12})],
             source_sequence_contiguous=True)
    anomalies = b.finish()["report"]["anomalies"]
    assert anomalies["DUPLICATE_SOURCE_SEQUENCE"] == 1 and anomalies["SOURCE_SEQUENCE_GAP"] == 1
    assert anomalies["SOURCE_SEQUENCE_REVERSAL"] == 1
    assert anomalies["EXCHANGE_TIMESTAMP_INVERSION"] == 1 and anomalies["RECEIVE_TIMESTAMP_INVERSION"] == 1
    with pytest.raises(LifecycleInputError, match="contiguous"):
        b.apply(E(sequence=9, kind="RESET"))


def test_snapshot_reset_censors_and_check_compares_without_repair():
    b = book([CENSUS, ("CANCEL", {"order_id": "b", "quantity": 2}),
              ("SNAPSHOT", {"snapshot_mode": "check", "orders": (CensusOrder("a", "BUY", 99, 10),
                            CensusOrder("b", "BUY", 99, 5), CensusOrder("c", "SELL", 101, 7))})])
    check = b.report.snapshot_checks[0]
    assert check["exact_order_matches"] == 2 and check["quantity_or_price_mismatch"] == 1
    assert b.order("b")["quantity"] == 3  # the check did not repair the replay
    b.apply(E(sequence=3, kind="SNAPSHOT", orders=(CensusOrder("q", "SELL", 102, 1),)))
    assert b.report.censored_by_reset == 3 and b.live_orders().keys() == {"q"}


def test_census_priority_order_agreement_is_measured():
    b = book([CENSUS, ("MODIFY", {"order_id": "a", "quantity": 20}),
              ("SNAPSHOT", {"snapshot_mode": "check", "orders": (CensusOrder("a", "BUY", 99, 20),
                            CensusOrder("b", "BUY", 99, 5), CensusOrder("c", "SELL", 101, 7))})])
    check = b.report.snapshot_checks[0]
    # Tracked rule moved 'a' behind 'b' after a size increase; the census lists 'a' first.
    assert check["adjacent_priority_discordant"] == 1 and check["adjacent_priority_concordant"] == 0


def test_gap_and_file_boundary_censoring():
    b = book([CENSUS, ("GAP", {}), ("CANCEL", {"order_id": "a"})])
    result = b.finish()
    assert result["report"]["censored_by_gap"] == 3 and result["report"]["anomalies"] == {"EVENT_AFTER_GAP": 1}
    b = book([CENSUS])
    assert b.finish()["report"]["right_censored_orders"] == 3
    b = book([("ADD", {"order_id": "early", "side": "BUY", "price": 1, "quantity": 1})])
    assert b.finish()["report"]["anomalies"] == {"EVENT_BEFORE_INITIAL_CENSUS": 1}


def test_crossed_book_is_counted():
    b = book([CENSUS, ("ADD", {"order_id": "x", "side": "BUY", "price": 101, "quantity": 1})])
    assert b.finish()["report"]["anomalies"] == {"CROSSED_BOOK_AFTER_EVENT": 1}


def test_mbo_to_l2_aggregation_is_exact_and_mismatches_are_typed():
    events = [E(0, "SNAPSHOT", orders=CENSUS[1]["orders"]), E(1, "EXECUTE", order_id="a", quantity=3)]
    references = [{"after_sequence": 0, "bids": [[99, 15]], "asks": [[101, 7]]},
                  {"after_sequence": 1, "bids": [[99, 12]], "asks": [[101, 7]]},
                  {"after_sequence": 1, "bids": [[99, 13]], "asks": [[101, 7]]},
                  {"after_sequence": 1, "bids": [[98, 12]], "asks": [[101, 7]]}]
    result = replay(events, references=references, depth=5)["reference_comparison"]
    assert (result["compared"], result["exact"], result["quantity_mismatch"], result["price_level_mismatch"]) == (4, 2, 1, 1)


def test_replay_is_deterministic():
    events = [E(0, "SNAPSHOT", orders=CENSUS[1]["orders"]), E(1, "EXECUTE", order_id="a", quantity=3),
              E(2, "ADD", order_id="n", side="SELL", price=102, quantity=4)]
    assert replay(events)["final_state_sha256"] == replay(list(events))["final_state_sha256"]
    changed = events[:2] + [E(2, "ADD", order_id="n", side="SELL", price=102, quantity=5)]
    assert replay(events)["final_state_sha256"] != replay(changed)["final_state_sha256"]


def test_malformed_events_are_invalid_not_anomalies():
    for kwargs in ({"kind": "ADD", "order_id": "x", "side": "BUY", "price": 0, "quantity": 1},
                   {"kind": "EXECUTE", "order_id": "x", "quantity": 0}, {"kind": "NOPE"},
                   {"kind": "ADD", "orders": (CensusOrder("a", "BUY", 1, 1),), "order_id": "a", "side": "BUY",
                    "price": 1, "quantity": 1}):
        with pytest.raises(LifecycleInputError):
            E(sequence=0, **kwargs)


def test_exact_units_refuses_rounding():
    assert exact_units("84080.09", "0.01", "price") == 8408009
    assert exact_units("0.00181036", "0.00000001", "amount") == 181036
    with pytest.raises(LifecycleInputError):
        exact_units("84080.095", "0.01", "price")


def _ws(channel, event, data, recv):
    return {"kind": "ws", "recv_ns": recv, "raw": json.dumps({"channel": channel, "event": event, "data": data})}


def _order(oid, side, price, amount, traded, ts_ms):
    return {"id_str": oid, "order_type": 0 if side == "BUY" else 1, "microtimestamp": str(ts_ms * 1000),
            "price_str": price, "amount_str": amount, "amount_traded": traded}


def _capture(tmp_path, rows):
    path = tmp_path / "capture.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return path


def test_bitstamp_adapter_on_synthetic_capture(tmp_path):
    census = {"microtimestamp": "1000500", "bids": [["100.00", "1.0", "b1"], ["99.99", "0.5", "b2"]],
              "asks": [["100.01", "2.0", "s1"]]}
    rows = [
        _ws("live_orders_btcusd", "order_created", _order("old", "BUY", "99.00", "1", "0", 999), 1),
        {"kind": "rest_snapshot", "recv_ns": 2, "body": json.dumps(census)},
        _ws("live_orders_btcusd", "order_created", _order("b2", "BUY", "99.99", "0.5", "0", 1000), 3),
        _ws("live_orders_btcusd", "order_changed", _order("s1", "SELL", "100.01", "1.5", "0.5", 1002), 4),
        _ws("live_trades_btcusd", "trade", {"microtimestamp": "1002000", "price_str": "100.01", "amount_str": "0.5",
                                            "type": 0, "buy_order_id": 77, "sell_order_id": "s1"}, 5),
        _ws("order_book_btcusd", "data", {"microtimestamp": "1002500", "bids": [["100.00", "1.0"], ["99.99", "0.5"]],
                                          "asks": [["100.01", "1.5"]]}, 6),
        _ws("live_orders_btcusd", "order_deleted", _order("b1", "BUY", "100.00", "1.0", "0", 1003), 7),
        _ws("live_orders_btcusd", "order_created", _order("n1", "SELL", "100.02", "0.25", "0", 1004), 8),
        _ws("live_orders_btcusd", "order_deleted", _order("s1", "SELL", "100.01", "0", "2.0", 1005), 9),
        {"kind": "capture_end", "recv_ns": 10, "status": "COMPLETE"},
    ]
    adapter = BitstampCaptureAdapter(_capture(tmp_path, rows))
    events, references = adapter.build()
    assert [e.kind for e in events] == ["SNAPSHOT", "EXECUTE", "TRADE", "CANCEL", "ADD", "EXECUTE"]
    assert adapter.stats["dropped_pre_census"] == 1 and adapter.stats["census_boundary_skipped"] == 1
    assert references[0]["after_sequence"] == 2
    result = validate_source(adapter, LifecycleSemantics())
    assert result["deterministic_replay"] and result["report"]["unexplained_anomalies"] == {}
    assert result["reference_comparison"]["exact"] == 1
    assert result["report"]["trades_with_maker_execution"] == 1
    assert result["report"]["full_executions"] == 1 and result["report"]["partial_executions"] == 1
    assert m1_gate(result)["status"] == "ESTABLISHED"  # synthetic mechanics only
    assert result["contract"]["fifo_established"] is False


def test_bitstamp_gap_marker_censors(tmp_path):
    census = {"microtimestamp": "1000000", "bids": [["1.00", "1", "a"]], "asks": [["2.00", "1", "b"]]}
    rows = [{"kind": "rest_snapshot", "recv_ns": 1, "body": json.dumps(census)},
            _ws("live_orders_btcusd", "bts:request_reconnect", {}, 2),
            _ws("live_orders_btcusd", "order_deleted", _order("a", "BUY", "1.00", "1", "0", 1001), 3)]
    rows[1]["raw"] = json.dumps({"event": "bts:request_reconnect", "channel": "", "data": {}})
    result = validate_source(BitstampCaptureAdapter(_capture(tmp_path, rows)), LifecycleSemantics())
    assert result["report"]["censored_by_gap"] == 2 and result["report"]["anomalies"] == {"EVENT_AFTER_GAP": 1}
    assert m1_gate(result)["status"] == "NOT_AVAILABLE"  # no reference books at all


def test_bitstamp_contract_hash_changes_with_alignment():
    assert bitstamp_contract(alignment="arrival").sha256() != bitstamp_contract().sha256()
    with pytest.raises(ValueError):
        bitstamp_contract(alignment="best_match")


def test_lobster_adapter_on_synthetic_files(tmp_path):
    messages = tmp_path / "msg.csv"
    books = tmp_path / "book.csv"
    messages.write_text("34200.000000001,1,11,100,1000000,1\n34200.5,1,12,50,1000100,-1\n"
                        "34201,4,11,40,1000000,1\n34201.25,5,0,10,1000050,1\n34202,3,99,20,999900,1\n"
                        "34203,2,12,10,1000100,-1\n", encoding="utf-8")
    books.write_text("9999999999,0,1000000,100\n1000100,50,1000000,100\n1000100,50,1000000,60\n"
                     "1000100,50,1000000,60\n1000100,50,1000000,60\n1000100,40,1000000,60\n", encoding="utf-8")
    adapter = LobsterAdapter(messages, books, ticker="SYNTH", levels=1)
    result = validate_source(adapter, LifecycleSemantics(census_complete=False, source_sequence_contiguous=True))
    assert result["reference_comparison"]["exact_fraction"] == 1.0
    assert result["report"]["anomalies"] == {"LEFT_CENSORED_UNKNOWN_ID": 1}
    assert result["report"]["kinds"]["TRADE"] == 1 and result["contract"]["fifo_established"] is False

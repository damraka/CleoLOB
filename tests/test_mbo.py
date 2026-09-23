"""MBO mechanics and epistemic boundaries: all inputs here are synthetic."""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lob.mbo import MBOBook, MBOEvent, MBOReplay
from lob.replay import EventSchemaError, ReconstructionError, SnapshotOrder
from lob.replay.l2 import L2_CAPABILITIES, L2Replay, L2State


def event(sequence=0, kind="RESET", **changes):
    fields = {"timestamp_ns": sequence * 10, "sequence": sequence,
              "event_type": kind, "symbol": "TEST", "venue": "SYNTHETIC"}
    if kind == "ADD":
        fields.update(order_id=f"o{sequence}", side="BUY", price_ticks=99, quantity=10)
    fields.update(changes)
    return MBOEvent(**fields)


def initialized(**limits):
    book = MBOBook(**limits)
    book.apply(event())
    return book


def source(tmp_path, events, compressed=False):
    path = tmp_path / ("events.jsonl.gz" if compressed else "events.jsonl")
    raw = b"".join(json.dumps(item.to_dict()).encode() + b"\n" for item in events)
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    return path


def test_schema_exchange_clock_reset_and_round_trip():
    item = event(1, "ADD", exchange_timestamp_ns=7)
    assert MBOEvent.from_mapping(item.to_dict()) == item
    assert MBOEvent.from_mapping(event().to_dict()) == event()
    with pytest.raises(FrozenInstanceError):
        item.timestamp_ns = 2
    book = initialized()
    book.apply(item)
    book.apply(event(2, "ADD", exchange_timestamp_ns=6))
    assert book.exchange_clock_regressions == 1
    assert book.snapshot()["exchange_timestamp_ns"] == 6


@pytest.mark.parametrize("field,value", [
    ("exchange_timestamp_ns", True), ("exchange_timestamp_ns", -1),
    ("exchange_timestamp_ns", 1.5), ("exchange_timestamp_ns", 2**63),
    ("sequence", True), ("quantity", 0), ("quantity", 1.5),
    ("price_ticks", "1e2"), ("side", "bid"), ("order_id", " padded"),
])
def test_schema_rejects_ambiguous_values(field, value):
    with pytest.raises(EventSchemaError):
        MBOEvent.from_mapping({**event(1, "ADD").to_dict(), field: value})


@pytest.mark.parametrize("changes", [
    {"unknown": 1}, {"orders": []}, {"order_id": "x"},
    {"price_ticks": 99}, {"quantity": 1},
])
def test_reset_rejects_unexpected_fields(changes):
    with pytest.raises(EventSchemaError):
        MBOEvent.from_mapping({**event().to_dict(), **changes})


def test_fifo_partial_fill_cancel_ahead_age_and_full_fill():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.apply(event(2, "ADD", quantity=6))
    book.watch("o2")
    book.apply(event(3, "ADD", quantity=4))
    assert book.queue("BUY", 99) == ("o1", "o2", "o3")
    assert book.queue_metrics("o2") == {
        "order_id": "o2", "timestamp_ns": 30, "sequence": 3,
        "side": "BUY", "price_ticks": 99, "quantity": 6, "position": 1,
        "orders_ahead": 1, "quantity_ahead": 10, "quantity_behind": 4,
        "age_ns": 10, "age_is_left_censored": False,
    }
    book.apply(event(4, "CANCEL", order_id="o1", quantity=3))
    assert book.queue_metrics("o2")["quantity_ahead"] == 7
    book.apply(event(5, "EXECUTE", order_id="o1", quantity=7))
    assert book.queue_metrics("o2")["position"] == 0
    book.apply(event(6, "EXECUTE", order_id="o2", quantity=2))
    assert book.queue_metrics("o2")["quantity"] == 4
    book.apply(event(7, "EXECUTE", order_id="o2", quantity=4))
    research = book.order_research("o2")
    assert research["position_at_submission"] == 1
    assert research["cancelled_ahead_quantity"] == 3
    assert research["filled_quantity"] == 6
    assert research["time_to_first_fill_ns"] == 40
    assert research["time_to_full_fill_ns"] == 50
    assert research["history"][-2]["queue_movement"] == 1
    assert research["terminal_reason"] == "FILLED"
    assert book.aggregate_l2() == {"bids": [[99, 4]], "asks": []}


def test_cancel_before_fill_and_nonmutating_print_are_not_fills():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")
    book.apply(event(2, "TRADE", price_ticks=99, quantity=10))
    assert book.aggregate_l2()["bids"] == [[99, 10]]
    book.apply(event(3, "CANCEL", order_id="o1", quantity=10))
    metrics = book.cohort_summary(20)
    assert metrics["any_passive_fill_frequency"] == 0
    assert metrics["full_fill_frequency"] == 0
    assert metrics["queue_survival_frequency"] == 0
    assert book.order_research("o1")["time_to_first_fill_ns"] is None
    assert book.summary()["trade_print_count"] == 1
    assert book.summary()["execution_count"] == 0


def test_cancelled_then_executed_remainder_is_not_full_initial_size_fill():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")
    book.apply(event(2, "CANCEL", order_id="o1", quantity=4))
    book.apply(event(3, "EXECUTE", order_id="o1", quantity=6))
    research = book.order_research("o1")
    assert research["terminal_reason"] == "FILLED"
    assert research["time_to_full_fill_ns"] is None
    assert research["initial_quantity"] == 10
    assert book.cohort_summary(20)["full_fill_frequency"] == 0
    assert book.cohort_summary(20)["any_passive_fill_frequency"] == 1


def test_initial_execution_threshold_is_distinct_from_actual_full_fill():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")

    # Increasing quantity loses priority and increases the then-resting size.
    book.apply(event(2, "MODIFY", order_id="o1", quantity=15))

    # Ten units have now executed: this reaches the originally submitted
    # quantity, but the amended order is still resting with five units.
    book.apply(event(3, "EXECUTE", order_id="o1", quantity=10))

    assert book.queue_metrics("o1")["quantity"] == 5

    research = book.order_research("o1")
    assert research["time_to_initial_quantity_executed_ns"] == 20
    assert research["time_to_full_fill_ns"] is None
    assert research["terminal_reason"] is None
    assert research["priority_reset"] is True
    assert research["priority_reset_reason"] == "SIZE_INCREASE"

    report = book.cohort_summary(20)
    assert report["full_fill_frequency"] == 0
    assert report["queue_survival_frequency"] == 1

    # The final recorded execution actually removes the order.
    book.apply(event(4, "EXECUTE", order_id="o1", quantity=5))

    research = book.order_research("o1")
    assert research["terminal_reason"] == "FILLED"
    assert research["time_to_full_fill_ns"] == 30
    assert research["time_to_initial_quantity_executed_ns"] == 20

    report = book.cohort_summary(30)
    assert report["full_fill_frequency"] == 1



def test_modify_retains_or_loses_priority_and_cancellation_ahead_definition():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.apply(event(2, "ADD"))
    book.watch("o2")
    book.apply(event(3, "MODIFY", order_id="o1", quantity=8))
    assert book.queue("BUY", 99) == ("o1", "o2")
    assert book.order_research("o2")["cancelled_ahead_quantity"] == 2
    book.apply(event(4, "MODIFY", order_id="o1", quantity=9))
    assert book.queue("BUY", 99) == ("o2", "o1")
    book.apply(event(5, "MODIFY", order_id="o2", price_ticks=98))
    assert book.queue("BUY", 98) == ("o2",)
    assert book.queue_metrics("o2")["age_ns"] == 30
    assert book.aggregate_l2() == {"bids": [[99, 9], [98, 10]], "asks": []}
    assert book.aggregate_l2(1) == {"bids": [[99, 9]], "asks": []}


def test_execution_uses_recorded_id_instead_of_inventing_head_fills():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.apply(event(2, "ADD"))
    book.apply(event(3, "EXECUTE", order_id="o2", quantity=3))
    assert book.orders["o1"].quantity == 10
    assert book.orders["o2"].quantity == 7


@pytest.mark.parametrize("bad", [
    event(1, "ADD", order_id="o1"),
    event(3, "ADD"),
    event(2, "ADD", timestamp_ns=9),
    event(2, "ADD", symbol="OTHER"),
    event(2, "CANCEL", order_id="o1", quantity=11),
    event(2, "EXECUTE", order_id="o1", quantity=11),
    event(2, "DELETE", order_id="missing"),
    event(2, "MODIFY", order_id="o1", side="SELL", quantity=2),
    event(2, "ADD", side="SELL", price_ticks=98),
])
def test_invalid_transition_leaves_book_and_research_atomic(bad):
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")
    before = book.summary(), book.order_research("o1")
    with pytest.raises(ReconstructionError):
        book.apply(bad)
    assert (book.summary(), book.order_research("o1")) == before
    book.apply(event(2, "EXECUTE", order_id="o1", quantity=1))
    assert book.orders["o1"].quantity == 9


def test_duplicate_ids_remain_forbidden_after_delete_and_reset():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.apply(event(2, "DELETE", order_id="o1"))
    book.apply(event(3))
    with pytest.raises(ReconstructionError, match="already been used"):
        book.apply(event(4, "ADD", order_id="o1"))


def test_initial_snapshot_required_and_snapshot_order_is_source_priority():
    book = MBOBook()
    with pytest.raises(ReconstructionError, match="initial SNAPSHOT"):
        book.apply(event(1, "ADD"))
    book.apply(event(1, "SNAPSHOT", orders=(SnapshotOrder("z", "BUY", 99, 2),
                                          SnapshotOrder("a", "BUY", 99, 3))))
    assert book.queue("BUY", 99) == ("z", "a")
    assert book.queue_metrics("a")["age_is_left_censored"]
    assert book.queue_metrics("a")["age_ns"] is None
    book.watch("a")
    assert book.order_research("a")["position_at_submission"] is None
    assert book.cohort_summary(0)["non_submission_watches"] == 1


@pytest.mark.parametrize("kind", ["RESET", "SNAPSHOT"])
def test_census_censors_unfinished_cohorts_instead_of_inventing_cancels(kind):
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")
    book.apply(event(2, kind))
    report = book.cohort_summary(20)
    assert report["evaluated_orders"] == 0
    assert report["censored_orders"] == 1
    assert report["any_passive_fill_frequency"] is None
    assert book.order_research("o1")["terminal_reason"] == "CENSORED_CENSUS"
    earlier = book.cohort_summary(5)
    assert earlier["evaluated_orders"] == 1
    assert earlier["queue_survival_frequency"] == 1


def test_snapshot_replacement_censors_trajectory_without_implying_zero_remaining_size():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")
    book.apply(event(2, "SNAPSHOT", orders=(SnapshotOrder("o1", "BUY", 99, 7),)))
    research = book.order_research("o1")
    assert book.orders["o1"].quantity == 7
    assert research["terminal_timestamp_ns"] == 20
    assert research["history"][-1]["quantity"] is None
    assert research["filled_quantity"] == 0
    assert research["cancelled_ahead_quantity"] == 0
    assert book.cohort_summary(10)["censored_orders"] == 1
    assert book.cohort_summary(9)["queue_survival_frequency"] == 1


def test_fixed_horizon_cohort_excludes_future_fills_and_tracks_prefix_censoring():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")
    book.apply(event(2, "ADD"))
    book.watch("o2")
    assert book.cohort_summary(20)["censored_orders"] == 2
    book.apply(event(3, "EXECUTE", order_id="o1", quantity=10))
    report = book.cohort_summary(15)
    assert report["evaluated_orders"] == 1
    assert report["censored_orders"] == 1
    assert report["any_passive_fill_frequency"] == 0
    assert report["queue_survival_frequency"] == 1
    report = book.cohort_summary(20)
    assert report["any_passive_fill_frequency"] == 1
    assert report["full_fill_frequency"] == 1
    assert report["queue_survival_frequency"] == 0
    assert report["counterfactual_fill_probability"] is None


def test_returned_research_and_orders_do_not_mutate_internal_state():
    book = initialized()
    book.apply(event(1, "ADD"))
    report = book.watch("o1")
    report["history"][0]["position"] = 100
    book.orders.clear()
    assert book.order_research("o1")["history"][0]["position"] == 0
    assert len(book.orders) == 1


@pytest.mark.parametrize("limit,value,events", [
    ("max_events", 1, [event(1, "ADD")]),
    ("max_orders", 1, [event(1, "ADD"), event(2, "ADD")]),
    ("max_total_order_records", 1, [event(1, "ADD"), event(2), event(3, "ADD")]),
])
def test_book_resource_limits_reject_atomically(limit, value, events):
    book = initialized(**{limit: value})
    for item in events[:-1]:
        book.apply(item)
    before = book.summary()
    with pytest.raises(ValueError, match=limit):
        book.apply(events[-1])
    assert book.summary() == before


def test_watch_bounds_and_late_registration_are_explicit():
    book = initialized(max_watches=1, max_observations=2)
    book.apply(event(1, "ADD"))
    book.apply(event(2, "ADD"))
    book.watch("o1")
    assert not book.order_research("o1")["observed_from_submission"]
    with pytest.raises(ValueError, match="max_watches"):
        book.watch("o2")
    book.apply(event(3, "CANCEL", order_id="o2", quantity=1))
    before = book.summary()
    with pytest.raises(ValueError, match="max_observations"):
        book.apply(event(4, "CANCEL", order_id="o2", quantity=1))
    assert book.summary() == before


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        MBOBook(max_events=value)
    with pytest.raises(ValueError, match="positive integer"):
        MBOReplay("unused.jsonl", max_file_bytes=value)


@pytest.mark.parametrize("compressed", [False, True])
def test_parser_hashes_determinism_and_complete_status(tmp_path, compressed):
    events = [event(), event(1, "ADD"), event(2, "EXECUTE", order_id="o1", quantity=3)]
    path = source(tmp_path, events, compressed)
    replay = MBOReplay(path)
    assert list(replay) == events
    assert replay.stats["complete"]
    assert replay.stats["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    other = MBOReplay(path)
    assert other.run() == 3
    assert other.summary() == replay.summary()
    with pytest.raises(ValueError, match="one-pass"):
        list(replay)


def test_gzip_and_plain_have_identical_canonical_hashes(tmp_path):
    events = [event(), event(1, "ADD")]
    first = MBOReplay(source(tmp_path, events))
    second = MBOReplay(source(tmp_path, events, True))
    first.run()
    second.run()
    assert first.stats["canonical_sha256"] == second.stats["canonical_sha256"]
    assert first.stats["source_sha256"] != second.stats["source_sha256"]


@pytest.mark.parametrize("raw", [
    b"", b"\n", b"[]\n", b"{broken}\n", b"\xff\n",
    b'{"sequence":1,"sequence":2}\n', b'{"quantity":NaN}\n',
])
def test_corrupt_jsonl_never_claims_complete_validation(tmp_path, raw):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(raw)
    replay = MBOReplay(path)
    with pytest.raises(ValueError):
        replay.run()
    assert not replay.stats["complete"]
    assert replay.stats["source_sha256"] is None


@pytest.mark.parametrize("limit,value", [
    ("max_file_bytes", 10), ("max_expanded_bytes", 20),
    ("max_events", 1), ("max_line_bytes", 10),
])
def test_parser_limits_fail_instead_of_truncating(tmp_path, limit, value):
    replay = MBOReplay(source(tmp_path, [event(), event(1, "ADD")], True), **{limit: value})
    with pytest.raises(ValueError, match=limit):
        replay.run()
    assert not replay.stats["complete"]
    assert replay.stats["source_sha256"] is None


def test_unconsumed_prefix_and_bad_gzip_trailer_are_not_full_validation(tmp_path):
    path = source(tmp_path, [event(), event(1, "ADD")], True)
    replay = MBOReplay(path)
    iterator = iter(replay)
    next(iterator)
    iterator.close()
    assert not replay.stats["complete"]
    raw = path.read_bytes()
    for bad in (raw[:-4], raw[:-8] + bytes([raw[-8] ^ 0xFF]) + raw[-7:]):
        path.write_bytes(bad)
        replay = MBOReplay(path)
        with pytest.raises(ValueError, match="fully read"):
            replay.run()
        assert not replay.stats["complete"]
        assert replay.stats["source_sha256"] is None


def test_l2_requires_explicitly_available_evidence(tmp_path):
    state = L2State(1, 1, "VENUE", "SYMBOL", (), (), True, 0, 0, 0)
    for obj in (state, L2Replay(tmp_path / "unused.csv")):
        obj.require_capability("aggregate_depth")
        for capability, available in obj.capabilities.items():
            if not available:
                with pytest.raises(ValueError, match="unavailable from aggregate L2"):
                    obj.require_capability(capability)
        obj.capabilities["order_identity"] = True
        assert not obj.capabilities["order_identity"]


def test_l2_evidence_capabilities_are_immutable():
    with pytest.raises(TypeError):
        L2_CAPABILITIES["order_identity"] = True
    with pytest.raises(ValueError, match="Unknown L2 capability"):
        L2State(1, 1, "VENUE", "SYMBOL", (), (), True, 0, 0, 0).require_capability("imaginary")


@pytest.mark.parametrize("seed", [2, 17, 71])
def test_randomized_lifecycle_matches_independent_aggregation_and_queue_oracle(seed):
    rng = random.Random(seed)
    book = initialized()
    twin = initialized()
    # Independent insertion-ordered oracle keyed by ID; delete/reinsert on
    # priority loss. Integer aggregation avoids asserting against book internals.
    oracle = {}
    for sequence in range(1, 151):
        choice = rng.choice(("ADD", "CANCEL", "EXECUTE", "MODIFY", "DELETE")) if oracle else "ADD"
        if choice == "ADD":
            side = rng.choice(("BUY", "SELL"))
            price = rng.choice((98, 99) if side == "BUY" else (101, 102))
            quantity = rng.randint(1, 30)
            item = event(sequence, "ADD", side=side, price_ticks=price, quantity=quantity)
            oracle[item.order_id] = (side, price, quantity)
        else:
            oid = rng.choice(list(oracle))
            side, price, quantity = oracle[oid]
            if choice in {"CANCEL", "EXECUTE"}:
                reduction = rng.randint(1, quantity)
                item = event(sequence, choice, order_id=oid, quantity=reduction)
                if reduction == quantity:
                    del oracle[oid]
                else:
                    oracle[oid] = (side, price, quantity - reduction)
            elif choice == "DELETE":
                item = event(sequence, choice, order_id=oid)
                del oracle[oid]
            else:
                new_price = rng.choice((98, 99) if side == "BUY" else (101, 102))
                new_quantity = rng.randint(1, 30)
                item = event(sequence, choice, order_id=oid, price_ticks=new_price, quantity=new_quantity)
                if new_price != price or new_quantity > quantity:
                    del oracle[oid]
                oracle[oid] = (side, new_price, new_quantity)
        book.apply(item)
        twin.apply(MBOEvent.from_mapping(item.to_dict()))
        expected = {}
        for side, key in (("BUY", "bids"), ("SELL", "asks")):
            prices = sorted({p for s, p, _ in oracle.values() if s == side}, reverse=side == "BUY")
            expected[key] = [[p, sum(q for s, level, q in oracle.values() if s == side and level == p)]
                             for p in prices]
            for price in prices:
                assert book.queue(side, price) == tuple(oid for oid, (s, p, _) in oracle.items()
                                                        if (s, p) == (side, price))
        assert book.aggregate_l2() == expected
        assert book.snapshot() == twin.snapshot()


def test_redistributable_fixture_is_explicitly_synthetic_and_reproducible():
    path = Path(__file__).resolve().parents[1] / "examples/data/mbo-events.jsonl"
    replay = MBOReplay(path)
    for item in replay:
        assert item.venue == "SYNTHETIC-FIXTURE"
        if item.event_type == "ADD":
            replay.book.watch(item.order_id)
    assert replay.stats["complete"]
    assert replay.stats["events"] == 12
    assert replay.book.summary()["executed_quantity"] == 12
    assert replay.book.aggregate_l2() == {"bids": [], "asks": [[102, 9]]}
    assert replay.book.order_research("b2")["time_to_full_fill_ns"] == 60


# --- v0.3 MBO amendment/full-fill regression tests ---


def test_modify_down_then_terminal_execute_is_observed_full_fill():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")

    # Same-price quantity reduction retains priority.
    book.apply(event(2, "MODIFY", order_id="o1", quantity=5))

    research = book.order_research("o1")
    assert research["priority_reset"] is False

    book.apply(event(3, "EXECUTE", order_id="o1", quantity=5))

    research = book.order_research("o1")
    assert research["terminal_reason"] == "FILLED"
    assert research["filled_quantity"] == 5

    # Only five units executed, so the original-size execution threshold was
    # never reached. Nevertheless the amended resting identity genuinely
    # terminated through an execution.
    assert research["time_to_initial_quantity_executed_ns"] is None
    assert research["time_to_full_fill_ns"] == 20

    assert book.cohort_summary(20)["full_fill_frequency"] == 1


def test_explicit_cancel_then_execute_remainder_is_not_full_fill():
    book = initialized()
    book.apply(event(1, "ADD"))
    book.watch("o1")

    book.apply(event(2, "CANCEL", order_id="o1", quantity=4))
    book.apply(event(3, "EXECUTE", order_id="o1", quantity=6))

    research = book.order_research("o1")

    assert research["terminal_reason"] == "FILLED"
    assert research["cancelled_quantity"] == 4
    assert research["filled_quantity"] == 6
    assert research["time_to_full_fill_ns"] is None
    assert research["time_to_initial_quantity_executed_ns"] is None

    report = book.cohort_summary(20)
    assert report["any_passive_fill_frequency"] == 1
    assert report["full_fill_frequency"] == 0


def test_reprice_makes_submission_queue_movement_unavailable():
    book = initialized()

    book.apply(event(1, "ADD", quantity=8))
    book.apply(event(2, "ADD", quantity=10))
    book.watch("o2")

    assert book.order_research("o2")["history"][-1]["queue_movement"] == 0

    # Repricing loses source FIFO priority. A queue position at another price
    # level must not be presented as advancement relative to submission.
    book.apply(event(3, "MODIFY", order_id="o2", price_ticks=98))

    research = book.order_research("o2")

    assert research["priority_reset"] is True
    assert research["priority_reset_reason"] == "REPRICE"
    assert research["priority_reset_timestamp_ns"] == 30
    assert research["history"][-1]["queue_movement"] is None


def test_same_price_size_increase_resets_priority_movement_baseline():
    book = initialized()

    book.apply(event(1, "ADD", quantity=10))
    book.watch("o1")
    book.apply(event(2, "ADD", quantity=7))

    # o1 starts ahead of o2.
    assert book.queue("BUY", 99) == ("o1", "o2")

    # Same-price increase loses priority in the canonical historical book,
    # moving o1 behind o2.
    book.apply(event(3, "MODIFY", order_id="o1", quantity=12))

    assert book.queue("BUY", 99) == ("o2", "o1")

    research = book.order_research("o1")
    assert research["priority_reset"] is True
    assert research["priority_reset_reason"] == "SIZE_INCREASE"
    assert research["history"][-1]["queue_movement"] is None

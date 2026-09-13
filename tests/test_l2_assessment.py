"""Independent streaming depth comparisons and causal historical statistics."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
from decimal import Decimal

import pytest

from lob.replay import assessment
from lob.replay.assessment import SNAPSHOT_COLUMNS, TRADE_COLUMNS, assess_l2, read_snapshots
from lob.replay.l2 import L2_COLUMNS, L2State


def csv_file(tmp_path, name, columns, records, *, compressed=False):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    raw = buffer.getvalue().encode()
    path = tmp_path / (name + (".csv.gz" if compressed else ".csv"))
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    return path


def identity(local=1_000_000, **changes):
    return {"exchange": "venue", "symbol": "BTCUSD", "timestamp": str(local),
            "local_timestamp": str(local), **changes}


def updates(local=1_000_000, bid="100", ask="102", amount="1", snapshot=False):
    return [{**identity(local), "is_snapshot": "true" if snapshot else "false",
             "side": side, "price": price, "amount": amount}
            for side, price in (("bid", bid), ("ask", ask))]


def snapshot(local=1_000_000, bid="100", ask="102", amount="1", **changes):
    result = dict.fromkeys(SNAPSHOT_COLUMNS, "")
    result.update(identity(local))
    result.update({"bids[0].price": bid, "bids[0].amount": amount,
                   "asks[0].price": ask, "asks[0].amount": amount})
    result.update(changes)
    return result


def trade(local=1_000_000, **changes):
    return {**identity(local), "id": str(local), "side": "buy", "price": "101",
            "amount": "0.25", **changes}


def inputs(tmp_path, update_rows=None, reference_rows=None, *, compressed=False):
    update_rows = updates(snapshot=True) if update_rows is None else update_rows
    reference_rows = [snapshot()] if reference_rows is None else reference_rows
    return (csv_file(tmp_path, "updates", L2_COLUMNS, update_rows, compressed=compressed),
            csv_file(tmp_path, "snapshots", SNAPSHOT_COLUMNS, reference_rows, compressed=compressed))


def test_exact_reconstruction_matches_reference_and_hashes_full_inputs(tmp_path):
    paths = inputs(tmp_path, updates(snapshot=True) + updates(2_000_000, amount="2"),
                   [snapshot(), snapshot(2_000_000, amount="2")], compressed=True)
    trades = csv_file(tmp_path, "trades", TRADE_COLUMNS,
                      [trade(), trade(2_000_000, side="sell")], compressed=True)
    result = assess_l2(*paths, trades)
    assert result["status"] == "PASS"
    comparison = result["reference_comparison"]
    assert comparison["status"] == "MATCH"
    assert comparison["compared"] == comparison["exact_matches"] == comparison["reference_rows"] == 2
    assert result["replay"]["complete"]
    assert result["regular_grid"]["metrics"]["mid_price"]["count"] == 2
    assert result["trades"]["count"] == 2
    assert result["trades"]["total_amount_native"] == "0.50"
    for key, path in zip(("updates", "snapshots", "trades"), (*paths, trades)):
        assert result["inputs"][key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert result["inputs"][key]["bytes"] == path.stat().st_size


def test_different_book_is_exact_decimal_mismatch_not_float_near_equality(tmp_path):
    paths = inputs(tmp_path, reference_rows=[snapshot(amount="1.000000000000000001")])
    result = assess_l2(*paths)
    assert result["status"] == "WARNING"
    assert result["reference_comparison"]["different_books"] == 1
    assert result["reference_comparison"]["exact_matches"] == 0
    assert result["reference_comparison"]["first_issues"] == [
        {"kind": "different_depth_or_price", "local_timestamp_us": 1_000_000}]


def test_equal_book_with_different_exchange_time_reports_timestamp_mismatch(tmp_path):
    paths = inputs(tmp_path, reference_rows=[snapshot(timestamp="999999")])
    result = assess_l2(*paths)
    comparison = result["reference_comparison"]
    assert result["status"] == "WARNING"
    assert comparison["status"] == "MISMATCH"
    assert comparison["exact_matches"] == 1
    assert comparison["different_books"] == 0
    assert comparison["exchange_timestamp_mismatches"] == 1
    assert comparison["first_issues"] == [
        {"kind": "different_exchange_timestamp", "local_timestamp_us": 1_000_000}]


def test_missing_changed_group_and_extra_reference_both_count(tmp_path):
    paths = inputs(tmp_path, updates(snapshot=True) + updates(3_000_000, amount="2"),
                   [snapshot(), snapshot(2_000_000)])
    result = assess_l2(*paths)
    comparison = result["reference_comparison"]
    assert result["status"] == "WARNING"
    assert comparison["exact_matches"] == 1
    assert comparison["reference_without_update_group"] == 1
    assert comparison["changed_top_without_reference"] == 1


def test_unchanged_top_does_not_require_extra_published_snapshot(tmp_path):
    paths = inputs(tmp_path, updates(snapshot=True) + updates(2_000_000), [snapshot()])
    result = assess_l2(*paths)
    assert result["status"] == "PASS"
    assert result["replay"]["states"] == 2
    assert result["reference_comparison"]["changed_top_without_reference"] == 0


def state(local, bid="100", ask="102"):
    return L2State(local, local, "venue", "BTCUSD",
                   ((Decimal(bid), Decimal(1)),) if bid else (),
                   ((Decimal(ask), Decimal(1)),) if ask else (), False,
                   int(bool(bid)), int(bool(ask)), 2)


def test_regular_grid_uses_previous_completed_group_and_never_future_book():
    sampler = assessment._Sampler()
    sampler.add(state(1_200_000, "100", "102"))
    sampler.add(state(2_400_000, "200", "202"))
    sampler.add(state(3_000_000, "300", "302"))
    result = sampler.result()
    mid = result["metrics"]["mid_price"]
    # Grid at t=2 uses the book from 1.2; t=3 uses its exact-time state.
    assert mid["count"] == 2
    assert mid["min"] == 101
    assert mid["max"] == 301
    assert mid["mean"] == 201
    assert result["maximum_sample_age_seconds"] == .8
    assert sampler.result() == result  # reading the report does not append a sample


def test_regular_grid_counts_invalid_and_stale_samples_without_eof_extrapolation():
    sampler = assessment._Sampler()
    sampler.add(state(1_000_000, bid="", ask=""))
    sampler.add(state(2_000_000))
    sampler.add(state(9_200_000))
    result = sampler.result()
    assert result["invalid_book_samples"] == 1
    assert result["metrics"]["mid_price"]["count"] == 8
    assert result["samples_older_than_5_seconds"] == 2
    assert result["maximum_sample_age_seconds"] == 7


@pytest.mark.parametrize("rows", [
    [snapshot(2), snapshot(1)],
    [snapshot(1), snapshot(1)],
])
def test_reference_local_regression_or_duplicate_timestamp_rejected(tmp_path, rows):
    path = csv_file(tmp_path, "snapshots", SNAPSHOT_COLUMNS, rows)
    with pytest.raises(ValueError, match="strictly increase"):
        list(read_snapshots(path))


@pytest.mark.parametrize("changes", [{"symbol": "OTHER"}, {"exchange": "OTHER"}])
@pytest.mark.parametrize("local", [500_000, 1_000_000, 2_000_000])
def test_identity_mismatch_rejected_even_without_matching_timestamp(tmp_path, changes, local):
    paths = inputs(tmp_path, reference_rows=[snapshot(local, **changes)])
    with pytest.raises(ValueError, match="exchange/symbol"):
        assess_l2(*paths)


def test_trade_dates_must_match_book_utc_day(tmp_path):
    paths = inputs(tmp_path)
    trades = csv_file(tmp_path, "trades", TRADE_COLUMNS, [trade(86_401_000_000)])
    with pytest.raises(ValueError, match="dates"):
        assess_l2(*paths, trades)


@pytest.mark.parametrize("changes,error", [
    ({"symbol": "OTHER"}, "exchange/symbol"),
    ({"side": "BUY"}, "side"),
    ({"amount": "-1"}, "decimal"),
])
def test_invalid_trade_identity_side_or_amount_rejected(tmp_path, changes, error):
    paths = inputs(tmp_path)
    trades = csv_file(tmp_path, "trades", TRADE_COLUMNS, [trade(**changes)])
    with pytest.raises(ValueError, match=error):
        assess_l2(*paths, trades)


def test_trade_capture_regression_rejected(tmp_path):
    paths = inputs(tmp_path)
    trades = csv_file(tmp_path, "trades", TRADE_COLUMNS, [trade(2_000_000), trade()])
    with pytest.raises(ValueError, match="regress"):
        assess_l2(*paths, trades)


def test_duplicate_trade_ids_warn_and_exchange_clock_regression_is_counted(tmp_path):
    paths = inputs(tmp_path)
    trades = csv_file(tmp_path, "trades", TRADE_COLUMNS,
                      [trade(id="same"), trade(2_000_000, id="same", timestamp="999999")])
    result = assess_l2(*paths, trades)
    assert result["status"] == "WARNING"
    assert result["trades"]["duplicate_trade_ids"] == 1
    assert result["trades"]["exchange_timestamp_regressions"] == 1


def test_trade_total_preserves_all_accepted_decimal_digits(tmp_path):
    paths = inputs(tmp_path)
    trades = csv_file(tmp_path, "trades", TRADE_COLUMNS,
                      [trade(amount="12345678901234567890.123456789012345678"),
                       trade(2_000_000, amount="0.000000000000000001")])
    result = assess_l2(*paths, trades)
    assert Decimal(result["trades"]["total_amount_native"]) == Decimal(
        "12345678901234567890.123456789012345679")


@pytest.mark.parametrize("file_index", [0, 1, 2])
def test_every_gzip_input_requires_valid_trailer(tmp_path, file_index):
    paths = list(inputs(tmp_path, compressed=True))
    paths.append(csv_file(tmp_path, "trades", TRADE_COLUMNS, [trade()], compressed=True))
    original = paths[file_index].read_bytes()
    paths[file_index].write_bytes(original[:-4])
    with pytest.raises(ValueError, match="truncated|fully read"):
        assess_l2(*paths)


def test_corrupt_deflate_payload_is_reported_as_invalid_input(tmp_path):
    paths = inputs(tmp_path, compressed=True)
    raw = bytearray(paths[1].read_bytes())
    raw[10] |= 0b110  # Invalid DEFLATE block type in the first compressed block.
    paths[1].write_bytes(raw)
    with pytest.raises(ValueError, match="invalid|truncated"):
        assess_l2(*paths)


def test_source_mutation_during_assessment_rejected(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    original = assessment.sha256_file
    count = {}

    def changed_on_recheck(path):
        count[path] = count.get(path, 0) + 1
        return original(path) if count[path] == 1 else "0" * 64

    monkeypatch.setattr(assessment, "sha256_file", changed_on_recheck)
    with pytest.raises(ValueError, match="changed during"):
        assess_l2(*paths)


def test_implementation_mutation_during_assessment_rejected(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    manifests = iter(({"module": "first"}, {"module": "changed"}))
    monkeypatch.setattr(assessment, "source_manifest", lambda: next(manifests))
    with pytest.raises(ValueError, match="implementation changed"):
        assess_l2(*paths)


@pytest.mark.parametrize("kind", ["updates", "snapshots", "trades"])
def test_empty_data_stream_cannot_report_pass(tmp_path, kind):
    paths = list(inputs(tmp_path, update_rows=[] if kind == "updates" else None,
                        reference_rows=[] if kind == "snapshots" else None))
    if kind == "trades":
        paths.append(csv_file(tmp_path, "trades", TRADE_COLUMNS, []))
    with pytest.raises(ValueError, match="snapshot|nonempty|no records"):
        assess_l2(*paths)


def test_explicit_empty_book_is_reported_as_invalid_sample(tmp_path):
    paths = inputs(tmp_path, updates(amount="0", snapshot=True),
                   [snapshot(bid="", ask="", amount="")])
    result = assess_l2(*paths)
    assert result["reference_comparison"]["status"] == "MATCH"
    assert result["status"] == "WARNING"
    assert result["regular_grid"]["invalid_book_samples"] == 1


def test_off_grid_empty_book_cannot_evade_overall_quality_warning(tmp_path):
    local = 1_200_000
    paths = inputs(tmp_path, updates(local, amount="0", snapshot=True),
                   [snapshot(local, bid="", ask="", amount="")])
    result = assess_l2(*paths)
    assert result["reference_comparison"]["status"] == "MATCH"
    assert result["regular_grid"]["invalid_book_samples"] == 0
    assert result["replay"]["empty_groups"] == 1
    assert result["status"] == "WARNING"


@pytest.mark.parametrize("field,value", [
    ("bids[0].price", "١٠٠"),
    ("timestamp", "١٠٠٠٠٠٠"),
    ("local_timestamp", "1.0"),
    ("bids[0].amount", "NaN"),
    ("asks[0].amount", "0"),
    ("asks[0].price", " 102"),
])
def test_strict_reference_numeric_fields(tmp_path, field, value):
    path = csv_file(tmp_path, "snapshots", SNAPSHOT_COLUMNS, [snapshot(**{field: value})])
    with pytest.raises(ValueError):
        list(read_snapshots(path))


def test_reference_scientific_decimal_has_same_exact_semantics_as_updates(tmp_path):
    paths = inputs(tmp_path, updates(amount="1e-8", snapshot=True),
                   [snapshot(amount="0.00000001")])
    assert assess_l2(*paths)["reference_comparison"]["status"] == "MATCH"
    paths = inputs(tmp_path, updates(amount="0.00000001", snapshot=True),
                   [snapshot(amount="1e-8")])
    assert assess_l2(*paths)["reference_comparison"]["status"] == "MATCH"


@pytest.mark.parametrize("changes,error", [
    ({"bids[0].amount": ""}, "paired trailing blanks"),
    ({"bids[2].price": "98", "bids[2].amount": "1"}, "paired trailing blanks"),
    ({"bids[1].price": "100", "bids[1].amount": "1"}, "duplicate or unsorted"),
    ({"bids[1].price": "101", "bids[1].amount": "1"}, "duplicate or unsorted"),
    ({"asks[0].price": "100"}, "crossed or locked"),
])
def test_invalid_reference_book_structure_rejected(tmp_path, changes, error):
    path = csv_file(tmp_path, "snapshots", SNAPSHOT_COLUMNS, [snapshot(**changes)])
    with pytest.raises(ValueError, match=error):
        list(read_snapshots(path))


@pytest.mark.parametrize("max_rows", [0, -1, True, 1.5, 20_000_001])
def test_assessment_row_bound_must_be_valid_integer(tmp_path, max_rows):
    with pytest.raises(ValueError, match="max_rows"):
        assess_l2(tmp_path / "no-updates", tmp_path / "no-snapshots", max_rows=max_rows)


def test_reference_row_limit_fails_without_silently_accepting_prefix(tmp_path):
    path = csv_file(tmp_path, "snapshots", SNAPSHOT_COLUMNS, [snapshot(), snapshot(2_000_000)])
    with pytest.raises(ValueError, match="row limit"):
        list(read_snapshots(path, max_rows=1))

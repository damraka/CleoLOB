"""Real-data adapter semantics: exact quantities and capture-group consistency."""
from __future__ import annotations

import csv
import gzip
import io
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from lob.replay.l2 import L2_COLUMNS, L2Replay, exact_decimal


def row(local=1, side="bid", price="100.1", amount="0.25", snapshot=True, **changes):
    result = dict(zip(L2_COLUMNS, (
        "venue", "BTCUSD", str(local), str(local),
        "true" if snapshot else "false", side, price, amount,
    )))
    result.update(changes)
    return result


def source(tmp_path, rows, *, compressed=False, header=L2_COLUMNS):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode("utf-8")
    path = tmp_path / ("incremental.csv.gz" if compressed else "incremental.csv")
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    return path


def initial(local=1):
    return [row(local), row(local, "ask", "100.2", "0.75")]


def test_exact_decimal_absolute_updates_and_zero_delete(tmp_path):
    path = source(tmp_path, initial() + [
        row(2, amount="0.3", snapshot=False),
        row(2, "bid", "100.0", "1e-8", False),
        row(3, "bid", "100.1", "0", False),
    ])
    replay = L2Replay(path)
    states = list(replay)
    assert len(states) == 3
    assert states[1].bids == ((Decimal("100.1"), Decimal("0.3")),
                              (Decimal("100.0"), Decimal("0.00000001")))
    assert states[2].bids == ((Decimal("100"), Decimal("0.00000001")),)
    assert states[0].bids == ((Decimal("100.1"), Decimal("0.25")),)
    assert replay.stats["complete"]
    assert replay.stats["rows"] == 5
    assert replay.stats["groups"] == replay.stats["states"] == 3
    assert replay.stats["snapshots"] == 1
    with pytest.raises(FrozenInstanceError):
        states[0].symbol = "OTHER"
    with pytest.raises(ValueError, match="one-pass"):
        list(replay)


def test_group_is_atomic_and_does_not_report_transient_crossing(tmp_path):
    replay = L2Replay(source(tmp_path, initial() + [
        row(2, "bid", "100.3", "0.5", False),  # temporarily crosses old ask
        row(2, "ask", "100.2", "0", False),
        row(2, "ask", "100.4", "0.5", False),
    ]))
    states = list(replay)
    assert len(states) == 2
    assert states[-1].rows_in_group == 3
    assert not states[-1].crossed
    assert replay.stats["crossed_groups"] == 0


def test_completed_crossed_state_is_retained_without_vendor_cleaning(tmp_path):
    replay = L2Replay(source(tmp_path, initial() + [row(2, "bid", "101", "1", False)]))
    states = list(replay)
    assert states[-1].bids[0][0] == Decimal("101")
    assert states[-1].asks[0][0] == Decimal("100.2")
    assert states[-1].crossed
    assert replay.stats["crossed_groups"] == 1


def test_presnapshot_updates_skipped_then_snapshot_blocks_reset_once(tmp_path):
    rows = [row(0, "bid", "999", "1", False), *initial(1),
            row(2, "bid", "99", "1", True),  # same true block, no reset
            row(3, "ask", "100.4", "1", False),
            row(4, "bid", "90", "1", True),  # new true block clears old book
            row(5, "ask", "91", "1", True)]
    replay = L2Replay(source(tmp_path, rows))
    states = list(replay)
    assert replay.stats["presnapshot_rows"] == 1
    assert replay.stats["groups"] == 6
    assert replay.stats["states"] == 5
    assert replay.stats["snapshots"] == 2
    assert states[1].bid_levels == 2
    assert states[3].one_sided
    assert states[3].bids == ((Decimal("90"), Decimal("1")),)
    assert states[3].asks == ()
    assert states[4].bids == states[3].bids
    assert states[4].asks == ((Decimal("91"), Decimal("1")),)
    assert replay.stats["one_sided_groups"] == 1


def test_depth_limits_export_but_keeps_deeper_levels_for_later_updates(tmp_path):
    rows = initial() + [row(1, "bid", "99", "1"),
                        row(2, "bid", "100.1", "0", False)]
    states = list(L2Replay(source(tmp_path, rows), depth=1))
    assert len(states[0].bids) == 1
    assert states[0].bid_levels == 2
    assert states[1].bids == ((Decimal("99"), Decimal("1")),)


def test_empty_books_unknown_deletions_and_group_gaps_counted(tmp_path):
    rows = initial() + [row(4, "bid", "100.1", "0", False),
                        row(4, "ask", "100.2", "0", False),
                        row(4, "ask", "999", "0", False)]
    replay = L2Replay(source(tmp_path, rows))
    assert list(replay)[-1].empty
    assert replay.stats["empty_groups"] == 1
    assert replay.stats["missing_level_deletes"] == 1
    assert replay.stats["max_local_gap_us"] == 3


def test_exchange_clock_regression_is_counted_without_reordering(tmp_path):
    rows = initial(10) + [row(11, snapshot=False, timestamp="9")]
    replay = L2Replay(source(tmp_path, rows))
    states = list(replay)
    assert states[-1].timestamp_us == 9
    assert states[-1].local_timestamp_us == 11
    assert replay.stats["exchange_clock_regressions"] == 1


@pytest.mark.parametrize("changes,error", [
    ({"local_timestamp": "0"}, "local_timestamp regressed"),
    ({"symbol": "OTHER"}, "one exchange and symbol"),
    ({"exchange": "OTHER"}, "one exchange and symbol"),
    ({"symbol": " PADDED"}, "identifier"),
    ({"price": "NaN"}, "decimal"),
    ({"price": "0"}, "positive"),
    ({"amount": "-1"}, "decimal"),
    ({"amount": "Infinity"}, "decimal"),
    ({"timestamp": "1.0"}, "integer microseconds"),
    ({"timestamp": "9223372036854775808"}, "64-bit"),
    ({"side": "BUY"}, "bid or ask"),
    ({"is_snapshot": "True"}, "true or false"),
])
def test_bad_rows_fail_with_incomplete_report(tmp_path, changes, error):
    replay = L2Replay(source(tmp_path, initial() + [row(2, snapshot=False, **changes)]))
    with pytest.raises(ValueError, match=error):
        list(replay)
    assert not replay.stats["complete"]


@pytest.mark.parametrize("limit,value,error", [
    ("max_file_bytes", 10, "max_file_bytes"),
    ("max_expanded_bytes", 120, "max_expanded_bytes"),
    ("max_rows", 1, "max_rows"),
    ("max_group_rows", 1, "max_group_rows"),
    ("max_levels", 1, "max_levels"),
    ("max_line_bytes", 10, "max_line_bytes"),
])
def test_limits_reject_instead_of_truncate(tmp_path, limit, value, error):
    replay = L2Replay(source(tmp_path, initial(), compressed=True), depth=1, **{limit: value})
    with pytest.raises(ValueError, match=error):
        list(replay)
    assert not replay.stats["complete"]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_invalid_resource_limit_configuration_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="positive integer"):
        L2Replay(tmp_path / "unused.csv", max_rows=value)


def test_gzip_crc_and_trailer_are_verified_when_fully_consumed(tmp_path):
    path = source(tmp_path, initial() + [row(2, snapshot=False)], compressed=True)
    original = path.read_bytes()
    replay = L2Replay(path)
    assert len(list(replay)) == 2
    assert replay.stats["complete"]
    for malformed in (original[:-4], original[:-8] + bytes([original[-8] ^ 0xFF]) + original[-7:]):
        path.write_bytes(malformed)
        replay = L2Replay(path)
        with pytest.raises(ValueError, match="fully read"):
            list(replay)
        assert not replay.stats["complete"]


def test_prefix_consumption_does_not_claim_full_validation(tmp_path):
    replay = L2Replay(source(tmp_path, initial() + [row(2, snapshot=False)]))
    iterator = iter(replay)
    next(iterator)
    assert not replay.stats["complete"]
    iterator.close()
    assert not replay.stats["complete"]


@pytest.mark.parametrize("raw,error", [
    (b"", "empty"),
    (b"exchange,exchange\n", "header"),
    ((",".join(L2_COLUMNS) + "\n").encode(), "no initial snapshot"),
    ((",".join(L2_COLUMNS) + "\n\n").encode(), "column count"),
    ((",".join(L2_COLUMNS) + "\n\xff\n").encode("latin-1"), "UTF-8"),
    ((",".join(L2_COLUMNS) + '\n"unterminated\n').encode(), "CSV"),
])
def test_malformed_csv_fails(tmp_path, raw, error):
    path = tmp_path / "bad.csv"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=error):
        list(L2Replay(path))


def test_all_updates_without_snapshot_rejected(tmp_path):
    replay = L2Replay(source(tmp_path, [row(snapshot=False)]))
    with pytest.raises(ValueError, match="no initial snapshot"):
        list(replay)
    assert replay.stats["presnapshot_rows"] == 1


@pytest.mark.parametrize("text", ["1e999", "1e-31", "1" * 39, " 1", "+1", "1.", ".1", "-0"])
def test_decimal_range_and_representation_are_bounded(text):
    with pytest.raises(ValueError):
        exact_decimal(text, "amount", allow_zero=True)


def test_stats_are_copied_and_cannot_forge_completion(tmp_path):
    replay = L2Replay(source(tmp_path, initial()))
    replay.stats["complete"] = True
    assert not replay.summary()["complete"]

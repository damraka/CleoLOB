"""v1 order-book reconstruction: schema, L2, gaps, timing, sync, observability, hidden, ensembles, validation."""
import numpy as np
import pytest

from lob.v1.data import schema as SC
from lob.v1.data.reconstruction import ensemble as EN, hidden as HD, l2, mbo, observability as OB, sync, timing, \
    validate as VA

K = SC.EventKind


def ev(rows):
    """rows: (ts_ms, seq, kind, side, price, size[, order_id, aux, recv_ms])"""
    out = SC.empty(len(rows))
    for i, r in enumerate(rows):
        out[i]["ts_exchange"], out[i]["seq"], out[i]["kind"] = r[0] * 1_000_000, r[1], r[2]
        out[i]["side"], out[i]["price"], out[i]["size"] = r[3], r[4], r[5]
        if len(r) > 6:
            out[i]["order_id"] = r[6]
        if len(r) > 7:
            out[i]["aux"] = r[7]
        if len(r) > 8:
            out[i]["ts_receive"] = r[8] * 1_000_000
    return out


def snapshot(ts, seq, bids, asks):
    rows = [(ts, seq, K.SNAPSHOT_BEGIN, 0, 0, 0.0)]
    rows += [(ts, seq, K.LEVEL_SET, 1, p, q) for p, q in bids] + [(ts, seq, K.LEVEL_SET, -1, p, q) for p, q in asks]
    return rows + [(ts, seq, K.SNAPSHOT_END, 0, 0, 0.0)]


def random_stream(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = snapshot(0, 1, [(1000 - i, 5.0) for i in range(20)], [(1001 + i, 5.0) for i in range(20)])
    bid, ask = 1000, 1001
    for u in range(2, n):
        side = 1 if rng.random() < 0.5 else -1
        best = bid if side > 0 else ask
        price = best - side * int(rng.integers(0, 15))
        size = float(rng.integers(0, 8))
        if side > 0 and price >= ask:
            price = ask - 1
        if side < 0 and price <= bid:
            price = bid + 1
        rows.append((u, u, K.LEVEL_SET, side, int(price), size))
        bid = max(bid, price) if side > 0 and size > 0 else bid
        ask = min(ask, price) if side < 0 and size > 0 else ask
    return ev(rows)


# ---------------------------------------------------------------------------------------------- M20 schema
def test_schema_validation_counts_structure():
    e = ev(snapshot(0, 1, [(10, 1.0)], [(11, 1.0)]) + [(1, 2, K.LEVEL_SET, 1, 10, -1.0)])
    v = SC.validate_events(e)
    assert v["negative_sizes"] == 1 and v["by_kind"]["LEVEL_SET"] == 3 and v["ts_backwards"] == 0
    with pytest.raises(TypeError):
        SC.validate_events(np.zeros(3))


# ---------------------------------------------------------------------------------------------- M21 L2
def test_l2_dense_equals_reference_on_random_streams():
    for seed in range(5):
        e = random_stream(seed=seed)
        trace = lambda b, ts, ok: (b.best(1), b.best(-1), tuple(b.visible(1)), tuple(b.visible(-1)))  # noqa: E731
        _, c1, t1 = l2.replay([e], l2.DenseBook(width=4096), every=trace)
        _, c2, t2 = l2.replay([e], l2.ReferenceBook(), every=trace)
        assert t1 == t2 and c1.messages == c2.messages


def test_l2_chunking_does_not_change_results():
    e = random_stream(seed=7)
    whole = l2.replay([e], l2.ReferenceBook(), every=lambda b, ts, ok: (b.best(1), b.best(-1)))[2]
    pieces = [e[i:i + 13] for i in range(0, len(e), 13)]
    assert l2.replay(pieces, l2.ReferenceBook(), every=lambda b, ts, ok: (b.best(1), b.best(-1)))[2] == whole


def test_l2_recentering_is_lossless_within_window():
    e = ev(snapshot(0, 1, [(100, 1.0)], [(101, 1.0)]) + [(1, 2, K.LEVEL_SET, 1, 3000, 2.0),
                                                       (2, 3, K.LEVEL_SET, -1, 3001, 2.0)])
    b, c, _ = l2.replay([e], l2.DenseBook(width=1024))
    assert c.recenters >= 1 and b.best(1) == 3000 and b.best(-1) == 3001


# ---------------------------------------------------------------------------------------------- M23 gaps
def test_gap_makes_book_unreliable_until_next_snapshot():
    rows = snapshot(0, 1, [(10, 1.0)], [(11, 1.0)])
    rows += [(1, 2, K.LEVEL_SET, 1, 9, 1.0), (2, 5, K.LEVEL_SET, 1, 8, 1.0), (3, 6, K.LEVEL_SET, 1, 7, 1.0)]
    rows += snapshot(4, 7, [(10, 2.0)], [(11, 2.0)]) + [(5, 8, K.LEVEL_SET, 1, 9, 3.0)]
    flags = l2.replay([ev(rows)], every=lambda b, ts, ok: ok)[2]
    assert flags == [True, True, False, False, True, True]
    _, c, _ = l2.replay([ev(rows)])
    assert c.sequence_gaps == 1 and c.unreliable_messages == 2 and len(c.gap_intervals) == 1


def test_gap_frames_are_invalid():
    rows = snapshot(0, 1, [(10, 1.0)], [(11, 1.0)]) + [(150, 9, K.LEVEL_SET, 1, 9, 1.0)]
    rows += snapshot(450, 10, [(10, 1.0)], [(11, 1.0)])
    f = l2.frames([ev(rows)], SC.empty(0), day="1970-01-01", bins=8)
    # state at bin start: bin 0 precedes the snapshot, bins 2-4 follow the gap, bin 5 follows the re-anchor
    assert f["valid"].tolist() == [False, True, False, False, False, True, True, True]


# ---------------------------------------------------------------------------------------------- M24 timing
def test_timing_alignment_offset_and_inversions():
    rows = [(t, -1, K.ORDER_ADD, 1, 10, 1.0, t, -1, t + 50 + (t % 7)) for t in range(0, 600_000, 1000)]
    rows.append((100, -1, K.ORDER_ADD, 1, 10, 1.0, 1, -1, 600_100))          # received late, out of order
    a = timing.alignment(ev(rows))
    assert a["status"] == "OK" and a["offset_ms"] == pytest.approx(50.0)
    assert a["inversions_total"] == 1 and a["corrected_delay_ms"]["0.0"] == 0.0
    assert timing.alignment(ev([(1, 1, K.LEVEL_SET, 1, 1, 1.0)] * 20))["status"] == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------------------------- M25 sync
def test_sync_classifies_prints_and_measures_book_lag():
    book = ev(snapshot(0, 1, [(10, 5.0)], [(11, 5.0), (12, 5.0)]) +
              [(60, 2, K.LEVEL_SET, -1, 11, 2.0), (200, 3, K.LEVEL_SET, -1, 12, 1.0)])
    trades = ev([(10, -1, K.TRADE, 1, 11, 3.0), (100, -1, K.TRADE, 1, 12, 4.0)])
    d = sync.diagnostics([book], trades)
    assert d["classification"]["at_touch"] == 1 and d["classification"]["through"] == 1
    assert d["matched"] == 2 and d["book_lag_ms"]["p50"] == pytest.approx(75.0)


# ---------------------------------------------------------------------------------------------- M26 observability
def test_observability_classes_and_l2_bounds():
    from lob.v1.data.connectors import base as CB
    assert OB.classify(CB.get("bybit").capability("linear", "BTCUSDT", "2026-01-01"))["partial_identification"]
    assert OB.classify(CB.get("bitstamp").capability("exchange", "btcusd", "2026-01-01"))["exact"]
    b = OB.l2_queue_bounds(10.0, np.array([10, 8, 5, 5]), np.array([0, 2, 0, 3]))
    assert b["ahead_optimistic"].tolist() == [10, 8, 5, 2]
    assert b["ahead_pessimistic"].tolist() == [10, 8, 8, 5]
    assert np.all(b["width"] >= 0)
    b2 = OB.l2_queue_bounds(4.0, np.array([4, 0]), np.array([0, 1]))
    assert b2["fill_start_optimistic"] == 1 and b2["fill_start_pessimistic"] is None


# ---------------------------------------------------------------------------------------------- M27 hidden
def test_hidden_liquidity_evidence_and_interval():
    e = HD.refill_evidence(np.array([5, 5, 2]), np.array([7, 3, 4]), np.array([1, 2, 0]))
    assert e["prints_with_excess"] == 1 and e["unseen_volume_lower_bound"] == 2.0
    iv = HD.hidden_interval(e)
    assert iv["lower"] == 0.0 and iv["point_lower_bound"] == pytest.approx(2 / 14) and iv["upper"] > 2 / 14
    l3 = HD.l3_evidence(np.array([True, False, True]), np.array([1.0, 2.0, 5.0]), np.array([1.0, 0.0, 3.0]))
    assert l3["unknown_maker"] == 1 and l3["oversized"] == 1 and l3["unseen_volume_lower_bound"] == 4.0
    assert HD.hidden_interval({"executed_volume": 0, "prints": 0, "unseen_volume_lower_bound": 0})["status"] == \
        "NOT_IDENTIFIED"


# ---------------------------------------------------------------------------------------------- M28 ensemble
def test_ensemble_members_span_attribution_ambiguity():
    n = 50
    frames = {"depth": np.zeros((n, 2, 10)), "volume": np.zeros((n, 2)), "bid_px": np.full(n, 10),
              "ask_px": np.full(n, 11), "valid": np.ones(n, bool)}
    frames["depth"][:, 0, 0] = np.where(np.arange(n) % 2 == 0, 10.0, 6.0)   # alternating decreases of 4
    frames["volume"][:, 1] = 2.0                                              # 2 lots sold per bin
    rec = EN.decrease_records(frames)
    out = EN.ensemble(rec, members=16, seed=1)
    s = out["spread"]["side0:cancel_volume"]
    assert s["p05"] < s["p95"] and len(out["members"]) == 16
    always = EN.ensemble(rec, members=4, seed=1, p_trade_first=(1.0, 1.0), merge_mean=1.0)
    assert always["spread"]["side0:cancel_volume"]["relative_width"] == 0.0


# ---------------------------------------------------------------------------------------------- M22 / M29 MBO
def mbo_stream():
    rows = [(0, -1, K.SNAPSHOT_BEGIN, 0, 0, 0.0), (0, -1, K.ORDER_ADD, 1, 100, 1.0, 1),
            (0, -1, K.ORDER_ADD, 1, 100, 2.0, 2), (0, -1, K.ORDER_ADD, -1, 101, 1.0, 3),
            (0, -1, K.SNAPSHOT_END, 0, 0, 0.0),
            (0, -1, K.ORDER_ADD, 1, 99, 9.0, 9),                        # pre-anchor (already in census): skipped
            (1, -1, K.ORDER_ADD, 1, 100, 3.0, 4),
            (2, -1, K.TRADE, -1, 100, 0.5, 1, 77),
            (2, -1, K.ORDER_MODIFY, 1, 100, 0.5, 1),                    # partial fill keeps priority
            (3, -1, K.ORDER_MODIFY, 1, 100, 5.0, 2),                    # increase loses priority
            (4, -1, K.ORDER_DELETE, -1, 101, 0.0, 3),
            (5, -1, K.ORDER_DELETE, -1, 101, 0.0, 42),                  # unknown order
            (6, -1, K.SNAPSHOT_BEGIN, 0, 0, 0.0), (6, -1, K.ORDER_ADD, 1, 100, 0.5, 1),
            (6, -1, K.ORDER_ADD, 1, 100, 3.0, 4), (6, -1, K.ORDER_ADD, 1, 100, 5.0, 2),
            (6, -1, K.ORDER_ADD, -1, 102, 1.0, 5), (6, -1, K.SNAPSHOT_END, 0, 0, 0.0)]
    return ev(rows)


def test_mbo_reconstruction_priority_and_reconciliation():
    positions = {}

    def on(book, r):
        if int(r["kind"]) == K.ORDER_MODIFY:
            positions[int(r["order_id"])] = book.queue_position(int(r["order_id"]))
    book, st = mbo.reconstruct([mbo_stream()], on_state=on)
    assert st.pre_anchor_skipped == 1 and st.unknown_deletes == 1 and st.priority_losses == 1
    assert st.trades_maker_known == 1
    assert positions[1]["orders_ahead"] == 0 and positions[1]["priority"] == "census"
    assert positions[2] == {"price": 100, "orders_ahead": 2, "size_ahead": 3.5, "priority": "observed"}
    rec = st.reconciliations[0]
    assert rec["census_orders"] == 4 and rec["missing"] == 1 and rec["extra"] == 0 and rec["mismatched"] == 0
    assert set(book.orders) == {1, 2, 4, 5}                    # re-anchored on the second census


def test_mbo_gap_invalidates_until_census():
    rows = [(0, -1, K.SNAPSHOT_BEGIN, 0, 0, 0.0), (0, -1, K.ORDER_ADD, 1, 100, 1.0, 1),
            (0, -1, K.SNAPSHOT_END, 0, 0, 0.0), (1, -1, K.GAP, 0, 0, 0.0), (2, -1, K.ORDER_ADD, 1, 99, 1.0, 2)]
    book, st = mbo.reconstruct([ev(rows)])
    assert st.gaps == 1 and st.adds == 0 and set(book.orders) == {1}


def test_validate_l2_snapshot_reference_and_mbo_vs_mbp():
    rows = snapshot(0, 1, [(10, 1.0), (9, 2.0)], [(11, 1.0)]) + [(1, 2, K.LEVEL_SET, 1, 10, 3.0)]
    rows += snapshot(2, 3, [(10, 3.0), (9, 2.0)], [(11, 1.0)])
    v = VA.l2_snapshot_check([ev(rows)])
    assert v["snapshots_checked"] == 1 and v["match_fraction"] == 1.0
    rows_bad = rows[:-4] + snapshot(2, 3, [(10, 4.0), (9, 2.0)], [(11, 1.0)])
    assert VA.l2_snapshot_check([ev(rows_bad)])["match_fraction"] < 1.0
    book = mbo.MBOBook()
    book.add(1, 1, 100, 1.0)
    book.add(2, 1, 100, 2.0)
    book.add(3, -1, 101, 1.0)
    ref = [{"ts": 0, "bids": [(100, 3.0)], "asks": [(101, 1.0)]}, {"ts": 1, "bids": [(100, 2.0)], "asks": [(101, 1.0)]}]
    out = VA.mbo_vs_mbp(lambda ts: book, ref)
    assert out["references"] == 2 and out["exact_fraction"] == 0.75 and out["price_presence_fraction"] == 1.0

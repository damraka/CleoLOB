"""M2: bounded historical counterfactual fills. All inputs are synthetic."""
from __future__ import annotations

from decimal import Decimal as D
import gzip

import pytest

from lob.capabilities import L2_CONTRACT, Capability, CapabilityContract, DataLevel
from lob.fill_bounds import ChildOrder, L2FillTracker, MBOFillTracker, classify, summarize
from lob.historical_execution import L2_WITH_TRADES, MBO_IDENTITY, run_l2, run_mbo, schedule
from lob.order_lifecycle import CensusOrder, LifecycleEvent as E, LifecycleSemantics

BUY = ChildOrder("h1", "BUY", D("100"), D("10"), 0, 100)


def test_classification_vocabulary():
    assert classify(10, 10, 10) == "GUARANTEED_FILL"
    assert classify(0, 0, 10) == "GUARANTEED_NON_FILL"
    assert classify(0, 4, 10) == "POSSIBLE_FILL"
    with pytest.raises(ValueError):
        classify(5, 4, 10)


def test_l2_requires_trades_and_refuses_identity_or_fifo_claims():
    with pytest.raises(ValueError):
        L2FillTracker(BUY, D(5), L2_CONTRACT)  # no trade prints declared
    fake = CapabilityContract(DataLevel.MBO, {Capability.ORDER_IDENTITY, Capability.AGGREGATE_DEPTH,
                                              Capability.TRADE_PRINTS}, "mbo")
    with pytest.raises(ValueError):
        L2FillTracker(BUY, D(5), fake)


def test_l2_prints_at_level_only_give_possible_fill_never_exact():
    t = L2FillTracker(BUY, D(20), L2_WITH_TRADES)
    t.on_print(D("100"), D(25), "SELL")      # consumes 20 ahead, 5 would reach us under FIFO
    t.on_level(D(0))
    r = t.result(L2_WITH_TRADES)
    assert r.classification == "POSSIBLE_FILL" and r.conservative_lower == 0
    assert (r.fifo_lower, r.fifo_upper, r.optimistic_upper) == (D(5), D(5), D(10))
    assert r.queue_evidence["exact_queue_position"].startswith("UNSUPPORTED")


def test_l2_print_through_price_is_a_conservative_fill():
    t = L2FillTracker(BUY, D(20), L2_WITH_TRADES)
    t.on_print(D("99.5"), D(4), "SELL")
    t.on_print(D("99"), D(30), "SELL")
    r = t.result(L2_WITH_TRADES)
    assert r.classification == "GUARANTEED_FILL" and r.conservative_lower == D(10)


def test_l2_same_side_prints_do_not_fill():
    t = L2FillTracker(BUY, D(0), L2_WITH_TRADES)
    t.on_print(D("100"), D(50), "BUY")
    r = t.result(L2_WITH_TRADES)
    assert r.classification == "GUARANTEED_NON_FILL" and r.optimistic_upper == 0


def test_l2_cancellation_attribution_brackets_fifo():
    t = L2FillTracker(BUY, D(20), L2_WITH_TRADES)
    t.on_level(D(35))   # +15 joins behind us
    t.on_level(D(22))   # 13 unexplained decrease: behind (lower path) or ahead (upper path)
    t.on_print(D("100"), D(12), "SELL")
    t.on_level(D(10))
    r = t.result(L2_WITH_TRADES)
    # upper path: ahead 20-13=7, print 12 -> 5 fill; lower path: ahead stays 20 -> 0 fill
    assert (r.conservative_lower, r.fifo_lower, r.fifo_upper, r.optimistic_upper) == (0, 0, D(5), D(10))
    assert r.conservative_lower <= r.fifo_lower <= r.fifo_upper <= r.optimistic_upper


def test_l2_forced_cancellation_from_ahead_when_level_shrinks_below_queue():
    t = L2FillTracker(BUY, D(20), L2_WITH_TRADES)
    t.on_level(D(6))    # nothing can be behind: at most 6 remain ahead on both paths
    t.on_print(D("100"), D(8), "SELL")
    r = t.result(L2_WITH_TRADES)
    assert r.fifo_lower == D(2) and r.fifo_upper == D(2)


def test_mbo_identity_fifo_point_and_priority_departure_evidence():
    ahead = {"a": 5, "b": 3}
    t = MBOFillTracker(ChildOrder("h", "BUY", 100, 10, 0, 100), ahead, MBO_IDENTITY)
    t.on_execute("a", "BUY", 100, 5)
    t.on_execute("late", "BUY", 100, 4)   # later identity executed while 'b' still rests ahead
    t.on_reduce("b", 3)
    t.on_execute("late2", "BUY", 100, 2)
    r = t.result(MBO_IDENTITY)
    assert (r.conservative_lower, r.fifo_lower, r.fifo_upper, r.optimistic_upper) == (0, 6, 6, 10)
    assert r.queue_evidence["later_executed_while_ahead_resting"] == 4
    assert r.queue_evidence["fifo_established_by_source"] is False
    assert r.fifo_classification == "POSSIBLE_FILL"


def test_mbo_through_execution_and_priority_loss():
    t = MBOFillTracker(ChildOrder("h", "SELL", 101, 4, 0, 100), {"x": 9}, MBO_IDENTITY)
    t.on_priority_loss("x")
    t.on_execute("y", "SELL", 101, 3)
    t.on_execute("z", "SELL", 102, 5)
    r = t.result(MBO_IDENTITY)
    assert r.classification == "GUARANTEED_FILL" and r.fifo_lower == 4


def test_summary_reports_shares_and_widths_not_point_estimates():
    t1 = L2FillTracker(BUY, D(0), L2_WITH_TRADES)
    t1.on_print(D("99"), D(20), "SELL")
    t2 = L2FillTracker(BUY, D(0), L2_WITH_TRADES)
    s = summarize([t1.result(L2_WITH_TRADES), t2.result(L2_WITH_TRADES)])
    assert s["classes"]["GUARANTEED_FILL"] == 1 and s["classes"]["GUARANTEED_NON_FILL"] == 1
    assert s["determinate_share_conservative"] == 1.0 and s["mean_conservative_width"] == 0.0


DESIGN = {"start_offset_s": 1, "interval_s": 2, "end_margin_s": 1, "sides": ["BUY", "SELL"], "sizes": ["5"],
          "lifetimes_s": [1.5], "price_rule": "join_best", "max_gap_s": 10, "max_staleness_s": 5}


def test_schedule_is_deterministic_and_inside_the_file():
    grid = schedule(DESIGN, 0, 10_000_000)
    assert grid[0] == (1_000_000, "BUY", "5", 1_500_000) and grid[-1][0] <= 9_000_000
    assert grid == schedule(DESIGN, 0, 10_000_000)


def _write(path, header, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        stream.write(header + "\n" + "\n".join(rows) + "\n")


def test_run_l2_streams_synthetic_tardis_files(tmp_path):
    updates, trades = tmp_path / "u.csv.gz", tmp_path / "t.csv.gz"
    header = "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"
    rows = ["x,S,0,0,true,bid,100,10", "x,S,0,0,true,ask,101,10",
            "x,S,1,1200000,false,bid,100,4",     # 6 left the bid: 6 printed + 0 cancel
            "x,S,2,2000000,false,ask,101,12",
            "x,S,3,3400000,false,ask,101,2",     # ask shrinks by 10 with 10 printed
            "x,S,4,5000000,false,bid,100,4", "x,S,5,6000000,false,bid,100,4"]
    _write(updates, header, rows)
    _write(trades, "exchange,symbol,timestamp,local_timestamp,id,side,price,amount",
           ["x,S,1,1100000,t1,sell,100,6", "x,S,3,3300000,t2,buy,101,10", "x,S,3,3350000,t3,buy,102,6"])
    result = run_l2(updates, trades, DESIGN, dataset_id="synthetic")
    orders = result["orders"]
    assert [o["side"] for o in orders[:2]] == ["BUY", "SELL"]
    buy = orders[0]
    assert buy["price"] == "100" and buy["classification"] == "POSSIBLE_FILL"  # print after submission at 1.0s
    assert buy["fifo_upper"] == "0" and buy["optimistic_upper"] == "5"
    later_sell = [o for o in orders if o["side"] == "SELL" and o["submit_ts"] == 3_000_000][0]
    assert later_sell["classification"] == "GUARANTEED_FILL"  # 6 printed through 101 at 3.35s
    assert later_sell["conservative_lower"] == "5"
    assert result["summary"]["orders"] == len(orders)


def test_run_mbo_observed_order_coverage():
    census = (CensusOrder("a", "BUY", 100, 5), CensusOrder("s", "SELL", 105, 5))
    s = 1_000_000_000
    events = [E(0, "SNAPSHOT", exchange_ts_ns=0, orders=census),
              E(1, "ADD", exchange_ts_ns=s, order_id="me", side="BUY", price=100, quantity=3),
              E(2, "EXECUTE", exchange_ts_ns=2 * s, order_id="a", quantity=5),
              E(3, "EXECUTE", exchange_ts_ns=3 * s, order_id="me", quantity=2),
              E(4, "ADD", exchange_ts_ns=9 * s, order_id="z", side="BUY", price=99, quantity=1)]
    design = dict(DESIGN, start_offset_s=0.5, interval_s=100, end_margin_s=0, lifetimes_s=[5], sizes=["2"])
    result = run_mbo(events, design, semantics=LifecycleSemantics(), dataset_id="synthetic",
                     observed_lifetime_s=5)
    coverage = result["observed_order_validation"]
    assert coverage["evaluable"] == 1 and coverage["coverage_fraction"] == 1.0
    assert coverage["fifo_point_agreement"] == 1.0
    hypo = result["orders"]
    assert hypo[0]["fifo_lower"] == 2 and hypo[0]["classification"] == "POSSIBLE_FILL"


def test_observed_order_with_zero_length_window_is_excluded_not_crashing():
    census = (CensusOrder("a", "BUY", 100, 5), CensusOrder("s", "SELL", 105, 5))
    s = 1_000_000_000
    events = [E(0, "SNAPSHOT", exchange_ts_ns=0, orders=census),
              E(1, "ADD", exchange_ts_ns=s, order_id="flash", side="BUY", price=100, quantity=3),
              E(2, "CANCEL", exchange_ts_ns=s, order_id="flash"),
              E(3, "ADD", exchange_ts_ns=9 * s, order_id="z", side="BUY", price=99, quantity=1)]
    design = dict(DESIGN, start_offset_s=0.5, interval_s=100, end_margin_s=0, lifetimes_s=[5], sizes=["2"])
    result = run_mbo(events, design, semantics=LifecycleSemantics(), dataset_id="synthetic", observed_lifetime_s=5)
    assert result["observed_order_validation"]["zero_length_windows_excluded"] == 1

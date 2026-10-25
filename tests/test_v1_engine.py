"""v1 matching engine v2 and exchange mechanics (M30-M39)."""
from fractions import Fraction

from hypothesis import given, settings, strategies as st
import pytest

from lob.v1.engines import accounting as AC, agent as AG, invariants as INV, journal as J, latency as LT
from lob.v1.engines.matching import BUY, SELL, MatchingEngine, Order
from lob.v1.engines.reference import ReferenceEngine
from lob.v1.engines.rules import Rules, VENUES


def lim(oid, side, price, qty, owner="a", **kw):
    return Order(oid=oid, owner=owner, side=side, qty=qty, kind="limit", price=price, **kw)


def mkt(oid, side, qty, owner="t"):
    return Order(oid=oid, owner=owner, side=side, qty=qty, kind="market")


def fills(reps, liquidity="taker"):
    return [(r.price, r.qty) for r in reps if r.kind == "fill" and r.liquidity == liquidity]


# ----------------------------------------------------------------------------------------- M30 / M34
def test_matching_price_time_priority_and_queue_advancement():
    e = MatchingEngine()
    e.submit(lim(1, SELL, 101, 5, "a"))
    e.submit(lim(2, SELL, 101, 3, "b"))
    e.submit(lim(3, SELL, 102, 4, "c"))
    assert e.queue_ahead(2) == {"orders_ahead": 1, "lots_ahead": 5, "price": 101, "level_total": 8}
    reps = e.submit(mkt(10, BUY, 7))
    assert fills(reps) == [(101, 5), (101, 2)]
    assert [(r.oid, r.remaining) for r in reps if r.kind == "fill" and r.liquidity == "maker"] == [(1, 0), (2, 1)]
    assert e.queue_ahead(2)["orders_ahead"] == 0 and e.depth(SELL) == [(101, 1), (102, 4)]
    INV.check(e)


def test_queue_cancellation_ahead_advances_and_additions_behind_do_not():
    e = MatchingEngine()
    for oid, q in ((1, 5), (2, 5), (3, 5)):
        e.submit(lim(oid, BUY, 100, q, f"o{oid}"))
    e.submit(lim(4, BUY, 100, 9, "late"))
    assert e.queue_ahead(3)["lots_ahead"] == 10
    e.cancel(1)
    assert e.queue_ahead(3)["lots_ahead"] == 5 and e.queue_ahead(4)["lots_ahead"] == 10


# ----------------------------------------------------------------------------------------- M32
def test_orders_ioc_fok_market_post_only_and_partial_fills():
    e = MatchingEngine()
    e.submit(lim(1, SELL, 101, 5))
    r = e.submit(lim(2, BUY, 101, 8, "b", tif="IOC"))
    assert fills(r) == [(101, 5)] and r[-1].kind == "expire" and r[-1].qty == 3
    e.submit(lim(3, SELL, 102, 2))
    r = e.submit(lim(4, BUY, 103, 3, "b", tif="FOK"))
    assert r[-1].kind == "expire" and r[-1].reason == "fok_not_fillable" and e.level_qty(SELL, 102) == 2
    r = e.submit(lim(5, BUY, 102, 1, "b", post_only=True))
    assert r[-1].reason == "post_only_would_cross" and e.orders[5].status == "rejected"
    e2 = MatchingEngine(Rules(post_only="slide"))
    e2.submit(lim(1, SELL, 101, 5))
    r = e2.submit(lim(2, BUY, 105, 1, "b", post_only=True))
    assert e2.best(BUY) == 100 and r[-1].reason == "post_only_slide"
    r = e.submit(mkt(6, BUY, 10))
    assert fills(r) == [(102, 2)] and r[-1].reason == "market_unfilled"
    assert e.submit(lim(7, BUY, None, 1))[0].reason == "limit_without_price"
    assert e.submit(lim(1, BUY, 99, 1))[0].reason == "duplicate_order_id"
    assert MatchingEngine(Rules(lot=5)).submit(lim(9, BUY, 99, 7))[0].reason == "invalid_quantity"


def test_orders_cancel_and_replace_priority_rules():
    e = MatchingEngine()
    e.submit(lim(1, BUY, 100, 5, "a"))
    e.submit(lim(2, BUY, 100, 5, "b"))
    r = e.replace(1, qty=3)
    assert r[0].reason == "kept_priority" and e.queue_ahead(2)["lots_ahead"] == 3
    r = e.replace(1, qty=4)
    assert r[0].reason == "lost_priority" and e.queue_ahead(1)["orders_ahead"] == 1
    e.submit(lim(3, SELL, 101, 2, "c"))
    r = e.replace(2, price=101)                      # re-priced into the ask: trades as a taker
    assert fills(r) == [(101, 2)] and e.best(BUY) == 101
    assert e.cancel(3)[0].reason == "too_late" and e.stats.cancel_too_late == 1
    assert e.cancel(999)[0].reason == "unknown"
    assert e.cancel(1, owner="zz")[0].reason == "not_owner"
    INV.check(e)


def test_orders_self_trade_prevention_modes():
    for mode, expect_fill, maker_status in (("cancel_taker", [], "live"), ("cancel_maker", [(101, 3)], "cancelled"),
                                            ("allow", [(101, 2)], "filled")):
        e = MatchingEngine(Rules(self_trade=mode))
        e.submit(lim(1, SELL, 101, 2, "me"))
        e.submit(lim(2, SELL, 101, 3, "other"))
        r = e.submit(lim(3, BUY, 101, 5, "me", tif="IOC"))
        got = fills(r)
        if mode == "allow":
            assert got == [(101, 2), (101, 3)]
        else:
            assert got == expect_fill
        assert e.orders[1].status == maker_status


def test_rules_pro_rata_allocation():
    e = MatchingEngine(VENUES["pro_rata_futures"])
    e.submit(lim(1, SELL, 101, 60, "a"))
    e.submit(lim(2, SELL, 101, 30, "b"))
    e.submit(lim(3, SELL, 101, 10, "c"))
    r = e.submit(mkt(4, BUY, 50))
    alloc = {x.oid: x.qty for x in r if x.kind == "fill" and x.liquidity == "maker"}
    assert alloc == {1: 30, 2: 15, 3: 5} and sum(alloc.values()) == 50
    INV.check(e)
    with pytest.raises(ValueError):
        Rules(allocation="lottery")


# ----------------------------------------------------------------------------------------- M33
def test_accounting_fees_positions_and_conservation():
    e = MatchingEngine()
    acc = AC.Accounting(AC.FeeSchedule(Fraction(-1), Fraction(5)), tick=0.1, lot=0.001)
    for o in (lim(1, SELL, 1000, 10, "mm"), lim(2, SELL, 1001, 10, "mm"), mkt(3, BUY, 15, "taker")):
        acc.register(o.oid, o.side)
        acc.on_reports(e.submit(o))
    t, m = acc.accounts["taker"], acc.accounts["mm"]
    assert t.position == 15 and m.position == -15 and t.taker_volume == 15 and m.maker_volume == 15
    notional = 1000 * 10 + 1001 * 5
    assert t.fees == Fraction(notional * 5, 10_000) and m.fees == Fraction(-notional, 10_000)
    # cash + mark is conserved across the two counterparties up to fees
    assert t.mark(1001) + m.mark(1001) == -(t.fees + m.fees)
    acc.register(4, SELL)
    acc.on_reports(e.submit(lim(4, SELL, 1002, 15, "taker")))
    acc.register(5, BUY)
    acc.on_reports(e.submit(lim(5, BUY, 1002, 20, "mm2")))   # takes mm 5 @1001, then the taker 15 @1002
    assert t.position == 0 and t.realized == Fraction(1002 * 15) - Fraction(notional)
    assert acc.summary(1002)["taker"]["position_lots"] == 0


# ----------------------------------------------------------------------------------------- M35
def test_latency_cancel_race_and_stale_view():
    e = MatchingEngine()
    gw = LT.Gateway(e, entry=LT.LatencyModel("constant", 1_000), md=LT.LatencyModel("constant", 5_000))
    gw.send(0, "submit", "mm", oid=1, side=SELL, qty=5, kind="limit", price=101)
    gw.run_until(2_000)
    gw.send(3_000, "submit", "t", oid=2, side=BUY, qty=5, kind="market")      # arrives at 4_000
    gw.send(3_500, "cancel", "mm", oid=1)                                      # arrives at 4_500: too late
    gw.run_until(10_000)
    assert gw.races == 1
    rep = gw.reports("mm", 10_000)
    assert [r.kind for r in rep] == ["ack", "fill", "reject"] and rep[-1].reason == "too_late"
    assert gw.view(5_500) is None                  # the first book (published at 1_000) is visible from 6_000
    assert gw.view(6_500)["ask"] == 101            # the fill at 4_000 is not visible before 9_000
    assert gw.view(9_500)["ask"] is None
    lm = LT.LatencyModel("lognormal", 100, 1_000, 0.5)
    import numpy as np
    draws = [lm.draw(np.random.default_rng(i)) for i in range(50)]
    assert min(draws) > 100


# ----------------------------------------------------------------------------------------- M36
def test_inventory_outstanding_and_self_impact_trace():
    e = MatchingEngine()
    for oid, p in ((1, 101), (2, 102), (3, 103)):
        e.submit(lim(oid, SELL, p, 2, "bg"))
    ag = AG.AgentState("me")
    AG.submit_tracked(e, ag, lim(10, BUY, 99, 4, "me"), 0)
    assert ag.outstanding == 4 and ag.inventory == 0
    AG.submit_tracked(e, ag, mkt(11, BUY, 5, "me"), 1)
    assert ag.inventory == 5 and ag.impact_trace[0]["touch_before"] == (99, 101)
    assert ag.impact_trace[0]["touch_after"] == (99, 103)
    assert ag.shortfall_ticks(100.0) == (101 - 100) * 2 + (102 - 100) * 2 + (103 - 100) * 1


# ----------------------------------------------------------------------------------------- M37
def test_journal_snapshot_and_recovery():
    e = MatchingEngine()
    j = J.attach(e)
    e.submit(lim(1, SELL, 101, 5))
    e.submit(lim(2, BUY, 99, 5, "b"))
    snap = J.snapshot(e)
    e.submit(mkt(3, BUY, 2))
    e.replace(2, qty=3)
    e.cancel(1)
    rebuilt = J.recover(snap, j)
    assert rebuilt.state() == e.state()
    snap["state"]["seq"] += 1
    with pytest.raises(ValueError):
        J.recover(snap, j)


# ----------------------------------------------------------------------------------------- M38 / M39
command = st.one_of(
    st.tuples(st.just("lim"), st.sampled_from([BUY, SELL]), st.integers(95, 105), st.integers(1, 6),
              st.sampled_from(["a", "b", "c"]), st.sampled_from(["GTC", "GTC", "IOC", "FOK"])),
    st.tuples(st.just("mkt"), st.sampled_from([BUY, SELL]), st.integers(1, 12), st.sampled_from(["a", "b", "c"])),
    st.tuples(st.just("cxl"), st.integers(1, 60)),
)


def _run(cmds, rules):
    e, ref = MatchingEngine(rules), ReferenceEngine(rules)
    all_reports = []
    for i, c in enumerate(cmds, 1):
        if c[0] == "lim":
            o = lim(i, c[1], c[2], c[3], c[4], tif=c[5])
            r1, r2 = e.submit(o), ref.submit(o)
        elif c[0] == "mkt":
            o = mkt(i, c[1], c[2], c[3])
            r1, r2 = e.submit(o), ref.submit(o)
        else:
            r1, r2 = e.cancel(c[1]), ref.cancel(c[1])
        INV.check(e)
        INV.check_reports(r1)
        all_reports.append((r1, r2))
        assert e.depth(BUY, 50) == ref.depth(BUY, 50) and e.depth(SELL, 50) == ref.depth(SELL, 50)
    return all_reports


@settings(max_examples=150, deadline=None)
@given(st.lists(command, min_size=1, max_size=60))
def test_differential_engine_equals_reference_price_time(cmds):
    for r1, r2 in _run(cmds, Rules(self_trade="cancel_taker")):
        assert [(r.kind, r.oid, r.price, r.qty, r.liquidity) for r in r1] == \
               [(r.kind, r.oid, r.price, r.qty, r.liquidity) for r in r2]


@settings(max_examples=100, deadline=None)
@given(st.lists(command, min_size=1, max_size=60), st.sampled_from(["cancel_maker", "allow"]))
def test_differential_other_self_trade_modes_agree_on_fills(cmds, mode):
    for r1, r2 in _run(cmds, Rules(self_trade=mode)):
        key = lambda rs: sorted((r.oid, r.counter, r.price, r.qty) for r in rs if r.kind == "fill")  # noqa: E731
        assert key(r1) == key(r2)


@settings(max_examples=100, deadline=None)
@given(st.lists(command, min_size=1, max_size=80))
def test_fuzz_pro_rata_invariants(cmds):
    e = MatchingEngine(VENUES["pro_rata_futures"])
    for i, c in enumerate(cmds, 1):
        if c[0] == "lim":
            r = e.submit(lim(i, c[1], c[2], c[3], c[4], tif=c[5]))
        elif c[0] == "mkt":
            r = e.submit(mkt(i, c[1], c[2], c[3]))
        else:
            r = e.cancel(c[1])
        INV.check(e)
        INV.check_reports(r)
        taker = sum(x.qty for x in r if x.kind == "fill" and x.liquidity == "taker")
        maker = sum(x.qty for x in r if x.kind == "fill" and x.liquidity == "maker")
        assert taker == maker

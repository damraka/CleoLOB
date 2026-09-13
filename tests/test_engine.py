"""Matching engine, latency queue and background-flow tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from conftest import assert_book_consistent, limit, market
from lob.engine import ExchangeSimulator, Side, SimConfig


# ------------------------------------------------------------- matching
def test_price_time_priority_fifo_within_level(book):
    limit(book, 1, Side.BUY, 100, 10, owner="A", t=1.0)
    limit(book, 2, Side.BUY, 100, 10, owner="B", t=2.0)
    trades = market(book, 3, Side.SELL, 15, t=3.0)
    assert [(t.maker_owner, t.qty) for t in trades] == [("A", 10), ("B", 5)]
    assert book.bid_vol[100] == 5 and book.orders[2].remaining == 5
    assert 1 not in book.orders                      # fully filled order purged
    assert_book_consistent(book)


def test_price_priority_over_time_priority(book):
    limit(book, 1, Side.BUY, 100, 10, owner="early", t=1.0)
    limit(book, 2, Side.BUY, 101, 10, owner="late-but-better", t=2.0)
    trades = market(book, 3, Side.SELL, 10, t=3.0)
    assert trades[0].maker_owner == "late-but-better" and trades[0].price == 101


def test_market_order_walks_levels_and_vwap_matches_walk_cost(book):
    for i, (px, q) in enumerate([(100, 30), (99, 30), (98, 30)], start=1):
        limit(book, i, Side.BUY, px, q)
    est_vwap, fillable = book.walk_cost(Side.SELL, 75)
    trades = market(book, 9, Side.SELL, 75)
    filled = sum(t.qty for t in trades)
    vwap = sum(t.price * t.qty for t in trades) / filled
    assert filled == fillable == 75
    assert vwap == pytest.approx(est_vwap)
    assert book.best_bid() == 98 and book.bid_vol[98] == 15
    assert_book_consistent(book)


def test_market_order_leftover_is_cancelled_not_rested(book):
    limit(book, 1, Side.BUY, 100, 10)
    trades = market(book, 2, Side.SELL, 25)
    assert sum(t.qty for t in trades) == 10
    assert book.best_bid() is None and not book.asks and 2 not in book.orders


def test_crossing_limit_fills_then_rests_remainder(book):
    limit(book, 1, Side.SELL, 101, 10)
    limit(book, 2, Side.SELL, 102, 10)
    trades = limit(book, 3, Side.BUY, 101, 25)            # crosses only the 101 level
    assert sum(t.qty for t in trades) == 10 and trades[0].price == 101
    assert book.best_bid() == 101 and book.bid_vol[101] == 15
    assert book.best_ask() == 102
    assert_book_consistent(book)


def test_non_crossing_limit_never_trades(book):
    limit(book, 1, Side.SELL, 105, 10)
    assert limit(book, 2, Side.BUY, 104, 10) == []
    assert book.spread() == 1


def test_trade_records_taker_and_maker(book):
    limit(book, 1, Side.SELL, 101, 5, owner="maker")
    (t,) = market(book, 2, Side.BUY, 5, owner="taker")
    assert (t.taker_owner, t.maker_owner, t.taker_side) == ("taker", "maker", Side.BUY)
    assert book.trades[-1] is t


# ------------------------------------------------------------- cancels
def test_cancel_removes_volume_and_cleans_empty_level(book):
    limit(book, 1, Side.BUY, 100, 10)
    limit(book, 2, Side.BUY, 100, 5)
    assert book.cancel(1) is True
    assert book.bid_vol[100] == 5 and 1 not in book.orders
    assert book.cancel(2) is True
    assert book.best_bid() is None and 100 not in book.bids and 100 not in book.bid_vol
    assert_book_consistent(book)


def test_cancel_of_filled_or_unknown_order_is_noop(book):
    limit(book, 1, Side.BUY, 100, 10)
    market(book, 2, Side.SELL, 10)
    assert book.cancel(1) is False
    assert book.cancel(999) is False


# ------------------------------------------------------------- book analytics
def test_microprice_and_imbalance(book):
    limit(book, 1, Side.BUY, 99, 30)
    limit(book, 2, Side.SELL, 101, 10)
    # heavier bid pushes microprice toward the ask
    assert book.mid() == 100.0
    assert book.microprice() == pytest.approx((99 * 10 + 101 * 30) / 40)
    assert book.imbalance(5) == pytest.approx((30 - 10) / 40)


def test_depth_is_best_first(book):
    for i, px in enumerate([98, 100, 99], start=1):
        limit(book, i, Side.BUY, px, 1)
    for i, px in enumerate([103, 101, 102], start=4):
        limit(book, i, Side.SELL, px, 1)
    bids, asks = book.depth(3)
    assert [p for p, _ in bids] == [100, 99, 98]
    assert [p for p, _ in asks] == [101, 102, 103]


# ------------------------------------------------------------- latency
def test_strategy_order_arrives_only_after_latency(warm_sim):
    sim = warm_sim
    cfg = sim.cfg
    t_submit = sim.t
    oid = sim.submit(Side.BUY, 1, "T", price_ticks=sim.book.best_bid() - 5)
    assert oid not in sim.book.orders                     # not yet at the exchange
    sim.step(cfg.latency_base * 0.5)
    assert oid not in sim.book.orders                     # still in flight (< base latency)
    sim.step(1.0)                                         # far beyond any jitter
    assert oid in sim.book.orders
    assert sim.book.orders[oid].ts_active >= t_submit + cfg.latency_base


def test_cancel_travels_with_latency_and_can_lose_race(warm_sim):
    sim = warm_sim
    px = sim.book.best_ask() + 500  # isolate cancel transport from stochastic fills
    oid = sim.submit(Side.SELL, 5, "T", price_ticks=px)   # join the ask queue
    sim.step(1.0)
    assert oid in sim.book.orders
    sim.cancel(oid)
    assert oid in sim.book.orders                         # cancel not yet effective
    sim.step(1.0)
    assert oid not in sim.book.orders                     # either cancelled or filled — gone


def test_events_are_processed_in_time_order():
    sim = ExchangeSimulator(SimConfig(seed=3))
    sim.step(2.0)
    times = [tr.time for tr in sim.book.trades]
    assert times == sorted(times)
    assert all(0.0 <= t <= 2.0 for t in times)


# ------------------------------------------------------------- flow & resilience
def test_background_flow_keeps_book_two_sided_and_consistent():
    sim = ExchangeSimulator(SimConfig(seed=11))
    for _ in range(300):
        sim.step(0.1)
        assert_book_consistent(sim.book)
        assert sim.book.best_bid() is not None and sim.book.best_ask() is not None
    assert len(sim.book.trades) > 100
    assert sim.book.spread() <= 3


def test_resilience_refills_after_liquidity_shock():
    # Freeze unrelated arrivals/cancels so this isolates resilience. Tracking fixed
    # old price levels in a moving market is not a deterministic recovery invariant.
    sim = ExchangeSimulator(SimConfig(seed=7, limit_rate=0, market_rate=0, cancel_rate=0))
    levels = [p for p, _ in sim.book.depth(3)[0]]                # the three best bid prices
    def vol_at():
        return sum(sim.book.bid_vol.get(p, 0) for p in levels)
    depth_before = vol_at()
    market(sim.book, 10**9, Side.SELL, 4000, owner="SHOCK", t=sim.t)   # sweep the bid side
    depth_after = vol_at()
    assert depth_after < 0.2 * depth_before                      # those levels were consumed
    for _ in range(60):
        sim.step(0.1)
    assert vol_at() > 0.3 * depth_before                          # …and have refilled


def test_higher_resilience_refills_faster():
    def top3_after_shock(res: float) -> int:
        sim = ExchangeSimulator(SimConfig(seed=5, resilience=res))
        for _ in range(50):
            sim.step(0.1)
        market(sim.book, 10**9, Side.SELL, 4000, owner="SHOCK", t=sim.t)
        for _ in range(30):
            sim.step(0.1)
        return sum(v for _, v in sim.book.depth(3)[0])
    assert top3_after_shock(2.0) > top3_after_shock(0.0)


# ------------------------------------------------------------- reproducibility
def test_same_seed_same_market_different_seed_different_market():
    def trade_signature(seed: int):
        sim = ExchangeSimulator(SimConfig(seed=seed))
        for _ in range(100):
            sim.step(0.1)
        return [(round(t.time, 6), t.price, t.qty) for t in sim.book.trades]
    assert trade_signature(42) == trade_signature(42)
    assert trade_signature(42) != trade_signature(43)


def test_fills_for_only_returns_own_trades(warm_sim):
    sim = warm_sim
    cursor = len(sim.book.trades)
    sim.submit(Side.SELL, 50, "ME")
    sim.step(1.0)
    mine, new_cursor = sim.fills_for("ME", cursor)
    assert new_cursor == len(sim.book.trades)
    assert mine and all("ME" in (t.taker_owner, t.maker_owner) for t in mine)
    assert sum(t.qty for t in mine) == 50

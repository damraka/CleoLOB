"""Order lifecycle, atomic matching, persistent-clock and randomized invariants."""
import math
import random

import pytest

from lob.engine import (ExchangeSimulator, Order, OrderBook, OrderStatus, OrderType,
                        Side, SimConfig, TimeInForce)


def limit(oid, side, qty=10, price=100, **kwargs):
    return Order(oid, side, qty, OrderType.LIMIT, str(oid), 0.0, price=price, **kwargs)


def quiet(**overrides):
    values = dict(limit_rate=0, market_rate=0, cancel_rate=0, resilience=0,
                  latency_base=0, latency_jitter=0)
    return ExchangeSimulator(SimConfig(**(values | overrides)))


@pytest.mark.parametrize("kwargs", [dict(qty=0), dict(qty=-1), dict(qty=1.5), dict(qty=True),
                                   dict(price=1.1), dict(price=0), dict(ts_submit=-1),
                                   dict(ts_submit=math.nan), dict(owner=""), dict(side="BUY")])
def test_invalid_orders_are_rejected_before_admission(kwargs):
    values = dict(order_id=1, side=Side.BUY, qty=10, otype=OrderType.LIMIT, owner="x", ts_submit=0, price=100)
    with pytest.raises(ValueError):
        Order(**(values | kwargs))


def test_duplicate_id_rejected_after_fill_and_no_mutation_on_rejection():
    book = OrderBook()
    book.process(limit(1, Side.SELL), 0)
    book.process(Order(2, Side.BUY, 10, OrderType.MARKET, "buyer", 0), 0)
    before = book.snapshot()
    with pytest.raises(ValueError, match="already"):
        book.process(limit(1, Side.SELL), 0)
    assert book.snapshot() == before and len(book.trades) == 1


def test_fok_is_all_or_none_and_respects_limit_price():
    book = OrderBook()
    book.process(limit(1, Side.SELL, 5, 100), 0)
    book.process(limit(2, Side.SELL, 20, 101), 0)
    failed = limit(3, Side.BUY, 6, 100, time_in_force=TimeInForce.FOK)
    assert book.process(failed, 0) == []
    assert failed.status is OrderStatus.CANCELLED and book.ask_vol[100] == 5
    success = limit(4, Side.BUY, 6, 101, time_in_force=TimeInForce.FOK)
    trades = book.process(success, 0)
    assert [(t.price, t.qty) for t in trades] == [(100, 5), (101, 1)]
    assert success.status is OrderStatus.FILLED
    book.assert_invariants()


def test_ioc_remainder_is_terminal_and_post_only_cross_has_no_fill():
    book = OrderBook()
    book.process(limit(1, Side.SELL, 5), 0)
    rejected = limit(2, Side.BUY, post_only=True)
    assert book.process(rejected, 0) == [] and rejected.status is OrderStatus.REJECTED
    ioc = limit(3, Side.BUY, time_in_force=TimeInForce.IOC)
    assert sum(t.qty for t in book.process(ioc, 0)) == 5
    assert ioc.status is OrderStatus.CANCELLED and ioc.remaining == 5 and ioc.filled_qty == 5
    assert not book.orders


def test_lot_size_tick_validation_and_zero_depth():
    book = OrderBook(tick_size=0.25, lot_size=5)
    with pytest.raises(ValueError, match="lot_size"):
        book.process(limit(1, Side.BUY, qty=3), 0)
    book.process(limit(2, Side.BUY, qty=10), 0)
    assert book.depth(0) == ([], []) and book.queue_position(2) == (0, 0)
    book.assert_invariants()


def test_modify_reduction_keeps_priority_increase_loses_it_and_overfill_fails_atomically():
    book = OrderBook()
    first, second = limit(1, Side.BUY), limit(2, Side.BUY)
    book.process(first, 0)
    book.process(second, 0)
    book.modify(1, 5, now=1)
    assert book.queue_position(1) == (0, 0) and book.queue_position(2) == (1, 5)
    book.modify(1, 15, now=2)
    assert book.queue_position(1) == (1, 10)
    book.process(Order(3, Side.SELL, 12, OrderType.MARKET, "taker", 3), 3)
    assert first.filled_qty == 2 and first.remaining == 13
    before = book.snapshot()
    with pytest.raises(ValueError, match="already filled"):
        book.modify(1, 1, now=3)
    assert book.snapshot() == before
    book.modify(1, 2, now=3)
    assert first.status is OrderStatus.CANCELLED and not book.orders


def test_book_and_order_timestamps_cannot_reverse():
    book = OrderBook()
    order = limit(1, Side.BUY)
    book.process(order, 2)
    with pytest.raises(ValueError, match="backwards"):
        book.cancel(1, 1)
    with pytest.raises(ValueError, match="backwards"):
        order.transition(OrderStatus.CANCEL_PENDING, 1)
    with pytest.raises(ValueError):
        order.transition(OrderStatus.CREATED, 3)


def test_gtd_expires_before_same_time_market_and_expired_on_late_arrival():
    sim = quiet()
    oid = sim.submit(Side.BUY, 5, "owner", price_ticks=10000, time_in_force=TimeInForce.GTD, expires_at=1)
    sim.step(0)
    sim.step(1)
    assert sim.orders[oid].status is OrderStatus.EXPIRED and oid not in sim.book.orders
    late = quiet(latency_base=2)
    oid = late.submit(Side.BUY, 5, "owner", price_ticks=10000, time_in_force=TimeInForce.GTD, expires_at=1)
    late.step(2)
    assert late.orders[oid].status is OrderStatus.EXPIRED
    assert not late.fills_for("owner", 0)[0]


def test_cancel_races_and_same_timestamp_fifo_are_deterministic():
    sim = quiet(latency_base=0.1)
    oid = sim.submit(Side.SELL, 5, "maker", price_ticks=10000)
    sim.step(0.1)
    sim.submit(Side.BUY, 5, "taker")  # enqueued first at 0.2
    sim.cancel(oid)
    sim.step(0.1)
    assert sim.orders[oid].status is OrderStatus.FILLED
    assert any(e["kind"] == "cancel" and not e["accepted"] for e in sim.events)
    sim = quiet(latency_base=0.1)
    oid = sim.submit(Side.SELL, 5, "maker", price_ticks=10000)
    sim.step(0.1)
    sim.cancel(oid)  # enqueued first at 0.2
    sim.submit(Side.BUY, 5, "taker")
    sim.step(0.1)
    assert sim.orders[oid].status is OrderStatus.CANCELLED
    assert not sim.fills_for("maker", 0)[0]


def test_cancel_before_original_arrival_is_rejected_without_killing_later_order():
    sim = quiet()
    draws = iter([1.0, 0.1])
    sim._latency = lambda: next(draws)
    oid = sim.submit(Side.BUY, 5, "owner", price_ticks=9990)
    sim.cancel(oid)
    sim.step(0.2)
    assert sim.orders[oid].status is OrderStatus.IN_FLIGHT
    sim.step(1)
    assert sim.orders[oid].status is OrderStatus.RESTING


def test_modify_after_execution_and_replace_unknown_are_noops():
    sim = quiet(latency_base=0.1)
    oid = sim.submit(Side.SELL, 5, "maker", price_ticks=10000)
    sim.step(0.1)
    sim.submit(Side.BUY, 5, "taker")
    sim.modify(oid, 20)
    replacement = sim.cancel_replace(oid, 10, 10002)
    sim.step(0.1)
    assert sim.orders[oid].status is OrderStatus.FILLED
    assert sim.orders[replacement].status is OrderStatus.REJECTED
    assert not sim.orders[replacement].filled_qty


def test_background_path_does_not_depend_on_step_partition():
    whole, split = ExchangeSimulator(SimConfig(seed=381)), ExchangeSimulator(SimConfig(seed=381))
    whole.step(10)
    for _ in range(80):
        split.step(0.125)
    assert whole.book.snapshot() == split.book.snapshot()
    assert whole.book.trades == split.book.trades
    assert whole.events == split.events
    assert whole.seed_manifest == split.seed_manifest


def test_no_effect_strategy_messages_do_not_change_background_flow():
    base, agent = ExchangeSimulator(SimConfig(seed=72)), ExchangeSimulator(SimConfig(seed=72))
    agent.submit(Side.BUY, 5, "strategy", price_ticks=1)
    base.step(4)
    agent.step(4)
    assert base.book.trades == agent.book.trades
    assert base.book.depth(5) == agent.book.depth(5)


@pytest.mark.parametrize("kwargs", [dict(tick_size=0), dict(lot_size=0), dict(market_rate=-1),
                                   dict(offset_p=1.1), dict(latency_base=math.nan), dict(seed=-1),
                                   dict(target_level_vol=3, lot_size=5)])
def test_invalid_simulation_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        SimConfig(**kwargs)


def test_hard_event_budget_stops_without_discarding_processed_events():
    sim = ExchangeSimulator(SimConfig(max_events=5))
    with pytest.raises(RuntimeError, match="max_events"):
        sim.step(20)
    assert sim.event_count == len(sim.events) == 5
    assert sim.t < 20
    sim.book.assert_invariants()


def test_randomized_order_sequences_preserve_book_and_fill_invariants():
    for seed in range(6):
        rng = random.Random(seed)
        book = OrderBook(check_invariants=True)
        for oid in range(1, 301):
            now = float(oid)
            if book.orders and rng.random() < 0.3:
                target = rng.choice(list(book.orders))
                book.cancel(target, now)
            else:
                side = rng.choice(list(Side))
                qty = rng.randint(1, 40)
                order = Order(oid, side, qty, OrderType.LIMIT, f"owner-{oid}", now,
                              price=rng.randint(95, 105), time_in_force=rng.choice([TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK]))
                trades = book.process(order, now)
                assert sum(t.qty for t in trades) <= qty
                assert order.filled_qty <= order.qty
            book.assert_invariants()

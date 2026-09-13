"""Shared fixtures. Run with:  pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lob.engine import ExchangeSimulator, Order, OrderBook, OrderType, Side, SimConfig  # noqa: E402


@pytest.fixture
def book() -> OrderBook:
    return OrderBook(tick_size=0.01)


@pytest.fixture
def warm_sim() -> ExchangeSimulator:
    """A simulator with 5 s of background flow already processed."""
    sim = ExchangeSimulator(SimConfig(seed=7))
    for _ in range(50):
        sim.step(0.1)
    return sim


def limit(book: OrderBook, oid: int, side: Side, price: int, qty: int,
          owner: str = "T", t: float = 0.0):
    """Submit a limit order straight into a book (no latency) and return its trades."""
    return book.process(Order(oid, side, qty, OrderType.LIMIT, owner, t, price=price), t)


def market(book: OrderBook, oid: int, side: Side, qty: int, owner: str = "T", t: float = 0.0):
    return book.process(Order(oid, side, qty, OrderType.MARKET, owner, t), t)


def assert_book_consistent(book: OrderBook) -> None:
    """Structural invariants that must hold after any sequence of events."""
    bb, ba = book.best_bid(), book.best_ask()
    if bb is not None and ba is not None:
        assert bb < ba, f"crossed book: bid {bb} >= ask {ba}"
    assert book.bid_prices == sorted(book.bid_prices)
    assert book.ask_prices == sorted(book.ask_prices)
    assert set(book.bid_prices) == set(book.bids) == set(book.bid_vol)
    assert set(book.ask_prices) == set(book.asks) == set(book.ask_vol)
    for side_book, vols in ((book.bids, book.bid_vol), (book.asks, book.ask_vol)):
        for price, queue in side_book.items():
            assert queue, f"empty level {price} left in book"
            live = sum(o.remaining for o in queue if o.active)
            assert live == vols[price], f"level {price}: vol {vols[price]} != live {live}"
            assert all(o.price == price for o in queue)
    for oid, o in book.orders.items():
        assert o.active and o.remaining > 0 and o.order_id == oid

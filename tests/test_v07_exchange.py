"""v0.7 venue rules, latency models, delayed feeds, race accounting and deterministic replay."""
from __future__ import annotations

from decimal import Decimal
import json

import numpy as np
import pytest

from lob.engine import ExchangeSimulator, SimConfig, Side
from lob.replay.l2 import L2Replay
from lob.v07.adapters.tardis import TardisAdapter
from lob.v07.exchange import latency
from lob.v07.exchange.venue import VenueRuleError, rules
from lob.v07.replay.deterministic import BookReplay
from tests.v07_fixtures import write_tardis


def test_venue_rules_validate_orders() -> None:
    btc = rules("deribit", "BTC-PERPETUAL")
    assert btc.validate(side="buy", qty=Decimal("20"), price=Decimal("10000.5"))["price"] == Decimal("10000.5")
    for kwargs in ({"qty": Decimal("15"), "price": Decimal("10000")}, {"qty": Decimal("10"), "price": Decimal("10000.2")},
                   {"qty": Decimal("10"), "price": None}):
        with pytest.raises(VenueRuleError):
            btc.validate(side="buy", **kwargs)
    with pytest.raises(VenueRuleError):
        btc.validate(side="buy", qty=Decimal("10"), price=Decimal("100"), tif="GTD")
    with pytest.raises(VenueRuleError):
        btc.validate(side="buy", qty=Decimal("10"), price=Decimal("100"), post_only=True, best_opposite=Decimal("99.5"))
    with pytest.raises(VenueRuleError):
        btc.validate(side="buy", qty=Decimal("10"), price=None, order_type="market", post_only=True)
    assert not btc.verified and rules("cleolob", "SIM").verified
    assert btc.sha256() != rules("bitmex", "XBTUSD").sha256()
    with pytest.raises(VenueRuleError):
        rules("binance", "BTCUSDT")


@pytest.mark.parametrize("name", sorted(latency.ZOO))
def test_latency_models_are_seed_deterministic_and_nonnegative(name) -> None:
    model = latency.ZOO[name]
    a = [model.draw(np.random.default_rng(3)) for _ in range(5)]
    b = [model.draw(np.random.default_rng(3)) for _ in range(5)]
    assert a == b and min(a) >= 0
    q = model.quantiles(2000)
    assert q["p50"] <= q["p90"] <= q["p99"]


def test_attached_latency_changes_arrival_but_stays_deterministic() -> None:
    def run(model):
        sim = latency.attach(ExchangeSimulator(SimConfig(seed=5)), model)
        sim.step(1.0)
        oid = sim.submit(Side.BUY, 10, "agent")
        sim.step(0.004)
        return sim.orders[oid].status.name
    assert run(latency.ZOO["zero"]) != "IN_FLIGHT"
    assert run(latency.ZOO["slow_fixed"]) == "IN_FLIGHT"
    assert run(latency.ZOO["heavy_tail"]) == run(latency.ZOO["heavy_tail"])


def test_delayed_feed_serves_stale_book() -> None:
    sim = ExchangeSimulator(SimConfig(seed=9))
    feed = latency.DelayedFeed(sim, delay=0.5)
    feed.step(2.0)
    seen = feed.observe()
    assert sim.t - seen["t"] >= 0.5 - 1e-9 and sim.t - seen["t"] < 0.52
    assert latency.DelayedFeed(sim, delay=0.0).observe()["t"] == sim.t


def test_race_report_counts_cancel_lost_to_fill() -> None:
    sim = latency.attach(ExchangeSimulator(SimConfig(seed=11)), latency.ZOO["slow_fixed"])
    sim.step(1.0)
    oid = sim.submit(Side.BUY, 10, "agent")  # marketable; fills on arrival after 50 ms
    sim.step(0.01)
    assert sim.cancel(oid)
    sim.step(1.0)
    report = latency.race_report(sim, "agent")
    assert report["orders"] == 1 and report["filled"] == 1 and report["cancel_lost_to_fill"] == 1


@pytest.fixture(scope="module")
def l2(tmp_path_factory):
    return write_tardis(tmp_path_factory.mktemp("replay"), seconds=120.0)


def test_replay_matches_reference_l2_reconstruction(l2) -> None:
    replay = BookReplay(levels=10).run(TardisAdapter("deribit", "ETH-PERPETUAL", l2=l2["l2"], trades=None).records())
    replay.finish()
    last = None
    for state in L2Replay(l2["l2"], depth=10):
        last = state
    bids, asks = replay.top()
    assert [(Decimal(p), Decimal(q)) for p, q in bids] == list(last.bids)
    assert [(Decimal(p), Decimal(q)) for p, q in asks] == list(last.asks)
    assert replay.crossed_groups == 0 and replay.groups > 100


def test_checkpoint_resume_reproduces_hash_chain(l2) -> None:
    adapter = TardisAdapter("deribit", "ETH-PERPETUAL", l2=l2["l2"], trades=l2["trades"])
    full = BookReplay().run(adapter.records()).finish()
    partial = BookReplay().run(adapter.records(), stop=1500)
    checkpoint = partial.checkpoint()
    resumed = BookReplay.resume(checkpoint, adapter.records()).finish()
    assert resumed == full
    assert BookReplay().run(adapter.records()).finish() == full
    stored = json.loads(checkpoint)
    stored["state"]["index"] = 1499
    tampered = json.dumps(stored)
    with pytest.raises(ValueError):
        BookReplay.resume(tampered, adapter.records())


# ----------------------------------------------------------------------------- matching fixtures (workstream 3)

from lob.engine import Order, OrderBook, OrderType, TimeInForce  # noqa: E402
from lob.v07.exchange.venue import semantics_matrix  # noqa: E402


def _limit(oid, side, qty, price, tif=TimeInForce.GTC, post_only=False, owner="X"):
    return Order(oid, side, qty, OrderType.LIMIT, owner, 0.0, price=price, time_in_force=tif, post_only=post_only)


def test_price_time_priority_partial_fills_and_sweep() -> None:
    book = OrderBook(check_invariants=True)
    book.process(_limit(1, Side.SELL, 5, 101), 0.0)
    book.process(_limit(2, Side.SELL, 5, 101), 0.0)
    book.process(_limit(3, Side.SELL, 5, 102), 0.0)
    trades = book.process(Order(4, Side.BUY, 12, OrderType.MARKET, "T", 0.0), 0.1)
    assert [(t.maker_order_id, t.price, t.qty) for t in trades] == [(1, 101, 5), (2, 101, 5), (3, 102, 2)]
    assert book.best_ask() == 102 and book.ask_vol[102] == 3


def test_ioc_fok_and_post_only_semantics() -> None:
    book = OrderBook(check_invariants=True)
    book.process(_limit(1, Side.SELL, 5, 101), 0.0)
    fok = _limit(2, Side.BUY, 10, 101, TimeInForce.FOK)
    assert book.process(fok, 0.1) == [] and fok.status.name == "CANCELLED"
    ioc = _limit(3, Side.BUY, 10, 101, TimeInForce.IOC)
    assert sum(t.qty for t in book.process(ioc, 0.2)) == 5 and ioc.status.name == "CANCELLED" and not book.asks
    book.process(_limit(4, Side.SELL, 5, 103), 0.3)
    post = _limit(5, Side.BUY, 1, 103, post_only=True)
    book.process(post, 0.4)
    assert post.status.name == "REJECTED" and post.reject_reason == "post_only_would_cross"


def test_resting_book_never_locked_or_crossed() -> None:
    book = OrderBook(check_invariants=True)
    book.process(_limit(1, Side.SELL, 5, 101), 0.0)
    book.process(_limit(2, Side.BUY, 3, 101), 0.1)        # marketable limit matches, never rests locked
    assert book.best_bid() is None and book.ask_vol[101] == 2


def test_cancel_replace_priority() -> None:
    book = OrderBook(check_invariants=True)
    book.process(_limit(1, Side.BUY, 5, 100), 0.0)
    book.process(_limit(2, Side.BUY, 5, 100), 0.0)
    book.modify(1, 3, None, 0.1)                          # reduce keeps priority
    assert book.queue_position(1) == (0, 0)
    book.modify(1, 8, None, 0.2)                          # increase resets priority
    assert book.queue_position(1) == (1, 5)


def test_semantics_matrix_marks_unavailable() -> None:
    matrix = semantics_matrix()
    assert matrix["bitmex/XBTUSD"]["auction"] == "NOT_AVAILABLE"
    assert matrix["deribit/BTC-PERPETUAL"]["price_time_priority"] == "DECLARED"
    assert matrix["cleolob/SIM"]["price_time_priority"] == "SIMULATED"
    assert matrix["deribit/ETH-PERPETUAL"]["self_trade_prevention"] == "NOT_AVAILABLE"

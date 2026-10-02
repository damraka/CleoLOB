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

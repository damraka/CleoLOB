"""v1 multi-fidelity twins: API, F0-F6, comparison, permissions and agreement (M60-M69, M134)."""
from pathlib import Path

import numpy as np
import pytest

from lob.v08.generators import PoissonGenerator
from lob.v1.execution import policies as XP
from lob.v1.generators.classical import book as B
from lob.v1.twins import agreement as AGR, api as TW

DEV = Path("data/v1/cache/bybit-linear-btcusdt-2025-08-22.arrow")


def test_api_capabilities_never_inherited():
    assert TW.CAPABILITIES["F0"]["execution"] is False and TW.CAPABILITIES["F1"]["execution"] is False
    t = TW.Twin("F1", "replay")
    with pytest.raises(PermissionError):
        t.episode(None)
    assert set(TW.FIDELITIES) == set(TW.CAPABILITIES)


def test_permissions_fail_closed():
    assert TW.permitted("F2", "historical_execution_cost")
    assert not TW.permitted("F4", "historical_execution_cost")
    assert not TW.permitted("F4", "execution_cost", {"gate": {"spread_dynamics": "REJECTED"}})
    assert TW.permitted("F4", "execution_cost", {"gate": {"spread_dynamics": "ACCEPTABLE"}})
    assert not TW.permitted("F0", "queue_position", {"queue_observability": "aggregate_level"})
    assert TW.permitted("F0", "queue_position", {"queue_observability": "exact_order_id"})
    assert not TW.permitted("F2", "causal_impact") and TW.permitted("F2", "causal_impact", {"identifying_design": 1})
    with pytest.raises(KeyError):
        TW.permitted("F9", "realism")


def _toy_cache(n=7200):
    rng = np.random.default_rng(0)
    bid = 1000 + np.cumsum(rng.integers(-1, 2, n) * (rng.random(n) < 0.02))
    return {"bid_px": bid, "ask_px": bid + 1, "depth": np.full((n, 2, 10), 5.0), "valid": np.ones(n, bool),
            "trade_t": np.sort(rng.random(300) * n / 10), "trade_px": np.full(300, 1000), "trade_q": np.ones(300),
            "trade_s": rng.choice([-1, 1], 300)}


def test_f0_observational_and_f1_replay():
    c = _toy_cache()
    assert TW.f0(c)["mean_spread_ticks"] == 1.0
    r = TW.f1_replay(c, 10.0, 60.0)
    assert len(r["bid_px"]) == 600 and np.all(np.diff(r["trades"]["trade_t"]) >= 0)


@pytest.mark.skipif(not DEV.exists(), reason="development cache not built")
def test_f2_bounded_twin_on_development_day():
    from lob.v1.data import daycache as DC
    tw = TW.f2_twin(DC.load(DEV), notional_usd=100_000)
    r = tw.episode(XP.make("twap"), window=3)
    assert r["status"] == "OK" and r["is_net_bps_lower"] <= r["is_net_bps_upper"]


@pytest.mark.skipif(not DEV.exists(), reason="development cache not built")
def test_f3_hybrid_twin_runs():
    from lob.v09.generators import hybrid as Hy
    from lob.v09.impact import models as IM
    from lob.v1.data import daycache as DC
    c = DC.load(DEV)
    bt = Hy.burst_targets(c)
    imp = IM.StateImpact(trees=10).fit(bt["X"], bt["Y"][:, 1])
    r = TW.f3_twin(c, notional_usd=100_000, impact_model=imp, persistent=0.5).episode(XP.make("twap"), window=2)
    assert r["fidelity"] == "F3" and r["status"] in ("OK", "NOT_EVALUABLE")


def _gen():
    base = B.BookModelV1(refill_mean=6.0, limit_size_mean=3.0, market_size_mean=2.0, mid_ticks=4000,
                         improve_by_spread=((2, 0.6), (3, 0.6), (4, 0.6)))
    return PoissonGenerator({**PoissonGenerator.default_params(), "market": 1.0}, base)


def test_f4_classical_and_f5_neural_twins():
    r = TW.f4_twin(_gen()).episode(XP.make("twap"), qty=20, seed=1)
    assert r["fidelity"] == "F4" and r["status"] == "OK"
    torch = pytest.importorskip("torch")  # noqa: F841
    from lob.v08.market.sim import simulate
    from lob.v1.generators.neural.counts import AutoregressiveCounts
    g = _gen()
    frames = [simulate(g, g.book, 600, seed=s, warmup_bins=50) for s in range(2)]
    nc = AutoregressiveCounts(steps=60, history=8, seed=1).fit(frames)
    nc.book = g.book
    r5 = TW.f5_twin(nc).episode(XP.make("twap"), qty=20, seed=1)
    assert r5["fidelity"] == "F5" and r5["status"] in ("OK", "NOT_EVALUABLE")


def test_f6_strategic_twin():
    r = TW.f6_twin().episode(XP.make("twap"), qty=40, seed=1, horizon_s=30)
    assert r["fidelity"] == "F6" and r["status"] == "OK"


def test_compare_and_agreement_metrics():
    twins = {"F6": TW.f6_twin()}
    pols = {"twap": lambda: XP.make("twap"), "sweep": lambda: XP.make("sweep")}
    table = AGR.compare(twins, pols, [{"qty": 40, "seed": s, "horizon_s": 30} for s in range(4)])
    assert set(table["F6"]) == {"twap", "sweep"} and len(table["F6"]["twap"]) == 4
    d = AGR.directions({"a": [1, 1.1, 0.9, 1.0, 1.05], "b": [3, 3.1, 2.9, 3.0, 3.2]})
    assert d == {"a>b": 1}
    a, n = AGR.agreement({"a>b": 1, "a>c": -1}, {"a>b": 1, "a>c": 1, "b>c": 0})
    assert a == 0.5 and n == 2
    assert AGR.consensus([{"x": 1}, {"x": 1}, {"x": -1}]) == {"x": 1}
    assert AGR.bounded_directions({"p": (0.2, 1.0), "q": (-1.0, 0.5)}) == {"p": 1, "q": 0}
    assert AGR.kendall({"a": [1], "b": [2], "c": [3]}, {"a": [1], "b": [2], "c": [3]}) == 1.0
    assert AGR.reality_gap({"C1": -0.7}, {"C1": 11.0}) == {"C1": pytest.approx(-11.7)}

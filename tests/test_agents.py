"""Execution-agent tests: schedules, completion, participation, reporting."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from lob.engine import ExchangeSimulator, Side, SimConfig, Trade
from lob.execution import (AlmgrenChrissAgent, POVAgent, TWAPAgent, VWAPAgent,
                           build_report, estimate_volume_profile)

Q, T = 10_000, 60.0


def _drive(agent, seed: int = 42, dt: float = 0.5, owner: str = "X"):
    """Warm up a market, then let ``agent`` trade for the full horizon."""
    sim = ExchangeSimulator(SimConfig(seed=seed))
    for _ in range(50):
        sim.step(0.1)
    agent.start_time = sim.t
    if hasattr(agent, "release_times"):
        agent.release_times = [sim.t + (r - 0.0) for r in agent.release_times]
    for _ in range(int(T / dt)):
        agent.on_step(sim, owner)
        sim.step(dt)
        agent.poll_fills(sim, owner)
    return sim


# ------------------------------------------------------------- schedules
def test_twap_targets_are_linear_and_slices_equal():
    a = TWAPAgent(Q, T, n_slices=20)
    assert a.targets[0] == Q and a.targets[-1] == 0
    slices = [a.targets[j] - a.targets[j + 1] for j in range(20)]
    assert all(s == 500 for s in slices)


@pytest.mark.parametrize("lam", [1e-7, 1e-6, 1e-5, 1e-4])
def test_ac_targets_monotone_and_complete(lam):
    a = AlmgrenChrissAgent(Q, T, n_slices=20, risk_aversion=lam)
    assert a.targets[0] == Q and a.targets[-1] == 0
    assert all(x >= y for x, y in zip(a.targets, a.targets[1:]))
    assert sum(a.targets[j] - a.targets[j + 1] for j in range(20)) == Q


def test_ac_front_loads_more_with_higher_risk_aversion():
    def first_slice(lam):
        return Q - AlmgrenChrissAgent(Q, T, risk_aversion=lam).targets[1]
    assert first_slice(1e-4) > first_slice(1e-6) > first_slice(1e-7)


def test_ac_with_tiny_risk_aversion_is_twap():
    ac = AlmgrenChrissAgent(Q, T, n_slices=20, risk_aversion=1e-12)
    assert ac.targets == TWAPAgent(Q, T, n_slices=20).targets


def test_vwap_flat_profile_equals_twap_and_profile_shapes_schedule():
    assert VWAPAgent(Q, T, n_slices=4).targets == TWAPAgent(Q, T, n_slices=4).targets
    v = VWAPAgent(Q, T, n_slices=4, volume_profile=[0.4, 0.1, 0.1, 0.4])
    assert v.targets == [Q, 6000, 5000, 4000, 0]
    with pytest.raises(ValueError):
        VWAPAgent(Q, T, n_slices=4, volume_profile=[1, 1])


def test_volume_profile_is_normalised_and_reproducible():
    cfg = SimConfig(seed=9)
    p1 = estimate_volume_profile(cfg, 20.0, 10, n_days=2)
    p2 = estimate_volume_profile(cfg, 20.0, 10, n_days=2)
    assert p1 == p2 and len(p1) == 10
    assert sum(p1) == pytest.approx(1.0) and all(w > 0 for w in p1)


# ------------------------------------------------------------- execution
@pytest.mark.parametrize("factory", [
    lambda: TWAPAgent(Q, T),
    lambda: VWAPAgent(Q, T),
    lambda: AlmgrenChrissAgent(Q, T, risk_aversion=1e-6),
    lambda: AlmgrenChrissAgent(Q, T, risk_aversion=1e-4),
    lambda: POVAgent(Q, T, participation=0.3, decision_dt=0.5),
])
def test_every_baseline_completes_within_horizon(factory):
    agent = factory()
    _drive(agent)
    assert agent.filled == Q and agent.remaining == 0
    assert sum(t.qty for t in agent.fills) == Q
    assert agent.child_orders >= 1


def test_schedule_agent_releases_exactly_one_child_per_slice():
    agent = TWAPAgent(Q, T, n_slices=20)
    _drive(agent)
    assert agent.child_orders == 20


def test_pov_tracks_participation_rate_mid_horizon():
    p = 0.3
    agent = POVAgent(Q, T, participation=p, decision_dt=0.5, catchup_frac=1.0)
    sim = ExchangeSimulator(SimConfig(seed=42))
    for _ in range(50):
        sim.step(0.1)
    agent.start_time = sim.t
    for _ in range(60):                       # first 30 s only: no catch-up regime
        agent.on_step(sim, "X")
        sim.step(0.5)
        agent.poll_fills(sim, "X")
    agent.on_step(sim, "X")                   # refresh market_volume
    share = agent.filled / (agent.filled + agent.market_volume)
    assert share == pytest.approx(p, abs=0.05)


def test_pov_rejects_invalid_rate():
    with pytest.raises(ValueError):
        POVAgent(Q, T, participation=1.0)


# ------------------------------------------------------------- reporting
def _fill(price: int, qty: int) -> Trade:
    return Trade(0.0, price, qty, Side.SELL, "X", "ZI")


def test_report_sign_convention_sell_below_arrival_is_cost():
    r = build_report("t", Side.SELL, 100, [_fill(9990, 100)], 10_000, 0.01, 1, 1.0)
    assert r.shortfall_bps == pytest.approx(10.0)
    assert r.total_cost == pytest.approx(10 * 100 * 0.01)
    r_buy = build_report("t", Side.BUY, 100, [_fill(10_010, 100)], 10_000, 0.01, 1, 1.0)
    assert r_buy.shortfall_bps == pytest.approx(10.0)


def test_effective_shortfall_prices_leftovers():
    fills = [_fill(10_000, 50)]                              # half filled at arrival
    lenient = build_report("t", Side.SELL, 100, fills, 10_000, 0.01, 1, 1.0)
    strict = build_report("t", Side.SELL, 100, fills, 10_000, 0.01, 1, 1.0,
                          leftover_vwap_ticks=9_900)         # leftovers 100 bps worse
    assert lenient.shortfall_bps == 0 and lenient.effective_shortfall_bps == 0
    assert strict.shortfall_bps == 0
    assert strict.effective_shortfall_bps == pytest.approx(50.0)
    assert strict.leftover_qty == 50
    assert "Effective IS (bps)" in strict.as_dict()

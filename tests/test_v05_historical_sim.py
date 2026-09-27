"""M8: historical replay behind the simulator interface with bounded passive fills (synthetic episodes)."""
from __future__ import annotations

import pytest

from lob.engine import Side, SimConfig
from lob.historical_sim import HistoricalEpisode, HistoricalSimulator
from lob.runner import run_episode
from lob.scenarios import scenario_params


def _episode(prints_at_bid=True, clock=1.0, sell_prints=40.0):
    updates, prints = [], []
    for i in range(80):
        t = i * 0.1
        bids = {999: 50.0, 998: 80.0, 997: 120.0, 996: 200.0, 995: 300.0}
        asks = {1001: 50.0, 1002: 80.0, 1003: 120.0, 1004: 200.0, 1005: 300.0}
        updates.append((t, bids, asks))
        if prints_at_bid and i % 5 == 2:
            prints.append((t + 0.05, 999, sell_prints, "SELL"))
            prints.append((t + 0.06, 1001, sell_prints, "BUY"))
    return HistoricalEpisode(0.0, updates, prints, 0.05, clock, label="synthetic")


def test_marketable_orders_walk_displayed_levels_only():
    sim = HistoricalSimulator(SimConfig(latency_base=0, latency_jitter=0), _episode(False), "conservative")
    oid = sim.submit(Side.BUY, 100, "ME")
    sim.step(0.01)
    fills = [t for t in sim.book.trades if t.taker_owner == "ME"]
    assert sum(t.qty for t in fills) == 100 and {t.price for t in fills} == {1001, 1002}
    assert sim.orders[oid].status.name == "FILLED"


def test_displayed_levels_reset_to_history_no_impact():
    sim = HistoricalSimulator(SimConfig(latency_base=0, latency_jitter=0), _episode(False), "conservative")
    sim.submit(Side.BUY, 60, "ME")
    sim.step(0.01)
    assert sim.book.depth(1)[1][0] == (1002, 70)
    sim.step(0.1)
    assert sim.book.depth(1)[1][0] == (1001, 50)


@pytest.mark.parametrize("mode,expected", [("conservative", 0), ("optimistic", 30)])
def test_passive_fills_follow_the_declared_bound(mode, expected):
    sim = HistoricalSimulator(SimConfig(latency_base=0, latency_jitter=0), _episode(), mode)
    sim.submit(Side.BUY, 30, "ME", price_ticks=999)
    sim.step(0.3)
    filled = sum(t.qty for t in sim.book.trades if t.maker_owner == "ME")
    assert filled == expected  # prints at 999 never print through the order's price


def test_prints_enter_the_tape_as_market_volume():
    sim = HistoricalSimulator(SimConfig(latency_base=0, latency_jitter=0), _episode(), "conservative")
    sim.step(1.0)
    assert any(t.taker_owner == "HIST" and t.maker_owner == "HIST" for t in sim.book.trades)


def test_clock_ratio_maps_simulated_to_historical_time():
    sim = HistoricalSimulator(SimConfig(latency_base=0, latency_jitter=0), _episode(clock=2.0), "optimistic")
    sim.step(0.13)  # 0.26 historical seconds: the first prints at 0.25/0.26 are consumed
    assert sim.stats["prints"] >= 1 and sim._hist_time(0.13) == pytest.approx(0.26)


@pytest.mark.parametrize("agent", ["twap", "pov", "heuristic", "random"])
def test_existing_agents_run_unchanged_on_history_and_bounds_are_ordered(agent):
    rows = {}
    for mode in ("conservative", "optimistic"):
        params = scenario_params("calm", 5)
        params.update(qty=120, horizon=2.0, dt=0.1, warmup_seconds=0.2, completion={"enabled": True},
                      settlement_timeout=0.5, latency_ms=0.0,
                      historical={"episode": _episode(), "fill_mode": mode})
        rows[mode] = run_episode(agent, params)
    for row in rows.values():
        assert row["mandate_final_filled_qty"] <= 120
    assert rows["optimistic"]["mandate_within_horizon_filled_qty"] >= rows["conservative"]["mandate_within_horizon_filled_qty"] \
        or agent == "random"


def test_historical_replay_refuses_flow_extensions():
    from lob.simulators import make_simulator
    with pytest.raises(ValueError):
        make_simulator(SimConfig(), {"imbalance_beta": 0.2}, {"episode": _episode(), "fill_mode": "conservative"})


def _iv(low, high):
    return {"mean": (low + high) / 2, "ci_low": low, "ci_high": high, "n": 144}


def test_transfer_classification_rules():
    from lob.policy_transfer import classify
    synthetic = {"mean": -1.0, "ci_low": -2.0, "ci_high": -0.1}
    assert classify(synthetic, {"conservative": _iv(-3, -1), "optimistic": _iv(-4, -2)}) == "agrees_survives_conservative"
    assert classify(synthetic, {"conservative": _iv(-1, 1), "optimistic": _iv(-4, -2)}) == "optimistic_assumption_dependent"
    assert classify(synthetic, {"conservative": _iv(1, 2), "optimistic": _iv(0.5, 3)}) == "reverses"
    assert classify(synthetic, {"conservative": _iv(-1, 1), "optimistic": _iv(-1, 2)}) == "indeterminate"
    assert classify(None, {"conservative": _iv(-3, -1), "optimistic": _iv(-3, -1)}) == "not_evaluable"
    assert classify(synthetic, {"conservative": None, "optimistic": _iv(-3, -1)}) == "not_evaluable"


def test_kendall_tau_between_rankings():
    from lob.policy_transfer import _kendall
    assert _kendall(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert _kendall(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert _kendall(["a"], ["a"]) is None

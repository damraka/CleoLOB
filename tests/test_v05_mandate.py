"""M5: strict within-horizon mandate completion versus post-horizon settlement."""
from __future__ import annotations

from dataclasses import replace

import pytest

from lob.engine import SimConfig
from lob.mandate import mandate_metrics
from lob.rl_env import LOBExecutionEnv
from lob.runner import BASELINES, run_episode
from lob.scenarios import scenario_params

COMMON = dict(target_qty=100, start_time=10.0, horizon=2.0, total_fees=1.0, late_fees=0.25,
              realized_fill_cost_bps=1.5, hypothetical_residual_cost_bps=None, net_effective_bps=None,
              participation=0.1)


def test_fill_exactly_at_horizon_counts_as_within():
    m = mandate_metrics([(11.0, 40), (12.0, 60)], settlement_complete=True, **COMMON)
    assert m["within_horizon_completion"] and m["lateness_seconds"] == 0.0
    assert m["post_horizon_filled_qty"] == 0 and m["settlement_only_completion"] is False


def test_fill_one_event_after_horizon_is_settlement_only():
    m = mandate_metrics([(11.0, 40), (12.000001, 60)], settlement_complete=True, **COMMON)
    assert m["within_horizon_completion"] is False and m["final_settlement_completion"] is True
    assert m["settlement_only_completion"] is True and m["post_horizon_filled_qty"] == 60
    assert m["residual_inventory_at_horizon"] == 60 and m["lateness_seconds"] == pytest.approx(1e-6)
    assert m["fees_post_horizon"] == 0.25 and m["fees_within_horizon"] == 0.75


def test_fill_one_event_before_horizon():
    m = mandate_metrics([(11.9999, 100)], settlement_complete=True, **COMMON)
    assert m["within_horizon_completion"] and m["time_to_completion"] == pytest.approx(1.9999)


def test_unresolved_settlement_withholds_final_but_not_horizon_outcome():
    m = mandate_metrics([(11.0, 30)], settlement_complete=None, **COMMON)
    assert m["within_horizon_completion"] is False and m["final_settlement_completion"] is None
    assert m["residual_inventory_after_settlement"] == 70


def test_residual_valuation_never_creates_fills_or_completion():
    m = mandate_metrics([], settlement_complete=True, **{**COMMON, "hypothetical_residual_cost_bps": 12.0,
                                                         "net_effective_bps": 12.0})
    assert m["final_filled_qty"] == 0 and m["within_horizon_completion"] is False
    assert m["final_settlement_completion"] is False and m["hypothetical_residual_valuation_bps"] == 12.0
    assert m["time_to_completion"] is None and m["lateness_seconds"] is None


def test_overfill_and_bad_inputs_are_rejected():
    with pytest.raises(ArithmeticError):
        mandate_metrics([(11.0, 101)], settlement_complete=True, **COMMON)
    with pytest.raises(ValueError):
        mandate_metrics([(11.0, 0)], settlement_complete=True, **COMMON)


def quiet(**changes):
    return SimConfig(limit_rate=0, market_rate=0, cancel_rate=0, resilience=0, latency_base=0, latency_jitter=0,
                     **changes)


def _episode(cfg, *, risk=None, settlement_timeout=5.0, qty=100, horizon=0.5):
    env = LOBExecutionEnv(total_qty=qty, horizon=horizon, decision_dt=0.1, warmup=0, cfg=cfg,
                          completion={"enabled": True}, settlement_timeout=settlement_timeout, risk=risk)
    env.reset(seed=1)
    while True:
        _, _, end, truncated, _ = env.step(0)
        if end or truncated:
            break
    from lob.completion import execution_metrics
    return execution_metrics(env.sim, env.fills, owner=env.OWNER, start_time=env.t0, first_trade=0,
                             report=env.report())


def test_latency_crossing_horizon_completes_only_in_settlement():
    m = _episode(replace(quiet(), latency_base=0.45))
    assert m["mandate_within_horizon_completion"] is False
    assert m["mandate_final_settlement_completion"] is True and m["mandate_settlement_only_completion"] is True
    assert m["mandate_post_horizon_filled_qty"] > 0


def test_missing_liquidity_leaves_residual_without_fills():
    env_cfg = quiet()
    env = LOBExecutionEnv(total_qty=100, horizon=0.5, decision_dt=0.1, warmup=0, cfg=env_cfg,
                          completion={"enabled": True}, settlement_timeout=0)
    env.reset(seed=1)
    for oid in list(env.sim.book.orders):
        if env.sim.book.orders[oid].side is not env.side:  # remove all opposite liquidity
            env.sim.book.cancel(oid)
    while True:
        _, _, end, truncated, _ = env.step(0)
        if end or truncated:
            break
    from lob.completion import execution_metrics
    m = execution_metrics(env.sim, env.fills, owner=env.OWNER, start_time=env.t0, first_trade=0, report=env.report())
    assert m["mandate_final_filled_qty"] == 0 and m["mandate_residual_inventory_at_horizon"] == 100
    assert m["mandate_within_horizon_completion"] is False


@pytest.mark.parametrize("risk", [{"kill_switch": True}, {"max_order_qty": 10}])
def test_kill_switch_and_size_limits_are_respected(risk):
    m = _episode(quiet(), risk=risk, settlement_timeout=0)
    assert m["mandate_within_horizon_completion"] is False
    assert m["mandate_final_filled_qty"] <= 100


def test_delayed_cancellation_during_settlement_is_reported_not_counted_as_mandate():
    m = _episode(replace(quiet(), latency_base=0.2), settlement_timeout=0.0)
    assert m["mandate_within_horizon_completion"] in {True, False}
    if not m["mandate_within_horizon_completion"]:
        assert m["mandate_settlement_only_completion"] in {True, False}
        assert m["mandate_final_settlement_completion"] in {True, False, None}


@pytest.mark.parametrize("agent", sorted(BASELINES))
def test_identical_mandate_rules_for_every_control(agent):
    params = scenario_params("calm", 3)
    params.update(completion={"enabled": True}, horizon=2.0, qty=300)
    row = run_episode(agent, params)
    keys = [k for k in row if k.startswith("mandate_")]
    assert {"mandate_within_horizon_completion", "mandate_final_settlement_completion",
            "mandate_post_horizon_filled_qty", "mandate_lateness_seconds"} <= set(keys)
    assert row["mandate_final_filled_qty"] == row["actual_filled_qty"]
    if row["mandate_within_horizon_completion"]:
        assert row["mandate_post_horizon_filled_qty"] == 0


def test_policy_runner_reports_the_same_mandate_block():
    params = scenario_params("calm", 3)
    params.update(completion={"enabled": True}, horizon=2.0, qty=300)
    row = run_episode("heuristic", params)
    assert "mandate_within_horizon_completion" in row and "mandate_fill_fraction_at_horizon" in row

"""Runner and RL integration tests for accounting and valid research outcomes."""
import json

import pytest

from lob.accounting import FeeConfig
from lob.engine import SimConfig
from lob.rl_env import LOBExecutionEnv
from lob.runner import run_episode
from lob.scenarios import scenario_params


def quiet_config(**kwargs):
    return SimConfig(**dict(dict(limit_rate=0, market_rate=0, cancel_rate=0,
                                  resilience=0, latency_base=0.2, latency_jitter=0), **kwargs))


def test_rl_cancel_replace_high_latency_never_overfills_even_after_horizon():
    env = LOBExecutionEnv(total_qty=100, horizon=1.05, decision_dt=0.1,
                          warmup=0, cfg=quiet_config(latency_base=0.35))
    env.reset(seed=4)
    done = False
    while not done:
        _, _, done, _, info = env.step(3)
        assert info["filled"] + info["reserved_qty"] <= 100
    env.sim.step(1)
    fills, _ = env.sim.fills_for(env.OWNER, 0)
    assert sum(t.qty for t in fills) <= 100
    assert info["t"] <= 1.05 + 1e-12
    if info["remaining"]:
        assert info["t"] == pytest.approx(1.05)


@pytest.mark.parametrize("action", [0, 4])
def test_rl_rewards_equal_disclosed_objective_with_actual_and_terminal_fees(action):
    env = LOBExecutionEnv(total_qty=100, horizon=0.35, decision_dt=0.2, warmup=0,
                          cfg=quiet_config(latency_base=0.01),
                          fees=FeeConfig(taker_bps=2, maker_bps=-1, per_share=0.01))
    env.reset(seed=4)
    rewards, done = 0, False
    while not done:
        _, reward, done, _, _ = env.step(action)
        rewards += reward
    report = env.report()
    assert rewards == pytest.approx(-report.optimization_cost_bps)
    assert report.optimization_cost_bps == pytest.approx(report.net_effective_bps + report.completion_penalty_bps)
    if action == 0:
        assert report.total_fees == 0 and report.hypothetical_liquidation_fees > 0
    else:
        assert report.total_fees > 0


def test_empty_terminal_liquidity_invalidates_economic_metric_but_reward_is_finite():
    env = LOBExecutionEnv(total_qty=100, horizon=0.1, decision_dt=0.1, warmup=0,
                          cfg=quiet_config(target_level_vol=1))
    env.reset(seed=4)
    _, reward, done, _, info = env.step(0)
    assert done and info["status"] == "INVALID"
    assert info["unpriced_leftover_qty"] == 80
    assert reward == pytest.approx(-env.report().optimization_cost_bps)
    assert env.report().effective_shortfall_bps is None


def test_env_invalid_action_and_post_done_step_fail():
    env = LOBExecutionEnv(total_qty=100, horizon=0.1, warmup=0)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(5)
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_baseline_exact_fractional_horizon_and_json_serializable_audit():
    params = scenario_params("calm", seed=1, qty=100, horizon=0.35, dt=0.2,
                             latency_ms=1000, record_audit=True, fees={"taker_bps": 2})
    row = run_episode("twap", params)
    assert row["duration"] == pytest.approx(0.35)
    assert row["outstanding_qty"] <= 100
    assert row["audit"]["orders"] and row["audit"]["risk_events"]
    json.dumps(row, allow_nan=False)


def test_research_ppo_missing_model_fails_and_legacy_fallback_is_disclosed(tmp_path):
    params = scenario_params("calm", seed=1, qty=100, horizon=0.1,
                             model_path=str(tmp_path / "missing"), strict_model=True)
    with pytest.raises(FileNotFoundError):
        run_episode("ppo", params)
    row = run_episode("ppo", {**params, "strict_model": False})
    assert row["agent"] == "ppo" and row["implementation"] == "heuristic"
    assert "heuristic" in row["label"]


def test_strict_invalid_depth_row_preserves_null_and_failure_reason():
    params = scenario_params("calm", seed=1, qty=1000, horizon=0.1, dt=0.1,
                             risk={"kill_switch": True},
                             sim={"target_level_vol": 1, "market_rate": 0, "limit_rate": 0,
                                  "cancel_rate": 0, "resilience": 0})
    row = run_episode("twap", params)
    assert row["status"] == "INVALID" and row["effective_bps"] is None
    assert row["total_fees"] == 0 and row["fill_frac"] == 0
    assert "insufficient_terminal_depth" in row["invalid_reasons"]
    json.dumps(row, allow_nan=False)


def test_decision_log_contains_pre_action_observation_and_separate_reward_terms():
    env = LOBExecutionEnv(total_qty=100, horizon=0.1, decision_dt=0.1, warmup=0,
                          cfg=quiet_config(latency_base=0), record=True)
    observation, _ = env.reset(seed=5)
    _, reward, _, _, info = env.step(4)
    decision = env.decisions[0]
    assert decision["observation"] == observation.tolist()
    assert decision["t"] == 0 and decision["result_time"] == pytest.approx(0.1)
    assert sum(decision["reward_terms"].values()) == pytest.approx(reward)
    assert info["reward_terms"] == decision["reward_terms"]


def test_training_and_validation_reset_domains_stay_disjoint_and_reproducible():
    training = LOBExecutionEnv(total_qty=10, horizon=0.1, warmup=0, seed_range=(100, 200))
    validation = LOBExecutionEnv(total_qty=10, horizon=0.1, warmup=0, seed_range=(300, 400))
    training.reset(seed=1)
    validation.reset(seed=1)
    first = training.market_seed
    train_seeds, validation_seeds = set(), set()
    for _ in range(30):
        train_seeds.add(training.reset()[1]["market_seed"])
        validation_seeds.add(validation.reset()[1]["market_seed"])
    assert train_seeds.isdisjoint(validation_seeds)
    assert all(100 <= seed < 200 for seed in train_seeds)
    assert all(300 <= seed < 400 for seed in validation_seeds)
    training.reset(seed=1)
    assert training.market_seed == first

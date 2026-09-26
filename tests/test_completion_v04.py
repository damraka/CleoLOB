"""Completion constraints cannot manufacture fills or consume held-out state."""
from dataclasses import replace
import json

import numpy as np
import pytest

from lob import policy_study as study
from lob.completion import CompletionConstraint, urgency_order
from lob.completion import execution_metrics
from lob.engine import Side, SimConfig
from lob.experiments.registry import sha256_file
from lob.observations import FEATURES, ObservationNormalization
from lob.rl_env import LOBExecutionEnv
from lob.runner import BASELINES, run_episode
from lob.scenarios import scenario_params
from tests.test_policy_study import synthetic_rows, tiny_design


def quiet(**changes):
    return SimConfig(limit_rate=0, market_rate=0, cancel_rate=0, resilience=0,
                     latency_base=0, latency_jitter=0, **changes)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_wait_policy_completion_override_creates_only_actual_fills(side):
    env = LOBExecutionEnv(total_qty=100, horizon=0.5, decision_dt=0.1, warmup=0,
                          cfg=quiet(), side=side, completion={"enabled": True}, record=True)
    env.reset(seed=1)
    done = False
    while not done:
        _, _, end, truncated, _ = env.step(0)
        done = end or truncated
    assert env.report().filled_qty == 100
    assert env.report().hypothetical_liquidation_cost == 0
    assert any(d["completion_override"] for d in env.decisions)
    assert all(env.sim.orders[o].ts_submit < env.t0 + env.horizon for o in env.risk.order_ids)


@pytest.mark.parametrize("case", ["empty", "partial", "halt", "limit", "latency"])
def test_completion_reports_unreachable_residual_without_inventing_fills(case):
    cfg = replace(quiet(), latency_base=2) if case == "latency" else quiet()
    env = LOBExecutionEnv(total_qty=100, horizon=0.5, decision_dt=0.1, warmup=0,
                          cfg=cfg, completion={"enabled": True}, settlement_timeout=0,
                          risk={"kill_switch": True} if case == "halt" else
                               {"max_order_qty": 10} if case == "limit" else None)
    env.reset(seed=1)
    if case in {"empty", "partial"}:
        for oid in list(env.sim.book.orders):
            order = env.sim.book.orders[oid]
            if order.side is Side.BUY:
                env.sim.book.cancel(oid)
        if case == "partial":
            env.sim.submit(Side.BUY, 7, "VISIBLE", price_ticks=9999)
            env.sim.step(0)
    while True:
        _, _, end, truncated, _ = env.step(0)
        if end or truncated:
            break
    report = env.report()
    expected = {"empty": 0, "partial": 7, "halt": 0, "limit": 10, "latency": 0}[case]
    assert report.filled_qty == expected
    assert report.leftover_qty == 100 - expected
    assert report.filled_qty == sum(fill.qty for fill in env.fills)
    if case in {"empty", "partial", "latency"}:
        assert report.status == "INVALID" and report.net_effective_bps is None


def test_urgency_never_releases_cancel_reservation_or_submits_after_horizon():
    env = LOBExecutionEnv(total_qty=100, horizon=1, warmup=0, cfg=replace(quiet(), latency_base=.3))
    env.reset(seed=1)
    oid = env.risk.submit(env.sim, env.OWNER, 100, 0, env.ledger, price_ticks=10010)
    assert urgency_order(env.sim, env.risk, env.ledger, env.OWNER, 0,
                         elapsed=.9, horizon=1, decision_dt=.1) is None
    assert env.risk.outstanding(env.sim) == 100
    assert env.risk.order_ids == [oid]
    assert urgency_order(env.sim, env.risk, env.ledger, env.OWNER, 0,
                         elapsed=1, horizon=1, decision_dt=.1) is None


@pytest.mark.parametrize("agent", BASELINES + ("heuristic", "random"))
def test_all_controls_use_same_completion_contract_and_actual_metrics(agent):
    params = scenario_params("calm", seed=1, qty=100, horizon=.5, dt=.1,
                             sim=quiet().__dict__, warmup_seconds=0,
                             completion={"enabled": True, "urgency_fraction": 0}, record_audit=True)
    row = run_episode(agent, params)
    assert row["actual_completion"] and row["actual_filled_qty"] == 100
    assert row["time_to_completion"] <= .5
    assert row["terminal_inventory"] == row["hypothetical_residual_midpoint_cost"] == 0
    assert row["participation"] == 1
    assert row["realized_fill_cost"] == row["total_cost"]
    assert all(o["submitted_at"] < .5 for o in row["audit"]["orders"])


def test_v04_observations_include_own_state_without_queue_and_are_bounded():
    scale = ObservationNormalization((0.,) * 32, (1.,) * 32)
    env = LOBExecutionEnv(total_qty=100, horizon=1, warmup=0, cfg=quiet(),
                          observation_version="v04", observation_normalization=scale)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (32,) and env.observation_space.contains(obs)
    assert not any("queue" in feature for feature in FEATURES)
    env.risk.submit(env.sim, env.OWNER, 30, 0, env.ledger, price_ticks=10010)
    env.sim.step(0)
    updated = env._obs()
    assert updated[24] == pytest.approx(.3)
    assert updated[25] == pytest.approx(.7)
    assert updated[26] == pytest.approx(.3)
    assert np.isfinite(updated).all() and env.observation_space.contains(updated)
    masked = study.mask_observation(updated, "no_book")
    assert not masked[:22].any() and masked[28] == masked[31] == 0
    assert masked[24:28].tolist() == updated[24:28].tolist()


def test_extreme_spread_and_large_residual_remain_finite_and_incomplete():
    env = LOBExecutionEnv(total_qty=100000, horizon=.5, decision_dt=.1, warmup=0,
                          cfg=quiet(), completion={"enabled": True}, observation_version="v04",
                          observation_normalization=ObservationNormalization((0.,) * 32, (1.,) * 32))
    env.reset(seed=1)
    for oid in list(env.sim.book.orders):
        env.sim.book.cancel(oid)
    env.sim.submit(Side.BUY, 3, "VISIBLE", price_ticks=1)
    env.sim.submit(Side.SELL, 3, "VISIBLE", price_ticks=19999)
    env.sim.step(0)
    assert env.observation_space.contains(env._obs())
    while True:
        obs, reward, end, truncated, _ = env.step(0)
        assert np.isfinite(reward) and env.observation_space.contains(obs)
        if end or truncated:
            break
    assert env.report().filled_qty == 3
    assert env.report().leftover_qty == 99997
    assert env.report().net_effective_bps is None
    metrics = execution_metrics(env.sim, env.fills, owner=env.OWNER, start_time=env.t0,
                                first_trade=env.first_trade, report=env.report())
    assert metrics["hypothetical_residual_midpoint_cost"] is None


def test_normalization_fit_uses_only_registered_training_domain_and_is_frozen():
    design = tiny_design(protocol_version="v04", normalization_seeds=[1900100, 1900101])
    fit = study.fit_training_normalization(design)
    changed = design.model_copy(update={"evaluation_seeds": [76010, 76011]})
    assert study.fit_training_normalization(changed) == fit
    env = study.environment(design, "main", training=False, normalization=fit["normalization"])
    before = env.unwrapped.normalization.as_dict()
    for seed in changed.evaluation_seeds:
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
    assert env.unwrapped.normalization.as_dict() == before
    with pytest.raises(ValueError, match="training market domain"):
        tiny_design(protocol_version="v04", normalization_seeds=[73001, 73002])


@pytest.fixture
def registered_v04(tmp_path, monkeypatch):
    name = "train_rl.py"
    monkeypatch.setattr(study, "source_manifest", lambda: {name: sha256_file(study.PROJECT_ROOT / name)})
    return study.register(tiny_design(protocol_version="v04", normalization_seeds=[1900100, 1900101]), tmp_path / "study")


def test_completion_is_not_conditioned_on_valid_economics_and_both_endpoints_corrected(registered_v04):
    _, plan, design = study.read_study(registered_v04)
    rows = synthetic_rows(design)
    for row in rows:
        row.update(actual_completion=True, time_to_completion=.3, realized_fill_cost_bps=1)
    failed = next(row for row in rows if row["agent"] == "ppo")
    failed.update(status="INVALID", net_effective_bps=None, actual_completion=False, terminal_inventory=20,
                  time_to_completion=None)
    result = study.summarize(rows, plan, design)
    assert plan["family_size"] == 96
    first = result["comparisons"][0]
    assert first["status"] == "WITHHELD"
    assert first["completion_comparison"]["status"] == "AVAILABLE"
    assert first["completion_comparison"]["confidence"] == pytest.approx(1 - .05 / 96)
    assert not first["joint_success_gate_passed"]
    summary = next(row for row in result["summaries"] if row["agent"] == "ppo")
    assert summary["actual_completion_rate"] == .75
    constant = next(c for c in result["comparisons"] if c["agent"] == "dqn")
    assert constant["completion_comparison"]["status"] == "INCONCLUSIVE"
    assert not constant["joint_success_gate_passed"]


def test_v04_real_training_evaluation_and_normalization_integrity(registered_v04):
    pytest.importorskip("stable_baselines3")
    result = study.train(registered_v04)
    assert len(result["models"]) == 12
    assert all(model["normalization_sha256"] for model in result["models"])
    evaluated = study.evaluate(registered_v04)
    assert evaluated["actual_episodes"] == 108
    assert study.verify(registered_v04)["valid"]
    result_path = registered_v04 / "result.json"
    corrupted = json.loads(result_path.read_text(encoding="utf-8"))
    corrupted["comparisons"][0]["mean_delta_bps"] = 9999
    result_path.write_text(json.dumps(corrupted), encoding="utf-8")
    manifest_path = registered_v04 / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["result.json"] = sha256_file(result_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "result differs from registered paired episode summary" in study.verify(registered_v04)["issues"]
    (registered_v04 / "normalization.json").write_text("{}", encoding="utf-8")
    assert not study.verify(registered_v04)["valid"]


def test_invalid_completion_and_normalization_contracts():
    with pytest.raises(ValueError):
        CompletionConstraint(True, 1.1)
    with pytest.raises(ValueError):
        ObservationNormalization((0.,) * 32, (0.,) * 32)

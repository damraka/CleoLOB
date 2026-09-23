import json

import numpy as np
import pytest

from lob import policy_study as study
from lob.config import ResearchConfig
from lob.experiments.registry import sha256_file


def tiny_design(**changes):
    return study.PolicyDesign(config=ResearchConfig(
        market={"limit_rate": 0.0, "market_rate": 0.0, "cancel_rate": 0.0,
                "resilience": 0.0, "latency_base": 0.0, "latency_jitter": 0.0},
        execution={"quantity": 20, "horizon": 0.5, "decision_dt": 0.1, "warmup_seconds": 0.0}),
        training_seeds=[83001, 83002], evaluation_seeds=[73001, 73002],
        identification_seeds=[53001, 53002], timesteps=256, bootstrap_samples=100,
        shifted_market={}, stress_market={}, **changes)


@pytest.fixture
def registered(tmp_path, monkeypatch):
    # Independent tests remain valid while unrelated modules are edited.
    name = "train_rl.py"
    monkeypatch.setattr(study, "source_manifest", lambda: {name: sha256_file(study.PROJECT_ROOT / name)})
    return study.register(tiny_design(), tmp_path / "policy")


@pytest.mark.parametrize("changes", [
    {"training_seeds": [1, 1]}, {"training_seeds": [True, 2]},
    {"evaluation_seeds": [83001, 83002]}, {"evaluation_seeds": [1000000, 1000001]},
    {"evaluation_seeds": [10, 1010]}, {"identification_seeds": [74000, 74001]},
    {"timesteps": 257}, {"shifted_market": {"fake_market_feature": 2.0}},
])
def test_seed_domains_and_finite_design_reject_leakage(changes):
    with pytest.raises(ValueError):
        study.PolicyDesign(**changes)


def test_plan_config_and_source_seal_detect_changes(registered):
    _, plan, _ = study.read_study(registered)
    assert plan["family_size"] == 48
    assert plan["dataset"]["calibrated"] is False
    assert "executable" not in plan["runtime"]
    path = registered / "preregistration.json"
    plan["design"]["timesteps"] = 512
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="plan changed"):
        study.read_study(registered)


def test_snapshot_corruption_rejected(registered):
    (registered / "source" / "train_rl.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source snapshot"):
        study.read_study(registered)


def test_environment_audit_observation_reward_inventory_and_fees():
    design = tiny_design()
    env = study.environment(design, "main", training=False)
    observation, info = env.reset(seed=73001)
    raw = env.unwrapped
    assert env.observation_space.shape == (24,)
    assert env.action_space.n == 5
    assert np.array_equal(observation, raw._obs())
    assert observation[-2:].tolist() == [1.0, 1.0]
    assert info["market_seed"] == 73001
    assert raw.fees.taker_bps == 1.0
    reward = 0.0
    while True:
        _, r, done, truncated, info = env.step(4)
        reward += r
        assert 0 <= raw.remaining <= raw.total_qty
        assert raw.risk.outstanding(raw.sim) <= raw.remaining
        if done or truncated:
            break
    report = raw.report()
    assert reward == pytest.approx(-report.optimization_cost_bps)
    assert info["status"] == "VALID"
    assert info["total_fees"] > 0
    assert report.decision_duration <= raw.horizon
    with pytest.raises(RuntimeError, match="episode ended"):
        env.step(0)
    env.close()


def test_terminal_ablation_preserves_economic_endpoint_and_mask_is_current_only():
    design = tiny_design()
    outcomes = []
    for arm in ("main", "no_terminal", "no_book"):
        env = study.environment(design, arm, training=False)
        obs, _ = env.reset(seed=73001)
        if arm == "no_book":
            assert not obs[:22].any()
            assert obs[-2:].tolist() == [1.0, 1.0]
        reward = 0.0
        while True:
            _, r, done, truncated, _ = env.step(0)
            reward += r
            if done or truncated:
                break
        outcomes.append((env.unwrapped.report(), reward))
    assert outcomes[0][0].net_effective_bps == outcomes[1][0].net_effective_bps
    assert outcomes[1][1] - outcomes[0][1] == pytest.approx(25.0)
    assert outcomes[0][0].filled_qty == 0
    assert outcomes[0][0].hypothetical_liquidation_cost > 0


def test_market_randomization_disjoint_and_reproducible():
    env = study.environment(tiny_design(), "main", training=True)
    first = [env.reset(seed=83001)[1]["market_seed"], env.reset()[1]["market_seed"]]
    second = [env.reset(seed=83001)[1]["market_seed"], env.reset()[1]["market_seed"]]
    assert first == second
    assert first[0] != first[1]
    assert all(study.TRAIN_MARKET_RANGE[0] <= s < study.TRAIN_MARKET_RANGE[1] for s in first)


def synthetic_rows(design):
    rows = []
    for regime in study.REGIMES:
        cells = [(c, "main", None) for c in study.CONTROLS]
        cells += [(a, arm, s) for a in study.ALGORITHMS for arm in study.ARMS for s in design.training_seeds]
        for agent, arm, seed in cells:
            for market in design.evaluation_seeds:
                value = 3.0 if seed is None else (1.0 if seed == design.training_seeds[0] else 3.0)
                rows.append({"regime": regime, "agent": agent, "arm": arm, "training_seed": seed,
                             "seed": market, "status": "VALID", "net_effective_bps": value,
                             "gross_effective_bps": value - 1, "terminal_inventory": 0, "fill_frac": 1.0})
    return rows


def test_crossed_uncertainty_seed_sensitivity_and_full_planned_family(registered):
    _, plan, design = study.read_study(registered)
    rows = synthetic_rows(design)
    result = study.summarize(rows, plan, design)
    assert result["actual_episodes"] == result["planned_episodes"] == 108
    assert len(result["comparisons"]) == 48
    first = result["comparisons"][0]
    assert first["mean_delta_bps"] == -1.0
    assert first["confidence"] == pytest.approx(1 - .05 / 48)
    assert first["ci_low_bps"] <= -2.0
    assert first["ci_high_bps"] >= 0.0
    learned = next(r for r in result["summaries"] if r["agent"] == "dqn")
    assert learned["mean_bps"] == 2.0
    assert learned["training_seed_means_bps"] == [1.0, 3.0]
    assert learned["training_seed_sd_bps"] == pytest.approx(np.sqrt(2))


def test_invalid_or_missing_rows_withhold_only_affected_comparisons(registered):
    _, plan, design = study.read_study(registered)
    rows = synthetic_rows(design)
    failed = next(r for r in rows if r["agent"] == "ppo")
    failed.update(status="INVALID", net_effective_bps=None)
    result = study.summarize(rows, plan, design)
    assert result["invalid_episodes"] == 1
    assert sum(c["status"] == "WITHHELD" for c in result["comparisons"]) == 8
    rows.remove(failed)
    missing = study.summarize(rows, plan, design)
    assert sum(c["status"] == "WITHHELD" for c in missing["comparisons"]) == 8
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        study.summarize(rows, plan, design)


def test_incomplete_classical_market_grid_withholds_summary_interval(registered):
    _, plan, design = study.read_study(registered)
    rows = synthetic_rows(design)
    rows.remove(next(r for r in rows if r["regime"] == "original" and r["agent"] == "twap"))
    result = study.summarize(rows, plan, design)
    summary = next(r for r in result["summaries"] if r["regime"] == "original" and r["agent"] == "twap")
    assert summary["mean_bps"] == 3.0
    assert "mean_ci_bps" not in summary
    assert "incomplete" in summary["mean_ci_withheld"]
    assert sum(c["status"] == "WITHHELD" for c in result["comparisons"]) == 2


def test_warning_economics_are_retained_and_reported(registered):
    _, plan, design = study.read_study(registered)
    rows = synthetic_rows(design)
    rows[0].update(status="WARNING", net_effective_bps=20.0)
    result = study.summarize(rows, plan, design)
    summary = next(r for r in result["summaries"] if r["regime"] == "original" and r["agent"] == "twap")
    assert summary["mean_bps"] == 11.5
    assert summary["valid_episodes"] == 2
    assert summary["warning_episodes"] == result["warning_episodes"] == 1
    assert result["invalid_episodes"] == 0
    assert all(c["status"] == "AVAILABLE" for c in result["comparisons"])


@pytest.mark.parametrize("change", [{"regime": "unplanned"}, {"seed": 99001},
                                    {"agent": "other"}, {"training_seed": 83003}])
def test_unregistered_cells_cannot_enter_summaries(registered, change):
    _, plan, design = study.read_study(registered)
    rows = synthetic_rows(design)
    rows[0].update(change)
    with pytest.raises(ValueError, match="unregistered evaluation"):
        study.summarize(rows, plan, design)


def test_failed_ac_identification_remains_visible(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unidentified")
    monkeypatch.setattr(study, "estimate_ac_parameters", fail)
    design = tiny_design()
    fits = study.fit_controls(design)
    assert all(f["fit"]["status"] == "FAIL" for f in fits.values())
    assert all(f["selected"]["risk_aversion"] == design.config.execution.risk_aversion for f in fits.values())


def test_real_ppo_dqn_training_frozen_evaluation_and_artifact_integrity(registered):
    pytest.importorskip("stable_baselines3")
    result = study.train(registered)
    assert len(result["models"]) == 12
    assert result["total_timesteps"] == 3072
    assert {m["algorithm"] for m in result["models"]} == {"ppo", "dqn"}
    assert all(m["reward_summary"]["episodes"] > 0 for m in result["models"])
    evaluated = study.evaluate(registered)
    assert evaluated["actual_episodes"] == evaluated["planned_episodes"] == 108
    assert evaluated["invalid_episodes"] == 0
    assert study.verify(registered)["valid"]
    with pytest.raises(FileExistsError, match="preserved evaluation"):
        study.evaluate(registered)
    model = registered / "models" / "dqn-main-83001.zip"
    with model.open("ab") as handle:
        handle.write(b"changed")
    assert not study.verify(registered)["valid"]


def test_training_failure_is_recorded_and_seed_never_recycled(registered, monkeypatch):
    pytest.importorskip("stable_baselines3")
    def fail(*args, **kwargs):
        raise ValueError("injected environment failure")
    monkeypatch.setattr(study, "environment", fail)
    with pytest.raises(ValueError, match="injected"):
        study.train(registered)
    failure = json.loads((registered / "models" / "ppo-main-83001.failure.json").read_text())
    assert "injected" in failure["error"]
    with pytest.raises(FileExistsError, match="preserved existing"):
        study.train(registered)
    assert not (registered / "training_summary.json").exists()


def test_dqn_save_load_action_contract_and_reproducibility(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    design = tiny_design()
    models = []
    for _ in range(2):
        env = study.environment(design, "no_book", training=True)
        model = sb3.DQN("MlpPolicy", env, seed=83001, learning_starts=32,
                        buffer_size=256, batch_size=32, train_freq=4, device="cpu")
        model.learn(256)
        models.append(model)
    assert all(torch.equal(v, models[1].policy.state_dict()[k]) for k, v in models[0].policy.state_dict().items())
    path = tmp_path / "dqn.zip"
    models[0].save(path)
    restored = sb3.DQN.load(path, device="cpu")
    obs, _ = study.environment(design, "main", training=False).reset(seed=73001)
    action = study.EvaluationPolicy(restored, "no_book").predict(obs)[0]
    assert action == models[0].predict(study.mask_observation(obs, "no_book"), deterministic=True)[0]

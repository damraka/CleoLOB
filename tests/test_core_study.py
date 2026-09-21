import json
from pathlib import Path

import numpy as np
import pytest

from lob import core_study as study
from lob.config import ResearchConfig
from lob.experiments.registry import seal_experiment, sha256_file, write_json
from lob.runner import cfg_from_params


def tiny_config(**execution):
    return ResearchConfig(
        market={"limit_rate": 0.0, "market_rate": 0.0, "cancel_rate": 0.0,
                "resilience": 0.0, "latency_base": 0.0, "latency_jitter": 0.0},
        execution={"quantity": 20, "horizon": 0.5, "decision_dt": 0.1,
                   "warmup_seconds": 0.0, **execution})


@pytest.fixture
def registered(tmp_path, monkeypatch):
    # Freeze a real, small source snapshot so concurrent unrelated repository
    # edits cannot invalidate a test fixture's prospective registration.
    name = "train_rl.py"
    monkeypatch.setattr(study, "source_manifest", lambda: {name: sha256_file(study.PROJECT_ROOT / name)})
    out = study.register_study(tiny_config(), tmp_path / "study", timesteps=256,
                               min_markets=20, max_markets=20, bootstrap_samples=100)
    return out


def test_training_and_evaluation_share_market_fees_risk_and_warmup():
    config = tiny_config(warmup_seconds=30.0)
    train = study.make_environment(config, 5.0, training=True)
    evaluation = study.make_environment(config, 5.0, training=False)
    params = study.runner_parameters(config, 55, 5.0)
    assert train.warmup == evaluation.warmup == params["warmup_seconds"] == 30.0
    assert train.base_cfg == evaluation.base_cfg
    assert cfg_from_params(params).__dict__ == {**train.base_cfg.__dict__, "seed": 55}
    assert train.fees == evaluation.fees
    assert train.risk_config == evaluation.risk_config
    assert train.terminal_penalty_bps == evaluation.terminal_penalty_bps == 5.0
    assert train.seed_range == study.TRAIN_MARKET_RANGE
    assert evaluation.seed_range is None


def test_seed_domains_are_disjoint_and_training_reproducible():
    env = study.make_environment(tiny_config(), 0.0, training=True)
    _, a = env.reset(seed=81001)
    _, b = env.reset(seed=81001)
    assert a["market_seed"] == b["market_seed"]
    assert study.TRAIN_MARKET_RANGE[0] <= a["market_seed"] < study.TRAIN_MARKET_RANGE[1]
    assert set(study.DIAGNOSTIC_SEEDS).isdisjoint(range(study.TEST_SEED_START, study.TEST_SEED_START + 256))
    assert study.TEST_SEED_START + 256 < study.TRAIN_MARKET_RANGE[0]


def test_registration_is_immutable_and_source_snapshot_checked(registered):
    out, plan, config = study._read_study(registered)
    assert plan["timesteps_per_model"] == 256
    assert len(plan["training_seeds"]) == 5
    assert plan["penalties_bps"] == [0.0, 5.0, 25.0, 100.0]
    assert plan["ac_secondary_risk_aversion"] == pytest.approx(
        config.execution.temp_impact / (config.execution.sigma * config.execution.horizon) ** 2)
    with pytest.raises(FileExistsError):
        study.register_study(config, out)
    (out / "source" / "train_rl.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot changed"):
        study._read_study(out, check_source=False)


def test_registration_seal_rejects_changed_plan(registered):
    path = registered / "preregistration.json"
    plan = json.loads(path.read_text())
    plan["primary_penalty_bps"] = 100.0
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="preregistration"):
        study._read_study(registered)


@pytest.mark.parametrize("budget", [0, True, 255, 300])
def test_registration_rejects_ambiguous_rollout_budgets(tmp_path, budget):
    with pytest.raises(ValueError, match="timesteps"):
        study.register_study(tiny_config(), tmp_path / "invalid", timesteps=budget)


def test_crossed_bootstrap_preserves_training_variability():
    delta = np.repeat(np.array([-4.0, -2.0, 0.0, 2.0, 4.0])[:, None], 200, axis=1)
    result = study.crossed_interval(delta, samples=1000)
    assert result == study.crossed_interval(delta, samples=1000)
    assert result["ci_low_bps"] < -1
    assert result["ci_high_bps"] > 1
    assert result["mean_delta_bps"] == 0
    assert result["training_seeds"] == 5


def test_power_retains_training_variance_floor_and_caps():
    delta = np.repeat(np.array([-4.0, -2.0, 0.0, 2.0, 4.0])[:, None], 20, axis=1)
    result = study.pilot_power(delta)
    assert result["required_market_count"] is None
    assert result["capped"]
    assert result["final_market_count"] == 256
    assert result["approximate_achievable_power"] < 0.8
    flat = study.pilot_power(np.ones((5, 20)))
    assert flat["final_market_count"] == 32
    assert flat["approximate_achievable_power"] == 1


@pytest.mark.parametrize("bad", [np.ones(10), np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones((1, 10))])
def test_crossed_statistics_reject_unidentified_design(bad):
    with pytest.raises(ValueError):
        study.crossed_interval(bad)
    with pytest.raises(ValueError):
        study.pilot_power(bad)


def test_invalid_proxy_is_only_for_settled_missing_depth():
    row = {"status": "INVALID", "net_effective_bps": None, "gross_cost": 2.0,
           "total_fees": 1.0, "arrival": 100.0, "target_quantity": 100,
           "fill_frac": 0.5, "invalid_reasons": ["insufficient_terminal_depth"],
           "settlement_complete": True, "outstanding_qty": 0, "settlement_pending_order_ids": []}
    assert np.isnan(study._outcome(row))
    assert study._outcome(row, 100.0) == 53.0
    for change in ({"settlement_complete": False}, {"outstanding_qty": 2},
                   {"invalid_reasons": ["execution_exception"]}, {"settlement_pending_order_ids": [1]}):
        assert np.isnan(study._outcome({**row, **change}, 100.0))


def test_reward_composition_separates_penalty_from_economic_cost():
    summary = study.reward_summary([
        {"reward_terms": {"fills": -2.0, "completion_penalty": -6.0}, "reward": -8.0, "status": "VALID"},
        {"reward_terms": {"fills": 2.0, "completion_penalty": -6.0}, "reward": -4.0, "status": "VALID"},
    ])
    assert summary["mean_reward_terms_bps"]["fills"] == 0.0
    assert summary["completion_penalty_absolute_share"] == 0.75
    assert summary["invalid_episodes"] == 0


def test_resume_reuses_evaluation_cells_and_checks_journal(registered, monkeypatch):
    sb3 = pytest.importorskip("stable_baselines3")
    _, plan, config = study._read_study(registered)
    plan["training_seeds"] = [1, 2]
    monkeypatch.setattr(sb3.PPO, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(study, "_check_model", lambda *args: (Path("unused.zip"), {}))
    calls = []

    def run(agent, params, model=None):
        calls.append((agent, params["seed"]))
        return {"agent": agent, "seed": params["seed"], "status": "VALID", "net_effective_bps": 1.0}

    monkeypatch.setattr(study, "run_episode", run)
    first = study._run_rows(registered, plan, config, [100, 101], [0.0], "final")
    assert len(first) == 8  # AC, kappa*T=1 AC and two PPO fits, each on two paths.
    assert len(calls) == 8
    second = study._run_rows(registered, plan, config, [100, 101], [0.0], "final")
    assert second == first
    assert len(calls) == 8
    path = registered / "final_episodes.jsonl"
    path.write_text(path.read_text().replace('"net_effective_bps":1.0', '"net_effective_bps":2.0'), encoding="utf-8")
    with pytest.raises(ValueError, match="journal changed"):
        study._run_rows(registered, plan, config, [100, 101], [0.0], "final")


def test_invalid_training_stops_and_preserves_failure(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    out = study.register_study(tiny_config(quantity=1_000_000), tmp_path / "invalid", timesteps=256)
    with pytest.raises(ValueError, match="INVALID training"):
        study.train_study(out)
    failure = json.loads((out / "models" / "penalty-0-seed-81001.failure.json").read_text())
    assert failure["episodes"][-1]["status"] == "INVALID"
    assert not (out / "models" / "penalty-0-seed-81001.zip").exists()
    with pytest.raises(ValueError, match="preserved failed"):
        study.train_study(out)


def test_sb3_train_save_load_contract_uses_same_resolved_environment(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    config = tiny_config()
    train = study.make_environment(config, 5.0, training=True)
    model = sb3.PPO("MlpPolicy", train, seed=81001, n_steps=64, batch_size=32,
                    n_epochs=1, device="cpu", verbose=0)
    model.learn(total_timesteps=256)
    path = tmp_path / "contract.zip"
    model.save(path)
    evaluation = study.make_environment(config, 5.0, training=False)
    restored = sb3.PPO.load(path, env=evaluation, device="cpu")
    observation, _ = evaluation.reset(seed=71000)
    original, _ = model.predict(observation, deterministic=True)
    loaded, _ = restored.predict(observation, deterministic=True)
    assert np.array_equal(original, loaded)
    total = 0.0
    while True:
        action, _ = restored.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = evaluation.step(int(action))
        total += reward
        if terminated or truncated:
            break
    assert info["status"] == "VALID"
    assert total == pytest.approx(-evaluation.report().optimization_cost_bps)
    assert info["market_seed"] == 71000
    train.close()
    evaluation.close()


def test_verifier_detects_missing_models_and_artifact_corruption(registered):
    seal_experiment(registered)
    assert not study.verify_study(registered)["valid"]
    (registered / "unexpected.txt").write_text("unsealed", encoding="utf-8")
    result = study.verify_study(registered)
    assert not result["valid"]
    assert "sealed artifact file set differs" in result["issues"]


def test_evaluation_locks_power_and_corrects_five_comparisons(registered, monkeypatch):
    pytest.importorskip("torch")
    write_json(registered / "training_summary.json", {"models": [], "total_timesteps": 0})
    monkeypatch.setattr(study, "_check_model", lambda *args: (Path("unused.zip"), {}))

    def run_rows(out, plan, config, markets, penalties, phase):
        if phase == "final":
            locked = json.loads((out / "power.json").read_text())
            assert locked["final_market_seeds"] == markets
        # A stable known improvement isolates inference and its family size.
        rows = [{"agent": agent, "training_seed": None, "seed": seed,
                 "penalty_bps": 0.0, "status": "VALID", "net_effective_bps": 3.0}
                for agent in (("ac", "ac_risk") if phase == "final" else ("ac",))
                for seed in markets]
        rows += [{"agent": "ppo", "training_seed": training_seed, "seed": seed,
                  "penalty_bps": penalty, "status": "VALID", "net_effective_bps": 2.0,
                  "fill_frac": 1.0, "completion_penalty_bps": 0.0}
                 for penalty in penalties for training_seed in plan["training_seeds"]
                 for seed in markets]
        (out / f"{phase}_episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return rows

    monkeypatch.setattr(study, "_run_rows", run_rows)
    result = study.evaluate_study(registered)
    assert result["planned_final_episodes"] == result["actual_final_episodes"] == 440
    assert result["primary"]["economic_interval"]["mean_delta_bps"] == -1.0
    assert result["primary"]["economic_interval"]["confidence"] == 0.95
    assert result["primary"]["family_interval"]["confidence"] == 0.99
    assert result["ac_risk_sensitivity"]["family_interval"]["confidence"] == 0.99
    assert result["invalid_final_episodes"] == 0
    assert study.verify_study(registered)["valid"]
    with pytest.raises(FileExistsError, match="already complete"):
        study.evaluate_study(registered)

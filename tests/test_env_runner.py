"""RL environment, runner payloads, scenarios and statistics tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from lob.rl_env import LOBExecutionEnv
from lob.runner import AGENTS, BASELINES, cfg_from_params, run_episode, run_pair
from lob.scenarios import DEFAULT_PARAMS, SCENARIOS, scenario_params
from lob.stats import bootstrap_ci, paired_vs_reference, sign_test_p, summarize


# ------------------------------------------------------------- environment
def test_env_passes_gymnasium_checker():
    check_env(LOBExecutionEnv(total_qty=2000, horizon=10.0), skip_render_check=True)


def test_env_reset_is_deterministic_per_seed():
    e1, e2 = LOBExecutionEnv(total_qty=2000, horizon=10.0), LOBExecutionEnv(total_qty=2000, horizon=10.0)
    o1, _ = e1.reset(seed=5)
    o2, _ = e2.reset(seed=5)
    assert np.array_equal(o1, o2) and o1.shape == (24,)
    o3, _ = e1.reset(seed=6)
    assert not np.array_equal(o1, o3)


def test_env_episode_terminates_and_accounts_inventory():
    env = LOBExecutionEnv(total_qty=3000, horizon=15.0)
    obs, _ = env.reset(seed=1)
    done, steps = False, 0
    while not done:
        obs, r, term, trunc, info = env.step(4)         # always market
        done = term or trunc
        steps += 1
    assert info["remaining"] == 0 and steps <= env.n_steps
    assert sum(t.qty for t in env.fills) == 3000
    assert env.report("x").filled_qty == 3000


def test_env_waiting_forever_is_penalised_at_horizon():
    env = LOBExecutionEnv(total_qty=3000, horizon=10.0)
    env.reset(seed=1)
    total, done = 0.0, False
    while not done:
        _, r, term, trunc, info = env.step(0)            # never trade
        total += r
        done = term or trunc
    assert info["remaining"] == 3000
    assert total <= -25.0                                # hard leftover penalty applied


# ------------------------------------------------------------- runner
def test_cfg_from_params_maps_ui_fields_and_sim_overrides():
    p = {**DEFAULT_PARAMS, "latency_ms": 20, "sim": {"limit_rate": 15.0}}
    cfg = cfg_from_params(p)
    assert cfg.latency_base == pytest.approx(0.01) and cfg.limit_rate == 15.0
    with pytest.raises(KeyError):
        cfg_from_params({**DEFAULT_PARAMS, "sim": {"not_a_field": 1}})


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_params_are_complete(name):
    p = scenario_params(name, seed=3)
    assert set(DEFAULT_PARAMS) <= set(p) and p["seed"] == 3
    cfg_from_params(p)                                      # must build a valid SimConfig


def test_run_pair_payload_contract():
    p = scenario_params("calm", seed=42, horizon=10.0, qty=2000)
    out = run_pair(p)
    assert set(out) == {"tick", "params", "decision_dt", "runs"}
    for key in ("baseline", "rl"):
        run = out["runs"][key]
        s = run["series"]
        n = len(s["t"])
        assert n >= 2 and all(len(s[k]) == n for k in s)
        assert s["remaining"][0] == 2000 and s["t"][0] == 0.0
        assert all(len(lvl) <= 12 for lvl in s["bids"])
        assert {"shortfall_bps", "effective_bps", "filled", "target"} <= set(run["raw"])
        assert run["report"]["Strategy"] == run["label"]


@pytest.mark.parametrize("baseline", BASELINES)
def test_run_pair_honours_selected_baseline(baseline):
    p = scenario_params("calm", seed=1, horizon=10.0, qty=2000, baseline=baseline)
    label = run_pair(p)["runs"]["baseline"]["label"]
    assert label.lower().replace("-", "").startswith(baseline.replace("ac", "almgren"))


@pytest.mark.parametrize("agent", AGENTS)
def test_run_episode_row_contract(agent):
    row = run_episode(agent, scenario_params("calm", seed=2, horizon=10.0, qty=2000))
    assert row["agent"] == agent and row["seed"] == 2
    assert 0.0 <= row["fill_frac"] <= 1.0
    assert np.isfinite(row["shortfall_bps"]) and np.isfinite(row["effective_bps"])
    assert row["effective_bps"] >= row["shortfall_bps"] - 1e-9 or row["fill_frac"] == 1.0


def test_same_seed_gives_same_episode_twice():
    p = scenario_params("calm", seed=9, horizon=10.0, qty=2000)
    assert run_episode("twap", p) == run_episode("twap", p)


# ------------------------------------------------------------- statistics
def test_bootstrap_ci_contains_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small, large = rng.normal(5, 1, 20), rng.normal(5, 1, 2000)
    lo, hi = bootstrap_ci(small)
    assert lo <= small.mean() <= hi
    lo2, hi2 = bootstrap_ci(large)
    assert (hi2 - lo2) < (hi - lo)
    assert bootstrap_ci([3.0]) == (3.0, 3.0)
    assert all(np.isnan(v) for v in bootstrap_ci([]))


def test_sign_test_values():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0
    assert sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert sign_test_p(9, 1) == pytest.approx(2 * 11 / 1024)


def _episodes(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for seed in range(n):
        base = rng.normal(4, 1)
        rows.append(dict(scenario="calm", agent="ac", label="AC", seed=seed,
                         effective_bps=base, shortfall_bps=base, fill_frac=1.0, children=20))
        rows.append(dict(scenario="calm", agent="good", label="Good", seed=seed,
                         effective_bps=base - 1.0, shortfall_bps=base - 1.0, fill_frac=1.0, children=20))
        rows.append(dict(scenario="calm", agent="noise", label="Noise", seed=seed,
                         effective_bps=base + rng.normal(0, 0.01), shortfall_bps=base,
                         fill_frac=1.0, children=20))
    return pd.DataFrame(rows)


def test_summarize_and_paired_comparison_detect_a_real_edge():
    df = _episodes()
    s = summarize(df).set_index("agent")
    assert s.loc["good", "mean"] == pytest.approx(s.loc["ac", "mean"] - 1.0)
    assert s.loc["good", "ci_lo"] <= s.loc["good", "mean"] <= s.loc["good", "ci_hi"]
    pr = paired_vs_reference(df, reference="ac").set_index("agent")
    assert pr.loc["good", "delta_mean"] == pytest.approx(-1.0)
    assert pr.loc["good", "win_rate"] == 1.0 and pr.loc["good", "p_sign"] < 1e-6
    assert bool(pr.loc["good", "significant_5pct"])
    assert not bool(pr.loc["noise", "significant_5pct"])
    assert "ac" not in pr.index

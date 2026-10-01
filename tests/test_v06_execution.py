"""v0.6 execution analysis: cells, equifinality, rank stability, execution-sensitive realism, model risk."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lob.v06 import execution as ex
from lob.v06.misspecification import interventions
from lob.v06.worlds import World, episode_params, research_config

MARKETS = list(range(40))
MANDATE = {"side": "buy", "quantity": 14, "horizon_s": 120.0, "decision_dt_s": 6.0, "warmup_s": 60.0,
           "settlement_timeout_s": 5.0, "fees": {"maker_bps": 0.0, "taker_bps": 1.0}, "terminal_penalty_bps": 25.0,
           "completion_urgency_fraction": 0.8, "pov_participation": 0.3}


def _rows(world: str, means: dict[str, float], *, noise: float = 0.5, seed: int = 0, invalid: dict | None = None):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 2.0, len(MARKETS))       # common market shock: pairs are paired
    rows = []
    for agent, mean in means.items():
        seeds = [1, 2] if ":" in agent else [None]
        for ts in seeds:
            for i, m in enumerate(MARKETS):
                bad = invalid and invalid.get(agent, 0) > i
                rows.append({"world": world, "agent": agent, "training_seed": ts, "seed": m,
                             "status": "INVALID" if bad else "VALID",
                             ex.COST: None if bad else float(mean + common[i] + rng.normal(0, noise)),
                             ex.COMPLETION: True})
    return rows


def _means(shift: float = 0.0) -> dict[str, float]:
    return {"twap": 3.0, "vwap": 3.2, "pov": 4.0 + shift, "ac": 2.5, "ppo:single": 3.5 - shift, "dqn:single": 3.8,
            "ppo:ensemble": 3.4, "dqn:ensemble": 3.9}


def test_cells_average_training_seeds_and_flag_invalid() -> None:
    rows = _rows("w", _means(), invalid={"pov": 3})
    cell_map = ex.cells(rows, MARKETS)
    assert cell_map[("w", "twap")]["status"] == "VALID"
    assert cell_map[("w", "pov")]["status"] == "INVALID"         # 3/40 = 7.5% > 5%
    assert cell_map[("w", "ppo:single")]["cost"].shape == (40,)
    assert set(cell_map[("w", "ppo:single")]["training_seed_means"]) == {"1", "2"}
    summary = ex.summarize_cell(cell_map[("w", "twap")], alpha=0.05, samples=200, seed=1)
    assert summary["mean_ci"]["status"] == "AVAILABLE"
    withheld = ex.paired(cell_map, "w", "pov", "twap", alpha=0.05, samples=100, seed=0)
    assert withheld["status"] == "WITHHELD"


def test_equifinality_detects_material_member_differences() -> None:
    rows = _rows("a", _means()) + _rows("b", {**_means(), "twap": 6.0}, seed=1)
    result = ex.equifinality(ex.cells(rows, MARKETS), ["a", "b"], alpha=0.05, samples=500, seed=3)
    assert result["family_size"] == 4 and result["status"] == "ESTABLISHED"
    assert any(c["agent"] == "twap" and c["material"] for c in result["contrasts"])
    same = _rows("a", _means()) + _rows("b", _means(), seed=1)
    assert ex.equifinality(ex.cells(same, MARKETS), ["a", "b"], alpha=0.05, samples=500, seed=3)["status"] == \
        "NOT_ESTABLISHED"
    assert ex.equifinality({}, ["a"], alpha=0.05, samples=10, seed=0)["status"] == "NOT_AVAILABLE"


def test_rank_stability_certifies_reversals_only_with_intervals() -> None:
    rows = _rows("selected", _means(0.0)) + _rows("m2", _means(2.0), seed=2)
    result = ex.rank_stability(ex.cells(rows, MARKETS), ["selected", "m2"], ex.PRIMARY, alpha=0.05, samples=500,
                               seed=1, reference="selected")
    assert "twap|ppo:single" in result["certified_reversals"] and result["status"] == "ESTABLISHED"
    assert "pov|ppo:single" not in result["certified_reversals"]          # same direction in both worlds
    pair = result["pairs"]["twap|ppo:single"]
    assert pair["fraction_left_costlier"] == 0.5 and pair["fraction_right_costlier"] == 0.5
    assert result["family_size"] == 15 * 2 and result["kendall_tau_min"] < 1
    stable = _rows("selected", _means()) + _rows("m2", _means(), seed=2)
    assert ex.rank_stability(ex.cells(stable, MARKETS), ["selected", "m2"], ex.PRIMARY, alpha=0.05, samples=500,
                             seed=1, reference="selected")["status"] == "NOT_ESTABLISHED"


def test_disagreement_and_execution_sensitive_realism() -> None:
    rows, families, worlds = [], {}, ["selected"]
    rows += _rows("selected", _means(0.0), noise=0.05)
    families["selected"] = {"spread": 0.0, "depth": 0.0}
    for k in range(1, 10):
        name = f"w{k}"
        worlds.append(name)
        rows += _rows(name, _means(0.4 * k), seed=k, noise=0.05)
        families[name] = {"spread": 0.1 * k, "depth": float(np.random.default_rng(k).random())}
    cell_map = ex.cells(rows, MARKETS)
    instability = ex.disagreement(cell_map, worlds, "selected", ex.PRIMARY, alpha=0.05, samples=300, seed=0)
    assert instability["selected"]["conclusion_disagreement"] == 0.0
    sequence = [instability[f"w{k}"]["conclusion_disagreement"] for k in range(1, 10)]
    assert sequence == sorted(sequence) and sequence[-1] > sequence[0]
    rng = np.random.default_rng(4)
    profile = {f"w{k}": {"conclusion_disagreement": k / 10, "mean_abs_shift_bps": float(k)} for k in range(1, 13)}
    errors = {f"w{k}": {"spread": 0.1 * k + rng.normal(0, 0.01), "depth": float(rng.random())} for k in range(1, 13)}
    result = ex.execution_sensitive_realism(errors, profile, families=("spread", "depth"), alpha=0.05, samples=400,
                                            seed=1)
    assert result["passing"] == ["spread"] and result["status"] == "ESTABLISHED"
    assert result["families"]["spread"]["disagreement"]["holm_adjusted_p"] < 0.05
    few = ex.execution_sensitive_realism(families, {"w1": instability["w1"]}, families=("spread",), alpha=0.05,
                                         samples=10, seed=0)
    assert few["status"] == "INCONCLUSIVE"


def test_decomposition_keeps_sources_separate() -> None:
    rows = _rows("selected", _means()) + _rows("m1", {**_means(), "twap": 5.0}, seed=1) + \
        _rows("i1", {**_means(), "twap": 9.0}, seed=2)
    risk = ex.decomposition(ex.cells(rows, MARKETS), selected="selected", members=["selected", "m1"],
                            interventions=["i1"], regime_worlds=[], agents=("twap", "ppo:single"),
                            fill_bounds={"twap": 1.5})
    twap = risk["agents"]["twap"]
    assert twap["calibration_ensemble_sd_bps"] > 1 and twap["structural_intervention_sd_bps"] is None
    assert twap["historical_fill_bound_width_bps"] == 1.5 and twap["training_seed_sd_bps"] is None
    assert risk["agents"]["ppo:single"]["training_seed_sd_bps"] is not None
    assert "not additive" in risk["rule"]


def test_pair_seeds_are_deterministic_and_distinct() -> None:
    assert ex.pair_seed(1, "a", "b") == ex.pair_seed(1, "a", "b") != ex.pair_seed(1, "b", "a")


def _world() -> World:
    from lob.v06.calibration import base_config, spec_from_unit
    spec = spec_from_unit(np.full(14, 0.4), base_config(3.79),
                          {"market_size_quantiles": [1.0] * 21, "limit_size_quantiles": [2.0] * 21})
    return World("selected", "selected", spec.config, spec.extensions)


def test_world_hash_roundtrip_and_tamper() -> None:
    world = _world()
    assert World.from_dict(world.to_dict()) == world
    tampered = {**world.to_dict(), "config": {**world.config, "limit_rate": 99.0}}
    with pytest.raises(ValueError, match="hash"):
        World.from_dict(tampered)


def test_episode_params_carry_world_and_mandate() -> None:
    world = _world()
    config = research_config(world, MANDATE)
    assert config.execution.quantity == 14 and config.market.limit_rate == pytest.approx(world.config["limit_rate"])
    params = episode_params(world, MANDATE, 670001, ac={"temp_impact": 0.1})
    assert params["seed"] == 670001 and params["flow_extensions"]["hawkes_decay"] == world.extensions["hawkes_decay"]
    assert params["completion"]["urgency_fraction"] == 0.8 and params["temp_impact"] == 0.1
    from lob.runner import run_episode
    row = run_episode("twap", params)
    assert row["status"] in {"VALID", "WARNING", "INVALID"} and ex.COST in row


def test_interventions_change_exactly_one_mechanism() -> None:
    world = _world()
    worlds = interventions(world)
    names = [w.name for w in worlds]
    assert len(worlds) == 14 and len(set(names)) == 14
    by = {w.name: w for w in worlds}
    assert by["cancel_x2"].config["cancel_rate"] == pytest.approx(2 * world.config["cancel_rate"])
    assert by["cancel_x2"].extensions == world.extensions
    assert by["impact_x2"].config == world.config
    assert by["impact_x0.5"].extensions["market_size_quantiles"] == [1.0] * 21
    assert by["persistence_removed"].extensions["hawkes_alpha"] == 0.0
    assert by["volatility_regime_x2"].extensions["regime_multiplier"] <= 20
    for w in worlds:
        w.spec.build(1).step(1.0)
    assert not math.isnan(by["depth_x0.5"].config["target_level_vol"])

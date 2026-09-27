"""M7: registered v0.5 policy-study design, endpoints, inference and gates (synthetic journals)."""
from __future__ import annotations

import json
import math

import pytest

from lob import policy_study_v05 as study
from lob.observations import FEATURES, OBSERVATION_CONTRACT


def design(markets=None, **changes):
    """Pilot-sized design with a fast bootstrap for tests, or the frozen design with invalid changes."""
    base, _ = study.load_design()
    if changes:
        return study.V05Design.model_validate_json(json.dumps({**base.model_dump(mode="json"), **changes}))
    document = study.pilot_design(base, training_seeds=2, markets=markets or 8, timesteps=1024).model_dump(mode="json")
    document["bootstrap_samples"] = 200
    return study.V05Design.model_validate_json(json.dumps(document))


def test_frozen_design_loads_exactly_and_matches_protocol_family():
    frozen, digest = study.load_design()
    assert len(frozen.evaluation_seeds) == 160 and study.family_size(frozen) == 128
    assert frozen.regimes == ["original", "shifted", "stress", "calibrated"] and frozen.evidence_level == "research"


@pytest.mark.parametrize("changes", [{"training_seeds": [85001, 85001]}, {"evaluation_seed_start": 85001},
                                     {"normalization_seeds": [5, 6]}, {"timesteps": 1000},
                                     {"regimes": ["stress", "original"]}])
def test_invalid_designs_are_rejected(changes):
    with pytest.raises(ValueError):
        design(**changes)


def test_pilot_is_labelled_and_smaller():
    pilot = design()
    assert pilot.evidence_level == "pilot" and len(pilot.training_seeds) == 2 and pilot.evaluation_seed_count == 8


def test_clopper_pearson_upper_limit():
    alpha = 0.05 / 128
    assert study.clopper_pearson_upper(0, 160, alpha) == pytest.approx(1 - alpha ** (1 / 160), rel=1e-9)
    assert study.clopper_pearson_upper(0, 40, alpha) > 0.05  # too few markets cannot pass the margin
    assert study.clopper_pearson_upper(160, 160, alpha) == 1.0
    assert study.clopper_pearson_upper(5, 160, alpha) > study.clopper_pearson_upper(1, 160, alpha)


def _rows(d, *, agent_cost=1.0, control_cost=2.0, agent_complete=True, invalid=None):
    rows = []
    for regime in d.regimes:
        for agent in study.CONTROLS:
            for m in d.evaluation_seeds:
                rows.append({"regime": regime, "agent": agent, "arm": "main", "training_seed": None, "seed": m,
                             "status": "VALID", study.COST: control_cost + 0.01 * (m % 3), study.COMPLETION: True})
        for algorithm in study.ALGORITHMS:
            for arm in study.ARMS:
                for s in d.training_seeds:
                    for m in d.evaluation_seeds:
                        complete = agent_complete(m, s) if callable(agent_complete) else agent_complete
                        row = {"regime": regime, "agent": algorithm, "arm": arm, "training_seed": s, "seed": m,
                               "status": "VALID", study.COST: agent_cost + 0.01 * ((m + s) % 5),
                               study.COMPLETION: complete}
                        if invalid and (regime, algorithm, arm, s, m) == invalid:
                            row.update(status="INVALID", **{study.COST: None, study.COMPLETION: None})
                        rows.append(row)
    return rows


def _plan(d):
    return {"family_size": study.family_size(d)}


def test_joint_gate_needs_enough_markets_even_when_everything_completes():
    small = design(markets=8)
    result = study.summarize(_rows(small), _plan(small), small)
    assert result["planned_comparisons"] == 64 and result["family_size"] == 128
    cheap = [c for c in result["comparisons"] if c["reference"] == "twap"]
    assert all(c["cost"]["ci_high_bps"] < 0 for c in cheap)
    # 8 markets cannot bound discordance below 5%: never a pass, never faked
    assert result["joint_gates_passed"] == 0


def test_joint_gate_passes_only_with_cost_and_completion_evidence():
    base, _ = study.load_design()
    big = design(markets=160)
    result = study.summarize(_rows(big), _plan(big), big)
    control_pairs = [c for c in result["comparisons"] if c["reference"] in study.CONTROLS]
    assert all(c["joint_success_gate_passed"] for c in control_pairs)
    worse = study.summarize(_rows(big, agent_cost=3.0), _plan(big), big)
    assert not any(c["joint_success_gate_passed"] for c in worse["comparisons"] if c["reference"] in study.CONTROLS)


def test_settlement_only_completion_counts_as_a_miss():
    base, _ = study.load_design()
    big = design(markets=160)
    rows = _rows(big, agent_complete=lambda m, s: not (m % 10 == 0 and s == big.training_seeds[0]))
    result = study.summarize(rows, _plan(big), big)
    twap = next(c for c in result["comparisons"] if c["reference"] == "twap" and c["regime"] == "original"
                and c["agent"] == "ppo:main")
    assert twap["completion"]["discordant_markets"] == 16 and not twap["joint_success_gate_passed"]


def test_invalid_cell_withholds_without_shrinking_family():
    small = design(markets=8)
    rows = _rows(small, invalid=("stress", "dqn", "main", small.training_seeds[0], small.evaluation_seeds[0]))
    result = study.summarize(rows, _plan(small), small)
    stress_dqn = [c for c in result["comparisons"] if c["regime"] == "stress" and c["agent"] == "dqn:main"]
    assert all(c["cost"]["status"] == "WITHHELD" for c in stress_dqn)
    assert result["planned_comparisons"] == 64 and result["invalid_episodes"] == 1


def test_calibrated_regime_requires_frozen_spec():
    frozen, _ = study.load_design()
    with pytest.raises(ValueError, match="NOT_AVAILABLE"):
        study.regime_setup(frozen, "calibrated", None)
    config, ext = study.regime_setup(frozen, "calibrated", {"config": {"limit_rate": 3.0, "tick_size": 0.02,
                                                                      "initial_mid_ticks": 5000,
                                                                      "latency_base": 9.0},
                                                           "extensions": {"imbalance_beta": 0.3}})
    assert config.market.limit_rate == 3.0 and config.market.latency_base == frozen.config.market.latency_base
    assert ext == {"imbalance_beta": 0.3}


def test_observations_exclude_leaking_information():
    forbidden = ("future", "seed", "queue", "hidden", "regime", "rate", "holdout")
    assert not [f for f in FEATURES if any(word in f.lower() for word in forbidden)]
    assert "exact_fifo_position" not in OBSERVATION_CONTRACT.to_dict()["available"]
    assert "hidden_order_quantity" not in OBSERVATION_CONTRACT.to_dict()["available"]


def test_episode_params_carry_identical_completion_and_extensions():
    frozen, _ = study.load_design()
    params = study.episode_params(frozen, "stress", None, 77000, 25.0, None)
    assert params["completion"] == {"enabled": True, "urgency_fraction": 0.8}
    assert params["flow_extensions"] is None and params["observation_version"] == "v04"
    assert math.isclose(params["sim"]["market_rate"], frozen.stress_market["market_rate"])

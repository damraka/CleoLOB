"""v0.7 risk metrics, robust selection, stresses and a planted adversarial world."""
from __future__ import annotations

import numpy as np
import pytest

from lob.sim_v2 import FlowExtensions, SimulatorSpec
from lob.v06.tape import tape_from_simulator
from lob.v07.robustness import analysis, stress
from lob.v07.uncertainty import risk

CONFIG = {"tick_size": 100 / 2640, "initial_mid_ticks": 2640, "target_level_vol": 26, "limit_rate": 2.9,
          "market_rate": 0.14, "cancel_rate": 0.03, "offset_p": 0.05, "limit_qty_mean": 35, "market_qty_mean": 40,
          "resilience": 0.2, "max_events": 3_000_000}
EXT = {"hawkes_alpha": 0.07, "hawkes_decay": 1.9, "imbalance_beta": 0.8}


def test_known_distribution_fixtures() -> None:
    x = np.arange(1, 101, dtype=float)
    assert risk.cvar(x, 0.95) == pytest.approx(98.0)
    assert risk.cvar(x, 0.99) == 100.0
    assert risk.downside_semivariance(np.r_[np.zeros(50), np.ones(50) * 2]) == pytest.approx(0.5)
    m = risk.risk_metrics(x, completion=[1, 1, 0.5, 1], time_to_completion=[1, 2, None, 3], residual_inventory=[0, 2])
    assert m["completion_risk"] == 0.25 and m["inventory_risk_mean_abs"] == 1.0 and m["tail_p95_bps"] > 90


def test_robust_selection_differs_from_expectation_when_tails_differ() -> None:
    means = {"w1": {"safe": 5.0, "risky": 2.0}, "w2": {"safe": 5.0, "risky": 2.0}, "w3": {"safe": 5.2, "risky": 7.0}}
    out = risk.robust_selection(means, q=0.3, radius=0.5)
    assert out["selected"]["expectation"] == "risky" and out["selected"]["worst_case"] == "safe"
    assert out["selected"]["cvar_worlds"] == "safe" and not out["agreement"]
    assert risk.worst_plausible(means["w3"] | {"x": 1.0})["world"] == "risky"


def test_planted_adversarial_world_is_found() -> None:
    rng = np.random.default_rng(0)
    cube = {f"c{i}": {"a": 5 + rng.normal(0, 1, 64), "b": 8 + rng.normal(0, 1, 64)} for i in range(10)}
    cube["planted"] = {"a": 9 + rng.normal(0, 1, 64), "b": 8 + rng.normal(0, 1, 64)}
    found = analysis.worst_plausible(cube, {"a|b": -1}, ("a", "b"))
    assert found["a|b"]["candidate"] == "planted" and found["a|b"]["difference"] > 0


@pytest.mark.parametrize("name", sorted(stress.STRESSES))
def test_stresses_build_valid_labelled_worlds(name) -> None:
    spec = stress.apply(SimulatorSpec(CONFIG, EXT), name)
    FlowExtensions(**spec.extensions)
    tape = tape_from_simulator(spec, 1, seconds=30.0, warmup=5.0)
    assert len(tape.t) == 300
    assert stress.manifest(name, "base").scenario == "SYNTHETIC_STRESS"
    assert stress.stressed_mandate({"fees": {}}, "fee_change")["fees"]["taker_bps"] == 5.0

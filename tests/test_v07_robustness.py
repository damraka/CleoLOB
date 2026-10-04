"""v0.7 model-risk analysis (H7, H8, H9) on constructed cost cubes."""
from __future__ import annotations

import numpy as np

from lob.v07.robustness import analysis as ra

AGENTS = ("a", "b", "c")


def _cube(world_shift: dict, n=200, seed=0, sd=2.0):
    rng = np.random.default_rng(seed)
    return {w: {a: base + shift.get(a, 0.0) + rng.normal(0, sd, n) for a, base in (("a", 10.0), ("b", 12.0),
                                                                                   ("c", 10.2))}
            for w, shift in world_shift.items()}


def test_h7_detects_material_world_spread_only_when_present() -> None:
    spread = _cube({f"w{i}": {"a": 3.0 * i, "b": 3.0 * i, "c": 3.0 * i} for i in range(6)})
    flat = _cube({f"w{i}": {} for i in range(6)}, seed=1)
    assert ra.h7(spread, AGENTS, alpha=0.05, samples=300, seed=2)["status"] == "ESTABLISHED"
    result = ra.h7(flat, AGENTS, alpha=0.05, samples=300, seed=2)
    assert result["status"] == "NOT_ESTABLISHED"
    assert all(not v["material"] or not v["holm_rejected"] for v in result["agents"].values())


def test_h8_edges() -> None:
    cube = _cube({f"w{i}": {} for i in range(5)})
    worst = ra.worst_plausible(_cube({f"x{i}": {} for i in range(4)}, seed=9), {"a|b": -1, "a|c": 0, "b|c": 1}, AGENTS)
    result = ra.h8(cube, AGENTS, alpha=0.05, samples=300, seed=3, worst=worst, reference="w0")
    assert result["edges"]["a|b"]["edge"] == "ROBUSTLY_BETTER"
    assert result["edges"]["b|c"]["edge"] == "ROBUSTLY_WORSE"
    assert result["edges"]["a|c"]["edge"] in {"EQUIVALENT_WITHIN_MARGIN", "INDETERMINATE"}
    assert result["status"] == "ESTABLISHED" and result["family_size"] == 3
    flipped = _cube({"w0": {}, "w1": {}, "w2": {"a": 6.0}}, seed=4)
    dep = ra.h8(flipped, AGENTS, alpha=0.05, samples=300, seed=3, worst=worst)
    assert dep["edges"]["a|b"]["edge"] in {"MODEL_DEPENDENT", "INDETERMINATE"}
    reversed_worst = {**worst, "a|b": {"candidate": "x", "difference": 0.5}}
    rev = ra.h8(cube, AGENTS, alpha=0.05, samples=300, seed=3, worst=reversed_worst)
    assert rev["edges"]["a|b"]["edge"] == "REVERSED"
    no_search = ra.h8(cube, AGENTS, alpha=0.05, samples=300, seed=3, worst=None)
    assert no_search["robust_pairs"] == []


def test_h9_and_topology() -> None:
    cube = _cube({"G0_point": {}, "w1": {"a": 6.0}, "w2": {}}, seed=5)
    h8 = ra.h8(cube, AGENTS, alpha=0.05, samples=200, seed=1, worst={})
    h9 = ra.h9(h8)
    assert h9["status"] == "ESTABLISHED" and "a|b" in h9["flagged_pairs"]
    topo = ra.ranking_topology(cube, AGENTS)
    assert topo["distinct_rankings"] >= 2 and topo["kendall_distance"]["pairs_total"] == 3
    dec = ra.decomposition(cube, AGENTS, {"G0_point": "point", "w1": "post", "w2": "post"})
    assert dec["a"]["market_seed_sd_bps"] > 1 and dec["a"]["model_class_sd_bps"] is not None

"""v0.7 final decision benchmark and cross-cutting analyses on constructed inputs."""
from __future__ import annotations

from pathlib import Path

from lob.v07.reports import coverage, decision

ROOT = Path(__file__).resolve().parents[1]


def _edge(mean, ci, per, edge="INDETERMINATE", opposite=()):
    return {"pooled_mean_bps": mean, "ci": ci, "edge": edge, "opposite_worlds": list(opposite),
            "per_world": {w: {"mean": m, "direction": d} for w, (m, d) in per.items()}}


def test_decision_benchmark_certifies_only_robust_pairs() -> None:
    edges = {"a|b": _edge(-2.0, [-3.0, -1.2], {"w1": (-2.1, -1), "w2": (-1.9, -1), "G2": (-2.0, -1)}, "ROBUSTLY_BETTER"),
             "a|c": _edge(-0.3, [-1.0, 0.4], {"w1": (-0.2, 0), "w2": (-0.4, 0), "G2": (0.1, 0)}),
             "b|c": _edge(1.5, [1.1, 2.0], {"w1": (2.0, 1), "w2": (-1.0, -1), "G2": (1.5, 1)}, "MODEL_DEPENDENT", ["w2"])}
    worst = {"a|b": {"difference": -0.5}, "a|c": {"difference": 0.5}, "b|c": {"difference": 0.4}}
    history = {"a|b": {"conservative": {"direction": -1}, "optimistic": {"direction": -1}}}
    out = decision.decision_benchmark(edges, margin=1.0, mde=0.9, worst=worst, history=history, regime_world="G2",
                                      queue_width_bps=0.7)
    assert out["pairs"]["a|b"]["final_status"] == "ROBUST_ACROSS_MODEL_UNCERTAINTY"
    assert out["pairs"]["a|c"]["final_status"] in {"NOT_ESTABLISHED", "INCONCLUSIVE"}
    assert out["pairs"]["b|c"]["final_status"] == "MODEL_DEPENDENT"
    assert out["pairs"]["a|b"]["fraction_worlds_negative"] == 1.0


def test_complexity_overfitting_scaling_and_value() -> None:
    models = {"G0": {"parameters": 14, "compute_s": 10, "development": 2.2, "selection": 2.5, "fresh": 2.9},
              "G1": {"parameters": 96, "compute_s": 14, "development": 4.0, "selection": 4.1, "fresh": 4.2},
              "G4": {"parameters": 4000, "compute_s": 9, "development": 1.7, "selection": 2.1, "fresh": 2.6}}
    pen = decision.complexity_penalty(models)
    assert pen["models"]["G1"]["status"] == "NOT_ESTABLISHED" and pen["models"]["G4"]["status"] == "ESTABLISHED"
    curve = decision.overfitting_curve(models)
    assert curve["G4"]["deterioration_dev_to_last"] > 0
    assert decision.scaling(models)["status"] == "EXPLORATORY"
    value = decision.compute_value(models, "G0")
    assert value["G1"]["verdict"] == "expensive complexity did not matter"


def test_realism_to_decision_ablation_and_failure_catalogue() -> None:
    edges = {f"p{i}": _edge(0.0, [-1, 1], {f"w{k}": (float(k) * (i + 1), 0) for k in range(6)}) for i in range(4)}
    out = decision.realism_to_decision(edges, {f"w{k}": float(k) for k in range(6)})
    assert out["verdict"] == "PREDICTIVE" and out["median_spearman"] > 0.9
    ab = decision.ablations({"posterior": {"complex": 3.3, "simple": 2.5}, "x": {"complex": None, "simple": 1}})
    assert ab["posterior"]["attribution"] == "component does not help" and ab["x"]["attribution"] == "NOT_AVAILABLE"
    cat = decision.failure_catalogue({"SUPPORT_FAILURE": ["m17"]})
    assert cat["SUPPORT_FAILURE"]["observed"] and len(cat) == 13


def test_requirement_coverage_has_exactly_100_rows() -> None:
    assert [r["id"] for r in coverage.ROWS] == list(range(1, 101))
    table = coverage.render()
    assert table.count("\n") == 101
    assert all((ROOT / p).exists() for r in coverage.ROWS for p in r["tests"])

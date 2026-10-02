"""v0.7 decision certification: every gate combination, abstention, ranking-graph consistency."""
from __future__ import annotations

from itertools import product

import pytest

from lob.v07.robustness.certification import ABSTAIN, Evidence, abstain, certify, ranking_graph


def _e(**kw):
    base = dict(registered=True, estimate=-2.0, ci=(-3.0, -1.0), margin=1.0, mde=0.5, opposite_worlds=[],
                worst_plausible=-0.5, seed_signs=None, regime_signs=None,
                history={"conservative": -1, "optimistic": -1})
    base.update(kw)
    return Evidence(**base)


def test_certified_only_when_every_gate_passes() -> None:
    assert certify(_e())["status"] == "ROBUST_ACROSS_MODEL_UNCERTAINTY"


@pytest.mark.parametrize("registered,significant,material,model_ok,history", list(
    product([True, False], [True, False], [True, False], [True, False], ["both", "one", "opposite", "none"])))
def test_all_logical_combinations(registered, significant, material, model_ok, history) -> None:
    hist = {"both": {"conservative": -1, "optimistic": -1}, "one": {"conservative": -1, "optimistic": 0},
            "opposite": {"conservative": 1, "optimistic": -1}, "none": None}[history]
    e = _e(registered=registered, ci=(-3.0, -1.0) if significant else (-3.0, 0.5),
           estimate=-2.0 if material else -0.5, opposite_worlds=[] if model_ok else ["post_03"], history=hist)
    if not material and significant:
        e.ci = (-0.8, -0.2)
    result = certify(e)
    certified = registered and significant and material and model_ok and history == "both"
    assert result["certified"] is certified
    if not registered:
        assert result["status"] == "INVALID"
    elif not significant:
        assert result["status"] in {"NOT_ESTABLISHED", "INCONCLUSIVE"}
    elif not material:
        assert result["status"] == "NOT_ESTABLISHED"
    elif not model_ok:
        assert result["status"] == "MODEL_DEPENDENT" and result["responsible_worlds"] == ["post_03"]
    elif history == "none":
        assert result["status"] == "INCONCLUSIVE" and ABSTAIN in result["message"]
    elif history != "both":
        assert result["status"] == "ASSUMPTION_DEPENDENT"


def test_underpowered_design_abstains() -> None:
    result = certify(_e(ci=(-2.5, 0.4), mde=2.8))
    assert result["status"] == "INCONCLUSIVE" and "underpowered" in result["message"]
    assert abstain("x")["message"].startswith(ABSTAIN)


def test_equivalence_only_inside_registered_margin() -> None:
    assert certify(_e(ci=(-0.5, 0.4), estimate=0.0, mde=0.3))["equivalence"] == "EQUIVALENT_WITHIN_MARGIN"
    assert certify(_e(ci=(-1.5, 0.4), estimate=0.0, mde=0.3))["equivalence"] == "NOT_ESTABLISHED"


def test_seed_regime_and_worst_plausible_gates() -> None:
    assert certify(_e(worst_plausible=0.3))["status"] == "MODEL_DEPENDENT"
    assert certify(_e(worst_plausible=None))["status"] == "MODEL_DEPENDENT"
    assert certify(_e(seed_signs=[-1, -1, 1]))["failed_gate"] == "seeds"
    assert certify(_e(regime_signs=[-1, 1]))["failed_gate"] == "regime"
    assert certify(_e(seed_signs=[-1, -1], regime_signs=[-1, 0]))["certified"]


def test_ranking_graph_cycles_and_antisymmetry() -> None:
    edges = {"a|b": "ROBUSTLY_BETTER", "b|c": "ROBUSTLY_BETTER", "a|c": "ROBUSTLY_WORSE", "c|d": "INDETERMINATE"}
    graph = ranking_graph(edges)
    assert graph["three_cycles"] == [["a", "b", "c"]] and graph["antisymmetric"]
    acyclic = ranking_graph({"a|b": "ROBUSTLY_BETTER", "b|c": "ROBUSTLY_BETTER", "a|c": "ROBUSTLY_BETTER"})
    assert acyclic["three_cycles"] == [] and acyclic["open_transitive_triples"] == 0
    assert ranking_graph({"a|b": "ROBUSTLY_BETTER", "b|c": "ROBUSTLY_BETTER"})["open_transitive_triples"] == 1

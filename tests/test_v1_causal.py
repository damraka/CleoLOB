"""v1 partial identification and causal estimand registry (M135, M136)."""
import pytest

from lob.v1.causal import bounds as CB


def grid():
    effects, factors = {}, {}
    for q in ("opt", "neu", "con"):
        for imp in ("none", "transient", "persistent"):
            k = f"{q}|{imp}"
            effects[k] = 2.0 - (1.5 if q == "con" else 0) - (0.8 if imp == "persistent" else 0)
            factors[k] = {"queue": q, "impact": imp}
    return effects, factors


def test_bounds_identified_set_and_indeterminate():
    e, _ = grid()
    s = CB.identified_set(e)
    assert s["status"] == "INDETERMINATE" and s["lower"] == pytest.approx(-0.3) and s["upper"] == pytest.approx(2.0)
    assert CB.identified_set({"a": 1.0, "b": 2.0})["status"] == "IDENTIFIED"
    assert CB.identified_set({"a": -1.0, "b": 2.0})["status"] == "INDETERMINATE"
    assert CB.identified_set({})["status"] == "NOT_EVALUABLE"


def test_bounds_factor_importance_and_breakdown():
    e, f = grid()
    imp = CB.factor_importance(e, f)
    assert imp["dominant"] == "queue"
    pos = {k: v + 1.0 for k, v in e.items()}                   # all positive: identified
    b = CB.breakdown(pos, f, {"queue": "con", "impact": "persistent"})
    assert b["status"] == "IDENTIFIED" and b["robustness"] >= 1
    m = CB.manski_fill(-1.0, 3.0, 0.2, 0.6)
    assert m["lower"] == pytest.approx(0.6) and m["upper"] == pytest.approx(2.2)


def test_estimands_registry_blocks_unidentified_claims():
    assert not CB.check("own_price_impact")["allowed"]
    assert not CB.check("queue_position", capability={"queue_observability": "aggregate_level"})["allowed"]
    assert CB.check("queue_position", capability={"queue_observability": "exact_order_id"},
                    diagnostics={"mbo_reconciliation": 1})["allowed"]
    assert not CB.check("policy_cost_difference_historical", diagnostics={"identified_set": 1})["allowed"]
    assert all(e["status"] in CB.STATUSES for e in CB.ESTIMANDS.values())

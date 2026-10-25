"""v1 research decisions, power and promotion (M6, M7, M131 controls)."""
import json
from pathlib import Path

import numpy as np
import pytest

from lob.v1.research import decide as DE, promotion as PR, registry as RG


def dates(n):
    return [f"2026-01-{i + 1:02d}" for i in range(n)]


def vals(xs):
    return dict(zip(dates(len(xs)), xs))


def test_hypotheses_registry_is_complete_and_consistent():
    assert RG.check() == []
    hs = RG.candidates()
    assert len(hs) == 48 and all(h["cluster"].startswith("calendar date") for h in hs)
    reps = [h["id"] for h in hs if h["promotion"] == "direct_replication"]
    assert reps == [f"H{i}" for i in range(1, 13)] + ["H45", "H46", "H48"]
    assert {h["v09_reference"] for h in hs if h["id"] in ("H1", "H8", "H12")} == {"H2", "H23", "H36"}


def test_power_feasibility_from_pilot_clusters():
    rec = {"clusters": 8, "mean": 0.3, "se": 0.02, "sd_cluster_means": 0.06, "planning_effect": 0.28,
           "compute_hours_per_cluster": 0.2}
    h = {"promotion": "pilot", "test": "superiority", "threshold": 0.02, "margin": None, "level": None}
    rules = {"min_pilot_clusters": 6, "min_power": 0.8, "max_confirm_clusters": 40, "compute_ceiling_hours": 60}
    d = PR.decide_one(h, rec, rules=rules, family_size=8, capability_ok=True)
    assert d["outcome"] == "PROMOTED" and d["confirm_clusters"] >= DE.MIN_CLUSTERS and d["expected_power"] >= 0.8
    small = {**rec, "planning_effect": 0.01}
    assert PR.decide_one(h, small, rules=rules, family_size=8, capability_ok=True)["outcome"] == \
        "NOT_PROMOTED_EFFECT_TOO_SMALL"
    noisy = {**rec, "sd_cluster_means": 5.0}
    assert PR.decide_one(h, noisy, rules=rules, family_size=8, capability_ok=True)["outcome"] == \
        "NOT_PROMOTED_INSUFFICIENT_POWER"
    assert PR.decide_one(h, {**rec, "clusters": 4}, rules=rules, family_size=8, capability_ok=True)["outcome"] == \
        "NOT_PROMOTED_TOO_FEW_PILOT_CLUSTERS"
    assert PR.decide_one(h, None, rules=rules, family_size=8, capability_ok=True)["outcome"] == "NOT_PROMOTED_NO_PILOT"
    assert PR.decide_one(h, rec, rules=rules, family_size=8, capability_ok=False)["outcome"] == \
        "NOT_PROMOTED_DATA_CAPABILITY"
    rep = {"promotion": "direct_replication", "confirm_clusters": 12}
    assert PR.decide_one(rep, None, rules=rules, family_size=7, capability_ok=True)["confirm_clusters"] == 12


def test_hypotheses_decision_rules_established_failed_inconclusive():
    rng = np.random.default_rng(0)
    hyp = {"A": {"test": "superiority", "threshold": 0.1}, "B": {"test": "superiority", "threshold": 0.1},
           "C": {"test": "superiority", "threshold": 0.1}}
    cv = {"A": vals(0.5 + 0.05 * rng.standard_normal(12)), "B": vals(-0.2 + 0.05 * rng.standard_normal(12)),
          "C": vals(0.12 + 0.3 * rng.standard_normal(12))}
    out = DE.decide_family(hyp, cv)
    assert out["A"]["status"] == "ESTABLISHED" and out["B"]["status"] == "FAILED" and out["C"]["status"] == \
        "INCONCLUSIVE"
    few = DE.decide_family({"A": hyp["A"]}, {"A": vals([1.0] * 5)})
    assert few["A"]["status"] == "NOT_EVALUABLE"


def test_hypotheses_sign_flip_is_required_conjunctively():
    # a large t but too few positive clusters for the exact test at a 1/8-family Holm level
    x = [5.0] * 7 + [4.0]
    out = DE.decide_family({f"H{i}": {"test": "superiority", "threshold": 0.0} for i in range(8)},
                           {f"H{i}": vals(x) for i in range(8)})
    # 2^-8 = 0.0039 <= 0.05 / 8 = 0.00625: the floor allows rejection when every cluster is positive
    assert all(r["status"] == "ESTABLISHED" for r in out.values())
    out9 = DE.decide_family({f"H{i}": {"test": "superiority", "threshold": 0.0} for i in range(16)},
                            {f"H{i}": vals(x) for i in range(16)})
    # with a 16-member family the smallest step is 0.05 / 16 = 0.0031 < 2^-8: not established by the sign flip
    assert any(r["status"] != "ESTABLISHED" for r in out9.values())


def test_hypotheses_noninferiority_agreement_equivalence_and_transfer():
    ni = {"test": "noninferiority", "threshold": 0.0, "margin": 0.02}
    ag = {"test": "agreement", "threshold": 0.15, "level": 0.5}
    eq = {"test": "equivalence", "margin": 0.5, "threshold": 0.5}
    rng = np.random.default_rng(1)
    out = DE.decide_family({"NI": ni, "AG": ag, "EQ": eq},
                           {"NI": vals(0.0 + 0.005 * rng.standard_normal(10)),
                            "AG": vals(0.9 + 0.02 * rng.standard_normal(10)),
                            "EQ": vals(0.05 * rng.standard_normal(10))})
    assert out["NI"]["status"] == "ESTABLISHED" and out["AG"]["status"] == "ESTABLISHED"
    assert out["EQ"]["status"] == "ESTABLISHED"
    tr = {"test": "transfer", "thresholds": {"C1": 1.0, "C2": 1.0}}
    tra = {"test": "transfer_any", "thresholds": {"C1": 1.0, "C2": 1.0}}
    c = {"C1": vals(5 + rng.standard_normal(10)), "C2": vals(0.2 * rng.standard_normal(10))}
    o = DE.decide_family({"T": tr, "TA": tra}, {"T": c, "TA": c})
    assert o["T"]["status"] == "INCONCLUSIVE" or o["T"]["status"] == "FAILED"
    assert o["TA"]["status"] == "ESTABLISHED" and o["TA"]["established_conclusions"] == ["C1"]


def test_hypotheses_mixed_family_uses_full_family_size():
    out = DE.decide_family({"S": {"test": "superiority", "threshold": 0.0},
                            "T": {"test": "transfer", "thresholds": {"C1": 1.0}}},
                           {"S": vals([1.0] * 10), "T": {"C1": vals([3.0] * 10)}})
    assert out["S"]["holm_step_level"] == pytest.approx(0.025)


def test_controls_null_data_rarely_established():
    """Negative control: under the null (true mean = threshold) the conjunctive rule rejects at most ~alpha."""
    rng = np.random.default_rng(7)
    rej = 0
    for k in range(300):
        out = DE.decide_family({"A": {"test": "superiority", "threshold": 0.0}},
                               {"A": vals(rng.standard_normal(10))}, seed=k)
        rej += out["A"]["status"] == "ESTABLISHED"
    assert rej / 300 <= 0.08


def test_controls_positive_control_detected():
    rng = np.random.default_rng(8)
    hits = sum(DE.decide_family({"A": {"test": "superiority", "threshold": 0.0}},
                                {"A": vals(1.5 + rng.standard_normal(12))}, seed=k)["A"]["status"] == "ESTABLISHED"
               for k in range(100))
    assert hits >= 85


def test_replication_registry_pools_are_disjoint_and_fresh():
    from lob.v1.data import firewall as FW
    seen = {}
    for p in RG.POOLS:
        for d in RG.pool_dates(p):
            assert d not in seen and FW.date_fresh(d)[0]
            seen[d] = p
    reg = RG.dataset_registry()
    assert len(reg["datasets"]) == sum((v["primaries"] + v["backups"]) * len(v["streams"]) for v in RG.POOLS.values())


def test_vault_dates_are_last_and_untouched():
    vault = RG.pool_dates("vault")
    assert min(vault) > max(RG.pool_dates("replicate")) > max(RG.pool_dates("confirm"))
    led = Path("configs/v1/ledger.jsonl")
    if led.exists():
        for line in led.read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "access" and e["dataset"] and e["dataset"][-10:] in vault:
                raise AssertionError(f"vault dataset accessed: {e['dataset']}")


def test_controls_module_negative_positive_falsification():
    from lob.v1.research import controls as CT
    rng = np.random.default_rng(11)
    cv = vals(rng.standard_normal(12))
    assert CT.negative_self({"a": 0.0, "b": 0.0})["passed"] and not CT.negative_self({"a": 1e-3})["passed"]
    assert CT.negative_placebo(cv, reps=100)["passed"]
    assert CT.positive_injection(cv, 2.0, reps=50)["passed"]
    assert CT.falsification(vals(rng.standard_normal(12) * 0.1))["passed"]
    assert not CT.falsification(vals(3 + rng.standard_normal(12) * 0.1))["passed"]

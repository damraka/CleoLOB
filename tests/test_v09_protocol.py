"""v0.9 protocol, ledger v2 firewall, seeds, incidents, amendments and the promotion gate (M1-M6)."""
from __future__ import annotations

import json
import math

import pytest

from lob.preregistration import ProtocolError
from lob.v09 import promotion, protocol as pr, registry
from tests.v09_fixtures import access, first, promote_dummy, v09_root, v09_template  # noqa: F401


def _protocol(root):
    return pr.load_protocol(root / pr.PROTOCOL_PATH)


def test_builder_is_deterministic_and_pools_do_not_overlap() -> None:
    a, b = registry.pool_days("confirm"), registry.pool_days("confirm")
    assert a == b and len(a) == len(set(a)) == 46
    seen = {}
    for name in registry.POOLS:
        for day in registry.pool_days(name):
            assert day not in seen, f"{day} in {seen.get(day)} and {name}"
            seen[day] = name
    assert registry.candidates() == registry.candidates()
    assert set(registry.candidates()["hypotheses"]) == {f"H{k}" for k in range(1, 37)}


def test_isolated_root_freezes_with_every_dataset_fresh(v09_root) -> None:
    report = pr.verify_protocol_files(v09_root)
    assert report["valid"], report["issues"]
    assert report["frozen"] and not report["promoted"] and not report["confirmatory_closed"]
    for role, row in report["datasets_by_role"].items():
        assert row["consumed"] == 0, role
    _, state = pr.state_of(v09_root)
    assert all(state.freshness(d["id"]) == "fresh" for d in _protocol(v09_root)["datasets"])
    assert any(d.startswith("deribit") for d in state.consumed), "v0.5-v0.8 consumption must be imported"


def test_explore_access_is_free_after_freeze_but_never_confirmatory(v09_root) -> None:
    p = _protocol(v09_root)
    ds = first(p, pool="explore")
    access(v09_root, ds, use="explore", design="pilot-anything")
    with pytest.raises(ProtocolError, match="not permitted"):
        access(v09_root, first(p, pool="explore", venue="bybit-linear"), use="confirm", design="x")


def test_confirmatory_data_and_seeds_require_promotion(v09_root) -> None:
    p = _protocol(v09_root)
    ds = first(p, pool="confirm")
    with pytest.raises(ProtocolError, match="sealed design"):
        access(v09_root, ds, use="confirm", design="c1")
    with pytest.raises(ProtocolError, match="before the promotion freeze"):
        pr.seal_design(v09_root, "c1", {"x": 1}, reads=[ds], namespaces=["confirm"])
    assert not (v09_root / pr.CONFIG_DIR / "designs" / "c1.json").exists(), "refused seals write no design file"
    promote_dummy(v09_root)
    pr.seal_design(v09_root, "c1", {"x": 1}, reads=[ds], namespaces=["confirm"])
    with pytest.raises(ProtocolError, match="has not opened"):
        pr.seeds("confirm", "study", 3, root=v09_root, design="c1")
    pr.open_namespace(v09_root, "confirm", design="c1", reason="test")
    s = pr.seeds("confirm", "study", 3, root=v09_root, design="c1")
    assert s == pr.seeds("confirm", "study", 3, root=v09_root, design="c1") and len(set(s)) == 3
    access(v09_root, ds, use="confirm", design="c1")
    with pytest.raises(ProtocolError, match="posthoc"):
        access(v09_root, ds, use="confirm_posthoc", design="c1")
    with pytest.raises(ProtocolError, match="exactly once"):
        promote_dummy(v09_root)


def test_final_vault_opens_once_after_close(v09_root) -> None:
    p = _protocol(v09_root)
    vault = first(p, pool="final")
    promote_dummy(v09_root)
    with pytest.raises(ProtocolError, match="before the confirmatory phase closed"):
        pr.seal_design(v09_root, "final", {"f": 1}, reads=[vault], namespaces=["final"])
    pr.append_event(v09_root / pr.LEDGER_PATH, "close_confirmatory", reason="all confirmatory runs sealed",
                    protocol=p, root=v09_root, payload={"runs": {"c1": "results/v09/confirm/x"}})
    pr.seal_design(v09_root, "final", {"f": 1}, reads=[vault], namespaces=["final"])
    pr.open_namespace(v09_root, "final", design="final", reason="vault")
    access(v09_root, vault, use="final", design="final")
    with pytest.raises(ProtocolError, match="consumed guarded data"):
        pr.seal_design(v09_root, "final-again", {"f": 2}, reads=[vault], namespaces=["final"])
    pr.seal_design(v09_root, "final-posthoc", {"f": 3}, reads=[vault], posthoc=True)
    access(v09_root, vault, use="final_posthoc", design="final-posthoc")
    other = [d["id"] for d in pr.datasets_by(p, pool="final")][1]
    pr.seal_design(v09_root, "second-reader", {"f": 4}, reads=[other])
    with pytest.raises(ProtocolError, match="exactly one registered design"):
        access(v09_root, other, use="final", design="second-reader")


def test_close_requires_promotion_and_happens_once(v09_root) -> None:
    p = _protocol(v09_root)
    with pytest.raises(ProtocolError, match="only after promotion"):
        pr.append_event(v09_root / pr.LEDGER_PATH, "close_confirmatory", reason="x", protocol=p, root=v09_root,
                        payload={"runs": {}})
    promote_dummy(v09_root)
    pr.append_event(v09_root / pr.LEDGER_PATH, "close_confirmatory", reason="x", protocol=p, root=v09_root,
                    payload={"runs": {}})
    with pytest.raises(ProtocolError, match="exactly once"):
        pr.append_event(v09_root / pr.LEDGER_PATH, "close_confirmatory", reason="x", protocol=p, root=v09_root,
                        payload={"runs": {}})


@pytest.mark.parametrize("mutation", ["edit", "truncate_middle", "reorder", "blank"])
def test_ledger_tampering_is_detected(v09_root, mutation) -> None:
    path = v09_root / pr.LEDGER_PATH
    lines = path.read_text(encoding="utf-8").splitlines()
    if mutation == "edit":
        entry = json.loads(lines[5])
        entry["reason"] = "rewritten"
        lines[5] = json.dumps(entry)
    elif mutation == "truncate_middle":
        del lines[10]
    elif mutation == "reorder":
        lines[3], lines[4] = lines[4], lines[3]
    else:
        lines.insert(7, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not pr.verify_protocol_files(v09_root)["valid"]


def test_protocol_and_candidate_tampering_are_detected(v09_root) -> None:
    cand = v09_root / pr.CANDIDATES_PATH
    doc = json.loads(cand.read_text(encoding="utf-8"))
    doc["hypotheses"]["H23"]["threshold"] = 0.1          # moving a threshold after the freeze
    cand.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    report = pr.verify_protocol_files(v09_root)
    assert not report["valid"] and any("hypotheses-candidates" in i for i in report["issues"])


def test_confirmatory_registry_cannot_change_without_amendment(v09_root) -> None:
    promote_dummy(v09_root)
    (v09_root / pr.CONFIRMATORY_PATH).write_text("{}\n", encoding="utf-8")
    report = pr.verify_protocol_files(v09_root)
    assert not report["valid"] and any("confirmatory registry changed" in i for i in report["issues"])


def test_confirmatory_amendment_refused_after_confirmatory_access(v09_root) -> None:
    p = _protocol(v09_root)
    ds = first(p, pool="confirm")
    promote_dummy(v09_root)
    pr.seal_design(v09_root, "c1", {"x": 1}, reads=[ds], namespaces=["confirm"])
    access(v09_root, ds, use="confirm", design="c1")
    with pytest.raises(ProtocolError, match="after confirmatory access"):
        pr.append_event(v09_root / pr.LEDGER_PATH, "amend", reason="x", protocol=p, root=v09_root,
                        payload={"protocol_sha256": pr.protocol_sha256(p), "affects": [], "affects_confirmatory": True,
                                 "confirmatory_sha256": "d" * 64})


def test_incidents_are_ledgered_and_validated(v09_root) -> None:
    entry = pr.record_incident(v09_root, "engineering_bug", description="test bug", affects_validity=True,
                               runs=["results/v09/x"])
    assert entry["event"] == "incident"
    with pytest.raises(ProtocolError, match="incident kind"):
        pr.record_incident(v09_root, "made_up", description="x", affects_validity=False)
    report = pr.verify_protocol_files(v09_root)
    assert report["valid"] and report["incidents"][0]["kind"] == "engineering_bug"


def test_source_bytes_cannot_change_after_first_access(v09_root) -> None:
    ds = first(_protocol(v09_root), pool="explore")
    access(v09_root, ds, use="explore", design="d", sources={"l2": "1" * 64})
    with pytest.raises(ProtocolError, match="source bytes changed"):
        access(v09_root, ds, use="explore", design="d", stage="parsed", sources={"l2": "2" * 64})


def test_v09_seed_streams_are_namespaced_and_differ_from_v08() -> None:
    from lob.v08 import protocol as v08
    a = pr.seeds("dev", "s", 4)
    assert a != pr.seeds("explore", "s", 4) != pr.seeds("pilot", "s", 4)
    assert a != v08.seeds("dev", "s", 4)
    with pytest.raises(ProtocolError):
        pr.seeds("nope", "s", 1)


def test_unsafe_paths_are_rejected() -> None:
    for bad in ("../x", "/abs", "C:\\x", "a//b", "a/./b"):
        with pytest.raises(ProtocolError):
            pr.portable(bad)


# ----------------------------------------------------------------------------- promotion gate

RULES = {"min_reliability": 0.5, "min_pilot_units": 6, "min_power": 0.8, "shrinkage_se": 1.0, "min_confirm_units": 8,
         "compute_hours_ceiling": 40.0}
FAMILY = {"alpha": 0.05, "correction": "holm"}


def _cand(**kw):
    base = {"test": "superiority", "threshold": 0.5, "max_units": 80, "required_capabilities": ["incremental_l2"],
            "family": "f"}
    return {**base, **kw}


def _pilot(**kw):
    base = {"data_available": True, "capabilities": {"incremental_l2": True}, "reliability": 0.9, "units": 16,
            "effect": 2.0, "sd": 2.0, "compute_hours_per_unit": 0.1}
    return {**base, **kw}


@pytest.mark.parametrize("pilot,outcome", [
    (None, "NOT_PROMOTED_OTHER_REGISTERED_REASON"),
    (_pilot(data_available=False), "NOT_PROMOTED_DATA_CAPABILITY"),
    (_pilot(capabilities={}), "NOT_PROMOTED_DATA_CAPABILITY"),
    (_pilot(reliability=0.2), "NOT_PROMOTED_MEASUREMENT"),
    (_pilot(units=3), "NOT_PROMOTED_MEASUREMENT"),
    (_pilot(effect=0.6, sd=2.0), "NOT_PROMOTED_EFFECT_TOO_SMALL"),
    (_pilot(effect=1.0, sd=4.0), "NOT_PROMOTED_EFFECT_TOO_SMALL"),
    (_pilot(effect=1.6, sd=3.0), "NOT_PROMOTED_INSUFFICIENT_POWER"),
    (_pilot(compute_hours_per_unit=10.0), "NOT_PROMOTED_COMPUTE"),
    (_pilot(), "PROMOTED"),
])
def test_promotion_gate_order_and_outcomes(pilot, outcome) -> None:
    d = promotion.decide("Hx", _cand(), pilot, rules=RULES, family=FAMILY, family_size=4)
    assert d["outcome"] == outcome, d["reason"]
    assert d["lifecycle"][0] == "CANDIDATE" and d["lifecycle"][-1] in ("PROMOTED", "NOT_PROMOTED")
    if outcome == "PROMOTED":
        assert d["required_units"] >= RULES["min_confirm_units"] and d["expected_power"] >= 0.8


def test_power_matches_the_decision_rule() -> None:
    # at a planning effect equal to the threshold the rule (estimate >= threshold) can never exceed 1/2
    assert promotion.power("superiority", 0.5, 1.0, 10_000, alpha=0.01, threshold=0.5) <= 0.5 + 1e-9
    p = [promotion.power("superiority", 1.0, 1.0, n, alpha=0.0125, threshold=0.5) for n in (8, 16, 32, 64)]
    assert p == sorted(p) and p[-1] > 0.95
    ni = promotion.power("noninferiority", 0.0, 0.05, 40, alpha=0.01, threshold=0.02)
    assert 0 < ni < 1 and math.isfinite(ni)
    n = promotion.required_units("superiority", 1.0, 1.0, alpha=0.0125, threshold=0.5, target=0.8, ceiling=200)
    assert promotion.power("superiority", 1.0, 1.0, n, alpha=0.0125, threshold=0.5) >= 0.8
    assert promotion.power("superiority", 1.0, 1.0, n - 1, alpha=0.0125, threshold=0.5) < 0.8


def test_noninferiority_gate_uses_the_margin() -> None:
    cand = _cand(test="noninferiority", threshold=0.02)
    assert promotion.decide("Hn", cand, _pilot(effect=-0.03, sd=0.02), rules=RULES, family=FAMILY,
                            family_size=5)["outcome"] == "NOT_PROMOTED_EFFECT_TOO_SMALL"
    assert promotion.decide("Hn", cand, _pilot(effect=0.0, sd=0.02), rules=RULES, family=FAMILY,
                            family_size=5)["outcome"] == "PROMOTED"


def test_promote_is_deterministic_and_registry_lists_only_promoted() -> None:
    cands = registry.candidates()
    caps = {c: True for c in ("incremental_l2", "trades_with_side", "queue_clearance_observable")}
    pilots = {"H23": _pilot(effect=3.0, sd=3.0, capabilities=caps), "H7": _pilot(effect=0.0001, sd=0.01, capabilities=caps)}
    fams = registry.families()
    a = promotion.promote(cands, pilots, rules=RULES, families=fams)
    b = promotion.promote(cands, dict(reversed(list(pilots.items()))), rules=RULES, families=fams)
    assert a == b
    reg = promotion.confirmatory_registry(cands, a)
    assert set(reg["hypotheses"]) == set(a["promoted"])
    assert "H23" in reg["hypotheses"]
    assert "H7" in reg["not_promoted"] and reg["not_promoted"]["H7"] == "NOT_PROMOTED_EFFECT_TOO_SMALL"


def test_split_half_consistency_screen_amendment_3() -> None:
    rules = {**RULES, "reliability_rule": "split_half_consistency"}
    homogeneous = _pilot(effect=2.0, sd=0.5, reliability=-0.1, odd_values=[2.1, 1.9, 2.0, 2.2, 1.8, 2.0],
                         even_values=[1.9, 2.1, 2.0, 1.8, 2.2, 2.0])
    d = promotion.decide("Hc", _cand(), homogeneous, rules=rules, family=FAMILY, family_size=4)
    assert d["outcome"] == "PROMOTED", d["reason"]          # strong homogeneous effect is not blocked
    assert promotion.decide("Hc", _cand(), homogeneous, rules=RULES, family=FAMILY,
                            family_size=4)["outcome"] == "NOT_PROMOTED_MEASUREMENT"   # the original screen would
    inconsistent = _pilot(odd_values=[3.0, 3.1, 2.9, 3.2, 3.0, 3.1], even_values=[0.1, 0.0, -0.1, 0.2, 0.0, 0.1])
    assert promotion.decide("Hc", _cand(), inconsistent, rules=rules, family=FAMILY,
                            family_size=4)["outcome"] == "NOT_PROMOTED_MEASUREMENT"
    assert promotion.decide("Hc", _cand(), _pilot(), rules=rules, family=FAMILY,
                            family_size=4)["outcome"] == "NOT_PROMOTED_MEASUREMENT"   # no halves recorded

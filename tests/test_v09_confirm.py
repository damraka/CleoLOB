"""v0.9 confirmatory machinery: decision rules, design building, backup substitution, unit selection."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v09 import confirm as CF, decide as D, protocol as pr
from lob.v09.studies import final as FA

from .v09_fixtures import promote_dummy, v09_root, v09_template  # noqa: F401


# ----------------------------------------------------------------------------- decision rules


def test_superiority_needs_holm_and_threshold() -> None:
    strong = D.evaluate("superiority", np.full(20, 1.0) + np.linspace(-0.1, 0.1, 20), threshold=0.5, seed=1)
    below = D.evaluate("superiority", np.full(20, 0.3) + np.linspace(-0.1, 0.1, 20), threshold=0.5, seed=2)
    out = D.family({"A": strong, "B": below})
    assert out["A"]["status"] == "ESTABLISHED"
    assert out["B"]["holm_rejected"] and out["B"]["status"] == "NOT_ESTABLISHED"


def test_opposite_direction_is_failed_not_relabelled() -> None:
    neg = D.evaluate("superiority", -1.0 + np.linspace(-0.1, 0.1, 12), threshold=0.1, seed=3)
    assert D.family({"A": neg})["A"]["status"] == "FAILED"


def test_noninferiority_and_agreement_rules() -> None:
    ni = D.evaluate("noninferiority", np.linspace(-0.004, 0.002, 30), threshold=0.02, seed=4)
    assert D.family({"N": ni})["N"]["status"] == "ESTABLISHED"
    bad = D.evaluate("noninferiority", np.linspace(-0.2, -0.1, 30), threshold=0.02, seed=5)
    assert D.family({"N": bad})["N"]["status"] == "FAILED"
    ag = D.evaluate("agreement", np.full(20, 0.8) + np.linspace(-0.02, 0.02, 20), threshold=0.15, level=0.5, seed=6)
    assert D.family({"G": ag})["G"]["status"] == "ESTABLISHED"
    weak = D.evaluate("agreement", np.full(20, 0.6) + np.linspace(-0.02, 0.02, 20), threshold=0.15, level=0.5, seed=7)
    assert D.family({"G": weak})["G"]["status"] == "NOT_ESTABLISHED"


def test_too_few_units_is_not_evaluable() -> None:
    st = D.evaluate("superiority", [1.0], threshold=0.0, seed=0)
    assert D.family({"X": st})["X"]["status"] == "NOT_EVALUABLE"


def _units(lower, upper):
    return {f"d{i}": {"lower": a, "upper": b, "status": "IDENTIFIED" if a > 0 or b < 0 else "INDETERMINATE"}
            for i, (a, b) in enumerate(zip(lower, upper))}


def test_transfer_rule_any_conclusion_and_failed_only_when_all_opposite() -> None:
    good = D.conclusion_stats(_units(np.linspace(1.5, 2.5, 15), np.linspace(3, 4, 15)), 1.0, seed=1)
    flat = D.conclusion_stats(_units(np.linspace(-0.5, 0.4, 15), np.linspace(0.5, 1.0, 15)), 1.0, seed=2)
    assert D.transfer({"C1": good, "C2": flat})["status"] == "ESTABLISHED"
    neg = D.conclusion_stats(_units(np.linspace(-3, -2, 15), np.linspace(-1.5, -1.0, 15)), 0.25, seed=3)
    assert D.transfer({"C2": neg})["status"] == "FAILED"
    assert D.transfer({"C2": neg, "C3": flat})["status"] == "NOT_ESTABLISHED"
    assert D.transfer({})["status"] == "NOT_AVAILABLE"


def test_holm_adjusted_pvalues_are_monotone() -> None:
    adj = D.holm_adjusted([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)


# ----------------------------------------------------------------------------- designs and substitution


def _registry(**units):
    return {"hypotheses": {h: {"test": "superiority"} for h in units}, "confirm_units": units,
            "carried_conclusions": {"C3": {"confirm_units": 10, "identifiability": "L2", "threshold_bps": 0.25}},
            "not_promoted": {}}


def test_designs_read_primaries_plus_backups_and_train_on_explore(v09_root) -> None:
    designs = CF.build(v09_root, _registry(H3=20, H16=12, H11=30), {"H16": {"pilot": {"selected_method": "replay"}}})
    protocol = pr.load_protocol(v09_root / pr.PROTOCOL_PATH)
    seq = designs["confirm-sequence-generator"]
    assert len(seq["document"]["test_datasets"]) == 20
    assert all(pr.dataset_declaration(protocol, d)["role"] == "explore" for d in seq["document"]["train_datasets"])
    prim, back = CF.slots(protocol, "confirm", "bybit-linear:BTCUSDT")
    assert set(back) <= set(seq["reads"]) and set(prim[:10]) <= set(seq["reads"])
    cont = designs["confirm-continual-calibration"]["document"]
    assert cont["method"] == "replay" and len(cont["sequences"]["bybit-linear:BTCUSDT"]) == 7
    days = cont["sequences"]["bybit-linear:BTCUSDT"]
    assert days == sorted(days, key=lambda d: d[-10:])
    syn = designs["confirm-strategic-response"]
    assert syn["reads"] == [] and len(syn["document"]["calibration_days"]) == 30
    assert len(designs["confirm-bounds"]["document"]["datasets"]) == 10
    assert "confirm-data" in designs


def test_sealing_requires_promotion(v09_root) -> None:
    with pytest.raises(pr.ProtocolError):
        CF.seal_all(v09_root)


def test_backup_substitution_follows_slot_order(v09_root) -> None:
    protocol = pr.load_protocol(v09_root / pr.PROTOCOL_PATH)
    prim, back = CF.slots(protocol, "confirm", "bybit-linear:ETHUSDT")
    promote_dummy(v09_root)
    for d in (prim[3], prim[1]):
        pr.append_event(v09_root / pr.LEDGER_PATH, "attempt", dataset=d, role="confirm", design="confirm-data",
                        reason="test failure", protocol=protocol, root=v09_root,
                        payload={"outcome": "NOT_AVAILABLE", "directory": None, "note": "HTTP 404",
                                 "no_directory_reason": "test"})
    _, state = pr.state_of(v09_root)
    doc = {"test_datasets": prim[:5], "train_datasets": ["x"]}
    new, mapping = CF.substitute(doc, protocol, state)
    assert mapping == {prim[1]: back[0], prim[3]: back[1]}
    assert new["test_datasets"] == [prim[0], back[0], prim[2], back[1], prim[4]]


def test_first_units_split_evenly_over_streams() -> None:
    order = [f"bybit-linear-btcusdt-2024-01-{i:02d}" for i in range(1, 6)] + \
            [f"bybit-linear-ethusdt-2024-01-{i:02d}" for i in range(1, 6)]
    got = FA.first_units(order, 5, synthetic=False)
    assert len(got) == 6 and got[:3] == order[:3] and got[3:] == order[5:8]
    assert FA.first_units(["b", "a", "c"], 2, synthetic=True) == ["b", "a"]

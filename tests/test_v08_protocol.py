"""v0.8 protocol, ledger, seed namespaces, dataset registry and fail-closed data access (M1, M2)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lob.preregistration import ProtocolError
from lob.v08 import datasets, protocol as pr, realdata
from lob.v08.smoke import isolated_root

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    r = isolated_root(tmp_path_factory.mktemp("v08root"))
    assert r is not None
    return r


@pytest.fixture
def root(template, tmp_path) -> Path:
    target = tmp_path / "root"
    shutil.copytree(template, target)
    return target


def test_repository_protocol_verifies_and_hash_is_deterministic() -> None:
    report = pr.verify_protocol_files(ROOT)
    assert report["valid"], report["issues"]
    p = pr.load_protocol(ROOT / pr.PROTOCOL_PATH, ROOT)
    assert pr.protocol_sha256(p) == pr.protocol_sha256(json.loads(json.dumps(p))) == report["protocol_sha256"]
    assert set(json.loads((ROOT / pr.CONFIG_DIR / "hypotheses.json").read_text())["hypotheses"]) >= {f"H{k}"
                                                                                                    for k in range(1, 22)}


def test_v07_history_is_pinned_and_unchanged() -> None:
    p = pr.load_protocol(ROOT / pr.PROTOCOL_PATH)
    h = pr.history_state(ROOT)
    for key in ("v07_protocol_sha256", "v07_ledger_entries", "v07_ledger_head_sha256"):
        assert p["history"][key] == h[key]


def test_isolated_freeze_and_replay(root) -> None:
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    assert state.frozen is not None and len(state.fresh) == 6
    assert pr.verify_protocol_files(root)["valid"]


def _rewrite(path: Path, mutate) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(mutate(lines)) + "\n", encoding="utf-8")


@pytest.mark.parametrize("mutation", ["edit", "truncate_middle", "reorder", "blank"])
def test_ledger_tampering_is_detected(root, mutation) -> None:
    path = root / pr.LEDGER_PATH

    def mutate(lines):
        if mutation == "edit":
            entry = json.loads(lines[3])
            entry["reason"] = "tampered"
            lines[3] = json.dumps(entry)
        elif mutation == "truncate_middle":
            del lines[2]
        elif mutation == "reorder":
            lines[1], lines[2] = lines[2], lines[1]
        else:
            lines.insert(2, "")
        return lines

    _rewrite(path, mutate)
    with pytest.raises(ProtocolError):
        pr.replay_ledger(pr.read_ledger(path), pr.load_protocol(root / pr.PROTOCOL_PATH))
    assert not pr.verify_protocol_files(root)["valid"]


def test_protocol_and_config_tampering_are_detected(root) -> None:
    doc = json.loads((root / pr.PROTOCOL_PATH).read_text(encoding="utf-8"))
    doc["research_question"] += " (edited)"
    (root / pr.PROTOCOL_PATH).write_text(json.dumps(doc), encoding="utf-8")
    assert any("protocol changed after freeze" in i for i in pr.verify_protocol_files(root)["issues"])


def test_referenced_config_change_without_amendment_is_detected(root) -> None:
    path = root / pr.CONFIG_DIR / "statistical-families.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["margins"]["policy_equivalence_bps"] = 5.0
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert any("referenced config changed" in i for i in pr.verify_protocol_files(root)["issues"])


def test_confirm_namespace_requires_a_sealed_design(root) -> None:
    assert len(pr.seeds("dev", "study", 3)) == 3
    with pytest.raises(ProtocolError):
        pr.seeds("confirm", "study", 3, root=root, design="nope")
    pr.seal_design(root, "d1", {"x": 1}, namespaces=["confirm"])
    with pytest.raises(ProtocolError):
        pr.seeds("confirm", "study", 3, root=root, design="d1")       # sealed but not opened
    pr.open_namespace(root, "confirm", design="d1", reason="test")
    a = pr.seeds("confirm", "study", 3, root=root, design="d1")
    assert a == pr.seeds("confirm", "study", 3, root=root, design="d1") and a != pr.seeds("dev", "study", 3)


def test_synthetic_holdout_opens_once(root) -> None:
    pr.seal_design(root, "h1", {"x": 1}, namespaces=["holdout"])
    pr.open_namespace(root, "holdout", design="h1", reason="first")
    with pytest.raises(ProtocolError):
        pr.seal_design(root, "h2", {"x": 2}, namespaces=["holdout"])          # must be posthoc
    pr.seal_design(root, "h3", {"x": 3}, namespaces=["holdout"], posthoc=True)


def test_designs_cannot_change_after_sealing(root) -> None:
    pr.seal_design(root, "d", {"a": 1})
    path = root / pr.CONFIG_DIR / "designs" / "d.json"
    path.write_text(json.dumps({"a": 2}), encoding="utf-8")
    with pytest.raises(ProtocolError):
        pr.sealed_design(root, "d")


def test_holdout_access_without_sealed_reader_is_refused(root) -> None:
    with pytest.raises(ProtocolError):
        realdata.acquire("deribit-eth-perp-2021-02-01", use="evaluate", design="unsealed", purpose="t", root=root)


def test_blocked_probe_records_not_available_and_keeps_holdout_fresh(root) -> None:
    ds = "deribit-eth-perp-2021-02-01"
    pr.seal_design(root, "reader", {"x": 1}, reads=[ds])
    calls = []

    def downloader(*a, **k):
        calls.append(a)
        raise AssertionError("no holdout byte may be requested after a failed probe")

    with pytest.raises(realdata.DataNotAvailable):
        realdata.acquire(ds, use="evaluate", design="reader", purpose="t", root=root,
                         prober=lambda: {"ok": False, "status": 403}, downloader=downloader)
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    assert not calls and state.freshness(ds) == "fresh"
    assert state.attempts[-1]["outcome"] == "NOT_AVAILABLE" and state.attempts[-1]["dataset"] == ds


def test_consumed_dataset_role_uses_are_enforced(root) -> None:
    with pytest.raises(ProtocolError):
        realdata.acquire("deribit-eth-perp-2020-10-01", use="evaluate", design="x", purpose="t", root=root)


def test_amendment_cannot_touch_accessed_holdout(root) -> None:
    doc = json.loads((root / pr.PROTOCOL_PATH).read_text(encoding="utf-8"))
    state = pr.LedgerState()
    state.first_access["deribit-eth-perp-2021-02-01"] = 1
    state.frozen = {"x": 1}
    entry = {"payload": {"protocol_sha256": "0" * 64, "affects": ["deribit-eth-perp-2021-02-01"]}}
    with pytest.raises(ProtocolError):
        state._on_amend(entry, entry["payload"], doc)


def test_registry_capabilities_fail_closed() -> None:
    report = datasets.report(ROOT)
    assert not report["issues"]
    caps = {d["id"]: d["capabilities"] for d in report["datasets"]}
    eth = caps["deribit-eth-perp-2020-10-01"]
    datasets.require(eth, "counterfactual_bounds", dataset="eth")
    with pytest.raises(datasets.CapabilityError):
        datasets.require(eth, "exact_queue_study", dataset="eth")
    with pytest.raises(datasets.CapabilityError):
        datasets.require(eth, "hidden_liquidity_study", dataset="eth")
    row = next(d for d in report["datasets"] if d["id"] == "deribit-eth-perp-2020-10-01")
    assert row["studies"]["exact_fill_replay"].startswith("NOT_AVAILABLE")
    assert not datasets.allowed(pr.dataset_declaration(pr.load_protocol(ROOT / pr.PROTOCOL_PATH),
                                                       "deribit-eth-perp-2020-10-01"), "exact_queue_position")


def test_unsafe_paths_are_rejected() -> None:
    for bad in ("../x", "C:\\x", "/abs", "a//b"):
        with pytest.raises(ProtocolError):
            pr.portable(bad)


def test_final_holdout_reuses_a_ledgered_failed_probe(root, monkeypatch) -> None:
    # regression: every final-analysis rerun probed the provider again for the H20 holdout
    from lob.v08.studies import final
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    pr.seal_design(root, "final-holdout-h20", {"hypotheses": ["H20"]}, reads=list(final.FINAL_HOLDOUT))
    for ds in final.FINAL_HOLDOUT:
        pr.append_event(root / pr.LEDGER_PATH, "attempt", design="final-holdout-h20", dataset=ds, reason="probe",
                        protocol=protocol, root=root,
                        payload={"outcome": "NOT_AVAILABLE", "directory": None, "note": "403",
                                 "no_directory_reason": "nothing was downloaded"})
    before = len(pr.read_ledger(root / pr.LEDGER_PATH))

    def no_probe(*a, **k):
        raise AssertionError("the provider must not be probed again")

    monkeypatch.setattr(realdata, "acquire", no_probe)
    out = final._final_holdout(None, root)
    assert out["status"] == "NOT_AVAILABLE" and set(out["datasets"]) == set(final.FINAL_HOLDOUT)
    assert len(pr.read_ledger(root / pr.LEDGER_PATH)) == before


def test_synthetic_holdout_validates_inputs_first_and_never_reopens(root, monkeypatch) -> None:
    # regression: H21 opened the holdout namespace before checking its worlds input, so a rerun could never open it
    from types import SimpleNamespace
    from lob.v08.studies import common as C, holdout
    args = SimpleNamespace(root=str(root), inputs={})
    before = len(pr.read_ledger(root / pr.LEDGER_PATH))
    with pytest.raises(ValueError, match="worlds"):
        holdout.evaluate_holdout(args, root, [{"source": "H9"}], None, None)
    assert len(pr.read_ledger(root / pr.LEDGER_PATH)) == before          # nothing sealed or opened
    pr.seal_design(root, holdout.DESIGN, {"x": 1}, namespaces=["holdout"])
    pr.open_namespace(root, "holdout", design=holdout.DESIGN, reason="retained attempt opened it")
    opened = len(pr.read_ledger(root / pr.LEDGER_PATH))

    class Reached(Exception):
        pass

    monkeypatch.setattr(C, "input_run", lambda a, k: root)
    monkeypatch.setattr(C, "load_world_set", lambda *a: (_ for _ in ()).throw(Reached()))
    with pytest.raises(Reached):
        holdout.evaluate_holdout(args, root, [{"source": "H9"}], None, None)
    assert len(pr.read_ledger(root / pr.LEDGER_PATH)) == opened          # no second open_namespace event

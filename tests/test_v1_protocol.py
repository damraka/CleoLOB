"""v1 protocol, ledger, namespace isolation, seeds and incidents (M2-M4, M8, M9)."""
import json

import pytest

from lob.v1.protocol import document as D, ledger as L, seeds as S
from tests.v1_fixtures import minimal_protocol, ws  # noqa: F401


def _design(datasets, namespace="confirm"):
    return {"study": "s", "namespace": namespace, "hypotheses": ["H1"], "datasets": list(datasets), "analysis": "a"}


def _to_promotion(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    ws.append("declare_fresh", dataset="d-confirm", reason="slot", payload={"role": "confirm", "pool": "confirm"})
    ws.append("declare_fresh", dataset="d-explore", reason="slot", payload={"role": "explore", "pool": "explore"})
    ws.append("declare_fresh", dataset="d-rep", reason="slot", payload={"role": "replicate", "pool": "replicate"})
    ws.append("declare_fresh", dataset="d-vault", reason="slot", payload={"role": "vault", "pool": "vault"})
    ws.append("promote", reason="promotion", payload={"promoted": ["H1"]})


def test_protocol_validation_and_freeze(ws):
    bad = minimal_protocol(statistics={"independent_unit": "date", "primary_test": "t", "min_clusters": 3,
                                       "alpha": 0.05, "correction": "holm"})
    assert any("sign-flip" in i for i in D.validate(bad))
    with pytest.raises(D.ProtocolError):
        D.freeze(ws, bad, reason="x")
    e = D.freeze(ws, minimal_protocol(), reason="freeze")
    assert e["event"] == "freeze"
    assert D.verify(ws)["valid"]
    with pytest.raises(D.ProtocolError):
        D.freeze(ws, minimal_protocol(), reason="again")


def test_protocol_tamper_is_detected(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    doc = json.loads(ws.protocol_path.read_text())
    doc["research_question"] = "changed"
    ws.protocol_path.write_text(json.dumps(doc))
    assert not D.verify(ws)["valid"]


def test_protocol_amendment_is_numbered_archived_and_posthoc_after_guarded_access(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    a1 = D.amend(ws, minimal_protocol(research_question="v2"), reason="clarify", affects=["research_question"])
    assert a1["payload"]["number"] == 1 and a1["payload"]["posthoc"] is False
    assert (ws.config / "protocol-history" / "protocol-0.json").exists()
    assert D.verify(ws)["valid"]
    with pytest.raises(D.ProtocolError):
        D.amend(ws, minimal_protocol(), reason="x", affects=[])
    # a forced non-posthoc amendment after guarded access is refused by the ledger rules
    ws.append("declare_fresh", dataset="d", reason="slot", payload={"role": "confirm", "pool": "confirm"})
    ws.append("promote", reason="p", payload={})
    D.seal_design(ws, "c", _design(["d"]), reason="seal")
    D.open_namespace(ws, "confirm", design="c", reason="open")
    D.record_access(ws, "d", namespace="confirm", stage="parsed", use="confirm", design="c", reason="read")
    with pytest.raises(L.LedgerError):
        D.amend(ws, minimal_protocol(research_question="v3"), reason="x", affects=["q"], posthoc=False)
    a2 = D.amend(ws, minimal_protocol(research_question="v3"), reason="x", affects=["q"])
    assert a2["payload"]["posthoc"] is True


def test_ledger_hash_chain_detects_edits(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    ws.append("declare_fresh", dataset="d1", reason="slot", payload={"role": "explore", "pool": "explore"})
    assert L.verify(ws.ledger_path)["valid"]
    lines = ws.ledger_path.read_text().splitlines()
    e = json.loads(lines[1])
    e["payload"]["role"] = "confirm"
    lines[1] = json.dumps(e)
    ws.ledger_path.write_text("\n".join(lines) + "\n")
    res = L.verify(ws.ledger_path)
    assert not res["valid"] and "hash mismatch" in res["issues"][0]


def test_ledger_rejects_reason_less_unknown_and_consumed_fresh(ws):
    with pytest.raises(L.LedgerError):
        ws.append("access", dataset="x", reason="  ")
    with pytest.raises(L.LedgerError):
        ws.append("bogus", reason="x")
    ws.append("import_consumed", dataset="old", reason="v0.9", payload={"version": "0.9"})
    with pytest.raises(L.LedgerError):
        ws.append("declare_fresh", dataset="old", reason="x", payload={"role": "confirm"})
    assert L.verify(ws.ledger_path)["state"]["entries"] == 1   # refused entries were never written


def test_ledger_acquire_hash_cannot_change(ws):
    ws.append("declare_fresh", dataset="d", reason="slot", payload={"role": "explore", "pool": "explore"})
    ws.append("acquire", dataset="d", reason="dl", payload={"sha256": {"l2": "a" * 64}})
    ws.append("acquire", dataset="d", reason="dl again", payload={"sha256": {"l2": "a" * 64, "trades": "b" * 64}})
    with pytest.raises(L.LedgerError):
        ws.append("acquire", dataset="d", reason="dl", payload={"sha256": {"l2": "c" * 64}})


def test_namespace_role_isolation(ws):
    _to_promotion(ws)
    D.record_access(ws, "d-explore", namespace="explore", stage="parsed", use="explore", reason="pilot")
    with pytest.raises(L.LedgerError):          # confirm data in an exploratory namespace
        D.record_access(ws, "d-confirm", namespace="explore", stage="parsed", use="peek", reason="x")
    with pytest.raises(L.LedgerError):          # confirm namespace not open
        D.record_access(ws, "d-confirm", namespace="confirm", stage="parsed", use="c", design="c", reason="x")
    D.seal_design(ws, "c", _design(["d-confirm"]), reason="seal")
    D.open_namespace(ws, "confirm", design="c", reason="open")
    with pytest.raises(L.LedgerError):          # not listed by the design
        D.record_access(ws, "d-explore", namespace="confirm", stage="parsed", use="c", design="c", reason="x")
    D.record_access(ws, "d-confirm", namespace="confirm", stage="parsed", use="c", design="c", reason="read")
    with pytest.raises(L.LedgerError):          # consumed data never in a guarded namespace
        ws.append("import_consumed", dataset="old", reason="v0.9", payload={})
        D.record_access(ws, "old", namespace="confirm", stage="parsed", use="c", design="c", reason="x")


def test_namespace_order_and_single_vault(ws):
    _to_promotion(ws)
    D.seal_design(ws, "v", _design(["d-vault"], "vault"), reason="seal")
    D.seal_design(ws, "r", _design(["d-rep"], "replicate"), reason="seal")
    D.seal_design(ws, "c", _design(["d-confirm"]), reason="seal")
    with pytest.raises(L.LedgerError):
        D.open_namespace(ws, "vault", design="v", reason="too early")
    with pytest.raises(L.LedgerError):
        D.open_namespace(ws, "replicate", design="r", reason="too early")
    D.open_namespace(ws, "confirm", design="c", reason="open")
    D.close_namespace(ws, "confirm", runs={"c": "results/v1/confirm/c"}, reason="close")
    with pytest.raises(L.LedgerError):
        D.open_namespace(ws, "confirm", design="c", reason="reopen")
    D.open_namespace(ws, "replicate", design="r", reason="open")
    D.close_namespace(ws, "replicate", runs={}, reason="close")
    D.open_namespace(ws, "vault", design="v", reason="open once")
    D.record_access(ws, "d-vault", namespace="vault", stage="evaluated", use="final", design="v", reason="read")
    D.close_namespace(ws, "vault", runs={}, reason="close")
    with pytest.raises(L.LedgerError):
        D.open_namespace(ws, "vault", design="v", reason="search for better results")
    assert D.verify(ws)["valid"]


def test_namespace_requires_promotion_and_sealed_design(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    D.seal_design(ws, "c", _design([]), reason="seal")
    with pytest.raises(L.LedgerError):
        D.open_namespace(ws, "confirm", design="c", reason="before promotion")
    with pytest.raises(D.ProtocolError):
        D.seal_design(ws, "c", _design([]), reason="twice")


def test_sealed_design_tamper_detected(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    D.seal_design(ws, "c", _design(["d"]), reason="seal")
    p = ws.designs_dir / "c.json"
    doc = json.loads(p.read_text())
    doc["analysis"] = "changed"
    p.write_text(json.dumps(doc))
    with pytest.raises(D.ProtocolError):
        D.sealed_design(ws, "c")
    assert not D.verify(ws)["valid"]


def test_seal_refused_before_freeze_leaves_no_file(ws):
    with pytest.raises(L.LedgerError):
        D.seal_design(ws, "c", _design([]), reason="seal")
    assert not (ws.designs_dir / "c.json").exists()


def test_dirty_tree_refuses_guarded_events(tmp_path):
    w = D.Workspace(tmp_path, allow_dirty=False)     # not a git repository -> treated as dirty
    with pytest.raises(L.LedgerError):
        D.freeze(w, minimal_protocol(), reason="freeze")


def test_seeds_are_deterministic_and_namespace_disjoint():
    a = S.seeds("explore", "tpp", 5)
    assert a == S.seeds("explore", "tpp", 5) and len(set(a)) == 5
    assert set(a).isdisjoint(S.seeds("calibrate", "tpp", 5))
    with pytest.raises(ValueError):
        S.seed("confirm", "tpp")
    assert S.seed("confirm", "tpp", design="c1") != S.seed("confirm", "tpp", design="c2")
    assert 0 <= S.seed("explore", "tpp") < 2**63


def test_incident_and_attempt_records(ws):
    D.record_incident(ws, "engineering_bug", description="unit order", affects_validity=False)
    D.record_attempt(ws, study="tpp", run="results/v1/explore/tpp", outcome="INVALID", namespace="explore")
    with pytest.raises(L.LedgerError):
        D.record_incident(ws, "nonsense", description="x", affects_validity=False)
    with pytest.raises(L.LedgerError):
        D.record_attempt(ws, study="tpp", run="r", outcome="GREAT", namespace="explore")
    st = ws.state()
    assert len(st.incidents) == 1 and len(st.attempts) == 1


def test_ledger_cache_sees_external_appends_and_tampering(ws):
    ws.append("declare_fresh", dataset="a", reason="slot", payload={"role": "explore", "pool": "p"})
    # an external writer appends a valid entry without going through this process's cache
    state = L.replay(L.read(ws.ledger_path))
    e = {"schema": L.SCHEMA, "index": state.entries, "at": L.utc_now(), "event": "declare_fresh", "dataset": "b",
         "reason": "external", "payload": {"role": "explore", "pool": "p"}, "git_commit": None, "git_dirty": True,
         "prev_sha256": state.head}
    e["sha256"] = L.entry_hash(e)
    with ws.ledger_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(L.canonical(e) + "\n")
    third = ws.append("declare_fresh", dataset="c", reason="slot", payload={"role": "explore", "pool": "p"})
    assert third["index"] == 2 and third["prev_sha256"] == e["sha256"]
    assert L.verify(ws.ledger_path)["valid"]
    with ws.ledger_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(L.canonical({**e, "index": 3, "dataset": "forged"}) + "\n")
    with pytest.raises(L.LedgerError):
        ws.append("declare_fresh", dataset="d", reason="slot", payload={"role": "explore", "pool": "p"})

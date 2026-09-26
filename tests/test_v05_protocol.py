"""M0: protocol freeze, holdout consumption and evidence binding."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lob import preregistration as pr

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs/v05/protocol.json").read_text(encoding="utf-8"))
CONSUMED = json.loads((ROOT / "configs/v04-consumed-data.json").read_text(encoding="utf-8"))
PROVENANCE = {"source_sha256": "a" * 64, "git_commit": "b" * 40, "git_dirty": False}


@pytest.fixture()
def ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    pr.initialize_ledger(path, PROTOCOL, ROOT, consumed_registry=CONSUMED)
    return path


def _access(path, dataset, digest="c" * 64):
    pr.append_event(path, "seal_design", {"analysis": f"design-{dataset}", "design_sha256": "d" * 64,
                                           "reads": [dataset]}, protocol=PROTOCOL)
    pr.append_event(path, "access", {"dataset_id": dataset, "source_sha256": {"l2.csv.gz": digest}},
                    protocol=PROTOCOL)


def test_committed_protocol_is_frozen_and_verifies():
    result = pr.verify_protocol_files(ROOT)
    assert result["valid"], result["issues"]
    assert "deribit-eth-perp-2020-07-01" in result["fresh"] or "deribit-eth-perp-2020-07-01" in result["consumed"]
    assert "deribit-eth-perp-2020-06-01" in result["consumed"]


def test_protocol_requires_statuses_roles_and_fill_vocabulary():
    pr.validate_protocol(copy.deepcopy(PROTOCOL))
    for mutate in (
        lambda p: p.update(statuses=p["statuses"][:-1]),
        lambda p: p["datasets"][3].update(freshness_at_freeze="consumed", consumed_evidence="x"),
        lambda p: p["historical_fill_uncertainty"].update(classes=["GUARANTEED_FILL"]),
        lambda p: p["statistics"]["M7"].update(family_size=96),
        lambda p: p.update(datasets=[d for d in p["datasets"] if d["role"] != "internal_holdout"]),
        lambda p: p["datasets"].append(dict(p["datasets"][0])),
        lambda p: p.update(referenced_configs=["../outside.json"]),
    ):
        changed = copy.deepcopy(PROTOCOL)
        mutate(changed)
        with pytest.raises(pr.ProtocolError):
            pr.validate_protocol(changed)


def test_protocol_mutation_changes_verification_identity(ledger):
    original = pr.protocol_sha256(PROTOCOL)
    changed = copy.deepcopy(PROTOCOL)
    changed["success_gates"]["M7"]["joint_gate"] = "cost upper bound < 0"
    assert pr.protocol_sha256(changed) != original
    with pytest.raises(pr.ProtocolError, match="protocol changed after freeze"):
        pr.replay_ledger(pr.read_ledger(ledger), changed)
    assert not pr.verify_ledger(ledger, protocol=changed)["valid"]


def test_holdout_freshness_cannot_be_restored(ledger):
    dataset = "deribit-eth-perp-2020-07-01"
    assert pr.replay_ledger(pr.read_ledger(ledger), PROTOCOL).freshness(dataset) == "fresh"
    _access(ledger, dataset)
    assert pr.replay_ledger(pr.read_ledger(ledger), PROTOCOL).freshness(dataset) == "consumed"
    with pytest.raises(pr.ProtocolError, match="cannot be restored"):
        pr.append_event(ledger, "declare_fresh", {"dataset_id": dataset}, protocol=PROTOCOL)
    anchor = pr.ledger_anchor(ledger)
    # Silently deleting the consumption entry breaks the anchor recorded by evidence.
    lines = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert not pr.verify_ledger(ledger, protocol=PROTOCOL, anchors=[anchor])["valid"]
    # Editing the consumed entry breaks the chain even without an anchor.
    tampered = [json.loads(line) for line in lines]
    tampered[-1]["event"] = "declare_fresh"
    ledger.write_text("\n".join(json.dumps(e) for e in tampered) + "\n", encoding="utf-8")
    assert not pr.verify_ledger(ledger, protocol=PROTOCOL)["valid"]


def test_previously_consumed_periods_can_never_be_declared_fresh(ledger):
    for dataset in ("deribit:ETH-PERPETUAL:2020-06-01", "deribit-eth-perp-2020-06-01"):
        with pytest.raises(pr.ProtocolError, match="cannot be restored"):
            pr.append_event(ledger, "declare_fresh", {"dataset_id": dataset}, protocol=PROTOCOL)


def test_holdout_access_requires_sealed_design(ledger):
    with pytest.raises(pr.ProtocolError, match="no sealed analysis design"):
        pr.append_event(ledger, "access", {"dataset_id": "deribit-btc-perp-2020-07-01",
                                           "source_sha256": {"f": "c" * 64}}, protocol=PROTOCOL)


def test_source_bytes_cannot_change_after_first_access(ledger):
    dataset = "deribit-eth-perp-2020-07-01"
    _access(ledger, dataset, "c" * 64)
    with pytest.raises(pr.ProtocolError, match="source bytes changed"):
        pr.append_event(ledger, "access", {"dataset_id": dataset, "source_sha256": {"l2.csv.gz": "e" * 64}},
                        protocol=PROTOCOL)


def test_amendment_cannot_follow_access_of_affected_data(ledger):
    pr.append_event(ledger, "amend", {"amendment_sha256": "1" * 64, "reason": "declared before access",
                                      "affects": ["deribit-eth-perp-2020-08-01"]}, protocol=PROTOCOL)
    _access(ledger, "deribit-eth-perp-2020-07-01")
    with pytest.raises(pr.ProtocolError, match="already consumed"):
        pr.append_event(ledger, "amend", {"amendment_sha256": "2" * 64, "reason": "too late",
                                          "affects": ["deribit-eth-perp-2020-07-01"]}, protocol=PROTOCOL)


def test_final_artifacts_bind_to_frozen_registration(ledger):
    _access(ledger, "deribit-eth-perp-2020-07-01")
    config, result = {"horizons": [1, 10]}, {"status": "FAILED", "loss": 0.9}
    binding = pr.bind_evidence(protocol=PROTOCOL, ledger_path=ledger, analysis="m3",
                               dataset_ids=["deribit-eth-perp-2020-07-01", "deribit-eth-perp-2020-04-01"],
                               config=config, result=result, provenance=PROVENANCE)
    ok = pr.verify_binding(binding, protocol=PROTOCOL, ledger_path=ledger, config=config, result=result,
                           source_sha256="a" * 64, expected_commit="b" * 40)
    assert ok["valid"], ok["issues"]
    # Later ledger growth does not invalidate an earlier anchored binding.
    pr.append_event(ledger, "seal_design", {"analysis": "later", "design_sha256": "f" * 64, "reads": []},
                    protocol=PROTOCOL)
    assert pr.verify_binding(binding, protocol=PROTOCOL, ledger_path=ledger, config=config,
                             result=result)["valid"]
    changed = copy.deepcopy(PROTOCOL)
    changed["hypotheses"]["RQ3"] = "rewritten after results"
    assert not pr.verify_binding(binding, protocol=changed, ledger_path=ledger, config=config, result=result)["valid"]
    for kwargs in ({"config": {"horizons": [1]}, "result": result},
                   {"config": config, "result": {"status": "ESTABLISHED", "loss": 0.9}}):
        assert not pr.verify_binding(binding, protocol=PROTOCOL, ledger_path=ledger, **kwargs)["valid"]
    assert not pr.verify_binding(binding, protocol=PROTOCOL, ledger_path=ledger, config=config,
                                 result=result, source_sha256="9" * 64)["valid"]
    assert not pr.verify_binding(binding, protocol=PROTOCOL, ledger_path=ledger, config=config,
                                 result=result, expected_commit="0" * 40)["valid"]


def test_dataset_identity_change_invalidates_binding(ledger):
    dataset = "deribit-eth-perp-2020-07-01"
    _access(ledger, dataset)
    config, result = {}, {"status": "NOT_ESTABLISHED"}
    binding = pr.bind_evidence(protocol=PROTOCOL, ledger_path=ledger, analysis="m4", dataset_ids=[dataset],
                               config=config, result=result, provenance=PROVENANCE)
    tampered = copy.deepcopy(binding)
    tampered["datasets"][dataset]["source_sha256"] = {"l2.csv.gz": "9" * 64}
    assert not pr.verify_binding(tampered, protocol=PROTOCOL, ledger_path=ledger, config=config,
                                 result=result)["valid"]
    # Changing the declared instrument changes the identity even with identical bytes.
    changed = copy.deepcopy(PROTOCOL)
    next(d for d in changed["datasets"] if d["id"] == dataset)["instrument"] = "BTC-PERPETUAL"
    declaration = pr.dataset_declaration(changed, dataset)
    assert (pr.dataset_identity_sha256(declaration, {"l2.csv.gz": "c" * 64})
            != binding["datasets"][dataset]["identity_sha256"])


def test_binding_refused_before_freeze(tmp_path):
    path = tmp_path / "ledger.jsonl"
    pr.append_event(path, "declare_fresh", {"dataset_id": "x"})
    with pytest.raises(pr.ProtocolError, match="before protocol freeze"):
        pr.bind_evidence(protocol=PROTOCOL, ledger_path=path, analysis="a", dataset_ids=[],
                         config={}, result={}, provenance=PROVENANCE)


def test_referenced_config_mutation_detected(tmp_path):
    root = tmp_path / "repo"
    for name in ["configs/v05/protocol.json", *PROTOCOL["referenced_configs"]]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes((ROOT / name).read_bytes())
    pr.initialize_ledger(root / "configs/v05/consumption-ledger.jsonl", PROTOCOL, root,
                         consumed_registry=CONSUMED)
    assert pr.verify_protocol_files(root)["valid"]
    target = root / "configs/v05/policy-study.json"
    target.write_text(target.read_text(encoding="utf-8").replace("16384", "4096"), encoding="utf-8")
    assert not pr.verify_protocol_files(root)["valid"]


def test_initialize_ledger_is_append_only(ledger):
    with pytest.raises(pr.ProtocolError, match="append-only"):
        pr.initialize_ledger(ledger, PROTOCOL, ROOT, consumed_registry=CONSUMED)


def test_renamed_alias_of_consumed_period_cannot_be_declared_fresh(ledger):
    with pytest.raises(pr.ProtocolError, match="cannot be restored"):
        pr.append_event(ledger, "declare_fresh", {"dataset_id": "eth-june-renamed",
                                                  "period_key": "deribit:eth-perpetual:2020-06-01"},
                        protocol=PROTOCOL)
    # A consumed v0.5 dataset also consumes its venue/instrument/date alias.
    _access(ledger, "deribit-btc-perp-2020-07-01")
    with pytest.raises(pr.ProtocolError, match="cannot be restored"):
        pr.append_event(ledger, "declare_fresh", {"dataset_id": "deribit:BTC-PERPETUAL:2020-07-01"},
                        protocol=PROTOCOL)


def test_cli_protocol_verify(capsys):
    from lob.cli import main
    assert main(["protocol", "verify", "--root", str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_preregistered_completion_bound_is_attainable():
    # With 160 independent markets and no discordance, the exact bound must fit the margin.
    alpha = 0.05 / PROTOCOL["statistics"]["M7"]["family_size"]
    design = json.loads((ROOT / "configs/v05/policy-study.json").read_text(encoding="utf-8"))
    assert 1 - alpha ** (1 / design["evaluation_seed_count"]) < PROTOCOL["statistics"]["M7"]["noninferiority_margin"]

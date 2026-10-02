"""v0.7 governance: protocol validation, stage-aware ledger, holdout protection, taxonomy and binding."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from lob.cli import main as cli_main
from lob.v07.protocol import core as pr
from lob.v07.protocol import taxonomy as tx

ROOT = Path(__file__).resolve().parents[1]
ETH_NOV, ETH_DEC = "deribit-eth-perp-2020-11-01", "deribit-eth-perp-2020-12-01"
BTC_NOV, BITMEX_NOV, ETH_JAN = "deribit-btc-perp-2020-11-01", "bitmex-xbtusd-2020-11-01", "deribit-eth-perp-2021-01-01"
DEV, SEL, VAL = "deribit-eth-perp-2020-04-01", "deribit-eth-perp-2020-05-01", "deribit-eth-perp-2020-06-01"
DIGEST = "a" * 64
COPIED = ("configs/v05/protocol.json", "configs/v05/consumption-ledger.jsonl", "configs/v06/protocol.json",
          "configs/v06/datasets.json", "configs/v06/consumption-ledger.jsonl")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    for name in COPIED + tuple(f"configs/v07/{p.name}" for p in (ROOT / "configs/v07").glob("*.json")):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    pr.initialize_ledger(tmp_path)
    return tmp_path


def _protocol(root: Path) -> dict:
    return pr.load_protocol(root / pr.PROTOCOL_PATH)


def _append(root: Path, event: str, **kwargs) -> dict:
    return pr.append_event(root / pr.LEDGER_PATH, event, protocol=_protocol(root), root=root,
                           reason=kwargs.pop("reason", "test"), **kwargs)


def _seal(root: Path, name: str, reads: list[str], posthoc: bool = False) -> None:
    _append(root, "seal_design", design=name, payload={"design_sha256": DIGEST, "reads": reads, "posthoc": posthoc})


def _access(root: Path, dataset: str, use: str, design: str, stage: str = "downloaded", sources=None) -> dict:
    role = pr.dataset_declaration(_protocol(root), dataset)["role"]
    return _append(root, "access", dataset=dataset, role=role, design=design,
                   payload={"use": use, "stage": stage, "source_sha256": sources or {}})


def test_repository_protocol_and_configs_validate() -> None:
    protocol = pr.load_protocol(ROOT / pr.PROTOCOL_PATH, ROOT)
    assert {d["id"] for d in protocol["datasets"] if d["freshness_at_freeze"] == "fresh"} == {
        ETH_NOV, ETH_DEC, BTC_NOV, BITMEX_NOV, ETH_JAN}
    hypotheses = json.loads((ROOT / "configs/v07/hypotheses.json").read_text(encoding="utf-8"))["hypotheses"]
    assert sorted(hypotheses, key=lambda h: int(h[1:])) == [f"H{i}" for i in range(1, 14)]


def test_initialized_ledger_imports_history_and_freezes(root: Path) -> None:
    report = pr.verify_protocol_files(root)
    assert report["valid"], report["issues"]
    assert report["fresh"] == sorted([ETH_NOV, ETH_DEC, BTC_NOV, BITMEX_NOV, ETH_JAN])
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), _protocol(root))
    for dataset in (DEV, SEL, VAL, "deribit-eth-perp-2020-09-01", "deribit-eth-perp-2020-10-01",
                    "deribit-btc-perp-2020-09-01", "bitstamp-btcusd-mbo-dev"):
        assert state.freshness(dataset) == "consumed"
    assert "deribit:eth-perpetual:2020-10-01" in state.consumed_keys
    entries = pr.read_ledger(root / pr.LEDGER_PATH)
    assert all({"index", "at", "event", "dataset", "role", "design", "reason", "git_commit", "prev_sha256",
                "sha256"} <= set(e) for e in entries)


def test_initialize_refuses_existing_ledger_and_changed_history(root: Path, tmp_path_factory) -> None:
    with pytest.raises(pr.ProtocolError):
        pr.initialize_ledger(root)
    other = tmp_path_factory.mktemp("other")
    for name in COPIED + tuple(f"configs/v07/{p.name}" for p in (ROOT / "configs/v07").glob("*.json")):
        (other / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, other / name)
    with (other / "configs/v06/consumption-ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(pr.ProtocolError):
        pr.initialize_ledger(other)


def test_v06_ledger_change_after_freeze_is_detected(root: Path) -> None:
    path = root / "configs/v06/consumption-ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert not pr.verify_protocol_files(root)["valid"]


def test_protocol_and_config_modification_are_detected(root: Path) -> None:
    protocol = _protocol(root)
    protocol["research_question"] += " (edited)"
    (root / pr.PROTOCOL_PATH).write_text(json.dumps(protocol), encoding="utf-8")
    assert not pr.verify_protocol_files(root)["valid"]


def test_referenced_config_change_is_detected(root: Path) -> None:
    path = root / "configs/v07/compute-budget.json"
    path.write_text(path.read_text(encoding="utf-8").replace("256", "128", 1), encoding="utf-8")
    report = pr.verify_protocol_files(root)
    assert not report["valid"] and any("compute-budget" in issue for issue in report["issues"])


def test_line_ending_changes_do_not_break_config_hashes(root: Path) -> None:
    path = root / "configs/v07/compute-budget.json"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert pr.verify_protocol_files(root)["valid"]


@pytest.mark.parametrize("mutation", ["edit", "delete", "reorder", "append_garbage"])
def test_ledger_tampering_is_detected(root: Path, mutation: str) -> None:
    path = root / pr.LEDGER_PATH
    lines = path.read_text(encoding="utf-8").splitlines()
    if mutation == "edit":
        entry = json.loads(lines[3])
        entry["reason"] = "rewritten"
        lines[3] = json.dumps(entry)
    elif mutation == "delete":
        del lines[2]
    elif mutation == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines.append("not json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not pr.verify_protocol_files(root)["valid"]


@pytest.mark.parametrize("dataset,use", [(DEV, "select"), (SEL, "develop"), (VAL, "develop"), (ETH_NOV, "develop"),
                                         (ETH_JAN, "select"), ("deribit-eth-perp-2020-09-01", "evaluate")])
def test_role_violations_are_refused(root: Path, dataset: str, use: str) -> None:
    with pytest.raises(pr.ProtocolError):
        _access(root, dataset, use, "d")


def test_access_requires_known_stage_and_matching_role(root: Path) -> None:
    with pytest.raises(pr.ProtocolError):
        _access(root, DEV, "develop", "d", stage="peeked")
    with pytest.raises(pr.ProtocolError):
        _append(root, "access", dataset=DEV, role="selection", design="d",
                payload={"use": "develop", "stage": "opened"})


def test_development_access_and_stages_are_recorded(root: Path) -> None:
    for stage in pr.STAGES:
        _access(root, DEV, "develop", "dev-design", stage=stage)
    _access(root, VAL, "validate", "dev-design")
    report = pr.verify_protocol_files(root)
    assert report["valid"]


def test_holdout_requires_sealed_design_before_first_access(root: Path) -> None:
    with pytest.raises(pr.ProtocolError):
        _access(root, ETH_NOV, "download", "unsealed")
    _seal(root, "holdout", [ETH_NOV])
    with pytest.raises(pr.ProtocolError):
        _access(root, ETH_DEC, "download", "holdout")
    for stage in pr.STAGES:
        _access(root, ETH_NOV, "download" if stage == "downloaded" else "evaluate", "holdout", stage=stage)
    _append(root, "inspect", dataset=ETH_NOV, payload={"run": "results/v07/x"})
    report = pr.verify_protocol_files(root)
    assert report["holdouts"][ETH_NOV]["freshness"] == "consumed"
    assert report["holdouts"][ETH_NOV]["stages"] == list(pr.STAGES)
    assert report["holdouts"][ETH_NOV]["outcome_inspected_index"] is not None
    assert ETH_NOV not in report["fresh"]


def test_inspection_before_access_is_refused(root: Path) -> None:
    with pytest.raises(pr.ProtocolError):
        _append(root, "inspect", dataset=ETH_DEC, payload={"run": "x"})


def test_consumed_holdout_requires_posthoc_design(root: Path) -> None:
    _seal(root, "first", [BTC_NOV])
    _access(root, BTC_NOV, "download", "first")
    with pytest.raises(pr.ProtocolError):
        _seal(root, "second", [BTC_NOV])
    _seal(root, "second", [BTC_NOV], posthoc=True)
    with pytest.raises(pr.ProtocolError):
        _access(root, BTC_NOV, "evaluate", "second", stage="evaluated")
    _access(root, BTC_NOV, "evaluate_posthoc", "second", stage="evaluated")


def test_posthoc_must_be_a_strict_boolean(root: Path) -> None:
    with pytest.raises(pr.ProtocolError):
        _append(root, "seal_design", design="x", payload={"design_sha256": DIGEST, "reads": [], "posthoc": None})


def test_consumed_period_cannot_be_declared_fresh_again(root: Path) -> None:
    with pytest.raises(pr.ProtocolError):
        _append(root, "declare_fresh", dataset="deribit-eth-perp-2020-10-01", role="fresh_temporal",
                payload={"period_key": "deribit:eth-perpetual:2020-10-01", "identity_sha256": DIGEST})


def test_source_bytes_cannot_change(root: Path) -> None:
    _access(root, DEV, "develop", "d", sources={"book.csv.gz": "b" * 64})
    with pytest.raises(pr.ProtocolError):
        _access(root, DEV, "develop", "d", stage="opened", sources={"book.csv.gz": "c" * 64})


def test_attempts_and_amendments(root: Path) -> None:
    _append(root, "attempt", design="m6", payload={"outcome": "FAILED", "directory": "results/v07/m6/a1",
                                                   "note": "diverged"})
    with pytest.raises(pr.ProtocolError):
        _append(root, "attempt", design="m6", payload={"outcome": "FAILED", "directory": None, "note": "x"})
    with pytest.raises(pr.ProtocolError):
        _append(root, "attempt", design="m6", payload={"outcome": "DONE", "directory": "x", "note": "x"})
    document = copy.deepcopy(_protocol(root))
    document["early_stopping"] += " Clarified."
    pr.amend_protocol(root, document, number=1, reason="clarify", affects=[])
    report = pr.verify_protocol_files(root)
    assert report["valid"] and report["amendments"] == 1 and report["attempts"][0]["outcome"] == "FAILED"
    _seal(root, "h", [ETH_DEC])
    _access(root, ETH_DEC, "download", "h")
    with pytest.raises(pr.ProtocolError):
        pr.amend_protocol(root, document, number=2, reason="late", affects=[ETH_DEC])


@pytest.mark.parametrize("edit,message", [
    (lambda p: p.update(statuses=p["statuses"][:-1]), "ten v0.7 statuses"),
    (lambda p: p["datasets"][11].update(freshness_at_freeze="consumed"), "holdout roles"),
    (lambda p: p["datasets"].append(dict(p["datasets"][0])), "duplicate"),
    (lambda p: p["referenced_configs"].remove("configs/v07/hypotheses.json"), "hypotheses"),
    (lambda p: p.update(base_commit="HEAD"), "base_commit"),
    (lambda p: p["referenced_configs"].append("../outside.json"), "unsafe"),
])
def test_protocol_validation_rejects_inconsistent_documents(edit, message: str) -> None:
    document = json.loads((ROOT / pr.PROTOCOL_PATH).read_text(encoding="utf-8"))
    edit(document)
    with pytest.raises(pr.ProtocolError, match=message):
        pr.validate_protocol(document)


def test_binding_roundtrip_and_tamper_detection(root: Path) -> None:
    _access(root, DEV, "develop", "d", sources={"book.csv.gz": "b" * 64})
    protocol, ledger = _protocol(root), root / pr.LEDGER_PATH
    binding = pr.bind_evidence(protocol=protocol, ledger_path=ledger, analysis="t", dataset_ids=[DEV],
                               config={"a": 1}, result={"r": 2}, provenance={"git_commit": "x"})
    assert pr.verify_binding(binding, protocol=protocol, ledger_path=ledger, config={"a": 1}, result={"r": 2}) == []
    assert pr.verify_binding(binding, protocol=protocol, ledger_path=ledger, config={"a": 2}, result={"r": 2})
    lines = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert pr.verify_binding(binding, protocol=protocol, ledger_path=ledger, config={"a": 1}, result={"r": 2})


def test_cli_protocol_v07(root: Path, capsys) -> None:
    assert cli_main(["protocol-v07", "verify", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    (root / pr.LEDGER_PATH).write_text("{}\n", encoding="utf-8")
    assert cli_main(["protocol-v07", "status", "--root", str(root)]) == 1


# ----------------------------------------------------------------------------- taxonomy


def test_result_taxonomy_rules() -> None:
    tx.Result("r", "H1", "ESTABLISHED", estimate=-1.0, ci_low=-2.0, ci_high=-0.5, qualifiers=("CONFIRMATORY",))
    with pytest.raises(tx.TaxonomyError):
        tx.Result("r", "H1", "PROBABLY")
    with pytest.raises(tx.TaxonomyError):
        tx.Result("r", "H1", "NOT_AVAILABLE")
    with pytest.raises(tx.TaxonomyError):
        tx.Result("r", "H1", "ESTABLISHED", qualifiers=("CONFIRMATORY",))
    with pytest.raises(tx.TaxonomyError):
        tx.Result("r", "H1", "FAILED", ci_low=1.0, ci_high=0.0)
    with pytest.raises(tx.TaxonomyError):
        tx.Result("r", "H1", "FAILED", estimate=float("nan"))


def test_multiplicity_registry_matches_config() -> None:
    document = json.loads((ROOT / "configs/v07/statistical-families.json").read_text(encoding="utf-8"))
    registry = tx.MultiplicityRegistry.from_config(document)
    for name, family in document["families"].items():
        assert registry.families[name].adjusted_alpha == pytest.approx(family["adjusted_alpha"])
        if family["comparisons"]:
            assert len(family["comparisons"]) == family["size"]
    assert registry.check("F8_robust_pairs", "twap|vwap").size == 28
    with pytest.raises(tx.TaxonomyError):
        registry.check("F8_robust_pairs", "twap|mpc")
    with pytest.raises(tx.TaxonomyError):
        registry.check("unregistered", "x")
    with pytest.raises(tx.TaxonomyError):
        registry.check("D1_posterior_shape", "x")
    holm = registry.holm("F7_model_uncertainty", {"twap": 0.001, "vwap": 0.008, "pov": 0.0071})
    assert holm == {"twap": True, "pov": True, "vwap": True}
    assert registry.holm("F7_model_uncertainty", {"twap": 0.001, "vwap": 0.008})["vwap"] is False


def test_equivalence_requires_margin() -> None:
    with pytest.raises(tx.TaxonomyError):
        tx.equivalence(0.1, 0.0, None)
    with pytest.raises(tx.TaxonomyError):
        tx.two_sided_equivalence(-0.1, 0.1, None)
    assert tx.equivalence(0.5, 0.1, 1.0) == "EQUIVALENT_WITHIN_MARGIN"
    assert tx.equivalence(3.0, 1.5, 1.0) == "FAILED_MARGIN"
    assert tx.two_sided_equivalence(-0.5, 0.5, 1.0) == "EQUIVALENT_WITHIN_MARGIN"
    assert tx.two_sided_equivalence(-0.5, 1.5, 1.0) == "NOT_ESTABLISHED"
    assert tx.two_sided_equivalence(None, 1.5, 1.0) == "NOT_EVALUABLE"


def test_interval_status() -> None:
    assert tx.interval_status(-2.0, -0.1) == "ESTABLISHED"
    assert tx.interval_status(0.1, 2.0) == "FAILED"
    assert tx.interval_status(-1.0, 1.0) == "NOT_ESTABLISHED"
    assert tx.interval_status(-2.0, -0.1, margin=0.5) == "NOT_ESTABLISHED"
    assert tx.interval_status(0.6, 2.0, direction=1, margin=0.5) == "ESTABLISHED"
    assert tx.interval_status(None, 1.0) == "INVALID"


def test_ledger_lock_serializes_concurrent_appends(root: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as executor:
        list(executor.map(lambda i: _append(root, "attempt", design=f"d{i}", payload={
            "outcome": "COMPLETED", "directory": f"results/v07/x{i}", "note": "concurrency"}), range(16)))
    report = pr.verify_protocol_files(root)
    assert report["valid"] and len(report["attempts"]) == 16
    assert not (root / (pr.LEDGER_PATH + ".lock")).exists()

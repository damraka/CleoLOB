"""v0.6 protocol, ledger rules, holdout protection, binding and ledger-gated acquisition."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from lob.cli import main as cli_main
from lob.v06 import data as v06_data
from lob.v06 import evidence
from lob.v06 import protocol as pr

ROOT = Path(__file__).resolve().parents[1]
ETH_SEP, BTC_SEP, ETH_OCT = "deribit-eth-perp-2020-09-01", "deribit-btc-perp-2020-09-01", "deribit-eth-perp-2020-10-01"
DEV, SEL, RETRO = "deribit-eth-perp-2020-04-01", "deribit-eth-perp-2020-05-01", "deribit-eth-perp-2020-06-01"
DIGEST = "a" * 64


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    for name in ("configs/v06/protocol.json", "configs/v06/datasets.json", "configs/v05/protocol.json",
                 "configs/v05/consumption-ledger.jsonl"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    pr.initialize_ledger(tmp_path)
    return tmp_path


def _protocol(root: Path) -> dict:
    return pr.load_protocol(root / pr.PROTOCOL_PATH)


def _ledger(root: Path) -> Path:
    return root / pr.LEDGER_PATH


def _append(root: Path, event: str, payload: dict) -> dict:
    return pr.append_event(_ledger(root), event, payload, protocol=_protocol(root))


def _seal(root: Path, name: str, reads: list[str], posthoc: bool = False) -> None:
    _append(root, "seal_design", {"analysis": name, "design_sha256": DIGEST, "reads": reads, "posthoc": posthoc})


def test_initialized_ledger_imports_v05_history_and_freezes(root: Path) -> None:
    report = pr.verify_protocol_files(root)
    assert report["valid"], report["issues"]
    assert report["fresh"] == sorted([ETH_SEP, BTC_SEP, ETH_OCT])
    state = pr.replay_ledger(pr.read_ledger(_ledger(root)), _protocol(root))
    for dataset in (DEV, SEL, RETRO, "deribit-eth-perp-2020-08-01", "deribit-btc-perp-2020-07-01"):
        assert state.freshness(dataset) == "consumed"
    assert "deribit:eth-perpetual:2020-08-01" in state.consumed_keys
    assert all(h["first_access_index"] is None for h in report["holdouts"].values())


def test_initialize_refuses_existing_ledger(root: Path) -> None:
    with pytest.raises(pr.ProtocolError):
        pr.initialize_ledger(root)


def test_initialize_refuses_changed_v05_history(tmp_path: Path) -> None:
    for name in ("configs/v06/protocol.json", "configs/v06/datasets.json", "configs/v05/protocol.json",
                 "configs/v05/consumption-ledger.jsonl"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, tmp_path / name)
    lines = (tmp_path / "configs/v05/consumption-ledger.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "configs/v05/consumption-ledger.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(pr.ProtocolError, match="pinned v0.5 history"):
        pr.initialize_ledger(tmp_path)


def test_protocol_modification_after_freeze_is_detected(root: Path) -> None:
    document = json.loads((root / pr.PROTOCOL_PATH).read_text(encoding="utf-8"))
    document["hypotheses"]["H1"]["threshold"] = "upper bound < 0.1"
    (root / pr.PROTOCOL_PATH).write_text(json.dumps(document), encoding="utf-8")
    report = pr.verify_protocol_files(root)
    assert not report["valid"] and any("protocol changed" in i for i in report["issues"])


def test_referenced_config_change_is_detected(root: Path) -> None:
    path = root / "configs/v06/datasets.json"
    path.write_text(path.read_text(encoding="utf-8").replace("UTC", "utc"), encoding="utf-8")
    report = pr.verify_protocol_files(root)
    assert any("referenced config changed" in i for i in report["issues"])


def test_v05_ledger_rewrite_or_extension_is_detected(root: Path) -> None:
    path = root / pr.V05_LEDGER_PATH
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[-1])
    entry["payload"]["purpose"] = "rewritten"
    lines[-1] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not pr.verify_protocol_files(root)["valid"]


@pytest.mark.parametrize("mutation", ["hash", "chain", "index", "blank", "malformed", "foreign"])
def test_ledger_tampering_is_detected(root: Path, mutation: str) -> None:
    path = _ledger(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[3])
    if mutation == "hash":
        entry["payload"]["evidence"] = "elsewhere"
        lines[3] = json.dumps(entry)
    elif mutation == "chain":
        entry["prev_sha256"] = "0" * 64
        lines[3] = json.dumps(entry)
    elif mutation == "index":
        lines[3], lines[4] = lines[4], lines[3]
    elif mutation == "blank":
        lines.insert(2, "")
    elif mutation == "malformed":
        lines[3] = lines[3][:-5]
    else:
        entry["schema"] = "cleolob-v05-ledger-1"
        lines[3] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = pr.verify_protocol_files(root)
    assert not report["valid"]


@pytest.mark.parametrize("dataset,use", [(DEV, "evaluate"), (DEV, "select"), (SEL, "develop"), (RETRO, "evaluate"),
                                         (RETRO, "develop"), (ETH_SEP, "develop"), (ETH_SEP, "select"),
                                         (ETH_SEP, "retrospective")])
def test_role_violations_are_refused(root: Path, dataset: str, use: str) -> None:
    with pytest.raises(pr.ProtocolError, match="not permitted"):
        _append(root, "access", {"dataset_id": dataset, "use": use, "analysis": "x"})


def test_permitted_development_and_retrospective_access(root: Path) -> None:
    _append(root, "access", {"dataset_id": DEV, "use": "develop", "analysis": "fit"})
    _append(root, "access", {"dataset_id": RETRO, "use": "retrospective", "analysis": "h1"})
    assert pr.verify_protocol_files(root)["valid"]


def test_holdout_requires_sealed_design_before_first_access(root: Path) -> None:
    with pytest.raises(pr.ProtocolError, match="without a sealed design"):
        _append(root, "access", {"dataset_id": ETH_SEP, "use": "evaluate", "analysis": "h2"})
    _seal(root, "h2", [ETH_SEP])
    with pytest.raises(pr.ProtocolError, match="without a sealed design"):
        _append(root, "access", {"dataset_id": BTC_SEP, "use": "evaluate", "analysis": "h2"})
    _append(root, "access", {"dataset_id": ETH_SEP, "use": "evaluate", "analysis": "h2"})
    report = pr.verify_protocol_files(root)
    assert report["holdouts"][ETH_SEP]["freshness"] == "consumed"
    assert ETH_SEP not in report["fresh"]


def test_consumed_holdout_cannot_be_reused_by_a_new_registered_design(root: Path) -> None:
    _seal(root, "h2", [ETH_SEP])
    _append(root, "access", {"dataset_id": ETH_SEP, "use": "evaluate", "analysis": "h2"})
    with pytest.raises(pr.ProtocolError, match="must be declared posthoc"):
        _seal(root, "h2-again", [ETH_SEP])
    _seal(root, "h2-posthoc", [ETH_SEP], posthoc=True)
    with pytest.raises(pr.ProtocolError, match="posthoc"):
        _append(root, "access", {"dataset_id": ETH_SEP, "use": "evaluate", "analysis": "h2-posthoc"})
    _append(root, "access", {"dataset_id": ETH_SEP, "use": "evaluate_posthoc", "analysis": "h2-posthoc"})
    with pytest.raises(pr.ProtocolError, match="posthoc"):
        _append(root, "access", {"dataset_id": ETH_SEP, "use": "evaluate_posthoc", "analysis": "h2"})


def test_consumed_period_cannot_be_declared_fresh_again() -> None:
    state = pr.LedgerState()
    entries = []
    head = "0" * 64
    for index, (event, payload) in enumerate([
            ("import_consumed", {"dataset_id": "old-id", "evidence": "x", "period_key": "deribit:eth-perpetual:2020-09-01"}),
            ("declare_fresh", {"dataset_id": "renamed", "role": "fresh_external", "period_key": "deribit:eth-perpetual:2020-09-01",
                               "identity_sha256": DIGEST})]):
        entry = {"schema": pr.LEDGER_SCHEMA, "index": index, "event": event, "payload": payload, "at": "t",
                 "prev_sha256": head}
        entry["sha256"] = pr._entry_hash(entry)
        head = entry["sha256"]
        entries.append(entry)
    state.apply(entries[0])
    with pytest.raises(pr.ProtocolError, match="freshness cannot be restored"):
        state.apply(entries[1])


def test_source_bytes_cannot_change(root: Path) -> None:
    _append(root, "access", {"dataset_id": DEV, "use": "develop", "analysis": "a", "source_sha256": {"f.csv.gz": DIGEST}})
    with pytest.raises(pr.ProtocolError, match="source bytes changed"):
        _append(root, "access", {"dataset_id": DEV, "use": "develop", "analysis": "a",
                                 "source_sha256": {"f.csv.gz": "b" * 64}})


def test_amendments(root: Path) -> None:
    _seal(root, "h2", [ETH_SEP])
    _append(root, "access", {"dataset_id": ETH_SEP, "use": "evaluate", "analysis": "h2"})
    with pytest.raises(pr.ProtocolError, match="already accessed"):
        _append(root, "amend", {"protocol_sha256": DIGEST, "reason": "r", "affects": [ETH_SEP]})
    document = json.loads((root / pr.PROTOCOL_PATH).read_text(encoding="utf-8"))
    document["exclusions"].append("amended exclusion")
    new_hash = pr.protocol_sha256(document)
    _append(root, "amend", {"protocol_sha256": new_hash, "reason": "clarify", "affects": [ETH_OCT]})
    (root / pr.PROTOCOL_PATH).write_text(json.dumps(document), encoding="utf-8")
    assert pr.verify_protocol_files(root)["valid"]


def test_attempts_are_recorded_and_reported(root: Path) -> None:
    with pytest.raises(pr.ProtocolError):
        _append(root, "attempt", {"analysis": "x", "outcome": "BROKEN", "directory": "d", "note": "n"})
    _append(root, "attempt", {"analysis": "x", "outcome": "INVALID", "directory": "results/v06/x/attempt1", "note": "bug"})
    assert pr.verify_protocol_files(root)["attempts"][0]["outcome"] == "INVALID"


@pytest.mark.parametrize("edit,message", [
    (lambda p: p["statuses"].pop(), "eight"),
    (lambda p: p["datasets"][6].update(freshness_at_freeze="consumed"), "holdout roles"),
    (lambda p: p["datasets"][2].update(freshness_at_freeze="fresh"), "holdout roles"),
    (lambda p: p["datasets"][1].update(role="development"), "exactly one"),
    (lambda p: p["statistics"]["families"]["F1_calibration"].update(adjusted_alpha=0.05), "adjusted alpha"),
    (lambda p: p["statistics"]["families"]["F7_regime"].update(correction="fdr_bh"), "Bonferroni or Holm"),
    (lambda p: p["hypotheses"]["H1"].pop("threshold"), "missing"),
    (lambda p: p["hypotheses"]["H2"].update(family="F99"), "unknown multiplicity"),
    (lambda p: p["hypotheses"]["H2"].update(datasets=["nope-2020-01-01"]), "undeclared"),
    (lambda p: p.update(base_commit="abc"), "full git SHA"),
    (lambda p: p["referenced_configs"].append("../secret"), "unsafe"),
])
def test_protocol_validation_rejects_inconsistent_documents(edit, message: str) -> None:
    document = copy.deepcopy(json.loads((ROOT / pr.PROTOCOL_PATH).read_text(encoding="utf-8")))
    edit(document)
    with pytest.raises(pr.ProtocolError, match=message):
        pr.validate_protocol(document)


def test_binding_roundtrip_and_tamper_detection(root: Path) -> None:
    out = evidence.new_run(root / "results/v06/run")
    evidence.finalize(out, analysis="unit", dataset_ids=[DEV], config={"a": 1}, result={"status": "ESTABLISHED"},
                      root=root, seeds={"s": [1]}, check_source=False)
    assert evidence.verify_run(out, root=root)["valid"]
    with pytest.raises(FileExistsError):
        evidence.new_run(out)
    (out / "result.json").write_text('{"status":"FAILED"}\n', encoding="utf-8")
    report = evidence.verify_run(out, root=root)
    assert not report["valid"]
    assert evidence.verify_tree(root / "results/v06", root=root)["invalid"] == ["run"]


def test_binding_survives_ledgered_amendment_but_not_silent_edit(root: Path) -> None:
    out = evidence.new_run(root / "results/v06/run")
    evidence.finalize(out, analysis="unit", dataset_ids=[DEV], config={}, result={}, root=root, check_source=False)
    document = json.loads((root / pr.PROTOCOL_PATH).read_text(encoding="utf-8"))
    document["exclusions"].append("silently edited")
    (root / pr.PROTOCOL_PATH).write_text(json.dumps(document), encoding="utf-8")
    assert not evidence.verify_run(out, root=root)["valid"]
    pr.append_event(_ledger(root), "amend", {"protocol_sha256": pr.protocol_sha256(document), "reason": "r",
                                             "affects": [], "number": 1})
    assert evidence.verify_run(out, root=root)["valid"]


def test_binding_detects_ledger_truncation(root: Path) -> None:
    _append(root, "access", {"dataset_id": DEV, "use": "develop", "analysis": "a"})
    out = evidence.new_run(root / "results/v06/run")
    evidence.finalize(out, analysis="unit", dataset_ids=[DEV], config={}, result={}, root=root, check_source=False)
    lines = _ledger(root).read_text(encoding="utf-8").splitlines()
    _ledger(root).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert not evidence.verify_run(out, root=root)["valid"]


def test_holdout_download_is_ledgered_before_any_request(root: Path) -> None:
    _seal(root, "h2", [ETH_SEP])
    seen = []

    def downloader(venue, symbol, date, kind, out, **_):
        entries = pr.read_ledger(_ledger(root))
        seen.append(entries[-1]["payload"].get("use"))
        out.mkdir(parents=True, exist_ok=True)
        (out / v06_data.file_name(venue, kind, date, symbol)).write_bytes(b"x")

    files = v06_data.acquire(ETH_SEP, use="evaluate", analysis="h2", purpose="test", root=root, downloader=downloader)
    assert seen == ["download", "download"] and set(files) == {"l2", "trades"}
    last = pr.read_ledger(_ledger(root))[-1]["payload"]
    assert last["use"] == "evaluate" and len(last["source_sha256"]) == 2


def test_failed_holdout_download_is_not_available_without_substitution(root: Path) -> None:
    _seal(root, "h2", [ETH_SEP])
    calls = []

    def downloader(*args, **kwargs):
        calls.append(args)
        raise OSError("network down")

    with pytest.raises(v06_data.DataNotAvailable):
        v06_data.acquire(ETH_SEP, use="evaluate", analysis="h2", purpose="test", root=root, downloader=downloader)
    assert len(calls) == 2 and all(c[2] == "2020-09-01" and c[1] == "ETH-PERPETUAL" for c in calls)
    report = pr.verify_protocol_files(root)
    assert report["attempts"][-1]["outcome"] == "NOT_AVAILABLE"
    assert report["holdouts"][ETH_SEP]["freshness"] == "consumed"


def test_acquire_refuses_unsealed_holdout_and_wrong_use(root: Path) -> None:
    with pytest.raises(pr.ProtocolError):
        v06_data.acquire(ETH_SEP, use="evaluate", analysis="none", purpose="t", root=root, downloader=lambda *a, **k: None)
    with pytest.raises(pr.ProtocolError):
        v06_data.acquire(DEV, use="evaluate", analysis="none", purpose="t", root=root)
    with pytest.raises(pr.ProtocolError):
        v06_data.acquire(ETH_SEP, use="download", analysis="none", purpose="t", root=root)


def test_cli_protocol_v06(root: Path, capsys) -> None:
    assert cli_main(["protocol-v06", "verify", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    (root / pr.LEDGER_PATH).write_text("{}\n", encoding="utf-8")
    assert cli_main(["protocol-v06", "status", "--root", str(root)]) == 1
    capsys.readouterr()
    assert cli_main(["verify-v06", str(root / "missing")]) == 1

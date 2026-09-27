"""M10: v0.5 evidence verification detects every registered class of tampering."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lob import preregistration as pr
from lob.v05_evidence import finalize, new_run, verify_run, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def repo(tmp_path):
    """A private copy of the protocol, referenced configs and ledger."""
    root = tmp_path / "repo"
    for name in ["configs/v05/protocol.json", "configs/v05/consumption-ledger.jsonl",
                 "configs/v04-consumed-data.json", "configs/v04-policy-study.json", "configs/v05/policy-study.json"]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    return root


@pytest.fixture()
def run(repo, tmp_path):
    out = new_run(tmp_path / "run")
    write_json(out, "detail.json", {"rows": [1, 2, 3]})
    finalize(out, analysis="tamper-test", dataset_ids=["deribit-eth-perp-2020-04-01"],
             config={"horizons": [1, 10]}, result={"status": "FAILED", "loss": 1.5}, root=repo)
    return out


def _mutate_json(path: Path, key: str, value) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data[key] = value
    path.write_text(json.dumps(data), encoding="utf-8")


def test_clean_run_verifies(run, repo):
    assert verify_run(run, root=repo)["valid"]


@pytest.mark.parametrize("name,key,value", [("result.json", "loss", 0.1), ("config.json", "horizons", [1]),
                                            ("detail.json", "rows", [])])
def test_result_config_and_detail_mutation(run, repo, name, key, value):
    _mutate_json(run / name, key, value)
    assert not verify_run(run, root=repo)["valid"]


def test_missing_and_extra_files(run, repo):
    (run / "detail.json").unlink()
    assert not verify_run(run, root=repo)["valid"]


def test_stale_artifact_added(run, repo):
    (run / "stale.json").write_text("{}", encoding="utf-8")
    assert "artifact file set changed" in verify_run(run, root=repo)["issues"]


def test_registration_protocol_mutation(run, repo):
    path = repo / "configs/v05/protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["success_gates"]["M3"]["selection"] = "changed after results"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    assert not verify_run(run, root=repo)["valid"]


def test_ledger_truncation_is_detected(run, repo):
    path = repo / "configs/v05/consumption-ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
    assert not verify_run(run, root=repo)["valid"]


def test_wrong_dataset_identity(run, repo):
    binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
    binding["datasets"]["deribit-eth-perp-2020-04-01"]["identity_sha256"] = "0" * 64
    protocol = pr.load_protocol(repo / "configs/v05/protocol.json")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    checked = pr.verify_binding(binding, protocol=protocol, ledger_path=repo / "configs/v05/consumption-ledger.jsonl",
                                config=config, result=result)
    assert not checked["valid"] and any("identity" in issue for issue in checked["issues"])


def test_wrong_commit_and_source(run, repo):
    binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
    protocol = pr.load_protocol(repo / "configs/v05/protocol.json")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    kwargs = dict(protocol=protocol, ledger_path=repo / "configs/v05/consumption-ledger.jsonl", config=config,
                  result=result)
    assert not pr.verify_binding(binding, expected_commit="f" * 40, **kwargs)["valid"]
    assert not pr.verify_binding(binding, source_sha256="e" * 64, **kwargs)["valid"]


def test_v05_policy_study_detects_checkpoint_normalization_lock_and_journal_mutation(tmp_path):
    pilot = ROOT / "results/v05/m7/pilot"
    if not (pilot / "manifest.json").exists():
        pytest.skip("local pilot study not present")
    from lob.policy_study_v05 import verify
    for target, change in (("models", "checkpoint"), ("normalization.json", "normalization"),
                           ("evaluation_lock.json", "lock"), ("episodes.jsonl", "journal")):
        copy = tmp_path / change
        shutil.copytree(pilot, copy)
        if target == "models":
            model = sorted((copy / "models").glob("*.zip"))[0]
            model.write_bytes(model.read_bytes() + b"x")
        elif target == "episodes.jsonl":
            lines = (copy / target).read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0])
            row["mandate_completion_adjusted_cost_bps"] = -99.0
            lines[0] = json.dumps(row)
            (copy / target).write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            _mutate_json(copy / target, "tampered", True)
        assert not verify(copy)["valid"], change

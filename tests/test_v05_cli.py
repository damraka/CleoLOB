"""M12: v0.5 CLI commands validate input, refuse misuse and keep v0.4 commands intact."""
from __future__ import annotations

import json

import pytest

from lob.cli import main, parser

V05 = ("validate-mbo-source", "fill-bounds", "historical-execution", "calibration-v2", "impact-study",
       "resilience-study", "regime-study", "policy-study-v05", "transfer-study", "transfer-regimes", "v05-benchmark", "verify-v05",
       "dataset-registry", "protocol")
V04 = ("validate-mbo", "multiperiod-study", "policy-study", "scaling-study", "verify-evidence", "smoke",
       "verify-artifact")


def test_all_v05_and_v04_commands_are_registered():
    choices = parser()._subparsers._group_actions[0].choices
    assert set(V05) <= set(choices) and set(V04) <= set(choices)


@pytest.mark.parametrize("argv", [["calibration-v2", "select", "--out", "x"],
                                  ["regime-study", "evaluate", "--select", "s", "--out", "x"],
                                  ["fill-bounds", "not-a-registered-dataset", "--out", "x"]])
def test_misuse_is_invalid_not_a_research_result(argv, capsys, tmp_path):
    argv = [a if a != "x" else str(tmp_path / "out") for a in argv]
    assert main(argv) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID"


def test_dataset_registry_and_protocol_commands(capsys):
    assert main(["dataset-registry"]) == 0
    registry = json.loads(capsys.readouterr().out)
    assert registry["deribit-eth-perp-2020-06-01"]["freshness"] == "consumed"
    assert main(["protocol", "status"]) == 0


def test_verify_v05_rejects_non_run(capsys, tmp_path):
    assert main(["verify-v05", str(tmp_path)]) == 1


def test_missing_optional_dependency_names_the_extra(capsys, tmp_path, monkeypatch):
    import lob.benchmarks_v05 as bench

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'stable_baselines3'", name="stable_baselines3")

    monkeypatch.setattr(bench, "run", missing)
    assert main(["v05-benchmark", "--out", str(tmp_path / "b")]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "MISSING_DEPENDENCY" and out["extra"] == "rl" and "cleolob[rl]" in out["install"]


def test_unrelated_missing_module_is_not_masked(tmp_path, monkeypatch):
    import lob.benchmarks_v05 as bench

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'nonexistent'", name="nonexistent")

    monkeypatch.setattr(bench, "run", missing)
    with pytest.raises(ModuleNotFoundError):
        main(["v05-benchmark", "--out", str(tmp_path / "b")])

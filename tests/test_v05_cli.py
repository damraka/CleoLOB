"""M12: v0.5 CLI commands validate input, refuse misuse and keep v0.4 commands intact."""
from __future__ import annotations

import json

import pytest

from lob.cli import main, parser

V05 = ("validate-mbo-source", "fill-bounds", "historical-execution", "calibration-v2", "impact-study",
       "resilience-study", "regime-study", "policy-study-v05", "transfer-study", "v05-benchmark", "verify-v05",
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

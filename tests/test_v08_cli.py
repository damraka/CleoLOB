"""v0.8 CLI: every command has --help; cheap commands run; failures are machine-readable; optional extras."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lob.cli import main as cli_main
from lob.v08.cli import HANDLERS, STUDIES

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("protocol-v08", "dataset-registry-v08", "validate-market-data-v08", "regime-study-v08",
            "change-point-study-v08", "adaptive-calibration-v08", "inference-v08", "identifiability-v3",
            "posterior-study-v08", "model-averaging-v08", "uncertainty-study-v08", "ood-study-v08",
            "conformal-study-v08", "calibration-utility-v08", "completion-study-v08", "robust-execution-v08",
            "mpc-study-v08", "policy-selection-v08", "policy-abstention-v08", "adversarial-stress-v08",
            "counterfactual-bounds-v08", "latency-study-v08", "transfer-study-v08", "ranking-transfer-v08",
            "benchmark-v08", "final-analysis-v08", "report-v08", "evidence-export-v08", "claim-audit-v08", "verify-v08")


def test_required_command_surface_is_registered() -> None:
    assert set(REQUIRED) <= set(HANDLERS)
    assert all(module for module, _, _ in STUDIES.values())


@pytest.mark.parametrize("command", sorted(HANDLERS))
def test_help(command, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main([command, "--help"])
    assert exc.value.code == 0 and command in capsys.readouterr().out


def test_protocol_verify_and_registry(capsys) -> None:
    assert cli_main(["protocol-v08", "verify", "--root", str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert cli_main(["dataset-registry-v08", "--root", str(ROOT)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "cleolob-v08-dataset-registry-3" and len(report["datasets"]) == 20


def test_validate_market_data_on_generated_tape(capsys) -> None:
    assert cli_main(["validate-market-data-v08", "--synthetic", "poisson", "--bins", "300", "--tick", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_missing_input_fails_with_machine_readable_status(capsys, tmp_path) -> None:
    code = cli_main(["validate-market-data-v08", "--l2", str(tmp_path / "missing.csv.gz")])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["status"] == "INVALID"


def test_confirm_namespace_without_design_fails_closed(capsys) -> None:
    code = cli_main(["mpc-study-v08", "--root", str(ROOT), "--out", "results/v08/never-created", "--namespace",
                     "confirm"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["status"] == "INVALID" and "design" in out["error"]
    assert not (ROOT / "results/v08/never-created").exists()


def test_study_requires_sealed_inputs(capsys, tmp_path) -> None:
    code = cli_main(["mpc-study-v08", "--root", str(ROOT), "--out", f"results/v08/dev/cli-test-{tmp_path.name}",
                     "--input", "worlds=results/v08/not-a-run"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and "not a sealed v0.8 run" in out["error"]


def test_verify_v08_on_the_repository(capsys) -> None:
    code = cli_main(["verify-v08", "--root", str(ROOT)])
    out = json.loads(capsys.readouterr().out)
    assert out["protocol"]["valid"] and code == (0 if out["valid"] else 1)


def test_optional_extras_are_declared() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for extra in ("inference", "neural", "distributed", "all"):
        assert f"{extra} = [" in text

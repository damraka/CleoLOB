"""v0.7 CLI: every command has working --help; bounded smoke runs of the cheap commands."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lob.cli import main as cli_main
from lob.v07.cli import HANDLERS
from tests.v07_fixtures import write_tardis

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", sorted(HANDLERS))
def test_help(command, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main([command, "--help"])
    assert exc.value.code == 0 and command in capsys.readouterr().out


def test_dataset_registry_reports_unavailable_capabilities(capsys) -> None:
    assert cli_main(["dataset-registry-v07", "--root", str(ROOT)]) == 0
    datasets = json.loads(capsys.readouterr().out)["datasets"]
    btc = next(d for d in datasets if d["id"] == "deribit-btc-perp-2020-11-01")
    assert btc["capabilities"]["exact_fifo_position"].startswith("NOT_AVAILABLE (reason: aggregate_l2")
    assert {d["role"] for d in datasets} >= {"development", "fresh_temporal", "final_transfer"}


def test_validate_and_replay_smoke(tmp_path, capsys) -> None:
    files = write_tardis(tmp_path, seconds=60.0)
    assert cli_main(["validate-market-data-v07", "--venue", "deribit", "--instrument", "ETH-PERPETUAL",
                     "--l2", str(files["l2"]), "--trades", str(files["trades"])]) == 0
    assert json.loads(capsys.readouterr().out)["records"] > 100
    args = ["replay-v07", "--venue", "deribit", "--instrument", "ETH-PERPETUAL", "--l2", str(files["l2"])]
    assert cli_main(args) == 0
    full = json.loads(capsys.readouterr().out)["chain"]
    assert cli_main(args + ["--checkpoint-at", "300"]) == 0
    assert json.loads(capsys.readouterr().out)["chain"] == full


def test_public_benchmark_reproduces(capsys) -> None:
    assert cli_main(["benchmark-v07", "public", "--root", str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)["reproduced"]


def test_study_run_defaults_point_to_sealed_runs() -> None:
    """Regression: defaults must name sealed runs, never retained aborted attempts."""
    from lob.cli import parser
    from lob.v07.evidence.runs import verify_run
    args = parser().parse_args(["transfer-study-v07", "seal", "--out", "unused"])
    for name in ("generator_run", "posterior_run", "ea_run", "bank_run", "selection_run", "policy_run",
                 "prediction_run", "execution_run"):
        assert verify_run(ROOT / getattr(args, name), root=ROOT)["valid"], name

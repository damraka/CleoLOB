import json
from pathlib import Path

from lob.cli import main, parser


def test_v04_cli_has_installed_package_entrypoints():
    help_text = parser().format_help()
    for command in ("validate-mbo", "multiperiod-study", "scaling-study", "verify-evidence"):
        assert command in help_text


def test_v04_cli_invalid_sources_are_explicit(tmp_path, capsys):
    assert main(["verify-evidence", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False
    assert main(["multiperiod-study", "--out", str(tmp_path / "run")]) == 1
    assert "--config" in json.loads(capsys.readouterr().out)["error"]


def test_v04_multi_period_cli_retains_scientific_failure(tmp_path, capsys):
    config = Path(__file__).resolve().parents[1] / "configs/v04-multiperiod-smoke.json"
    out = tmp_path / "multi"
    assert main(["multiperiod-study", "--config", str(config), "--out", str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["real_market_generalization"] == "NOT_ESTABLISHED"
    assert main(["multiperiod-study", "--out", str(out), "--verify"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]

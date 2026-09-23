import json
from pathlib import Path

from lob.cli import main


def test_portable_pipeline_smoke_seals_and_rejects_tampering(tmp_path, monkeypatch, capsys):
    # No current-directory configs or examples; same behavior required of a wheel.
    monkeypatch.chdir(tmp_path)
    assert main(["smoke", "--out", "smoke"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["execution_episodes"] == 12
    assert main(["verify-artifact", "smoke"]) == 0
    result = json.loads((tmp_path / "smoke/result.json").read_text())
    assert result["mbo"]["complete"] and result["calibration"]["external_used_for_selection"] is False
    (tmp_path / "smoke/result.json").write_text("{}")
    assert main(["verify-artifact", "smoke"]) == 1
    assert main(["smoke", "--out", "smoke"]) == 1  # No overwritten evidence.


def test_mbo_cli_and_resource_refusal(capsys):
    fixture = Path(__file__).parents[1] / "examples/data/mbo-events.jsonl"
    assert main(["mbo-replay", str(fixture)]) == 0
    assert json.loads(capsys.readouterr().out)["complete"]
    assert main(["mbo-replay", str(fixture), "--max-events", "1"]) == 1

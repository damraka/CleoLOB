"""Compatibility harness keeps failures and never silently substitutes PPO."""
from legacy.evaluate import _run_task


def test_legacy_evaluation_records_missing_ppo_without_stopping_baselines(tmp_path):
    rows = _run_task("calm", 1, ["ppo", "twap"], {
        "model_path": str(tmp_path / "missing"), "qty": 100, "horizon": 0.1, "dt": 0.1})
    assert rows[0]["status"] == "FAILED" and rows[0]["agent"] == "ppo"
    assert "FileNotFoundError" in rows[0]["error"]
    assert rows[1]["implementation"] == "twap" and rows[1]["status"] in {"VALID", "WARNING"}

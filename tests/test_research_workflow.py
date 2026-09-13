"""Actual CLI and registry workflows, not mocked successful metrics."""
import json
from pathlib import Path

import pytest

from lob.cli import main
from lob.config import ResearchConfig
from lob.experiments import read_experiment, reproduce, run_experiment, verify_experiment
from lob.experiments.design import design_experiment


def small_config(**extra):
    base = {"name": "mechanics", "market": {"limit_rate": 0.0, "market_rate": 0.0,
              "cancel_rate": 0.0, "resilience": 0.0, "latency_base": 0.001, "latency_jitter": 0.0},
            "execution": {"quantity": 100, "horizon": 1.0, "decision_dt": 0.2},
            "evaluation": {"agents": ["twap", "ac"], "reference": "twap", "seeds": [1, 2], "bootstrap_samples": 100}}
    return ResearchConfig.model_validate(base | extra)


def test_end_to_end_experiment_is_sealed_audited_and_reproduces_exactly(tmp_path):
    cfg = small_config()
    run = run_experiment(cfg, tmp_path)
    assert verify_experiment(run)["valid"]
    saved = read_experiment(run)
    assert saved["metadata"]["config_sha256"]
    assert saved["result"]["coverage"]["episodes_attempted"] == 4
    assert len(saved["result"]["paired"]) == 1
    audit = json.loads((run / "logs" / "1-twap.json").read_text())
    assert audit["orders"] and audit["fills"] and audit["risk_events"]
    replay = reproduce(run, tmp_path)
    assert replay["valid_reproduction"] and replay["episode_bytes_identical"]
    assert Path(replay["comparison_file"]).is_file()
    assert Path(replay["reproduction"]) != run


def test_corrupted_results_or_config_fail_verification(tmp_path):
    run = run_experiment(small_config(), tmp_path)
    with (run / "episodes.csv").open("a") as handle:
        handle.write("forged row\n")
    assert not verify_experiment(run)["valid"]
    with pytest.raises(ValueError, match="integrity"):
        reproduce(run, tmp_path)


def test_invalid_depth_preserves_every_episode_and_withholds_all_inference(tmp_path):
    cfg = small_config(market={"limit_rate": 0.0, "market_rate": 0.0, "cancel_rate": 0.0,
                               "resilience": 0.0, "target_level_vol": 1}, risk={"kill_switch": True})
    run = run_experiment(cfg, tmp_path)
    saved = read_experiment(run)["result"]
    assert saved["status"] == "INVALID" and saved["summary"] == [] and saved["paired"] == []
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(row["status"] == "INVALID" for row in rows)
    assert all(row["effective_bps"] is None for row in rows)
    assert "Inference withheld" in " ".join(saved["warnings"])


def test_failures_not_discarded_and_remaining_agents_still_run(tmp_path, monkeypatch):
    from lob.experiments import runner
    original = runner.run_episode
    def fail_one(agent, params):
        if agent == "ac" and params["seed"] == 1:
            raise ArithmeticError("deliberate regression fixture")
        return original(agent, params)
    monkeypatch.setattr(runner, "run_episode", fail_one)
    run = run_experiment(small_config(), tmp_path)
    result = read_experiment(run)["result"]
    assert result["status"] == "FAILED" and result["outcome_counts"]["FAILED"] == 1
    assert sum(result["outcome_counts"].values()) == 4 and not result["paired"]


def test_runtime_limit_records_planned_but_unstarted_episodes(tmp_path):
    cfg = small_config(resources={"max_runtime_seconds": 1e-12})
    run = run_experiment(cfg, tmp_path)
    result = read_experiment(run)["result"]
    assert result["status"] == "PARTIAL" and result["outcome_counts"] == {"PARTIAL": 4}
    assert result["coverage"]["episodes_attempted"] == 0


def test_report_escapes_names_and_never_claims_alpha(tmp_path):
    run = run_experiment(small_config(name="<script>alert(1)</script>"), tmp_path)
    report = (run / "report.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in report and "&lt;script&gt;" in report
    assert "INSUFFICIENT EVIDENCE FOR ALPHA" in report and "NOT RUN" in report


def test_design_samples_huge_spaces_with_bounded_unique_configs():
    factors = {"market.limit_rate": list(range(100)), "market.market_rate": list(range(100)),
               "market.resilience": [i / 100 for i in range(100)]}
    cfg = small_config()
    design = design_experiment(cfg, factors, seed=42)
    assert design["theoretical_configurations"] == 1_000_000 and design["selected_configurations"] == 4
    assert len({d["index"] for d in design["designs"]}) == 4
    assert design == design_experiment(cfg, factors, seed=42)
    with pytest.raises(ValueError, match="exhaustive"):
        design_experiment(cfg, factors, budget="exhaustive")
    with pytest.raises(ValueError):
        design_experiment(cfg, {"market.bogus": [1]})


def test_cli_config_replay_and_invalid_exit_codes(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(small_config().model_dump_json())
    assert main(["config", "validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "VALID"
    assert main(["config", "validate", str(path), "--set", "execution.quantity=-1"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID"
    fixture = Path(__file__).resolve().parents[1] / "examples/data/canonical-events.jsonl"
    assert main(["replay", str(fixture)]) == 0
    assert json.loads(capsys.readouterr().out)["complete"]

"""Redistributable synthetic end-to-end smoke runnable from an installed wheel."""
from __future__ import annotations

from pathlib import Path

from .artifacts import portable_provenance, seal_artifacts, verify_artifacts
from .config import ResearchConfig
from .experiments.registry import write_json
from .generalization import run_study, verify_study
from .mbo import MBOBook, MBOEvent, MBOReplay
from .performance import _fixture
from .replay.l2 import L2Replay
from .runner import run_episode


def run_smoke(out: str | Path) -> dict:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    provenance = portable_provenance()
    write_json(root / "provenance.json", provenance)
    # Generated from this source, without exchange data or a network connection.
    _fixture(root / "synthetic-l2.csv", 5, 12)
    l2 = L2Replay(root / "synthetic-l2.csv")
    states = list(l2)
    assert len(states) == 13 and l2.stats["complete"]
    try:
        states[-1].require_capability("exact_fifo_position")
    except ValueError:
        pass
    else:
        raise AssertionError("aggregate L2 incorrectly claims exact FIFO")
    events = [MBOEvent(0, 0, "RESET", "SMOKE", "SYNTHETIC"),
              MBOEvent(1, 1, "ADD", "SMOKE", "SYNTHETIC", "ahead", "BUY", 100, 5),
              MBOEvent(2, 2, "ADD", "SMOKE", "SYNTHETIC", "watched", "BUY", 100, 4),
              MBOEvent(3, 3, "CANCEL", "SMOKE", "SYNTHETIC", "ahead", quantity=5),
              MBOEvent(4, 4, "EXECUTE", "SMOKE", "SYNTHETIC", "watched", quantity=2),
              MBOEvent(5, 5, "EXECUTE", "SMOKE", "SYNTHETIC", "watched", quantity=2)]
    import json
    with (root / "synthetic-mbo.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
    book = MBOBook()
    for event in events:
        book.apply(event)
        if event.sequence == 2:
            book.watch("watched")
    mbo = MBOReplay(root / "synthetic-mbo.jsonl")
    assert mbo.run() == 6 and mbo.book.aggregate_l2() == book.aggregate_l2()
    calibration = {
        "schema_version": 1, "kind": "synthetic_smoke", "exchange": "SYNTHETIC", "symbol": "SMOKE",
        "interval_us": 1000000, "embargo_us": 2000000, "initial_train_rows": 360,
        "validation_rows": 120, "internal_rows": 120, "folds": 2, "simulation_rows": 256,
        "max_pool_rows": 64, "fit_seed": 91401, "validation_seeds": [91501],
        "internal_seeds": [91601], "external_seeds": [91701],
        "development": {"seed": 91101, "rows": 720, "start_us": 0, "shift": 1.0},
        "external": {"seed": 91102, "rows": 120, "start_us": 86400000000, "shift": 1.8}}
    calibrated = run_study(calibration, root / "calibration")
    assert verify_study(root / "calibration")["valid"]
    config = ResearchConfig.model_validate({"name": "portable_smoke",
        "execution": {"quantity": 100, "horizon": 1.0, "decision_dt": .1, "warmup_seconds": .2},
        "evaluation": {"agents": ["twap", "vwap", "pov", "ac", "heuristic", "random"],
                       "seeds": [95101, 95102], "bootstrap_samples": 100}})
    rows = [run_episode(agent, config.runner_params(seed))
            for seed in config.evaluation.seeds for agent in config.evaluation.agents]
    write_json(root / "execution-config.json", config.model_dump(mode="json"))
    # Small diagnostic table; no checkpoint or multi-megabyte run directory.
    write_json(root / "execution.json", rows)
    result = {"status": "COMPLETE", "dataset": "generated synthetic fixtures; no historical date",
              "symbol": "SMOKE", "l2": l2.summary(), "mbo": mbo.summary(),
              "observed_queue": book.order_research("watched"),
              "calibration_status": calibrated["status"], "calibration": calibrated,
              "execution_episodes": len(rows),
              "invalid_execution_episodes": sum(row["status"] == "INVALID" for row in rows),
              "rl": "Optional; run the separately registered policy-study commands with .[rl].",
              "interpretation": "Pipeline smoke only. A calibration FAIL is retained, not a software failure."}
    write_json(root / "result.json", result)
    seal_artifacts(root)
    assert verify_artifacts(root)["valid"]
    return {"status": result["status"], "calibration_status": calibrated["status"],
            "execution_episodes": len(rows), "invalid_execution_episodes": result["invalid_execution_episodes"]}

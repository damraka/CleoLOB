"""Run finite designs; preserve invalid, failed and unstarted episodes explicitly."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import ResearchConfig, canonical_json
from ..runner import run_episode
from ..stats import paired_vs_reference, summarize
from .registry import (create_experiment, read_experiment, runtime_metadata, seal_experiment,
                       sha256_file, source_manifest, utc_now, write_json)
from .report import write_report


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def run_experiment(config: ResearchConfig, out: str | Path = "results/research", *,
                   reproduced_from: str | None = None,
                   progress: Callable[[int, int, dict[str, Any]], None] | None = None) -> Path:
    # Validation is repeated instead of trusting model_construct/model_copy bypasses.
    config = ResearchConfig.model_validate(config.model_dump(mode="json"))
    run, metadata = create_experiment(config, out, reproduced_from)
    start = time.monotonic()
    rows: list[dict[str, Any]] = []
    interrupted = False
    for seed in config.evaluation.seeds:
        for agent in config.evaluation.agents:
            identity = {"scenario": config.name, "agent": agent, "seed": seed, "label": agent}
            if interrupted or time.monotonic() - start >= config.resources.max_runtime_seconds:
                row = dict(identity, status="PARTIAL", error="not started: interrupt or wall-clock budget reached")
            else:
                try:
                    params = config.runner_params(seed)
                    params["record_audit"] = True
                    row = dict(run_episode(agent, params), scenario=config.name)
                    value = row.get("effective_bps")
                    if row.get("status") not in {"VALID", "WARNING", "INVALID"}:
                        raise ValueError("episode omitted a recognized validity status")
                    if row["status"] != "INVALID" and (value is None or not math.isfinite(value)):
                        raise ValueError("valid episode returned a missing/nonfinite economic outcome")
                    audit = row.pop("audit", {})
                    write_json(run / "logs" / f"{seed}-{agent}.json", audit)
                except KeyboardInterrupt:
                    interrupted = True
                    row = dict(identity, status="PARTIAL", error="interrupted during episode")
                except Exception as exc:
                    row = dict(identity, status="FAILED", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            # Durable append after each episode, so a killed process leaves evidence.
            with (run / "episodes.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(row) + "\n")
                handle.flush()
            if progress:
                progress(len(rows), config.episode_count, row)
    frame = pd.DataFrame(rows)
    frame.to_csv(run / "episodes.csv", index=False)
    complete = all(row["status"] in {"VALID", "WARNING"} for row in rows)
    summary, paired = pd.DataFrame(), pd.DataFrame()
    evaluation = config.evaluation
    if complete:
        kwargs = dict(n_boot=evaluation.bootstrap_samples, alpha=evaluation.confidence_alpha,
                      seed=evaluation.statistics_seed)
        summary = summarize(frame, **kwargs)
        paired = paired_vs_reference(frame, evaluation.reference, correction=evaluation.correction, **kwargs)
    warnings = config.warnings()
    if not complete:
        warnings.append("Inference withheld because at least one planned episode failed, is invalid, or was not run. No seed intersection or favorable filtering was used.")
    counts = frame["status"].value_counts().to_dict()
    status = "INVALID" if "INVALID" in counts else "FAILED" if "FAILED" in counts else "PARTIAL" if "PARTIAL" in counts else "WARNING"
    executed = sum(row["status"] != "PARTIAL" for row in rows)
    result = {
        "name": config.name, "status": status, "ended_at": utc_now(),
        "elapsed_seconds": time.monotonic() - start, "outcome_counts": counts,
        "coverage": {"market_configurations_theoretical": 1, "market_configurations_selected": 1,
                     "sampling_method": "full factorial of configured agents × configured seeds",
                     "agents": len(evaluation.agents), "seeds": len(evaluation.seeds),
                     "episodes_planned": config.episode_count, "episodes_attempted": executed,
                     "episodes_valid_or_warning": sum(row["status"] in {"VALID", "WARNING"} for row in rows),
                     "unexplored_space": "All market parameters outside this frozen configuration; no broader coverage claimed.",
                     "estimated_events": config.estimated_events},
        "guardrails": {"config_validated": "YES", "data_validated": "N/A — synthetic only",
                       "dedicated_leakage_test": "NOT RUN", "train_test_separated": "NOT RUN — no training in this experiment",
                       "baselines_included": "YES" if evaluation.reference in {"twap", "vwap", "ac", "pov"} and len(evaluation.agents) > 1 else "NO",
                       "costs_modeled": "YES", "message_latency_modeled": "YES", "feed_latency_modeled": "NO",
                       "risk_controls_active": "YES — parent reservation and configured pretrade limits",
                       "multiple_seeds": "YES" if len(evaluation.seeds) > 1 else "NO",
                       "stress_tested": "NOT RUN", "ood_tested": "NOT RUN",
                       "multiple_testing": evaluation.correction if complete and not paired.empty else "NOT RUN",
                       "reproduced": "NO — provenance saved; use cleo reproduce"},
        "summary": _records(summary), "paired": _records(paired), "warnings": warnings,
        "conclusion": "This study measures execution mechanics under one uncalibrated synthetic configuration. "
                      "It provides insufficient evidence for robust execution alpha or live profitability. "
                      + ("All planned economic outcomes are available; inspect effect sizes, uncertainty and failures before drawing conclusions."
                         if complete else "The planned comparison is incomplete or invalid; statistical claims are withheld."),
    }
    if metadata["source_manifest"] != source_manifest():
        result["status"] = "INVALID"
        result["warnings"].append("Research source changed during execution; this run is not reproducible.")
    model = metadata["model"]
    if model and (not Path(model["path"]).is_file() or sha256_file(Path(model["path"])) != model["sha256"]):
        result["status"] = "INVALID"
        result["warnings"].append("PPO checkpoint is missing or changed during execution.")
    if result["status"] == "INVALID" and complete:
        result["summary"], result["paired"] = [], []
        summary, paired = pd.DataFrame(), pd.DataFrame()
        result["guardrails"]["multiple_testing"] = "WITHHELD — provenance invalid"
        result["conclusion"] = "The recorded implementation or model provenance is invalid. Comparative inference is withheld."
    write_json(run / "result.json", result)
    summary.to_csv(run / "summary.csv", index=False)
    paired.to_csv(run / "paired.csv", index=False)
    write_report(run, metadata, result, frame, summary, paired)
    seal_experiment(run)
    return run


def reproduce(path: str | Path, out: str | Path = "results/research") -> dict[str, Any]:
    original = Path(path).resolve(strict=True)
    stored = read_experiment(original)
    metadata = stored["metadata"]
    if metadata["source_manifest"] != source_manifest():
        raise ValueError("research source differs from recorded run; restore the recorded source before reproduction")
    current = runtime_metadata()
    expected = metadata["runtime"]
    for key in ("python", "implementation", "machine", "packages"):
        if current[key] != expected[key]:
            raise ValueError(f"runtime {key} differs from recorded run; exact reproduction is unavailable")
    model = metadata.get("model")
    if model and (not Path(model["path"]).is_file() or sha256_file(Path(model["path"])) != model["sha256"]):
        raise ValueError("recorded model checkpoint is missing or changed")
    config = ResearchConfig.model_validate(stored["resolved_config"])
    rerun = run_experiment(config, out, reproduced_from=metadata["experiment_id"])
    matched = (original / "episodes.jsonl").read_bytes() == (rerun / "episodes.jsonl").read_bytes()
    all_valid = stored["result"]["status"] not in {"INVALID", "FAILED", "PARTIAL"}
    comparison = {"original": str(original), "reproduction": str(rerun),
                  "episode_bytes_identical": matched, "valid_reproduction": matched and all_valid}
    comparison_dir = Path(out).resolve() / "reproductions"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_file = comparison_dir / f"{rerun.name}.json"
    write_json(comparison_file, comparison)
    return {**comparison, "comparison_file": str(comparison_file)}

"""Purged chronological splits and registered execution stress families."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from .config import ResearchConfig, merge_settings
from .experiments import read_experiment, run_experiment
from .experiments.registry import PROJECT_ROOT, sha256_file, source_manifest, runtime_metadata, write_json
from .stats import paired_vs_reference, summarize


def _time_axis(frame: pd.DataFrame) -> np.ndarray:
    if "timestamp_us" not in frame or len(frame) < 3:
        raise ValueError("at least three timestamped records required")
    values = frame["timestamp_us"].to_numpy()
    if values.dtype.kind not in "iu" or (values < 0).any() or (values > 2**63 - 1).any():
        raise ValueError("timestamp_us must contain nonnegative signed-64-bit integers")
    values = values.astype(np.int64)
    if (np.diff(values) <= 0).any():
        raise ValueError("timestamps must strictly increase without sorting or duplicates")
    return values


def chronological_split(frame: pd.DataFrame, *, train_fraction: float = .6,
                        validation_fraction: float = .2, embargo_seconds: float = 60,
                        outcome_horizon_seconds: float = 1) -> dict[str, pd.DataFrame]:
    """Expanding time order with forward-outcome purging and boundary embargo.

    Each retained record has information interval [timestamp, timestamp+horizon].
    Earlier partitions end strictly before the next boundary minus embargo. The
    last horizon of each partition is removed, including test's unknowable tail.
    Fractions select row boundaries; gaps are measured in elapsed time.
    """
    times = _time_axis(frame)
    numbers = (train_fraction, validation_fraction, embargo_seconds, outcome_horizon_seconds)
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in numbers):
        raise ValueError("split settings must be finite numbers")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("train and validation fractions must leave a nonempty test fraction")
    if min(embargo_seconds, outcome_horizon_seconds) < 0:
        raise ValueError("embargo/horizon cannot be negative")
    gap, horizon = int(embargo_seconds * 1e6), int(outcome_horizon_seconds * 1e6)
    a, b = int(len(times) * train_fraction), int(len(times) * (train_fraction + validation_fraction))
    if not 0 < a < b < len(times):
        raise ValueError("too few rows for requested fractions")
    # Subtract from boundaries rather than add to timestamps, avoiding int64 overflow.
    masks = {"train": times < int(times[a]) - gap - horizon,
             "validation": (times >= times[a]) & (times < int(times[b]) - gap - horizon),
             "test": (times >= times[b]) & (times <= int(times[-1]) - horizon)}
    result = {key: frame.loc[mask].copy() for key, mask in masks.items()}
    if any(part.empty for part in result.values()):
        raise ValueError("purging/embargo removed an entire partition")
    for part in result.values():
        if "log_return" in part:
            part.iloc[0, part.columns.get_loc("log_return")] = np.nan
    return result


def walk_forward_splits(frame: pd.DataFrame, *, initial_train_rows: int,
                        test_rows: int, embargo_seconds: float = 60,
                        outcome_horizon_seconds: float = 1, max_folds: int = 20):
    """Nonoverlapping test blocks, expanding training, no future training rows."""
    times = _time_axis(frame)
    for value in (initial_train_rows, test_rows, max_folds):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("fold counts must be positive integers")
    for value in (embargo_seconds, outcome_horizon_seconds):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("gap/horizon must be nonnegative finite numbers")
    gap, horizon = int(embargo_seconds * 1e6), int(outcome_horizon_seconds * 1e6)
    starts = range(initial_train_rows, len(frame) - test_rows + 1, test_rows)
    if not len(starts) or len(starts) > max_folds:
        raise ValueError("zero folds or fold count exceeds max_folds")
    for fold, start in enumerate(starts):
        stop = start + test_rows
        train = frame.loc[times < int(times[start]) - gap - horizon].copy()
        test = frame.iloc[start:stop].copy()
        test = test.loc[test.timestamp_us <= int(times[stop - 1]) - horizon]
        if train.empty or test.empty:
            raise ValueError("purge removes an entire walk-forward partition")
        if "log_return" in test:
            test.iloc[0, test.columns.get_loc("log_return")] = np.nan
        yield {"fold": fold, "train": train, "test": test}


def _new_run(out: str | Path, plan: dict) -> tuple[Path, dict]:
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12])
    run.mkdir()
    source = source_manifest()
    write_json(run / "plan.json", plan)
    write_json(run / "provenance.json", {"source_manifest": source, "runtime": runtime_metadata()})
    for name, expected in source.items():
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write((PROJECT_ROOT / name).read_bytes())
        if sha256_file(target) != expected:
            raise ValueError("source changed while archiving the study implementation")
    return run, source


def _seal(run: Path) -> None:
    write_json(run / "manifest.json", {"algorithm": "sha256", "files": {
        p.relative_to(run).as_posix(): sha256_file(p) for p in sorted(run.rglob("*"))
        if p.is_file()}})


def verify_study(path: str | Path) -> dict:
    """Verify a sealed plan/provenance study, including every nested manifest."""
    run = Path(path).resolve(strict=True)
    manifest_path = run / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 16 * 1024**2:
        raise ValueError("unsafe or excessive study manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if manifest.get("algorithm") != "sha256" or not isinstance(files, dict) or len(files) > 100_000:
        raise ValueError("invalid study manifest")
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file() and p != manifest_path}
    issues = []
    if actual != set(files):
        issues.append("study file set differs from sealed manifest")
    for name, expected in files.items():
        target = (run / name).resolve()
        if not target.is_relative_to(run) or not target.is_file() or target == manifest_path:
            issues.append(f"missing/unsafe study artifact: {name}")
        elif sha256_file(target) != expected:
            issues.append(f"study artifact checksum mismatch: {name}")
    required = {"plan.json", "provenance.json", "result.json"}
    if not required <= set(files):
        issues.append("study is missing required plan/provenance/result artifacts")
    if not issues:
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        for name, expected in provenance["source_manifest"].items():
            if files.get("source/" + name) != expected:
                issues.append(f"archived implementation differs from provenance: {name}")
    return {"valid": not issues, "issues": issues, "study_id": run.name}


STRESS_PROFILES: dict[str, dict[str, Any]] = {
    "reference": {},
    "thin_depth": {"market": {"target_level_vol": 60, "resilience": .2, "limit_rate": 12.0}},
    "aggressive_flow": {"market": {"market_rate": 18.0, "market_qty_mean": 80.0}},
    "slow_messages": {"market": {"latency_base": .1, "latency_jitter": .3}},
    "combined": {"market": {"target_level_vol": 60, "resilience": .1, "limit_rate": 12.0,
                              "market_rate": 18.0, "market_qty_mean": 80.0,
                              "latency_base": .1, "latency_jitter": .3}},
    "liquidity_exhaustion": {"market": {"target_level_vol": 10, "limit_rate": 0.0,
                                       "market_rate": 0.0, "cancel_rate": 0.0, "resilience": 0.0}},
}


def run_stress_suite(config: ResearchConfig, out: str | Path = "results/stress", *,
                     profiles: dict[str, dict] | None = None, max_total_episodes: int = 2000) -> Path:
    """Freeze all scenarios before running; correct across the entire family.

    A failed/invalid/unstarted planned outcome withholds ALL family inference.
    Valid outcomes remain descriptively visible; missing outcomes are never dropped.
    """
    profiles = STRESS_PROFILES if profiles is None else profiles
    if not isinstance(profiles, dict) or not 1 <= len(profiles) <= 32:
        raise ValueError("one to 32 named stress profiles required")
    if isinstance(max_total_episodes, bool) or not isinstance(max_total_episodes, int) or max_total_episodes < 1:
        raise ValueError("max_total_episodes must be a positive integer")
    configs = []
    for name, overlay in profiles.items():
        if not isinstance(name, str) or not name.isidentifier() or len(name) > 80 or not isinstance(overlay, dict):
            raise ValueError("profile names must be bounded identifiers and overlays mappings")
        # Freeze agent/seed family: scenario overlays may not select favorable candidates/seeds.
        if set(overlay) - {"market", "execution", "fees", "risk"}:
            raise ValueError("stress overlays may alter only market, execution, fees or risk")
        raw = merge_settings(config.model_dump(mode="json"), overlay)
        raw["name"] = name
        configs.append(ResearchConfig.model_validate(raw))
    planned = sum(c.episode_count for c in configs)
    if planned > max_total_episodes:
        raise ValueError("stress family exceeds max_total_episodes")
    model_snapshot = None
    if "ppo" in config.evaluation.agents:
        checkpoint = Path(config.evaluation.model_path).resolve()
        if not checkpoint.is_file() and checkpoint.with_suffix(".zip").is_file():
            checkpoint = checkpoint.with_suffix(".zip")
        model_snapshot = {"path": str(checkpoint),
                          "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None}
    plan = {"kind": "registered_execution_stress", "configs": [c.model_dump(mode="json") for c in configs],
            "episodes": planned, "model": model_snapshot,
            "correction": "holm across all planned scenario/candidate comparisons",
            "assumption": "synthetic FIFO engine; empirical L2 calibration does not identify latent FIFO dynamics"}
    run, source = _new_run(out, plan)
    outcomes = []
    children = []
    child_provenance_ok = True
    for cfg in configs:
        child = run_experiment(cfg, run / "experiments")
        stored = read_experiment(child)
        if stored["metadata"]["source_manifest"] != source or stored["metadata"]["model"] != model_snapshot:
            child_provenance_ok = False
        children.append({"scenario": cfg.name, "path": child.relative_to(run).as_posix(),
                         "status": stored["result"]["status"]})
        child_outcomes = []
        for line in (child / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["scenario"] = cfg.name
            outcomes.append(row)
            child_outcomes.append(row)
        if (stored["result"]["status"] == "INVALID"
                and all(row["status"] in {"VALID", "WARNING"} for row in child_outcomes)):
            child_provenance_ok = False
    frame = pd.DataFrame(outcomes)
    complete = len(outcomes) == planned and all(r["status"] in {"VALID", "WARNING"} for r in outcomes)
    checkpoint_ok = model_snapshot is None or (Path(model_snapshot["path"]).is_file()
                    and sha256_file(Path(model_snapshot["path"])) == model_snapshot["sha256"])
    provenance_ok = source == source_manifest() and child_provenance_ok and checkpoint_ok
    descriptive, comparisons = [], []
    if complete and provenance_ok:
        kwargs = dict(n_boot=config.evaluation.bootstrap_samples, alpha=config.evaluation.confidence_alpha,
                      seed=config.evaluation.statistics_seed)
        descriptive = json.loads(summarize(frame, **kwargs).to_json(orient="records"))
        comparisons = json.loads(paired_vs_reference(frame, config.evaluation.reference,
                                correction="holm", **kwargs).to_json(orient="records"))
    frame.to_csv(run / "episodes.csv", index=False)
    result = {"status": "COMPLETE" if provenance_ok and not frame.status.isin(["FAILED", "PARTIAL"]).any() else "INVALID",
              "episodes_planned": planned, "episodes_recorded": len(outcomes),
              "outcomes": frame.status.value_counts().to_dict(), "provenance_valid": provenance_ok,
              "family_inference": "AVAILABLE" if complete and provenance_ok else "WITHHELD",
              "planned_comparisons": len(configs) * (len(config.evaluation.agents) - 1),
              "children": children, "summary": descriptive, "paired": comparisons,
              "conclusion": "Stress controls and economic failures are reported for every frozen scenario. "
                            "Unavailable liquidation or unsettled orders invalidate outcomes. No robust alpha claim."}
    write_json(run / "result.json", result)
    report = ["# Registered execution stress test", "", f"Status: {result['status']}",
              f"Episodes: {len(outcomes)} / {planned}; family inference: {result['family_inference']}", "",
              "| Scenario | Outcome counts |", "|---|---|"]
    for name, rows in frame.groupby("scenario", sort=False):
        report.append(f"| {name} | {rows.status.value_counts().to_dict()} |")
    report.extend(["", result["conclusion"], "", "See episodes.csv and child audit logs for every planned result.", ""])
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    _seal(run)
    return run


def run_chronological_calibration(train_path: str | Path, validation_path: str | Path | None,
                                  test_path: str | Path, out: str | Path = "results/calibration") -> Path:
    """Train once on the first file; fixed validation and later untouched test.

    Also runs four expanding walk-forward folds strictly within the training day.
    Later files never select/fix model parameters. The plan precedes extraction.
    """
    from .calibration import extract_features, fit_calibration, evaluate_calibration, save_calibration, generate_features
    paths = [Path(p).resolve(strict=True) for p in (train_path, validation_path, test_path) if p is not None]
    hashes = [sha256_file(p) for p in paths]
    if len(set(hashes)) != len(paths):
        raise ValueError("dataset files must be distinct")
    roles = ("train", "validation", "test") if validation_path is not None else ("development", "external_test")
    plan = {"kind": "chronological_observable_calibration", "sample_interval_seconds": 1,
            "max_staleness_seconds": 5, "max_pool_rows": 4096, "embargo_seconds": 60,
            "outcome_horizon_seconds": 1, "selection": "no tuning or selection on validation/test",
            "fit_seed": 0, "generation_seed": 1729,
            "split_fractions": {"train": .6, "validation": .2, "internal_test": .2},
            "walk_forward": {"initial_train_fraction": .5, "test_block_fraction": .125, "max_folds": 5},
            "diagnostic_thresholds": {"distance_warning_above": .25, "distance_fail_above": .5,
                                      "coverage_warning_below": .90, "coverage_fail_below": .75,
                                      "valid_fraction_warning_below": .99, "valid_fraction_fail_below": .90},
            "split": "three ordered files" if validation_path is not None else "development day 60/20/20 purged train/validation/internal test; later file external test",
            "datasets": [dict(role=role, path=str(p), sha256=h) for role, p, h in
                         zip(roles, paths, hashes)]}
    run, source = _new_run(out, plan)
    metadata = []
    frames = []
    for p in paths:
        frame, meta = extract_features(p)
        _time_axis(frame)
        if frames and int(frames[-1].timestamp_us.max()) + 61_000_000 >= int(frame.timestamp_us.min()):
            raise ValueError("files overlap, are out of chronological order, or violate the purge/embargo gap")
        metadata.append(meta)
        frames.append(frame)
    internal_test = None
    if validation_path is None:
        split = chronological_split(frames[0])
        train, validation, internal_test = (split[k] for k in ("train", "validation", "test"))
        test = frames[1]
        validation_metadata, test_metadata = metadata[0], metadata[1]
    else:
        train, validation, test = frames
        validation_metadata, test_metadata = metadata[1], metadata[2]
    model = fit_calibration(train, metadata[0], max_pool_rows=4096)
    save_calibration(model, run / "model.json")
    frozen = sha256_file(run / "model.json")
    validation_result = evaluate_calibration(model, validation, validation_metadata)
    # No fit, tuning, selection, or model update occurs between validation and test.
    test_result = evaluate_calibration(model, test, test_metadata)
    internal_result = evaluate_calibration(model, internal_test, metadata[0]) if internal_test is not None else None
    folds = []
    for fold in walk_forward_splits(train, initial_train_rows=len(train) // 2,
                                    test_rows=len(train) // 8, max_folds=5):
        fitted = fit_calibration(fold["train"], metadata[0], max_pool_rows=4096)
        folds.append({"fold": fold["fold"], "train_rows": len(fold["train"]),
                      "test_rows": len(fold["test"]),
                      "train_last_timestamp_us": int(fold["train"].timestamp_us.max()),
                      "test_first_timestamp_us": int(fold["test"].timestamp_us.min()),
                      "scorecard": evaluate_calibration(fitted, fold["test"], metadata[0])})
    generated = generate_features(model, 1000, seed=1729)
    generated.to_csv(run / "generated_observables.csv", index=False)
    if source != source_manifest() or hashes != [sha256_file(p) for p in paths] or sha256_file(run / "model.json") != frozen:
        raise ValueError("source, dataset or frozen model changed during calibration study")
    result = {"status": "COMPLETE", "scope": "calibrated aggregate observables, not latent FIFO arrivals or strategy fills",
              "frozen_model_sha256": frozen, "extraction": metadata,
              "rows": {"train": len(train), "validation": len(validation), "test": len(test),
                       "internal_test": len(internal_test) if internal_test is not None else 0},
              "validation": validation_result, "test": test_result, "internal_test": internal_result, "walk_forward": folds,
              "generated_samples": len(generated),
              "conclusion": "A fitted observable model was frozen before later-date evaluation. "
                            "Drift and coverage are diagnostics; a completed study does not mean the model generalizes."}
    write_json(run / "result.json", result)
    (run / "report.md").write_text("# Chronological calibration study\n\n" + result["conclusion"] +
        "\n\nTraining, validation and test use disjoint ordered rows with a 60-second embargo plus a one-second outcome horizon. "
        "Four purged expanding folds exercise within-training-day stability. "
        "No parameters are updated after validation.\n\n" +
        "Detailed drift/coverage scorecards and all provenance are in result.json, plan.json and model.json.\n",
        encoding="utf-8")
    _seal(run)
    return run

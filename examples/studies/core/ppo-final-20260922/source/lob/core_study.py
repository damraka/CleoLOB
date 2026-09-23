"""Prospectively locked, crossed-seed PPO versus AC experiment.

The primary estimand is PPO minus AC mean net execution cost in bps, averaged
over independent training runs and common held-out market paths. Training
completion penalties never enter that endpoint. The fixed optimization budget
is not evidence of PPO convergence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .config import ResearchConfig, canonical_json, load_config
from .engine import Side
from .experiments.registry import (PROJECT_ROOT, runtime_metadata, seal_experiment,
                                   sha256_file, source_manifest, utc_now, write_json)
from .rl_env import LOBExecutionEnv
from .runner import cfg_from_params, run_episode

TRAINING_SEEDS = (81001, 81002, 81003, 81004, 81005)
PENALTIES = (0.0, 5.0, 25.0, 100.0)
TRAIN_MARKET_RANGE = (1_000_000, 2_000_000)
DIAGNOSTIC_SEEDS = tuple(range(61000, 61020))
TEST_SEED_START = 71000


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def runner_parameters(config: ResearchConfig, seed: int, penalty: float) -> dict[str, Any]:
    """One resolution path shared by training and every evaluation phase."""
    params = config.runner_params(config.evaluation.seeds[0])
    params["seed"] = seed
    params["sim"] = {**params["sim"], "seed": seed, "record_events": False}
    params["terminal_penalty_bps"] = penalty
    return params


def make_environment(config: ResearchConfig, penalty: float, *, training: bool) -> LOBExecutionEnv:
    params = runner_parameters(config, 0, penalty)
    return LOBExecutionEnv(
        total_qty=params["qty"], horizon=params["horizon"], decision_dt=params["dt"],
        side=Side.BUY if params["side"] == "buy" else Side.SELL,
        warmup=params["warmup_seconds"], cfg=cfg_from_params(params), fees=params["fees"], risk=params["risk"],
        terminal_penalty_bps=penalty, settlement_timeout=params["settlement_timeout"],
        settlement_poll_dt=params["settlement_poll_dt"],
        seed_range=TRAIN_MARKET_RANGE if training else None,
    )


def register_study(config: ResearchConfig, out: str | Path, *, timesteps: int = 8192,
                   min_markets: int = 32, max_markets: int = 256,
                   bootstrap_samples: int = 5000, evidence: dict[str, Any] | None = None) -> Path:
    """Write the design and source snapshot once, before any PPO fitting or test."""
    if isinstance(timesteps, bool) or timesteps < 256 or timesteps % 256:
        raise ValueError("timesteps must be a positive multiple of the 256-step rollout")
    if not 20 <= min_markets <= max_markets <= 10000 or bootstrap_samples < 100:
        raise ValueError("invalid study resources")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_manifest()
    secondary_risk = (config.execution.temp_impact / (config.execution.sigma * config.execution.horizon) ** 2
                      if config.execution.sigma > 0 else None)
    plan = {
        "schema_version": 1, "registered_at": utc_now(), "question": "PPO minus simulator-fitted AC net execution cost",
        "primary_penalty_bps": 0.0, "penalties_bps": list(PENALTIES),
        "training_seeds": list(TRAINING_SEEDS), "training_market_seed_range": list(TRAIN_MARKET_RANGE),
        "diagnostic_market_seeds": list(DIAGNOSTIC_SEEDS),
        "final_market_seed_start": TEST_SEED_START, "minimum_final_markets": min_markets,
        "maximum_final_markets": max_markets, "mde_bps": 0.5, "target_power": 0.8,
        "alpha": 0.05, "bootstrap_samples": bootstrap_samples, "statistics_seed": 91001,
        "timesteps_per_model": timesteps,
        "ppo": {"learning_rate": 0.0003, "n_steps": 256, "batch_size": 64,
                "n_epochs": 10, "gamma": 0.999, "ent_coef": 0.005, "device": "cpu"},
        "optimization_budget": "Fixed resource bound; no final-test checkpoint selection, no convergence claim.",
        "invalid_training_policy": "Abort and preserve a failure artifact at the first INVALID terminal episode; never optimize a missing residual value as zero, and never silently replace a failed training seed.",
        "endpoint": "net_effective_bps, including hypothetical residual book liquidation and fees; excluding completion penalty",
        "inference": "Two-way crossed percentile bootstrap: independently resample training seeds and common market seeds.",
        "multiplicity": "Primary 95% interval; Bonferroni family intervals cover four penalty arms and the additional AC risk sensitivity when identified.",
        "ac_primary_risk_aversion": config.execution.risk_aversion,
        "ac_primary_interpretation": "At risk_aversion=0 the analytical Almgren-Chriss expected-cost solution equals TWAP; fitted impact and volatility do not make a risk-neutral schedule front-loaded.",
        "ac_secondary_risk_aversion": secondary_risk,
        "ac_secondary_interpretation": "Prespecified kappa*T=1 schedule sensitivity, not an additional selected primary benchmark; unavailable at zero volatility.",
        "invalid_policy": "No family veto. Each arm retains failures. Complete-case descriptive intervals and labeled 100/500 bps residual proxy sensitivities; no missing economic outcome is asserted observed.",
        "power_method": "Pilot crossed random-effects ANOVA; solve normal-approximation variance target with five fixed training replicates; report training variance floor and power when capped.",
        "config_sha256": _digest(config), "source_manifest": source,
        "source_sha256": _digest(source), "runtime": runtime_metadata(), "evidence": evidence or {},
    }
    write_json(out / "resolved_config.json", config.model_dump(mode="json"))
    write_json(out / "preregistration.json", plan)
    write_json(out / "registration_seal.json", {"preregistration_sha256": _digest(plan), "config_sha256": _digest(config)})
    for name in source:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT_ROOT / name).read_bytes())
    (out / "models").mkdir()
    return out


def _read_study(out: str | Path, *, check_source: bool = True) -> tuple[Path, dict[str, Any], ResearchConfig]:
    out = Path(out).resolve(strict=True)
    plan = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    document = json.loads((out / "resolved_config.json").read_text(encoding="utf-8"))
    seal = json.loads((out / "registration_seal.json").read_text(encoding="utf-8"))
    if _digest(plan) != seal["preregistration_sha256"] or _digest(document) != seal["config_sha256"]:
        raise ValueError("preregistration or resolved configuration changed")
    if plan["config_sha256"] != _digest(document):
        raise ValueError("plan/configuration hash mismatch")
    if _digest(plan["source_manifest"]) != plan["source_sha256"]:
        raise ValueError("registered source manifest changed")
    for name, expected in plan["source_manifest"].items():
        path = (out / "source" / name).resolve()
        if not path.is_relative_to(out / "source") or sha256_file(path) != expected:
            raise ValueError(f"registered source snapshot changed: {name}")
    if check_source and source_manifest() != plan["source_manifest"]:
        raise ValueError("research source changed after registration; preserve this study and preregister a new one")
    return out, plan, ResearchConfig.model_validate(document)


def _model_stem(penalty: float, training_seed: int) -> str:
    return f"penalty-{penalty:g}-seed-{training_seed}"


def _check_model(out: Path, plan: dict[str, Any], penalty: float, training_seed: int) -> tuple[Path, dict[str, Any]]:
    stem = _model_stem(penalty, training_seed)
    path = out / "models" / f"{stem}.zip"
    metadata = json.loads((out / "models" / f"{stem}.json").read_text(encoding="utf-8"))
    if (metadata["model_sha256"] != sha256_file(path)
            or metadata["preregistration_sha256"] != _digest(plan)
            or metadata["training_seed"] != training_seed or metadata["penalty_bps"] != penalty
            or metadata["config_sha256"] != plan["config_sha256"]
            or metadata["timesteps"] != plan["timesteps_per_model"]
            or metadata["training_trace_sha256"] != sha256_file(out / "models" / f"{stem}.training.json")):
        raise ValueError(f"model provenance mismatch: {stem}")
    if ("training_attempt_sha256" in metadata
            and metadata["training_attempt_sha256"] != sha256_file(out / "models" / f"{stem}.attempt.json")):
        raise ValueError(f"training attempt provenance mismatch: {stem}")
    return path, metadata


def reward_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Report objective composition without conflating penalties with execution costs."""
    if not episodes:
        return {"episodes": 0, "mean_return_bps": None, "completion_penalty_absolute_share": None}
    keys = sorted({key for episode in episodes for key in episode["reward_terms"]})
    means = {key: float(np.mean([e["reward_terms"].get(key, 0.0) for e in episodes])) for key in keys}
    absolute = {key: float(np.mean([abs(e["reward_terms"].get(key, 0.0)) for e in episodes])) for key in keys}
    denominator = sum(absolute.values())
    return {"episodes": len(episodes), "mean_return_bps": float(np.mean([e["reward"] for e in episodes])),
            "mean_reward_terms_bps": means, "mean_absolute_reward_terms_bps": absolute,
            "completion_penalty_absolute_share": absolute.get("completion_penalty", 0.0) / denominator if denominator else 0.0,
            "invalid_episodes": sum(e["status"] == "INVALID" for e in episodes),
            "first_quarter_mean_return_bps": float(np.mean([e["reward"] for e in episodes[:max(1, len(episodes) // 4)]])),
            "last_quarter_mean_return_bps": float(np.mean([e["reward"] for e in episodes[-max(1, len(episodes) // 4):]]))}


def _training_state(out: Path, plan: dict[str, Any], penalty: float,
                    training_seed: int) -> dict[str, Any] | None:
    """Validate an existing fit or reject a failed/interrupted seed before fitting."""
    stem = _model_stem(penalty, training_seed)
    if (out / "models" / f"{stem}.failure.json").exists():
        raise ValueError(f"preserved failed training run {stem}; do not recycle its seed")
    if (out / "models" / f"{stem}.json").exists():
        return _check_model(out, plan, penalty, training_seed)[1]
    if ((out / "models" / f"{stem}.zip").exists()
            or (out / "models" / f"{stem}.attempt.json").exists()):
        raise FileExistsError(f"unsealed training attempt {stem}; preserve the interrupted study")
    return None


def _train_model(out: str | Path, penalty: float, training_seed: int) -> dict[str, Any]:
    """Fit one model in its own seeded process, with an exclusive attempt record."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import torch

    out, plan, config = _read_study(out)
    if penalty not in plan["penalties_bps"] or training_seed not in plan["training_seeds"]:
        raise ValueError("model is not in the registered training design")
    existing = _training_state(out, plan, penalty, training_seed)
    if existing is not None:
        return existing
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    stem = _model_stem(penalty, training_seed)
    attempt_path = out / "models" / f"{stem}.attempt.json"
    attempt = {"training_seed": training_seed, "penalty_bps": penalty,
               "preregistration_sha256": _digest(plan), "started_at": utc_now(),
               "worker_pid": os.getpid(), "torch_threads": torch.get_num_threads()}
    # Exclusive creation prevents two callers from fitting the same registered
    # cell concurrently; a process killed mid-fit leaves a non-retryable record.
    with attempt_path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(attempt))

    class Trace(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.episodes: list[dict[str, Any]] = []
            self.terms: dict[str, float] = {}
            self.rollouts: list[dict[str, Any]] = []

        def _on_step(self) -> bool:
            info = self.locals["infos"][0]
            for key, value in info.get("reward_terms", {}).items():
                self.terms[key] = self.terms.get(key, 0.0) + float(value)
            if self.locals["dones"][0]:
                self.episodes.append({"step": self.num_timesteps, "market_seed": info["market_seed"],
                                      "reward_terms": self.terms, "remaining": info["remaining"],
                                      "status": info["status"], "reward": info["episode"]["r"]})
                self.terms = {}
                if info["status"] == "INVALID":
                    raise ValueError("INVALID training economic outcome; missing residual value cannot enter PPO as zero")
            return True

        def _on_rollout_end(self) -> None:
            values = {key: float(value) for key, value in self.logger.name_to_value.items()
                      if key.startswith("train/") and isinstance(value, (int, float, np.number))
                      and math.isfinite(float(value))}
            self.rollouts.append({"step": self.num_timesteps, "previous_update_metrics": values,
                                  **reward_summary(self.episodes)})
            if self.num_timesteps % 1024 == 0:
                print(canonical_json({"training_progress_steps": self.num_timesteps,
                                      "model": stem,
                                      "episodes": len(self.episodes)}), flush=True)

    started = time.monotonic()
    env, model, trace = None, None, Trace()
    try:
        env = Monitor(make_environment(config, penalty, training=True))
        model = PPO("MlpPolicy", env, seed=training_seed, verbose=0, **plan["ppo"])
        model.learn(total_timesteps=plan["timesteps_per_model"], callback=trace)
        path = out / "models" / f"{stem}.zip"
        model.save(path)
        write_json(out / "models" / f"{stem}.training.json",
                   {"episodes": trace.episodes, "rollouts": trace.rollouts,
                    "partial_episode_reward_terms": trace.terms})
        metadata = {"training_seed": training_seed, "penalty_bps": penalty,
                    "timesteps": model.num_timesteps, "episodes": len(trace.episodes),
                    "elapsed_seconds": time.monotonic() - started,
                    "model_sha256": sha256_file(path), "config_sha256": plan["config_sha256"],
                    "training_trace_sha256": sha256_file(out / "models" / f"{stem}.training.json"),
                    "training_attempt_sha256": sha256_file(attempt_path),
                    "worker_pid": os.getpid(), "torch_threads": torch.get_num_threads(),
                    "reward_summary": reward_summary(trace.episodes),
                    "preregistration_sha256": _digest(plan), "completed_at": utc_now()}
        write_json(out / "models" / f"{stem}.json", metadata)
        print(canonical_json({"training_completed": stem, **metadata}), flush=True)
        return metadata
    except BaseException as exc:
        write_json(out / "models" / f"{stem}.failure.json",
                   {"error": f"{type(exc).__name__}: {exc}",
                    "timesteps": model.num_timesteps if model is not None else 0,
                    "training_seed": training_seed, "penalty_bps": penalty,
                    "preregistration_sha256": _digest(plan), "episodes": trace.episodes,
                    "failed_at": utc_now()})
        raise
    finally:
        if env is not None:
            env.close()


def train_study(out: str | Path, *, workers: int = 1) -> dict[str, Any]:
    """Fit the registered models in isolated processes; seal only complete fits.

    Process scheduling cannot share Torch/NumPy RNG state. Each task uses the
    same single-model implementation and deterministic one-thread CPU training.
    A failed task cancels queued work; already running tasks preserve their own
    outcomes. No summary is emitted unless every planned model verifies.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer from 1 to 32")
    out, plan, _ = _read_study(out)
    cells = [(penalty, seed) for penalty in plan["penalties_bps"] for seed in plan["training_seeds"]]
    existing = [_training_state(out, plan, penalty, seed) for penalty, seed in cells]
    if (out / "training_summary.json").exists():
        if any(metadata is None for metadata in existing):
            raise ValueError("training summary exists without every planned model")
        return json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
    missing = [cell for cell, metadata in zip(cells, existing) if metadata is None]
    if workers == 1:
        for penalty, seed in missing:
            _train_model(out, penalty, seed)
    elif missing:
        executor = ProcessPoolExecutor(max_workers=min(workers, len(missing)),
                                       mp_context=multiprocessing.get_context("spawn"))
        futures = []
        try:
            futures = [executor.submit(_train_model, out, penalty, seed) for penalty, seed in missing]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    # Canonical design order, independent of completion order. Verify artifacts
    # again in the parent rather than trusting worker return values.
    summaries = [_check_model(out, plan, penalty, seed)[1] for penalty, seed in cells]
    result = {"models": summaries, "total_timesteps": sum(s["timesteps"] for s in summaries),
              "execution": {"requested_workers": workers, "torch_threads_per_worker": 1,
                            "process_start_method": "spawn" if workers > 1 else "current_process"},
              "completed_at": utc_now()}
    write_json(out / "training_summary.json", result)
    return result


def crossed_interval(delta: np.ndarray, *, samples: int = 5000, alpha: float = 0.05,
                     seed: int = 91001) -> dict[str, Any]:
    """Resample both independent axes, preserving within-cell pairing."""
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or min(delta.shape) < 2 or not np.isfinite(delta).all():
        raise ValueError("crossed inference requires a finite training-seed by market-seed matrix")
    if not 0 < alpha < 1 or samples < 100:
        raise ValueError("invalid bootstrap controls")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for i in range(samples):
        rows = rng.integers(0, delta.shape[0], delta.shape[0])
        cols = rng.integers(0, delta.shape[1], delta.shape[1])
        draws[i] = delta[np.ix_(rows, cols)].mean()
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"mean_delta_bps": float(delta.mean()), "ci_low_bps": float(low), "ci_high_bps": float(high),
            "confidence": 1 - alpha, "training_seeds": delta.shape[0], "market_seeds": delta.shape[1],
            "training_seed_means_bps": delta.mean(axis=1).tolist()}


def pilot_power(delta: np.ndarray, *, mde: float = 0.5, power: float = 0.8,
                alpha: float = 0.05, minimum: int = 32, maximum: int = 256) -> dict[str, Any]:
    """Design on diagnostics only, retaining the training variance floor."""
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or min(delta.shape) < 2 or not np.isfinite(delta).all():
        raise ValueError("power pilot requires finite crossed outcomes")
    if not 0 < alpha < 1 or not 0.5 < power < 1 or mde <= 0 or not 2 <= minimum <= maximum:
        raise ValueError("invalid power design")
    r, n = delta.shape
    mean = delta.mean()
    row, col = delta.mean(axis=1), delta.mean(axis=0)
    interaction = delta - row[:, None] - col[None, :] + mean
    residual = float(np.sum(interaction ** 2) / ((r - 1) * (n - 1)))
    training = max(0.0, (float(n * np.sum((row - mean) ** 2) / (r - 1)) - residual) / n)
    market = max(0.0, (float(r * np.sum((col - mean) ** 2) / (n - 1)) - residual) / r)
    normal = NormalDist()
    z = normal.inv_cdf(1 - alpha / 2)
    target_variance = (mde / (z + normal.inv_cdf(power))) ** 2
    floor, numerator = training / r, market + residual / r
    required = max(minimum, math.ceil(numerator / (target_variance - floor))) if target_variance > floor else None
    chosen = min(maximum, required) if required is not None else maximum
    variance = floor + numerator / chosen
    standardized = mde / math.sqrt(variance) if variance else math.inf
    achieved = normal.cdf(standardized - z) + normal.cdf(-standardized - z)
    return {"final_market_count": chosen, "required_market_count": required,
            "capped": required is None or required > maximum, "mde_bps": mde, "target_power": power,
            "approximate_achievable_power": achieved, "variance_training": training,
            "variance_market": market, "variance_interaction": residual,
            "variance_floor_from_five_training_seeds": floor,
            "target_variance": target_variance, "pilot_training_seeds": r, "pilot_market_seeds": n,
            "limitation": "Pilot variances and normal approximation are uncertain; five training seeds limit precision."}


def _outcome(row: dict[str, Any], proxy: float | None = None) -> float:
    value = row.get("net_effective_bps")
    if value is not None and math.isfinite(value) and row.get("status") != "INVALID":
        return float(value)
    reasons = set(row.get("invalid_reasons") or ())
    if (proxy is None or not math.isfinite(proxy) or proxy < 0
            or reasons != {"insufficient_terminal_depth"}
            or row.get("settlement_complete") is not True or row.get("outstanding_qty", 0) != 0
            or row.get("settlement_pending_order_ids") or row.get("gross_cost") is None
            or row.get("total_fees") is None or row.get("arrival", 0) <= 0
            or row.get("target_quantity", 0) <= 0 or not 0 <= row.get("fill_frac", -1) <= 1):
        return math.nan
    # Scenario assumption, not an actual fill or a universal adverse-price bound.
    denominator = row["target_quantity"] * row["arrival"]
    filled_cost = (row["gross_cost"] + row["total_fees"]) / denominator * 1e4
    return filled_cost + (1 - row["fill_frac"]) * proxy


def _matrix(rows: list[dict[str, Any]], plan: dict[str, Any], markets: list[int],
            arm: float, imputation: float | None = None, reference: str = "ac") -> np.ndarray:
    index = {(r["agent"], r.get("training_seed"), r["seed"], r["penalty_bps"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate crossed episode cell")
    return np.array([[_outcome(index[("ppo", training_seed, market, arm)], imputation)
                      - _outcome(index[(reference, None, market, 0.0)], imputation)
                      for market in markets] for training_seed in plan["training_seeds"]])


def _run_rows(out: Path, plan: dict[str, Any], config: ResearchConfig,
              markets: list[int], penalties: list[float], phase: str) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO

    expected = [("ac", None, market, 0.0) for market in markets]
    secondary = phase == "final" and plan.get("ac_secondary_risk_aversion") is not None
    if secondary:
        expected += [("ac_risk", None, market, 0.0) for market in markets]
    expected += [("ppo", training_seed, market, arm) for arm in penalties
                 for training_seed in plan["training_seeds"] for market in markets]
    path = out / f"{phase}_episodes.jsonl"
    rows: list[dict[str, Any]] = []
    previous_digest = _digest({"phase": phase, "preregistration": _digest(plan)})
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            digest = row.pop("journal_sha256")
            if digest != _digest({"previous": previous_digest, "row": row}):
                raise ValueError("episode journal changed or was truncated")
            previous_digest = digest
            rows.append(row)
        actual = [(r["agent"], r.get("training_seed"), r["seed"], r["penalty_bps"]) for r in rows]
        if actual != expected[:len(actual)]:
            raise ValueError("episode journal does not match the preregistered evaluation order")
    completed = {(r["agent"], r.get("training_seed"), r["seed"], r["penalty_bps"]) for r in rows}
    with path.open("a" if path.exists() else "x", encoding="utf-8", newline="\n") as handle:
        def run(agent: str, market: int, arm: float, training_seed: int | None = None,
                model: Any = None) -> None:
            nonlocal previous_digest
            if (agent, training_seed, market, arm) in completed:
                return
            try:
                params = runner_parameters(config, market, arm)
                if agent == "ac_risk":
                    params["risk_aversion"] = plan["ac_secondary_risk_aversion"]
                row = run_episode("ac" if agent == "ac_risk" else agent, params, model=model)
            except Exception as exc:
                row = {"agent": agent, "seed": market, "status": "INVALID",
                       "net_effective_bps": None, "error": f"{type(exc).__name__}: {exc}"}
            row.update(agent=agent, training_seed=training_seed, penalty_bps=arm,
                       target_quantity=config.execution.quantity, phase=phase)
            previous_digest = _digest({"previous": previous_digest, "row": row})
            handle.write(canonical_json({**row, "journal_sha256": previous_digest}) + "\n")
            handle.flush()
            rows.append(row)

        for market in markets:
            run("ac", market, 0.0)
        if secondary:
            for market in markets:
                run("ac_risk", market, 0.0)
        for arm in penalties:
            for training_seed in plan["training_seeds"]:
                path, _ = _check_model(out, plan, arm, training_seed)
                model = PPO.load(path, device="cpu")
                for market in markets:
                    run("ppo", market, arm, training_seed, model)
                print(canonical_json({"phase": phase, "evaluated": _model_stem(arm, training_seed),
                                      "market_count": len(markets)}), flush=True)
    return rows


def evaluate_study(out: str | Path) -> dict[str, Any]:
    """Pilot first; lock final count before opening any final test episode."""
    import torch

    out, plan, config = _read_study(out)
    if not (out / "training_summary.json").is_file():
        raise ValueError("all preregistered training runs must complete before evaluation")
    torch.set_num_threads(1)
    if (out / "result.json").exists():
        raise FileExistsError("final inference is already complete; preserve its sealed result")
    for arm in plan["penalties_bps"]:
        for training_seed in plan["training_seeds"]:
            _check_model(out, plan, arm, training_seed)
    diagnostic = _run_rows(out, plan, config, plan["diagnostic_market_seeds"], [0.0], "diagnostic")
    delta = _matrix(diagnostic, plan, plan["diagnostic_market_seeds"], 0.0)
    if np.isfinite(delta).all():
        design = pilot_power(delta, mde=plan["mde_bps"], power=plan["target_power"], alpha=plan["alpha"],
                             minimum=plan["minimum_final_markets"], maximum=plan["maximum_final_markets"])
    else:
        design = {"final_market_count": plan["maximum_final_markets"], "capped": True,
                  "approximate_achievable_power": None, "reason": "Missing diagnostic economic outcomes; no imputation used for power.",
                  "invalid_pilot_cells": int((~np.isfinite(delta)).sum())}
    design.update(locked_at=utc_now(), diagnostic_episodes_sha256=sha256_file(out / "diagnostic_episodes.jsonl"))
    markets = list(range(plan["final_market_seed_start"], plan["final_market_seed_start"] + design["final_market_count"]))
    design["final_market_seeds"] = markets
    if (out / "power.json").exists():
        recorded = json.loads((out / "power.json").read_text(encoding="utf-8"))
        if {k: v for k, v in recorded.items() if k != "locked_at"} != {k: v for k, v in design.items() if k != "locked_at"}:
            raise ValueError("locked power design changed; no test-set recycling")
        design = recorded
    else:
        write_json(out / "power.json", design)
    rows = _run_rows(out, plan, config, markets, plan["penalties_bps"], "final")
    family_size = len(PENALTIES) + int(plan.get("ac_secondary_risk_aversion") is not None)
    arms = []
    for arm in plan["penalties_bps"]:
        delta = _matrix(rows, plan, markets, arm)
        complete = np.isfinite(delta).all(axis=0)
        result: dict[str, Any] = {"penalty_bps": arm, "planned_markets": len(markets),
                                  "missing_cells": int((~np.isfinite(delta)).sum()),
                                  "complete_markets": int(complete.sum()), "sensitivities": []}
        if complete.sum() >= 2:
            result["economic_interval"] = crossed_interval(delta[:, complete], samples=plan["bootstrap_samples"],
                                                           alpha=plan["alpha"], seed=plan["statistics_seed"])
            result["family_interval"] = crossed_interval(delta[:, complete], samples=plan["bootstrap_samples"],
                                                         alpha=plan["alpha"] / family_size, seed=plan["statistics_seed"])
            result["inference_status"] = "PRIMARY_IDENTIFIED" if complete.all() else "COMPLETE_CASE_DESCRIPTIVE_ONLY"
        else:
            result["inference_status"] = "INSUFFICIENT_OBSERVED_PAIRS"
        for proxy in (100.0, 500.0):
            sensitivity = _matrix(rows, plan, markets, arm, proxy)
            if np.isfinite(sensitivity).all():
                result["sensitivities"].append({"label": "Assumed residual adverse-cost proxy; not observed or guaranteed worst case",
                                                "residual_cost_bps": proxy,
                                                **crossed_interval(sensitivity, samples=plan["bootstrap_samples"],
                                                                   alpha=plan["alpha"] / family_size, seed=plan["statistics_seed"])})
        arm_rows = [r for r in rows if r["agent"] == "ppo" and r["penalty_bps"] == arm]
        for label, key in (("mean_fill_fraction", "fill_frac"), ("mean_completion_penalty_bps", "completion_penalty_bps")):
            values = [r[key] for r in arm_rows if r.get(key) is not None]
            result[label] = float(np.mean(values)) if values else None
        result["invalid_episodes"] = sum(r["status"] == "INVALID" for r in arm_rows)
        arms.append(result)
    secondary_result: dict[str, Any] = {"status": "UNAVAILABLE_ZERO_VOLATILITY"}
    if plan.get("ac_secondary_risk_aversion") is not None:
        delta = _matrix(rows, plan, markets, 0.0, reference="ac_risk")
        complete = np.isfinite(delta).all(axis=0)
        secondary_result = {"risk_aversion": plan["ac_secondary_risk_aversion"], "kappa_times_horizon": 1.0,
                            "status": "IDENTIFIED" if complete.all() else "COMPLETE_CASE_DESCRIPTIVE_ONLY",
                            "complete_markets": int(complete.sum()), "planned_markets": len(markets)}
        if complete.sum() >= 2:
            secondary_result["family_interval"] = crossed_interval(
                delta[:, complete], samples=plan["bootstrap_samples"], alpha=plan["alpha"] / family_size,
                seed=plan["statistics_seed"])
    result = {"question": plan["question"], "primary_penalty_bps": 0.0,
              "primary": arms[0], "penalty_ablations": arms, "power": design,
              "ac_risk_sensitivity": secondary_result,
              "planned_final_episodes": len(markets) * (len(PENALTIES) * len(TRAINING_SEEDS) + 1 + int(plan.get("ac_secondary_risk_aversion") is not None)),
              "actual_final_episodes": len(rows), "invalid_final_episodes": sum(r["status"] == "INVALID" for r in rows),
              "completed_at": utc_now(), "config_sha256": plan["config_sha256"],
              "training_summary": json.loads((out / "training_summary.json").read_text(encoding="utf-8")),
              "limitations": [f"{plan['timesteps_per_model']}-step PPO budget is bounded optimization, not convergence evidence.",
                              "Synthetic execution results require the separately reported market calibration gates.",
                              "Five training seeds leave uncertainty about rare optimization failures.",
                              "Residual liquidation is hypothetical; fill fractions are reported separately."]}
    write_json(out / "result.json", result)
    seal_experiment(out)
    return result


def verify_study(out: str | Path) -> dict[str, Any]:
    """Verify sealed evidence offline, without requiring today's source or runtime."""
    out = Path(out).resolve(strict=True)
    issues: list[str] = []
    try:
        _, plan, _ = _read_study(out, check_source=False)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
        if manifest.get("algorithm") != "sha256" or files != set(manifest["files"]):
            issues.append("sealed artifact file set differs")
        for name, expected in manifest["files"].items():
            path = (out / name).resolve()
            if not path.is_relative_to(out) or not path.is_file() or sha256_file(path) != expected:
                issues.append(f"artifact checksum mismatch: {name}")
        for penalty in plan["penalties_bps"]:
            for training_seed in plan["training_seeds"]:
                _check_model(out, plan, penalty, training_seed)
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        if result["config_sha256"] != plan["config_sha256"]:
            issues.append("result/configuration hash mismatch")
        if result["actual_final_episodes"] != result["planned_final_episodes"]:
            issues.append("final episode design incomplete")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(f"{type(exc).__name__}: {exc}")
    return {"valid": not issues, "issues": issues, "study": str(out)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    registration = sub.add_parser("register", help="Lock resolved configuration and analysis before training")
    registration.add_argument("--config", required=True)
    registration.add_argument("--out", required=True)
    registration.add_argument("--timesteps", type=int, default=8192)
    registration.add_argument("--max-markets", type=int, default=256)
    for phase in ("train", "evaluate", "verify"):
        command = sub.add_parser(phase)
        command.add_argument("--study", required=True)
        if phase == "train":
            command.add_argument("--workers", type=int, default=1,
                                 help="Isolated seeded CPU processes; each uses one Torch thread")
    args = parser.parse_args(argv)
    if args.phase == "register":
        print(register_study(load_config(args.config, environ={}), args.out,
                             timesteps=args.timesteps, max_markets=args.max_markets))
    elif args.phase == "train":
        print(canonical_json(train_study(args.study, workers=args.workers)))
    elif args.phase == "evaluate":
        print(canonical_json(evaluate_study(args.study)))
    else:
        result = verify_study(args.study)
        print(canonical_json(result))
        if not result["valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Prospectively locked, crossed-seed PPO versus AC experiment.

The primary estimand is PPO minus AC mean net execution cost in bps, averaged
over independent training runs and common held-out market paths. Training
completion penalties never enter that endpoint. The fixed optimization budget
is not evidence of PPO convergence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
        warmup=5.0, cfg=cfg_from_params(params), fees=params["fees"], risk=params["risk"],
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
        "endpoint": "net_effective_bps, including hypothetical residual book liquidation and fees; excluding completion penalty",
        "inference": "Two-way crossed percentile bootstrap: independently resample training seeds and common market seeds.",
        "multiplicity": "Primary 95% interval; additionally Bonferroni 98.75% intervals for all four penalty arms.",
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
            or metadata["training_seed"] != training_seed or metadata["penalty_bps"] != penalty):
        raise ValueError(f"model provenance mismatch: {stem}")
    return path, metadata


def train_study(out: str | Path) -> dict[str, Any]:
    """Fit 20 fixed-budget models; reuse only hash-verified completed fits."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import torch

    out, plan, config = _read_study(out)
    if (out / "training_summary.json").exists():
        raise FileExistsError("training is already complete")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    class Trace(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.episodes: list[dict[str, Any]] = []
            self.terms: dict[str, float] = {}

        def _on_step(self) -> bool:
            info = self.locals["infos"][0]
            for key, value in info.get("reward_terms", {}).items():
                self.terms[key] = self.terms.get(key, 0.0) + float(value)
            if self.locals["dones"][0]:
                self.episodes.append({"step": self.num_timesteps, "market_seed": info["market_seed"],
                                      "reward_terms": self.terms, "remaining": info["remaining"],
                                      "status": info["status"], "reward": info["episode"]["r"]})
                self.terms = {}
            return True

    summaries = []
    for penalty in plan["penalties_bps"]:
        for training_seed in plan["training_seeds"]:
            stem = _model_stem(penalty, training_seed)
            if (out / "models" / f"{stem}.json").exists():
                _, metadata = _check_model(out, plan, penalty, training_seed)
                summaries.append(metadata)
                continue
            if (out / "models" / f"{stem}.zip").exists():
                raise FileExistsError(f"unsealed checkpoint {stem}; preserve the interrupted study")
            started = time.monotonic()
            env = Monitor(make_environment(config, penalty, training=True))
            trace = Trace()
            model = PPO("MlpPolicy", env, seed=training_seed, verbose=0, **plan["ppo"])
            model.learn(total_timesteps=plan["timesteps_per_model"], callback=trace)
            path = out / "models" / f"{stem}.zip"
            model.save(path)
            env.close()
            metadata = {"training_seed": training_seed, "penalty_bps": penalty,
                        "timesteps": model.num_timesteps, "episodes": len(trace.episodes),
                        "elapsed_seconds": time.monotonic() - started,
                        "model_sha256": sha256_file(path), "config_sha256": plan["config_sha256"],
                        "preregistration_sha256": _digest(plan), "completed_at": utc_now()}
            write_json(out / "models" / f"{stem}.training.json", trace.episodes)
            write_json(out / "models" / f"{stem}.json", metadata)
            summaries.append(metadata)
            print(canonical_json({"training_completed": stem, **metadata}), flush=True)
    result = {"models": summaries, "total_timesteps": sum(s["timesteps"] for s in summaries),
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
    if proxy is None or row.get("gross_cost") is None or row.get("arrival", 0) <= 0:
        return math.nan
    # Scenario assumption, not an actual fill or a universal adverse-price bound.
    denominator = row["target_quantity"] * row["arrival"]
    filled_cost = (row["gross_cost"] + row["total_fees"]) / denominator * 1e4
    return filled_cost + (1 - row["fill_frac"]) * proxy


def _matrix(rows: list[dict[str, Any]], plan: dict[str, Any], markets: list[int],
            arm: float, imputation: float | None = None) -> np.ndarray:
    index = {(r["agent"], r.get("training_seed"), r["seed"], r["penalty_bps"]): r for r in rows}
    return np.array([[_outcome(index[("ppo", training_seed, market, arm)], imputation)
                      - _outcome(index[("ac", None, market, 0.0)], imputation)
                      for market in markets] for training_seed in plan["training_seeds"]])


def _run_rows(out: Path, plan: dict[str, Any], config: ResearchConfig,
              markets: list[int], penalties: list[float], phase: str) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO

    rows: list[dict[str, Any]] = []
    with (out / f"{phase}_episodes.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        def run(agent: str, market: int, arm: float, training_seed: int | None = None,
                model: Any = None) -> None:
            try:
                row = run_episode(agent, runner_parameters(config, market, arm), model=model)
            except Exception as exc:
                row = {"agent": agent, "seed": market, "status": "INVALID",
                       "net_effective_bps": None, "error": f"{type(exc).__name__}: {exc}"}
            row.update(training_seed=training_seed, penalty_bps=arm,
                       target_quantity=config.execution.quantity, phase=phase)
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            rows.append(row)

        for market in markets:
            run("ac", market, 0.0)
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
    if (out / "power.json").exists() or (out / "diagnostic_episodes.jsonl").exists():
        raise FileExistsError("evaluation already started; no rerun or test-set recycling")
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
    write_json(out / "power.json", design)
    rows = _run_rows(out, plan, config, markets, plan["penalties_bps"], "final")
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
                                                         alpha=plan["alpha"] / len(PENALTIES), seed=plan["statistics_seed"])
            result["inference_status"] = "PRIMARY_IDENTIFIED" if complete.all() else "COMPLETE_CASE_DESCRIPTIVE_ONLY"
        else:
            result["inference_status"] = "INSUFFICIENT_OBSERVED_PAIRS"
        for proxy in (100.0, 500.0):
            sensitivity = _matrix(rows, plan, markets, arm, proxy)
            if np.isfinite(sensitivity).all():
                result["sensitivities"].append({"label": "Assumed residual adverse-cost proxy; not observed or guaranteed worst case",
                                                "residual_cost_bps": proxy,
                                                **crossed_interval(sensitivity, samples=plan["bootstrap_samples"],
                                                                   alpha=plan["alpha"] / len(PENALTIES), seed=plan["statistics_seed"])})
        arm_rows = [r for r in rows if r["agent"] == "ppo" and r["penalty_bps"] == arm]
        for label, key in (("mean_fill_fraction", "fill_frac"), ("mean_completion_penalty_bps", "completion_penalty_bps")):
            values = [r[key] for r in arm_rows if r.get(key) is not None]
            result[label] = float(np.mean(values)) if values else None
        result["invalid_episodes"] = sum(r["status"] == "INVALID" for r in arm_rows)
        arms.append(result)
    result = {"question": plan["question"], "primary_penalty_bps": 0.0,
              "primary": arms[0], "penalty_ablations": arms, "power": design,
              "planned_final_episodes": len(markets) * (len(PENALTIES) * len(TRAINING_SEEDS) + 1),
              "actual_final_episodes": len(rows), "invalid_final_episodes": sum(r["status"] == "INVALID" for r in rows),
              "completed_at": utc_now(), "config_sha256": plan["config_sha256"],
              "limitations": [f"{plan['timesteps_per_model']}-step PPO budget is bounded optimization, not convergence evidence.",
                              "Synthetic execution results require the separately reported market calibration gates.",
                              "Five training seeds leave uncertainty about rare optimization failures.",
                              "Residual liquidation is hypothetical; fill fractions are reported separately."]}
    write_json(out / "result.json", result)
    seal_experiment(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    registration = sub.add_parser("register", help="Lock resolved configuration and analysis before training")
    registration.add_argument("--config", required=True)
    registration.add_argument("--out", required=True)
    registration.add_argument("--timesteps", type=int, default=8192)
    registration.add_argument("--max-markets", type=int, default=256)
    for phase in ("train", "evaluate"):
        command = sub.add_parser(phase)
        command.add_argument("--study", required=True)
    args = parser.parse_args()
    if args.phase == "register":
        print(register_study(load_config(args.config, environ={}), args.out,
                             timesteps=args.timesteps, max_markets=args.max_markets))
    elif args.phase == "train":
        print(canonical_json(train_study(args.study)))
    else:
        print(canonical_json(evaluate_study(args.study)))


if __name__ == "__main__":
    main()

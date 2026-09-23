"""Finite, prospectively registered PPO/DQN synthetic execution studies.

This extends the existing environment, controls and crossed-seed statistics.
No held-out trajectory is used for fitting, checkpoint selection or normalisation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from pydantic import Field, model_validator

from .config import ResearchConfig, Settings, canonical_json
from .controls import estimate_ac_parameters
from .core_study import (TRAIN_MARKET_RANGE, crossed_interval, make_environment,
                         reward_summary, runner_parameters)
from .experiments.registry import (PROJECT_ROOT, runtime_metadata, seal_experiment,
                                   sha256_file, source_manifest, utc_now, write_json)
from .runner import cfg_from_params, run_episode
from .stats import bootstrap_ci

CONTROLS = ("twap", "vwap", "pov", "ac", "heuristic", "random")
ALGORITHMS = ("ppo", "dqn")
ARMS = ("main", "no_book", "no_terminal")
REGIMES = ("original", "shifted", "stress")


class PolicyDesign(Settings):
    """A deliberately finite design; all resource bounds precede evaluation."""

    config: ResearchConfig = Field(default_factory=ResearchConfig)
    training_seeds: list[int] = Field(default_factory=lambda: [83001, 83002, 83003], min_length=2, max_length=10)
    evaluation_seeds: list[int] = Field(default_factory=lambda: list(range(73000, 73008)), min_length=2, max_length=100)
    identification_seeds: list[int] = Field(default_factory=lambda: [53001, 53002, 53003, 53004], min_length=2, max_length=20)
    timesteps: int = Field(default=1024, ge=256, le=1_048_576)
    bootstrap_samples: int = Field(default=20000, ge=100, le=100000)
    statistics_seed: int = Field(default=93001, ge=0, lt=2**32)
    evidence_level: Literal["smoke", "research"] = "smoke"
    shifted_market: dict[str, float | int] = Field(default_factory=lambda: {
        "market_rate": 9.0, "cancel_rate": 0.25, "resilience": 0.5})
    stress_market: dict[str, float | int] = Field(default_factory=lambda: {
        "market_rate": 15.0, "cancel_rate": 0.5, "resilience": 0.2,
        "target_level_vol": 120, "latency_base": 0.025, "latency_jitter": 0.025})

    @model_validator(mode="after")
    def finite_design(self) -> PolicyDesign:
        groups = (self.training_seeds, self.evaluation_seeds, self.identification_seeds)
        for group in groups:
            if len(set(group)) != len(group) or any(not 0 <= s < 2**31 for s in group):
                raise ValueError("seed lists must contain distinct nonnegative 31-bit integers")
        if any(set(left) & set(right) for i, left in enumerate(groups) for right in groups[i + 1:]):
            raise ValueError("training, identification and evaluation seeds must be disjoint")
        auxiliary = {seed + 1000 + day for seed in self.evaluation_seeds for day in range(5)}
        reserved = set(self.evaluation_seeds + self.identification_seeds + self.training_seeds)
        if auxiliary & reserved or any(TRAIN_MARKET_RANGE[0] <= s < TRAIN_MARKET_RANGE[1]
                                       for s in reserved | auxiliary):
            raise ValueError("evaluation/identification/VWAP seeds overlap a reserved training domain")
        if max(auxiliary) >= 2**31 or self.timesteps % 256:
            raise ValueError("timesteps must be a multiple of 256; auxiliary seeds must fit 31 bits")
        if self.config.execution.horizon < 0.5:
            raise ValueError("horizon must be at least 0.5 seconds for AC identification")
        for overlay in (self.shifted_market, self.stress_market):
            document = self.config.model_dump(mode="json")
            document["market"].update(overlay)
            ResearchConfig.model_validate(document)
        return self


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def regime_config(design: PolicyDesign, regime: str) -> ResearchConfig:
    if regime not in REGIMES:
        raise ValueError("unknown registered regime")
    document = design.config.model_dump(mode="json")
    if regime != "original":
        document["market"].update(getattr(design, f"{regime}_market"))
    return ResearchConfig.model_validate(document)


def mask_observation(observation: np.ndarray, arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError("unknown registered ablation")
    observation = np.array(observation, dtype=np.float32, copy=True)
    if arm == "no_book":
        observation[..., :22] = 0  # Preserve inventory and remaining time only.
    return observation


class ObservationAblation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, arm: str) -> None:
        super().__init__(env)
        self.arm = arm

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return mask_observation(observation, self.arm)


class EvaluationPolicy:
    """Apply exactly the training mask while reusing the audited policy runner."""

    def __init__(self, model: Any, arm: str) -> None:
        self.model, self.arm = model, arm

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Any:
        return self.model.predict(mask_observation(observation, self.arm), deterministic=deterministic)


def environment(design: PolicyDesign, arm: str, *, training: bool) -> gym.Env:
    penalty = 0.0 if arm == "no_terminal" else design.config.execution.terminal_penalty_bps
    return ObservationAblation(make_environment(design.config, penalty, training=training), arm)


def register(design: PolicyDesign, out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    source = source_manifest()
    runtime = runtime_metadata()
    runtime.pop("executable", None)  # No personal absolute paths in portable evidence.
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                             text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    plan = {
        "schema_version": 1, "registered_at": utc_now(), "design": design.model_dump(mode="json"),
        "git_commit": commit, "git_dirty": dirty, "source_manifest": source, "runtime": runtime,
        "dataset": {"kind": "synthetic", "generator": "poisson_fifo_v2", "symbol": "SYNTHETIC",
                    "period": "simulated seconds; no historical date", "calibrated": False},
        "training_market_range": list(TRAIN_MARKET_RANGE), "algorithms": ALGORITHMS,
        "arms": ARMS, "regimes": REGIMES, "controls": CONTROLS,
        "ppo": {"n_steps": 256, "batch_size": 64, "n_epochs": 5, "learning_rate": 0.0003,
                "gamma": 0.999, "ent_coef": 0.005, "device": "cpu"},
        "dqn": {"buffer_size": 10000, "learning_starts": 64, "batch_size": 64,
                "train_freq": 4, "gradient_steps": 1, "target_update_interval": 128,
                "exploration_fraction": 0.4, "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05, "learning_rate": 0.0003, "gamma": 0.999,
                "device": "cpu"},
        "checkpoint": "Final fixed-budget checkpoint only; no early stopping or held-out selection.",
        "evaluation": "Deterministic policy; common exogenous seeds, endogenous books can differ.",
        "ac_fit": "Separate identification seeds, per regime; existing fit PASS gate; otherwise preserve failure and use registered config defaults. Positive fitted sigma uses kappa*T=1 risk sensitivity.",
        "planned_comparisons": "Main PPO/DQN versus all six controls in all regimes; both ablations versus the same algorithm's main arm in all regimes.",
        "family_size": len(REGIMES) * len(ALGORITHMS) * (len(CONTROLS) + len(ARMS) - 1),
        "inference": "Crossed training-seed/market-seed bootstrap, Bonferroni family intervals at 0.05/family_size; any missing/INVALID cell withholds that comparison. Finite-bootstrap intervals are approximate, not exact coverage guarantees.",
        "metrics": {"net_effective_bps": "Fee-inclusive shortfall including hypothetical residual book liquidation; excludes completion objective penalty.",
                    "completion_rate": "Share of episodes with zero actual terminal remaining quantity, after settlement.",
                    "impact_proxy_bps": "Gross effective shortfall, including residual book walk. Mixes spread, impact, timing and stochastic price moves; not causal permanent impact.",
                    "terminal_inventory": "Unexecuted parent quantity after settlement; hypothetical valuation does not count as fills.",
                    "tail": "Empirical p95, worst cost, and mean of costs at/above empirical p95."},
        "interpretation": "Smoke exercises pipeline only. No convergence, superiority, calibrated realism, historical counterfactual PnL or live alpha claim.",
    }
    write_json(out / "preregistration.json", plan)
    write_json(out / "registration_seal.json", {"plan_sha256": digest(plan)})
    for name in source:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT_ROOT / name).read_bytes())
    return out


def read_study(out: str | Path, *, check_source: bool = True) -> tuple[Path, dict, PolicyDesign]:
    out = Path(out).resolve(strict=True)
    plan = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    seal = json.loads((out / "registration_seal.json").read_text(encoding="utf-8"))
    if digest(plan) != seal["plan_sha256"]:
        raise ValueError("registered plan changed")
    for name, expected in plan["source_manifest"].items():
        path = (out / "source" / name).resolve()
        if not path.is_relative_to(out / "source") or sha256_file(path) != expected:
            raise ValueError(f"source snapshot changed: {name}")
    if check_source and source_manifest() != plan["source_manifest"]:
        raise ValueError("source changed after registration; preserve run and register a new study")
    return out, plan, PolicyDesign.model_validate(plan["design"])


def model_stem(algorithm: str, arm: str, seed: int) -> str:
    return f"{algorithm}-{arm}-{seed}"


def model_metadata(out: Path, plan: dict, stem: str) -> dict:
    metadata = json.loads((out / "models" / f"{stem}.json").read_text(encoding="utf-8"))
    if (metadata["plan_sha256"] != digest(plan)
            or model_stem(metadata["algorithm"], metadata["arm"], metadata["seed"]) != stem
            or metadata["timesteps"] != plan["design"]["timesteps"]):
        raise ValueError(f"model registration mismatch: {stem}")
    for suffix, key in (("zip", "model_sha256"), ("training.json", "training_sha256"),
                        ("attempt.json", "attempt_sha256")):
        if sha256_file(out / "models" / f"{stem}.{suffix}") != metadata[key]:
            raise ValueError(f"model artifact mismatch: {stem}/{suffix}")
    return metadata


def train(out: str | Path) -> dict:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import torch

    out, plan, design = read_study(out)
    if (out / "training_summary.json").exists():
        raise FileExistsError("training already completed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fits = []
    for algorithm in ALGORITHMS:
        for arm in ARMS:
            for seed in design.training_seeds:
                stem = model_stem(algorithm, arm, seed)
                if list((out / "models").glob(f"{stem}.*")):
                    raise FileExistsError(f"preserved existing training attempt: {stem}")
                attempt = out / "models" / f"{stem}.attempt.json"
                write_json(attempt, {"plan_sha256": digest(plan), "seed": seed,
                                     "started_at": utc_now(), "algorithm": algorithm, "arm": arm})

                class Trace(BaseCallback):
                    def __init__(self) -> None:
                        super().__init__()
                        self.episodes: list[dict] = []
                        self.terms: dict[str, float] = {}
                        self.updates: list[dict] = []

                    def _on_step(self) -> bool:
                        info = self.locals["infos"][0]
                        for key, value in info["reward_terms"].items():
                            self.terms[key] = self.terms.get(key, 0.0) + float(value)
                        if self.locals["dones"][0]:
                            self.episodes.append({"step": self.num_timesteps, "market_seed": info["market_seed"],
                                                  "reward": info["episode"]["r"], "reward_terms": self.terms,
                                                  "status": info["status"], "remaining": info["remaining"]})
                            self.terms = {}
                            if info["status"] == "INVALID":
                                raise ValueError("INVALID training economics; missing residual valuation")
                        if self.num_timesteps % 256 == 0:
                            self.updates.append({"step": self.num_timesteps,
                                "prior_update": {k: float(v) for k, v in self.logger.name_to_value.items()
                                                 if k.startswith("train/") and isinstance(v, (int, float, np.number))
                                                 and math.isfinite(float(v))},
                                **reward_summary(self.episodes)})
                        return True

                trace, env, model = Trace(), None, None
                started = time.monotonic()
                try:
                    env = Monitor(environment(design, arm, training=True))
                    model = {"ppo": PPO, "dqn": DQN}[algorithm]("MlpPolicy", env, seed=seed,
                                                                 verbose=0, **plan[algorithm])
                    model.learn(total_timesteps=design.timesteps, callback=trace)
                    model.save(out / "models" / f"{stem}.zip")
                    write_json(out / "models" / f"{stem}.training.json",
                               {"episodes": trace.episodes, "updates": trace.updates,
                                "partial_episode_reward_terms": trace.terms})
                    metadata = {"algorithm": algorithm, "arm": arm, "seed": seed,
                                "timesteps": model.num_timesteps, "plan_sha256": digest(plan),
                                "elapsed_seconds": time.monotonic() - started,
                                "model_sha256": sha256_file(out / "models" / f"{stem}.zip"),
                                "training_sha256": sha256_file(out / "models" / f"{stem}.training.json"),
                                "attempt_sha256": sha256_file(attempt), "reward_summary": reward_summary(trace.episodes)}
                    write_json(out / "models" / f"{stem}.json", metadata)
                    fits.append(metadata)
                    print(canonical_json({"trained": stem, "timesteps": model.num_timesteps}), flush=True)
                except BaseException as exc:
                    write_json(out / "models" / f"{stem}.failure.json",
                               {"error": f"{type(exc).__name__}: {exc}", "episodes": trace.episodes,
                                "plan_sha256": digest(plan), "failed_at": utc_now()})
                    raise
                finally:
                    if env is not None:
                        env.close()
    result = {"models": fits, "total_timesteps": sum(f["timesteps"] for f in fits)}
    write_json(out / "training_summary.json", result)
    return result


def fit_controls(design: PolicyDesign) -> dict:
    """Identify on separate paths only; expose failures without changing the design."""
    result = {}
    for regime in REGIMES:
        config = regime_config(design, regime)
        params = runner_parameters(config, design.identification_seeds[0], 0.0)
        try:
            fit = estimate_ac_parameters(cfg_from_params(params), design.identification_seeds,
                horizon=config.execution.horizon, sample_dt=config.execution.horizon / 10,
                execution_interval=config.execution.horizon / 20,
                warmup_seconds=config.execution.warmup_seconds)
        except Exception as exc:
            fit = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        selected = {"temp_impact": config.execution.temp_impact, "sigma": config.execution.sigma,
                    "risk_aversion": config.execution.risk_aversion}
        if fit["status"] == "PASS":
            selected.update(temp_impact=fit["temp_impact"], sigma=fit["sigma"])
            if fit["sigma"] > 0:
                selected["risk_aversion"] = fit["temp_impact"] / (fit["sigma"] * config.execution.horizon) ** 2
        result[regime] = {"fit": fit, "selected": selected,
                          "selection": "fitted_kappa_T_1" if fit["status"] == "PASS" and fit["sigma"] > 0
                          else "zero_volatility_fit" if fit["status"] == "PASS" else "registered_defaults_after_failed_fit"}
    return result


def _value(row: dict, metric: str = "net_effective_bps") -> float:
    value = row.get(metric)
    return float(value) if row.get("status") in {"VALID", "WARNING"} and value is not None and math.isfinite(value) else math.nan


def describe(rows: list[dict], design: PolicyDesign, learned: bool) -> dict:
    valid = [r for r in rows if math.isfinite(_value(r))]
    values = np.array([_value(r) for r in valid])
    result: dict = {"episodes": len(rows), "valid_episodes": len(valid),
                   "invalid_episodes": len(rows) - len(valid),
                   "warning_episodes": sum(r.get("status") == "WARNING" for r in rows),
                   "conditional_on_valid": len(valid) != len(rows)}
    expected = {(s, m) for s in design.training_seeds for m in design.evaluation_seeds} if learned else {
        (None, m) for m in design.evaluation_seeds}
    actual = {(r["training_seed"], r["seed"]) for r in valid}
    complete = actual == expected and len(valid) == len(rows) == len(expected)
    if not complete:
        result["mean_ci_withheld"] = "incomplete or INVALID registered training/market seed grid"
    if not len(values):
        return result
    p95 = float(np.quantile(values, .95))
    result.update(mean_bps=float(values.mean()), median_bps=float(np.median(values)),
                  sd_bps=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                  p95_bps=p95, tail_mean_bps=float(values[values >= p95].mean()), worst_bps=float(values.max()),
                  completion_rate=sum(r["terminal_inventory"] == 0 for r in valid) / len(valid),
                  mean_terminal_inventory=float(np.mean([r["terminal_inventory"] for r in valid])),
                  mean_fill_fraction=float(np.mean([r["fill_frac"] for r in valid])),
                  mean_impact_proxy_bps=float(np.mean([r["gross_effective_bps"] for r in valid])))
    if complete:
        if learned:
            by_cell = {(r["training_seed"], r["seed"]): _value(r) for r in rows}
            matrix = np.array([[by_cell.get((s, m), math.nan) for m in design.evaluation_seeds]
                               for s in design.training_seeds])
            if not np.isfinite(matrix).all():
                result["mean_ci_withheld"] = "incomplete registered training/market seed grid"
                return result
            interval = crossed_interval(matrix, samples=design.bootstrap_samples, seed=design.statistics_seed)
            result.update(mean_ci_bps=[interval["ci_low_bps"], interval["ci_high_bps"]],
                          training_seed_means_bps=matrix.mean(axis=1).tolist(),
                          training_seed_sd_bps=float(matrix.mean(axis=1).std(ddof=1)))
        else:
            result["mean_ci_bps"] = list(bootstrap_ci(values, n_boot=design.bootstrap_samples,
                                                      seed=design.statistics_seed))
    return result


def summarize(rows: list[dict], plan: dict, design: PolicyDesign) -> dict:
    index = {(r["regime"], r["agent"], r["arm"], r["training_seed"], r["seed"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate evaluation cells")
    policies = [(c, "main", None) for c in CONTROLS] + [
        (a, arm, s) for a in ALGORITHMS for arm in ARMS for s in design.training_seeds]
    expected = {(regime, agent, arm, training_seed, market_seed)
                for regime in REGIMES for agent, arm, training_seed in policies
                for market_seed in design.evaluation_seeds}
    if set(index) - expected:
        raise ValueError("unregistered evaluation cells")
    summaries, comparisons = [], []
    for regime in REGIMES:
        for agent, arm in [(c, "main") for c in CONTROLS] + [(a, arm) for a in ALGORITHMS for arm in ARMS]:
            group = [r for r in rows if (r["regime"], r["agent"], r["arm"]) == (regime, agent, arm)]
            summaries.append({"regime": regime, "agent": agent, "arm": arm,
                              **describe(group, design, agent in ALGORITHMS)})
        for algorithm in ALGORITHMS:
            for arm, reference in [("main", c) for c in CONTROLS] + [(a, algorithm) for a in ARMS[1:]]:
                matrix = []
                for train_seed in design.training_seeds:
                    values = []
                    for market_seed in design.evaluation_seeds:
                        left = index.get((regime, algorithm, arm, train_seed, market_seed), {})
                        right = index.get((regime, reference, "main", train_seed if reference == algorithm else None,
                                           market_seed), {})
                        values.append(_value(left) - _value(right))
                    matrix.append(values)
                comparison = {"regime": regime, "agent": algorithm, "arm": arm, "reference": reference,
                              "delta": "agent arm minus reference main; negative means cheaper",
                              "family_size": plan["family_size"]}
                if np.isfinite(matrix).all():
                    comparison.update(status="AVAILABLE", **crossed_interval(np.array(matrix),
                        samples=design.bootstrap_samples, alpha=.05 / plan["family_size"], seed=design.statistics_seed))
                else:
                    comparison.update(status="WITHHELD", reason="Missing/INVALID planned paired outcome")
                comparisons.append(comparison)
    return {"evidence_level": design.evidence_level, "summaries": summaries, "comparisons": comparisons,
            "planned_episodes": len(REGIMES) * len(design.evaluation_seeds)
            * (len(CONTROLS) + len(ALGORITHMS) * len(ARMS) * len(design.training_seeds)),
            "actual_episodes": len(rows), "invalid_episodes": sum(not math.isfinite(_value(r)) for r in rows),
            "warning_episodes": sum(r.get("status") == "WARNING" for r in rows),
            "interpretation": plan["interpretation"], "calibrated_regimes": "UNAVAILABLE: no validated historical calibration accepted by this protocol"}


def evaluate(out: str | Path) -> dict:
    from stable_baselines3 import DQN, PPO
    import torch

    out, plan, design = read_study(out)
    if (out / "evaluation_lock.json").exists():
        raise FileExistsError("preserved evaluation attempt; no rerun or holdout selection")
    if not (out / "training_summary.json").exists():
        raise ValueError("all registered training must complete before evaluation")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = {}
    for algorithm in ALGORITHMS:
        for arm in ARMS:
            for seed in design.training_seeds:
                stem = model_stem(algorithm, arm, seed)
                model_metadata(out, plan, stem)
                models[algorithm, arm, seed] = EvaluationPolicy(
                    {"ppo": PPO, "dqn": DQN}[algorithm].load(out / "models" / f"{stem}.zip", device="cpu"), arm)
    controls = fit_controls(design)
    write_json(out / "evaluation_lock.json", {"plan_sha256": digest(plan), "controls": controls,
               "training_summary_sha256": sha256_file(out / "training_summary.json"), "locked_at": utc_now()})
    rows = []
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as handle:
        for regime in REGIMES:
            config = regime_config(design, regime)
            cells = [(c, "main", None) for c in CONTROLS] + list(models)
            for agent, arm, training_seed in cells:
                for market_seed in design.evaluation_seeds:
                    penalty = 0.0 if arm == "no_terminal" else config.execution.terminal_penalty_bps
                    params = runner_parameters(config, market_seed, penalty)
                    if agent == "ac":
                        params.update(controls[regime]["selected"])
                    try:
                        row = run_episode("ppo" if agent in ALGORITHMS else agent, params,
                                          model=models.get((agent, arm, training_seed)))
                        row["terminal_inventory"] = int(round(config.execution.quantity * (1 - row["fill_frac"])))
                        row["implementation"] = agent
                        row["label"] = f"{agent.upper()} / {arm}"
                    except Exception as exc:
                        row = {"status": "INVALID", "net_effective_bps": None,
                               "error": f"{type(exc).__name__}: {exc}"}
                    row.update(regime=regime, agent=agent, arm=arm, training_seed=training_seed,
                               seed=market_seed, target_quantity=config.execution.quantity)
                    handle.write(canonical_json(row) + "\n")
                    handle.flush()
                    rows.append(row)
            print(canonical_json({"evaluated_regime": regime, "episodes": len(rows)}), flush=True)
    result = summarize(rows, plan, design)
    result["ac_identification_status"] = {r: c["fit"]["status"] for r, c in controls.items()}
    write_json(out / "result.json", result)
    seal_experiment(out)
    return result


def verify(out: str | Path) -> dict:
    issues = []
    try:
        out, plan, design = read_study(out, check_source=False)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["files"]
        actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
        if actual != set(manifest):
            issues.append("sealed artifact file set differs")
        for name, expected in manifest.items():
            path = (out / name).resolve()
            if not path.is_relative_to(out) or not path.is_file() or sha256_file(path) != expected:
                issues.append(f"artifact mismatch: {name}")
        for algorithm in ALGORITHMS:
            for arm in ARMS:
                for seed in design.training_seeds:
                    model_metadata(out, plan, model_stem(algorithm, arm, seed))
        rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        if len(rows) != result["planned_episodes"] or len(rows) != result["actual_episodes"]:
            issues.append("incomplete registered evaluation")
    except (OSError, ValueError, KeyError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "train", "evaluate", "verify"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    if args.command == "register":
        if not args.config:
            parser.error("register requires --config")
        design = PolicyDesign.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
        result = {"registered": str(register(design, args.out))}
    else:
        result = {"train": train, "evaluate": evaluate, "verify": verify}[args.command](args.out)
    print(canonical_json(result))
    if args.command == "verify" and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

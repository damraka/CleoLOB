"""M7: registered v0.5 execution-policy study on the frozen environment.

The frozen design (``configs/v05/policy-study.json``) is verified byte-for-byte
against the M0 ledger freeze, and the calibrated regime against the M6
environment freeze, before registration. PPO/DQN (main, no_book, no_terminal
arms) and six controls share one completion mechanism and the M5 mandate
endpoints. Primary endpoints: E6 completion-adjusted cost and E5 within-horizon
completion. Every comparison in the 128-hypothesis Bonferroni family is kept;
missing or INVALID cells withhold a comparison without shrinking the family.

Training uses only the original regime and the training market-seed domain;
observation normalization is fitted on registered training-domain exploration
and sealed before any fit. Final fixed-budget checkpoints only. Pilot runs are
labelled ``pilot`` and are never the registered study.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import time
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .config import ResearchConfig, Settings, canonical_json
from .core_study import TRAIN_MARKET_RANGE, crossed_interval, runner_parameters
from .experiments.registry import (PROJECT_ROOT, runtime_metadata, seal_experiment, sha256_file, source_manifest,
                                   utc_now, write_json)
from .observations import FEATURES, OBSERVATION_CONTRACT, fit_normalization
from .policy_study import ARMS, CONTROLS, EvaluationPolicy, ObservationAblation
from . import preregistration as pr
from .runner import run_episode
from .stats import bootstrap_ci

ALGORITHMS = ("ppo", "dqn")
REFERENCES = CONTROLS + ("no_book", "no_terminal")
DESIGN_PATH = "configs/v05/policy-study.json"
FREEZE_PATH = "configs/v05/environment-freeze.json"
COST = "mandate_completion_adjusted_cost_bps"
COMPLETION = "mandate_within_horizon_completion"


class V05Design(Settings):
    protocol_version: Literal["v05"]
    config: ResearchConfig
    training_seeds: list[int] = Field(min_length=2, max_length=16)
    evaluation_seed_start: int = Field(ge=0, lt=2**31)
    evaluation_seed_count: int = Field(ge=2, le=1000)
    identification_seeds: list[int] = Field(min_length=2, max_length=20)
    normalization_seeds: list[int] = Field(min_length=2, max_length=20)
    completion_urgency_fraction: float = Field(ge=0, le=1)
    completion_noninferiority_margin: float = Field(ge=0, le=1)
    timesteps: int = Field(ge=256, le=1_048_576)
    bootstrap_samples: int = Field(ge=100, le=100_000)
    statistics_seed: int = Field(ge=0, lt=2**32)
    evidence_level: Literal["pilot", "research"]
    regimes: list[Literal["original", "shifted", "stress", "calibrated"]]
    shifted_market: dict[str, float | int]
    stress_market: dict[str, float | int]
    calibrated_market: dict[str, str]
    ppo: dict[str, float | int | str]
    dqn: dict[str, float | int | str]
    checkpoint: str

    @property
    def evaluation_seeds(self) -> list[int]:
        return list(range(self.evaluation_seed_start, self.evaluation_seed_start + self.evaluation_seed_count))

    @model_validator(mode="after")
    def finite_design(self) -> V05Design:
        groups = (self.training_seeds, self.evaluation_seeds, self.identification_seeds)
        for group in groups:
            if len(set(group)) != len(group):
                raise ValueError("seed lists must be distinct")
        if any(set(a) & set(b) for i, a in enumerate(groups) for b in groups[i + 1:]):
            raise ValueError("training, identification and evaluation seeds must be disjoint")
        auxiliary = {s + 1000 + d for s in self.evaluation_seeds for d in range(5)}
        reserved = set(self.evaluation_seeds) | set(self.identification_seeds) | set(self.training_seeds)
        if auxiliary & reserved or any(TRAIN_MARKET_RANGE[0] <= s < TRAIN_MARKET_RANGE[1] for s in reserved | auxiliary):
            raise ValueError("evaluation/identification/VWAP seeds overlap a reserved or training domain")
        if any(not TRAIN_MARKET_RANGE[0] <= s < TRAIN_MARKET_RANGE[1] for s in self.normalization_seeds):
            raise ValueError("normalization seeds must lie in the training market domain")
        if self.timesteps % 256 or self.regimes[0] != "original" or len(set(self.regimes)) != len(self.regimes):
            raise ValueError("timesteps must be a multiple of 256; regimes distinct, starting with original")
        return self


def family_size(design: V05Design) -> int:
    return len(design.regimes) * len(ALGORITHMS) * len(REFERENCES) * 2


def load_design(path: str | Path = DESIGN_PATH, root: Path = PROJECT_ROOT) -> tuple[V05Design, str]:
    text = (root / path).read_text(encoding="utf-8")
    return V05Design.model_validate_json(text), pr.text_sha256(root / path)


def pilot_design(design: V05Design, *, training_seeds: int, markets: int, timesteps: int) -> V05Design:
    document = design.model_dump(mode="json")
    document.update(training_seeds=document["training_seeds"][:training_seeds], evaluation_seed_count=markets,
                    timesteps=timesteps, evidence_level="pilot")
    return V05Design.model_validate_json(json.dumps(document))


def regime_setup(design: V05Design, regime: str, calibrated: dict | None) -> tuple[ResearchConfig, dict | None]:
    document = design.config.model_dump(mode="json")
    extensions = None
    if regime in {"shifted", "stress"}:
        document["market"].update(getattr(design, f"{regime}_market"))
    elif regime == "calibrated":
        if calibrated is None:
            raise ValueError("calibrated regime NOT_AVAILABLE")
        fields = set(document["market"]) - {"latency_base", "latency_jitter"}
        document["market"].update({k: v for k, v in calibrated["config"].items() if k in fields})
        extensions = calibrated["extensions"] or None
    return ResearchConfig.model_validate(document), extensions


def episode_params(design: V05Design, regime: str, calibrated: dict | None, seed: int, penalty: float,
                   normalization: dict | None) -> dict:
    config, extensions = regime_setup(design, regime, calibrated)
    params = runner_parameters(config, seed, penalty)
    params.update(completion={"enabled": True, "urgency_fraction": design.completion_urgency_fraction},
                  observation_version="v04", observation_normalization=normalization, flow_extensions=extensions)
    return params


def _environment(design: V05Design, arm: str, normalization: dict | None, *, training: bool):
    from .engine import Side
    from .completion import CompletionConstraint
    from .rl_env import LOBExecutionEnv
    from .runner import cfg_from_params
    penalty = 0.0 if arm == "no_terminal" else design.config.execution.terminal_penalty_bps
    params = runner_parameters(design.config, 0, penalty)
    env = LOBExecutionEnv(total_qty=params["qty"], horizon=params["horizon"], decision_dt=params["dt"],
                          side=Side.BUY if params["side"] == "buy" else Side.SELL, warmup=params["warmup_seconds"],
                          cfg=cfg_from_params(params), fees=params["fees"], risk=params["risk"],
                          terminal_penalty_bps=penalty, settlement_timeout=params["settlement_timeout"],
                          settlement_poll_dt=params["settlement_poll_dt"],
                          seed_range=TRAIN_MARKET_RANGE if training else None,
                          completion=CompletionConstraint(True, design.completion_urgency_fraction),
                          observation_version="v04", observation_normalization=normalization)
    return ObservationAblation(env, arm)


# ----------------------------------------------------------------------------- registration

def read_freeze(root: Path = PROJECT_ROOT) -> dict:
    freeze = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    protocol = pr.load_protocol(root / "configs/v05/protocol.json")
    state = pr.replay_ledger(pr.read_ledger(root / "configs/v05/consumption-ledger.jsonl"), protocol)
    sealed = state.designs.get("environment-freeze")
    if sealed is None or sealed["design_sha256"] != pr.document_sha256(freeze):
        raise ValueError("environment freeze is missing from, or differs from, the ledger seal")
    for name, digest in freeze["implementation_sha256"].items():
        if pr.text_sha256(root / name) != digest:
            raise ValueError(f"frozen implementation changed: {name}")
    return freeze


def register(out: str | Path, *, root: Path = PROJECT_ROOT, pilot: dict | None = None) -> Path:
    design, design_hash = load_design(root=root)
    protocol = pr.load_protocol(root / "configs/v05/protocol.json")
    state = pr.replay_ledger(pr.read_ledger(root / "configs/v05/consumption-ledger.jsonl"), protocol)
    if state.frozen["referenced_configs"].get(DESIGN_PATH) != design_hash:
        raise ValueError("policy-study design differs from the M0 protocol freeze")
    if pilot and not (root / FREEZE_PATH).exists():
        freeze = {"note": "pilot before the environment freeze; calibrated regime NOT_AVAILABLE"}
    else:
        freeze = read_freeze(root)
    if pilot:
        design = pilot_design(design, **pilot)
    calibrated = freeze.get("calibrated_regime")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    runtime = runtime_metadata()
    runtime.pop("executable", None)
    plan = {
        "schema": "cleolob-v05-policy-study-1", "registered_at": utc_now(), "design": design.model_dump(mode="json"),
        "design_file_sha256": design_hash, "evidence_level": design.evidence_level,
        "environment_freeze_sha256": pr.document_sha256(freeze), "calibrated_regime": calibrated,
        "calibrated_status": "AVAILABLE" if calibrated else "NOT_AVAILABLE",
        "source_manifest": source_manifest(), "runtime": runtime, "algorithms": ALGORITHMS, "arms": ARMS,
        "controls": CONTROLS, "references": REFERENCES, "regimes": design.regimes,
        "training": "original regime, training market-seed domain only; final fixed-budget checkpoint",
        "primary_endpoints": {"E6": COST, "E5": COMPLETION}, "family_size": family_size(design),
        "cost_inference": "crossed training-seed x market-seed percentile bootstrap at 1 - 0.05/family",
        "completion_inference": ("market-level Clopper-Pearson: Z_m = 1 if any policy training seed misses "
                                 "within-horizon completion where the reference completes; lower bound = -CP upper "
                                 "limit at one-sided alpha 0.05/family over independent market seeds"),
        "gate": {"cost": "upper bound < 0", "completion": f"lower bound >= -{design.completion_noninferiority_margin}",
                 "scope": "per contrast; no global best-policy label"},
        "observation": {"features": FEATURES, "capabilities": OBSERVATION_CONTRACT.to_dict(),
                        "excluded": ["future state", "holdout statistics", "queue position", "hidden liquidity",
                                     "market seed", "regime label", "simulator-internal flow parameters"]},
        "ppo": design.ppo, "dqn": design.dqn}
    write_json(out / "preregistration.json", plan)
    write_json(out / "registration_seal.json", {"plan_sha256": pr.document_sha256(plan)})
    if design.evidence_level == "research":
        pr.append_event(root / "configs/v05/consumption-ledger.jsonl", "seal_design", {
            "analysis": "m7-registration", "design_sha256": pr.document_sha256(plan), "reads": [],
            "note": "M7 plan registered on the frozen environment before any training or evaluation."},
            protocol=protocol)
    return out


def read_plan(out: Path) -> tuple[dict, V05Design]:
    plan = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    seal = json.loads((out / "registration_seal.json").read_text(encoding="utf-8"))
    if pr.document_sha256(plan) != seal["plan_sha256"]:
        raise ValueError("registered plan changed")
    if source_manifest() != plan["source_manifest"]:
        raise ValueError("implementation changed after registration; register a new study")
    return plan, V05Design.model_validate_json(json.dumps(plan["design"]))


# ----------------------------------------------------------------------------- training

def _fit_normalization(design: V05Design) -> dict:
    env = _environment(design, "main", None, training=False)
    observations = []
    try:
        for seed in design.normalization_seeds:
            rng = np.random.default_rng(seed)
            obs, _ = env.reset(seed=seed)
            observations.append(obs)
            while True:
                obs, _, done, truncated, _ = env.step(int(rng.integers(0, 5)))
                observations.append(obs)
                if done or truncated:
                    break
    finally:
        env.close()
    return {**fit_normalization(observations), "seeds": design.normalization_seeds,
            "source": "original-regime training-domain random exploration; frozen before any policy fit"}


def _train_one(task: tuple) -> dict:
    out, plan, algorithm, arm, seed = task
    import torch
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(out)
    design = V05Design.model_validate_json(json.dumps(plan["design"]))
    normalization = json.loads((out / "normalization.json").read_text(encoding="utf-8"))["normalization"]
    stem = f"{algorithm}-{arm}-{seed}"
    write_json(out / "models" / f"{stem}.attempt.json", {"started_at": utc_now(), "seed": seed})
    started = time.monotonic()
    env = Monitor(_environment(design, arm, normalization, training=True))
    try:
        hyper = dict(plan[algorithm])
        model = {"ppo": PPO, "dqn": DQN}[algorithm]("MlpPolicy", env, seed=seed, verbose=0, **hyper)
        model.learn(total_timesteps=design.timesteps)
        model.save(out / "models" / f"{stem}.zip")
        rewards = [float(x) for x in env.get_episode_rewards()]
        metadata = {"algorithm": algorithm, "arm": arm, "seed": seed, "timesteps": int(model.num_timesteps),
                    "elapsed_seconds": time.monotonic() - started, "episodes": len(rewards),
                    "reward_first_decile_mean": float(np.mean(rewards[: max(1, len(rewards) // 10)])) if rewards else None,
                    "reward_last_decile_mean": float(np.mean(rewards[-max(1, len(rewards) // 10):])) if rewards else None,
                    "model_sha256": sha256_file(out / "models" / f"{stem}.zip"),
                    "normalization_sha256": sha256_file(out / "normalization.json"),
                    "plan_sha256": pr.document_sha256(plan)}
        write_json(out / "models" / f"{stem}.json", metadata)
        return metadata
    except BaseException as exc:
        write_json(out / "models" / f"{stem}.failure.json", {"error": f"{type(exc).__name__}: {exc}", "failed_at": utc_now()})
        return {"algorithm": algorithm, "arm": arm, "seed": seed, "failed": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


def _workers() -> int:
    return int(os.environ.get("CLEOLOB_WORKERS", max(1, min(12, (os.cpu_count() or 2) - 2))))


def train(out: str | Path) -> dict:
    out = Path(out)
    plan, design = read_plan(out)
    if (out / "training_summary.json").exists():
        raise FileExistsError("training already completed")
    fitted = {**_fit_normalization(design), "plan_sha256": pr.document_sha256(plan)}
    write_json(out / "normalization.json", fitted)
    write_json(out / "normalization_seal.json", {"sha256": pr.document_sha256(fitted)})
    tasks = [(str(out), plan, a, arm, s) for a in ALGORITHMS for arm in ARMS for s in design.training_seeds]
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        results = list(pool.map(_train_one, tasks))
    summary = {"models": results, "failed": [r for r in results if "failed" in r],
               "total_timesteps": sum(r.get("timesteps", 0) for r in results)}
    write_json(out / "training_summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- evaluation

def _evaluate_chunk(task: tuple) -> list[dict]:
    out, plan, regime, agent, arm, training_seed, markets = task
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(out)
    design = V05Design.model_validate_json(json.dumps(plan["design"]))
    normalization = json.loads((out / "normalization.json").read_text(encoding="utf-8"))["normalization"]
    lock = json.loads((out / "evaluation_lock.json").read_text(encoding="utf-8"))
    model = None
    if agent in ALGORITHMS:
        from stable_baselines3 import DQN, PPO
        stem = f"{agent}-{arm}-{training_seed}"
        path = out / "models" / f"{stem}.zip"
        metadata = json.loads((out / "models" / f"{stem}.json").read_text(encoding="utf-8"))
        if sha256_file(path) != metadata["model_sha256"]:
            raise ValueError(f"checkpoint changed: {stem}")
        model = EvaluationPolicy({"ppo": PPO, "dqn": DQN}[agent].load(path, device="cpu"), arm)
    rows = []
    for market in markets:
        penalty = 0.0 if arm == "no_terminal" else design.config.execution.terminal_penalty_bps
        try:
            params = episode_params(design, regime, plan["calibrated_regime"], market, penalty, normalization)
            if agent == "ac":
                params.update(lock["controls"][regime]["selected"])
            row = run_episode("ppo" if agent in ALGORITHMS else agent, params, model=model)
            row.pop("audit", None)
        except Exception as exc:  # retained, never rerun
            row = {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}", COST: None, COMPLETION: None}
        row.update(regime=regime, agent=agent, arm=arm, training_seed=training_seed, seed=market)
        rows.append(json.loads(canonical_json(row)))
    return rows


def _fit_controls(design: V05Design, calibrated: dict | None) -> dict:
    from .controls import estimate_ac_parameters
    from .runner import cfg_from_params
    result = {}
    for regime in design.regimes:
        try:
            config, extensions = regime_setup(design, regime, calibrated)
        except ValueError as exc:
            result[regime] = {"fit": {"status": "NOT_AVAILABLE", "error": str(exc)}, "selected": {}}
            continue
        params = runner_parameters(config, design.identification_seeds[0], 0.0)
        try:
            fit = estimate_ac_parameters(cfg_from_params(params), design.identification_seeds,
                                         horizon=config.execution.horizon, sample_dt=config.execution.horizon / 10,
                                         execution_interval=config.execution.horizon / 20,
                                         warmup_seconds=config.execution.warmup_seconds, flow_extensions=extensions)
        except Exception as exc:
            fit = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        selected = {"temp_impact": config.execution.temp_impact, "sigma": config.execution.sigma,
                    "risk_aversion": config.execution.risk_aversion}
        if fit.get("status") == "PASS":
            selected.update(temp_impact=fit["temp_impact"], sigma=fit["sigma"])
            if fit["sigma"] > 0:
                selected["risk_aversion"] = fit["temp_impact"] / (fit["sigma"] * config.execution.horizon) ** 2
        result[regime] = {"fit": fit, "selected": selected}
    return result


def evaluate(out: str | Path) -> dict:
    out = Path(out)
    plan, design = read_plan(out)
    if (out / "evaluation_lock.json").exists():
        raise FileExistsError("preserved evaluation attempt; no rerun")
    summary = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
    controls = _fit_controls(design, plan["calibrated_regime"])
    write_json(out / "evaluation_lock.json", {"plan_sha256": pr.document_sha256(plan), "controls": controls,
                                              "training_summary_sha256": sha256_file(out / "training_summary.json"),
                                              "locked_at": utc_now()})
    trained = {(m["algorithm"], m["arm"], m["seed"]) for m in summary["models"] if "failed" not in m}
    cells = [(c, "main", None) for c in CONTROLS]
    cells += [(a, arm, s) for a in ALGORITHMS for arm in ARMS for s in design.training_seeds]
    tasks, withheld = [], []
    for regime in design.regimes:
        for agent, arm, seed in cells:
            if regime == "calibrated" and plan["calibrated_regime"] is None:
                withheld.append((regime, agent, arm, seed, "calibrated regime NOT_AVAILABLE"))
                continue
            if agent in ALGORITHMS and (agent, arm, seed) not in trained:
                withheld.append((regime, agent, arm, seed, "training failed"))
                continue
            tasks.append((str(out), plan, regime, agent, arm, seed, design.evaluation_seeds))
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        chunks = list(pool.map(_evaluate_chunk, tasks))
    rows = [row for chunk in chunks for row in chunk]
    for regime, agent, arm, seed, reason in withheld:
        for market in design.evaluation_seeds:
            rows.append({"status": "INVALID", "error": reason, COST: None, COMPLETION: None, "regime": regime,
                         "agent": agent, "arm": arm, "training_seed": seed, "seed": market})
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    result = summarize(rows, plan, design)
    result["ac_identification_status"] = {r: c["fit"].get("status") for r, c in controls.items()}
    write_json(out / "result.json", result)
    seal_experiment(out)
    return result


# ----------------------------------------------------------------------------- statistics

def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """One-sided exact upper confidence limit for a binomial proportion."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0

    def cdf(p: float) -> float:
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    low, high = k / n, 1.0
    for _ in range(200):
        mid = (low + high) / 2
        if cdf(mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def _cost(row: dict) -> float:
    value = row.get(COST)
    return float(value) if row.get("status") in {"VALID", "WARNING"} and value is not None and math.isfinite(value) else math.nan


def _completion(row: dict) -> float:
    value = row.get(COMPLETION)
    return float(value) if isinstance(value, bool) else math.nan


def describe(rows: list[dict]) -> dict:
    n = len(rows)
    costs = np.array([_cost(r) for r in rows])
    finite = costs[np.isfinite(costs)]
    within = [r.get(COMPLETION) for r in rows]
    final = [r.get("mandate_final_settlement_completion") for r in rows]
    result = {"episodes": n, "invalid_or_missing_cost": int(np.isnan(costs).sum()),
              "within_horizon_completion_rate": float(np.mean([w is True for w in within])) if n else None,
              "within_horizon_unknown": sum(w is None for w in within),
              "final_settlement_completion_rate": float(np.mean([f is True for f in final])) if n else None,
              "settlement_only_completions": sum(bool(r.get("mandate_settlement_only_completion")) for r in rows),
              "mean_residual_at_horizon": float(np.mean([r.get("mandate_residual_inventory_at_horizon") or 0 for r in rows]))
              if n else None,
              "mean_residual_after_settlement": float(np.mean([r.get("mandate_residual_inventory_after_settlement") or 0
                                                               for r in rows])) if n else None,
              "mean_post_horizon_fill_qty": float(np.mean([r.get("mandate_post_horizon_filled_qty") or 0 for r in rows]))
              if n else None,
              "max_lateness_s": max([r.get("mandate_lateness_seconds") or 0.0 for r in rows], default=None)}
    if len(finite):
        p95 = float(np.quantile(finite, 0.95))
        result.update(mean_cost_bps=float(finite.mean()), median_cost_bps=float(np.median(finite)),
                      p95_cost_bps=p95, tail_mean_cost_bps=float(finite[finite >= p95].mean()),
                      mean_realized_fill_cost_bps=_mean_of(rows, "mandate_realized_fill_cost_bps"),
                      mean_hypothetical_residual_bps=_mean_of(rows, "mandate_hypothetical_residual_valuation_bps"))
    return result


def _finite_mean(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _mean_of(rows: list[dict], key: str) -> float | None:
    """Mean over rows where the value exists (zero is a value); None when none do."""
    values = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return float(np.mean(values)) if values else None


def summarize(rows: list[dict], plan: dict, design: V05Design) -> dict:
    index = {(r["regime"], r["agent"], r["arm"], r["training_seed"], r["seed"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate evaluation cells")
    family = plan["family_size"]
    alpha = 0.05 / family
    markets = design.evaluation_seeds
    summaries, comparisons = [], []
    for regime in design.regimes:
        for agent, arm in [(c, "main") for c in CONTROLS] + [(a, arm) for a in ALGORITHMS for arm in ARMS]:
            group = [r for r in rows if (r["regime"], r["agent"], r["arm"]) == (regime, agent, arm)]
            entry = {"regime": regime, "agent": agent, "arm": arm, **describe(group)}
            if agent in ALGORITHMS:
                entry["training_seed_means_bps"] = {
                    str(s): _finite_mean([_cost(r) for r in group if r["training_seed"] == s])
                    for s in design.training_seeds}
                entry["training_seed_within_horizon_rates"] = {
                    str(s): float(np.mean([r.get(COMPLETION) is True for r in group if r["training_seed"] == s]))
                    for s in design.training_seeds}
            elif group:
                values = np.array([_cost(r) for r in group])
                if np.isfinite(values).all():
                    entry["mean_ci_bps"] = list(bootstrap_ci(values, n_boot=design.bootstrap_samples,
                                                             seed=design.statistics_seed))
            summaries.append(entry)
        for algorithm in ALGORITHMS:
            for reference in REFERENCES:
                # Controls: the algorithm's main arm versus the control (one control row per market).
                # Ablations: the ablated arm versus the same training seed's main arm.
                agent_arm = "main" if reference in CONTROLS else reference
                cost = np.full((len(design.training_seeds), len(markets)), np.nan)
                agent_miss = np.zeros(cost.shape, dtype=bool)
                ref_complete = np.zeros(cost.shape, dtype=bool)
                known = np.ones(cost.shape, dtype=bool)
                agent_completion = np.full(cost.shape, np.nan)
                for i, s in enumerate(design.training_seeds):
                    for j, m in enumerate(markets):
                        left = index.get((regime, algorithm, agent_arm, s, m), {})
                        right = (index.get((regime, reference, "main", None, m), {}) if reference in CONTROLS
                                 else index.get((regime, algorithm, "main", s, m), {}))
                        cost[i, j] = _cost(left) - _cost(right)
                        lc, rc = _completion(left), _completion(right)
                        known[i, j] = math.isfinite(lc) and math.isfinite(rc)
                        agent_miss[i, j] = lc == 0.0
                        ref_complete[i, j] = rc == 1.0
                        agent_completion[i, j] = lc
                comparison = {"regime": regime, "agent": f"{algorithm}:{agent_arm}",
                              "reference": reference if reference in CONTROLS else f"{algorithm}:main",
                              "delta": "agent minus reference; negative cost favours the agent", "family_size": family}
                if np.isfinite(cost).all():
                    comparison["cost"] = {"status": "AVAILABLE", **crossed_interval(
                        cost, samples=design.bootstrap_samples, alpha=alpha, seed=design.statistics_seed)}
                else:
                    comparison["cost"] = {"status": "WITHHELD", "reason": "missing or INVALID planned outcome"}
                if known.all():
                    z = (agent_miss & ref_complete).any(axis=0)
                    upper = clopper_pearson_upper(int(z.sum()), len(markets), alpha)
                    comparison["completion"] = {"status": "AVAILABLE", "discordant_markets": int(z.sum()),
                                                "markets": len(markets), "cp_upper_pi": upper, "lower_bound": -upper,
                                                "observed_agent_within_horizon_rate": float(np.mean(agent_completion))}
                else:
                    comparison["completion"] = {"status": "WITHHELD", "reason": "missing within-horizon outcome"}
                comparison["joint_success_gate_passed"] = bool(
                    comparison["cost"]["status"] == comparison["completion"]["status"] == "AVAILABLE"
                    and comparison["cost"]["ci_high_bps"] < 0
                    and comparison["completion"]["lower_bound"] >= -design.completion_noninferiority_margin)
                comparisons.append(comparison)
    return {"evidence_level": design.evidence_level, "family_size": family, "alpha_per_hypothesis": alpha,
            "planned_comparisons": len(comparisons), "summaries": summaries, "comparisons": comparisons,
            "joint_gates_passed": sum(c["joint_success_gate_passed"] for c in comparisons),
            "cost_intervals_excluding_zero": sum(c["cost"]["status"] == "AVAILABLE" and
                                                 (c["cost"]["ci_high_bps"] < 0 or c["cost"]["ci_low_bps"] > 0)
                                                 for c in comparisons),
            "planned_episodes": len(design.regimes) * len(markets) * (len(CONTROLS) + len(ALGORITHMS) * len(ARMS)
                                                                      * len(design.training_seeds)),
            "actual_episodes": len(rows), "invalid_episodes": sum(r.get("status") == "INVALID" for r in rows),
            "interpretation": ("Per-contrast gates only; no global best policy. Synthetic simulator evidence; "
                               "no historical, live or profitability claim follows.")}


def verify(out: str | Path) -> dict:
    out = Path(out)
    issues = []
    try:
        plan = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
        seal = json.loads((out / "registration_seal.json").read_text(encoding="utf-8"))
        if pr.document_sha256(plan) != seal["plan_sha256"]:
            issues.append("registered plan changed")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["files"]
        actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
        if actual != set(manifest):
            issues.append("sealed artifact file set differs")
        for name, digest in manifest.items():
            if sha256_file(out / name) != digest:
                issues.append(f"artifact mismatch: {name}")
        design = V05Design.model_validate_json(json.dumps(plan["design"]))
        rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        expected = summarize(rows, plan, design)
        if canonical_json(expected) != canonical_json({k: result.get(k) for k in expected}):
            issues.append("result differs from the registered episode journal")
        for model in json.loads((out / "training_summary.json").read_text(encoding="utf-8"))["models"]:
            if "failed" in model:
                continue
            stem = f"{model['algorithm']}-{model['arm']}-{model['seed']}"
            if sha256_file(out / "models" / f"{stem}.zip") != model["model_sha256"]:
                issues.append(f"checkpoint mismatch: {stem}")
    except (OSError, ValueError, KeyError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("register", "train", "evaluate", "verify"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="labelled pilot: 2 seeds, 8 markets, 1024 steps")
    args = parser.parse_args(argv)
    if args.operation == "register":
        result = {"registered": str(register(args.out, pilot={"training_seeds": 2, "markets": 8, "timesteps": 1024}
                                              if args.pilot else None))}
    else:
        result = {"train": train, "evaluate": evaluate, "verify": verify}[args.operation](args.out)
    summary = {k: v for k, v in result.items() if k not in {"summaries", "comparisons", "models"}} \
        if isinstance(result, dict) else result
    print(json.dumps(summary, indent=2, default=str))
    if args.operation == "verify" and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

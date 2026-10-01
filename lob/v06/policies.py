"""v0.6 learned execution policies: single-world and ensemble (domain-randomized) PPO/DQN.

Both variants use the unchanged v0.5 environment (``lob.rl_env.LOBExecutionEnv``)
under the frozen v0.6 mandate, the v0.4 observation contract and the v0.5 M7
hyperparameters. ``single`` trains in the selected v3 world; ``ensemble`` draws
one plausible ensemble member uniformly at every episode reset, with a member
stream seeded from the training seed (domain randomization derived from v0.6
calibration uncertainty, not arbitrary ranges). Observation normalization is
fitted on random exploration in the variant's training worlds and sealed before
any fit. Final fixed-budget checkpoints only; failures are retained.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from ..core_study import TRAIN_MARKET_RANGE
from ..experiments.registry import sha256_file, utc_now
from ..observations import fit_normalization
from .evidence import write_json
from .worlds import World, episode_params

ALGORITHMS = ("ppo", "dqn")
VARIANTS = ("single", "ensemble")


def make_env(world: World, mandate: dict, normalization: dict | None, *, training: bool):
    from ..completion import CompletionConstraint
    from ..engine import Side
    from ..rl_env import LOBExecutionEnv
    from ..runner import cfg_from_params
    params = episode_params(world, mandate, 0, normalization=normalization)
    return LOBExecutionEnv(total_qty=params["qty"], horizon=params["horizon"], decision_dt=params["dt"],
                           side=Side.BUY if params["side"] == "buy" else Side.SELL,
                           warmup=params["warmup_seconds"], cfg=cfg_from_params(params), fees=params["fees"],
                           risk=params["risk"], terminal_penalty_bps=params["terminal_penalty_bps"],
                           settlement_timeout=params["settlement_timeout"],
                           settlement_poll_dt=params["settlement_poll_dt"],
                           seed_range=TRAIN_MARKET_RANGE if training else None,
                           completion=CompletionConstraint(True, mandate["completion_urgency_fraction"]),
                           observation_version="v04", observation_normalization=normalization,
                           flow_extensions=params["flow_extensions"])


class WorldMixture(gym.Env):
    """Delegates each episode to one world drawn uniformly from a seeded member stream."""

    metadata = {"render_modes": []}

    def __init__(self, envs: list[gym.Env], seed: int) -> None:
        super().__init__()
        if not envs:
            raise ValueError("at least one world is required")
        self.envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        for env in envs[1:]:
            if env.observation_space.shape != self.observation_space.shape or env.action_space != self.action_space:
                raise ValueError("all worlds must share observation and action spaces")
        self._members = np.random.default_rng(seed)
        self.active = 0
        self.history: list[int] = []

    def reset(self, *, seed=None, options=None):
        self.active = int(self._members.integers(0, len(self.envs)))
        self.history.append(self.active)
        return self.envs[self.active].reset(seed=seed, options=options)

    def step(self, action):
        return self.envs[self.active].step(action)

    def close(self) -> None:
        for env in self.envs:
            env.close()


def training_env(variant: str, worlds: list[World], mandate: dict, normalization: dict | None, seed: int,
                 offset: int) -> gym.Env:
    if variant == "single":
        return make_env(worlds[0], mandate, normalization, training=True)
    envs = [make_env(w, mandate, normalization, training=True) for w in worlds]
    return WorldMixture(envs, seed + offset)


def fit_observation_normalization(variant: str, worlds: list[World], mandate: dict, seeds: list[int]) -> dict:
    """Random-action exploration in the variant's training worlds (member = seed index modulo worlds)."""
    observations = []
    for index, seed in enumerate(seeds):
        world = worlds[0] if variant == "single" else worlds[index % len(worlds)]
        env = make_env(world, mandate, None, training=False)
        try:
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
    return {**fit_normalization(observations), "seeds": list(seeds), "variant": variant,
            "worlds": [w.name for w in (worlds[:1] if variant == "single" else worlds)]}


def train_one(task: tuple) -> dict:
    """Train one (algorithm, variant, seed) model; failures are written and returned, never raised."""
    out, plan, algorithm, variant, seed = task
    import torch
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(out)
    worlds = [World.from_dict(w) for w in plan["training_worlds"][variant]]
    normalization = json.loads((out / f"normalization-{variant}.json").read_text(encoding="utf-8"))["normalization"]
    stem = f"{algorithm}-{variant}-{seed}"
    write_json(out, f"models/{stem}.attempt.json", {"started_at": utc_now(), "seed": seed})
    started = time.monotonic()
    env = Monitor(training_env(variant, worlds, plan["mandate"], normalization, seed,
                               plan["domain_randomization_seed_offset"]))
    try:
        model = {"ppo": PPO, "dqn": DQN}[algorithm]("MlpPolicy", env, seed=seed, verbose=0, **plan["hyperparameters"][algorithm])
        model.learn(total_timesteps=plan["timesteps"])
        model.save(out / "models" / f"{stem}.zip")
        rewards = [float(x) for x in env.get_episode_rewards()]
        members = getattr(env.env, "history", [])
        metadata = {"algorithm": algorithm, "variant": variant, "seed": seed, "timesteps": int(model.num_timesteps),
                    "elapsed_seconds": time.monotonic() - started, "episodes": len(rewards),
                    "reward_first_decile_mean": float(np.mean(rewards[: max(1, len(rewards) // 10)])) if rewards else None,
                    "reward_last_decile_mean": float(np.mean(rewards[-max(1, len(rewards) // 10):])) if rewards else None,
                    "member_counts": np.bincount(members, minlength=len(worlds)).tolist() if members else None,
                    "model_sha256": sha256_file(out / "models" / f"{stem}.zip")}
        write_json(out, f"models/{stem}.json", metadata)
        return metadata
    except BaseException as exc:  # retained, never rerun
        write_json(out, f"models/{stem}.failure.json", {"error": f"{type(exc).__name__}: {exc}", "failed_at": utc_now()})
        return {"algorithm": algorithm, "variant": variant, "seed": seed, "failed": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


def load_policy(out: Path, algorithm: str, variant: str, seed: int):
    from stable_baselines3 import DQN, PPO
    from ..policy_study import EvaluationPolicy
    stem = f"{algorithm}-{variant}-{seed}"
    metadata = json.loads((out / "models" / f"{stem}.json").read_text(encoding="utf-8"))
    path = out / "models" / f"{stem}.zip"
    if sha256_file(path) != metadata["model_sha256"]:
        raise ValueError(f"checkpoint changed: {stem}")
    return EvaluationPolicy({"ppo": PPO, "dqn": DQN}[algorithm].load(path, device="cpu"), "main")

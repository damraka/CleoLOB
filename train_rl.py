"""Train a PPO execution agent on LOBExecutionEnv (stable-baselines3).

Usage:
    python train_rl.py --timesteps 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from lob.rl_env import LOBExecutionEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--qty", type=int, default=10_000)
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="models/ppo_lob")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="TensorBoard log directory (view with: tensorboard --logdir logs)")
    args = parser.parse_args()

    env = Monitor(LOBExecutionEnv(total_qty=args.qty, horizon=args.horizon,
                                  decision_dt=args.dt))
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                n_steps=2048, batch_size=256, gamma=0.999, ent_coef=0.005,
                tensorboard_log=args.logdir)
    model.learn(total_timesteps=args.timesteps, tb_log_name="ppo_lob")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved -> {args.out}.zip")
    print(f"training curves -> tensorboard --logdir {args.logdir}")

    # Quick sanity evaluation on 3 fresh seeds.
    for seed in (101, 102, 103):
        obs, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            total += r
            done = terminated or truncated
        print(f"seed={seed}  return={total:+.2f} bps-equivalent  "
              f"leftover={info['remaining']}")


if __name__ == "__main__":
    main()
"""Train a PPO execution agent on LOBExecutionEnv (stable-baselines3).

Usage:
    python train_rl.py --timesteps 200000

To train in safe chunks instead of one long run (recommended on machines that can't
sustain hours of 100% CPU), reload and continue from a saved checkpoint:

    python train_rl.py --timesteps 300000                    # first chunk -> models/ppo_lob.zip
    python train_rl.py --timesteps 300000 --resume            # picks up where it left off
    python train_rl.py --timesteps 300000 --resume            # ...and again

Each run also saves a checkpoint automatically if interrupted with Ctrl+C, so a chunk
in progress isn't fully lost.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from lob.rl_env import LOBExecutionEnv

TRAIN_SEED_RANGE = (1_000_000, 2**31 - 1)
VALIDATION_SEED_RANGE = (900_000, 1_000_000)
# Validation influences model selection. It is not an untouched final test set.
# Explicit domains prevent reuse of market seeds between these two environments;
# they do not establish out-of-distribution validation of the market model.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="additional steps to train this run (not a lifetime total)")
    parser.add_argument("--qty", type=int, default=10_000)
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help="PPO and environment RNG initialization seed")
    parser.add_argument("--out", type=str, default="models/ppo_lob")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="TensorBoard log directory (view with: tensorboard --logdir logs)")
    parser.add_argument("--resume", action="store_true",
                        help="load --out and continue training instead of starting fresh")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="run held-out evaluation every N training steps (0 disables it)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="held-out episodes per evaluation — same market config as "
                             "training, but seeds the training run has never seen")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_freq < 0 or args.eval_episodes <= 0 or args.seed < 0:
        parser.error("timesteps/eval-episodes must be positive; eval-freq and seed nonnegative")

    env = Monitor(LOBExecutionEnv(total_qty=args.qty, horizon=args.horizon,
                                  decision_dt=args.dt, seed_range=TRAIN_SEED_RANGE))
    ckpt = Path(args.out).with_suffix(".zip")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    callback = None
    if args.eval_freq > 0:
        eval_env = Monitor(LOBExecutionEnv(total_qty=args.qty, horizon=args.horizon,
                                           decision_dt=args.dt, seed_range=VALIDATION_SEED_RANGE))
        eval_env.reset(seed=args.seed)
        callback = EvalCallback(eval_env, eval_freq=args.eval_freq,
                                n_eval_episodes=args.eval_episodes, deterministic=True,
                                log_path=args.logdir, best_model_save_path=None, verbose=0)
        print(f"held-out eval every {args.eval_freq:,} steps on {args.eval_episodes} episodes "
             f"(market seeds in {VALIDATION_SEED_RANGE}) — watch 'eval/mean_reward' in TensorBoard "
             f"alongside 'rollout/ep_rew_mean': if training keeps climbing while eval "
             f"plateaus or drops, that's overfitting to this training regime")

    if args.resume:
        if not ckpt.exists():
            raise SystemExit(f"--resume given but no checkpoint at {ckpt} — drop --resume "
                             f"to start a fresh model")
        model = PPO.load(str(ckpt), env=env)
        model.tensorboard_log = args.logdir
        prior = model.num_timesteps
        print(f"resuming {ckpt} from {prior:,} steps -> training {args.timesteps:,} more "
              f"(total {prior + args.timesteps:,})")
    else:
        if ckpt.exists():
            print(f"! {ckpt} already exists and will be overwritten — pass --resume to "
                 f"continue training it instead, or --out for a different path")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                    n_steps=2048, batch_size=256, gamma=0.999, ent_coef=0.005,
                    tensorboard_log=args.logdir, seed=args.seed)

    try:
        model.learn(total_timesteps=args.timesteps, tb_log_name="ppo_lob",
                   reset_num_timesteps=not args.resume, callback=callback)
    except KeyboardInterrupt:
        model.save(args.out)
        print(f"\ninterrupted — checkpoint saved -> {args.out}.zip "
             f"({model.num_timesteps:,} steps total). Resume with --resume.")
        raise SystemExit(130)

    model.save(args.out)
    print(f"saved -> {args.out}.zip  ({model.num_timesteps:,} steps total)")
    print(f"training curves -> tensorboard --logdir {args.logdir}")

    # These are diagnostics, not a reserved final test set.
    diagnostic_env = LOBExecutionEnv(total_qty=args.qty, horizon=args.horizon, decision_dt=args.dt)
    for seed in (101, 102, 103):
        obs, _ = diagnostic_env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = diagnostic_env.step(int(action))
            total += r
            done = terminated or truncated
        print(f"seed={seed}  return={total:+.2f} bps-equivalent  "
              f"leftover={info['remaining']}")


if __name__ == "__main__":
    main()

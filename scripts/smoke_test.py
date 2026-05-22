"""Short PPO smoke-test on the continuum-robot reach env.

Trains for a few thousand steps on CPU and prints mean episode reward / final
tip-to-target distance before and after training. Not meant to converge --
just to prove the env + algorithm wire up end-to-end.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from continuum_rl import ContinuumReachEnv  # noqa: E402


ALGOS = {"ppo", "sac"}


def evaluate(model, n_episodes: int = 10, seed: int = 123) -> tuple[float, float, float]:
    env = ContinuumReachEnv()
    returns, final_dists, successes = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        last_dist = float("nan")
        success = False
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            last_dist = info["dist"]
            success = success or bool(info["is_success"])
            done = terminated or truncated
        returns.append(total)
        final_dists.append(last_dist)
        successes.append(float(success))
    env.close()
    return float(np.mean(returns)), float(np.mean(final_dists)), float(np.mean(successes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=sorted(ALGOS), default="sac")
    parser.add_argument("--timesteps", type=int, default=15_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    print(f"[smoke] sb3 {args.algo.upper()} on ContinuumReachEnv | timesteps={args.timesteps}")

    print("[smoke] evaluating random policy ...")
    r0, d0, s0 = evaluate(None, n_episodes=10)
    print(f"[smoke]   random:  mean_return={r0:+.3f}  mean_final_dist={d0*1000:.2f} mm  success_rate={s0:.2f}")

    if args.algo == "ppo":
        env = make_vec_env(ContinuumReachEnv, n_envs=args.n_envs, seed=args.seed)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.98,
            gae_lambda=0.95,
            ent_coef=0.0,
            policy_kwargs=dict(net_arch=[64, 64]),
            device="cpu",
            seed=args.seed,
            verbose=0,
        )
    else:
        env = ContinuumReachEnv()
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=50_000,
            batch_size=256,
            tau=0.02,
            gamma=0.98,
            train_freq=1,
            gradient_steps=1,
            learning_starts=500,
            ent_coef=0.005,
            policy_kwargs=dict(net_arch=[64, 64]),
            device="cpu",
            seed=args.seed,
            verbose=0,
        )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    dt = time.time() - t0
    print(f"[smoke] trained {args.algo.upper()} in {dt:.1f}s ({args.timesteps/dt:.0f} steps/s)")

    print("[smoke] evaluating trained policy ...")
    r1, d1, s1 = evaluate(model, n_episodes=10)
    print(f"[smoke]   trained: mean_return={r1:+.3f}  mean_final_dist={d1*1000:.2f} mm  success_rate={s1:.2f}")

    improvement_mm = (d0 - d1) * 1000.0
    print(f"[smoke] tip-to-target distance improved by {improvement_mm:+.2f} mm")

    if args.save:
        model.save(args.save)
        print(f"[smoke] model saved to {args.save}")


if __name__ == "__main__":
    main()

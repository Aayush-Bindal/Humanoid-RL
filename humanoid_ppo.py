"""Train and evaluate a PPO baseline on the MuJoCo Humanoid state vector.

Install with: python -m pip install "stable-baselines3>=2.7,<3" "gymnasium[mujoco]>=1.2,<2"
Train with:   python humanoid_ppo.py train --run-dir runs/humanoid-seed0
Evaluate:     python humanoid_ppo.py evaluate runs/humanoid-seed0/policy.zip

Each training run saves a policy and one episode log per environment.
"""

import argparse
from pathlib import Path


ENV_ID = "Humanoid-v5"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from torch import set_num_threads

    # PPO uses a small MLP here. One Torch thread avoids CPU overhead per update.
    set_num_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    env = make_vec_env(
        ENV_ID,
        n_envs=args.envs,
        seed=args.seed,
        monitor_dir=str(args.run_dir),
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=64,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
        model.learn(total_timesteps=args.steps)
        checkpoint = args.run_dir / "policy.zip"
        model.save(checkpoint)
        print(f"Saved {checkpoint} after {model.num_timesteps:,} steps.")
    finally:
        env.close()


def evaluate(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    model = PPO.load(args.checkpoint, device="cpu")
    render_mode = "human" if args.render else None
    # Use a fresh environment and a different default seed from training.
    env = make_vec_env(ENV_ID, seed=args.seed, env_kwargs={"render_mode": render_mode})
    try:
        mean_return, std_return = evaluate_policy(
            model, env, n_eval_episodes=args.episodes, deterministic=True
        )
        print(f"{ENV_ID} | {args.episodes} episodes | seed {args.seed}")
        print(f"Mean return: {mean_return:.2f} (SD {std_return:.2f})")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="train PPO and save a checkpoint")
    training.add_argument("--run-dir", type=Path, required=True, help="new output directory")
    training.add_argument("--steps", type=positive_int, default=1_000_000,
                          help="minimum training timesteps (rounded up to a PPO rollout)")
    training.add_argument("--envs", type=positive_int, default=4)
    training.add_argument("--seed", type=int, default=0)
    training.set_defaults(run=train)

    evaluation = commands.add_parser("evaluate", help="evaluate a saved PPO checkpoint")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--episodes", type=positive_int, default=10)
    evaluation.add_argument("--seed", type=int, default=1000)
    evaluation.add_argument("--render", action="store_true", help="show the MuJoCo viewer")
    evaluation.set_defaults(run=evaluate)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

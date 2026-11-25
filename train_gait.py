"""Fine-tune the existing PPO policy on MuJoCo routes around collidable boxes.

Use separate course seeds for fitting and for comparison. Checkpoint selection
favors reliable arrival before comparing speed and gait. The optional phase
policy augments the original inputs with a stride clock and steering command.
"""

import argparse
from functools import partial
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from torch import no_grad, set_num_threads

from gait_support import GAIT_FREQUENCY, PHASE_FEATURES, walking_observation
from tabletop_demo import make_course, plan_route, steer


BASELINE = Path("runs/baseline-seed0/policy.zip")
VALIDATION_SEED = 40_000
TEST_SEED = 70_000


def obstacle_layout(seed):
    rng = np.random.default_rng(seed)
    boxes = []
    # Leave room between the boxes even after the planner adds body clearance.
    for index, x in enumerate((2.55, 5.45)):
        x += rng.uniform(-0.08, 0.08)
        center = (1.85 if index == 0 else 3.20) + rng.uniform(-0.48, 0.48)
        half_width = rng.uniform(0.26, 0.37)
        half_depth = rng.uniform(0.38, 0.66)
        boxes.append((x - half_width, center - half_depth,
                      x + half_width, center + half_depth))
    return boxes


def forward_distance(rotation, displacement):
    """Project movement onto the torso's horizontal forward axis."""
    heading = rotation[:2, 0]
    return float(np.dot(heading, displacement) / max(np.linalg.norm(heading), 1e-8))


class ObstacleWalk(gym.Env):
    def __init__(self, layout_seed, grounded_reward=False, lower_lift=False,
                 phase_gait=False, phase_support=False, pace_scale=1.0,
                 heading_weight=0.0):
        super().__init__()
        self.grounded_reward = grounded_reward
        self.lower_lift = lower_lift
        self.phase_gait = phase_gait
        self.phase_support = phase_support or phase_gait
        self.pace_scale = pace_scale
        self.heading_weight = heading_weight
        self.boxes = obstacle_layout(layout_seed)
        with TemporaryDirectory() as directory:
            self.course = make_course(self.boxes, [], directory, render=False, overlay=True)
        self.sim = self.course.unwrapped
        self.action_space = self.sim.action_space
        self.observation_space = self.sim.observation_space
        if self.phase_gait:
            size = self.observation_space.shape[0] + PHASE_FEATURES
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), dtype=np.float64)
        self.torso = self.sim.model.body("torso").id
        self.floor = self.sim.model.geom("floor").id
        self.feet = [self.sim.model.geom(name).id for name in ("left_foot", "right_foot")]
        self.obstacles = {self.sim.model.geom(f"obstacle_{i}").id for i in range(len(self.boxes))}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.course.reset(seed=seed)
        rng = self.np_random
        for _ in range(24):
            start = (rng.uniform(0.8, 1.15), rng.uniform(1.05, 3.95))
            goal = (rng.uniform(6.95, 7.2), rng.uniform(1.05, 3.95))
            if rng.random() < 0.35:
                start, goal = goal, start
            try:
                self.route = np.asarray(plan_route(self.boxes, origin=start, destination=goal,
                                                   clearance=rng.uniform(0.81, 0.96)))
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("Could not sample a route around the obstacles.")

        self.target = 1
        self.speed = float(rng.choice([0.78, 1.0, 1.28])) * self.pace_scale
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = start
        self.sim.set_state(qpos, np.zeros(self.sim.model.nv))
        self.start = np.asarray(start)
        self.previous_action = np.zeros(self.action_space.shape)
        self.previous_feet = self.sim.data.geom_xpos[self.feet, :2].copy()
        self.last_single_support = None
        self.last_switch_step = 0
        self.steps = 0
        self.sums = dict(action_change=0.0, flight=0.0, single_support=0.0, lean=0.0,
                         slip=0.0, up_speed=0.0, left_support=0.0, right_support=0.0,
                         switches=0, distance=0.0, forward_distance=0.0)
        return self.observation(), {}

    def observation(self):
        delta = self.route[self.target] - self.sim.data.qpos[:2]
        return walking_observation(self.sim, self.torso, delta, self.speed, self.phase_gait)

    def tracking_reward(self, grounded, command_speed):
        # Both feet share one stride cycle and are half a cycle apart. The
        # target keeps a foot on the floor through support and lifts it on swing.
        phases = (self.sim.data.time * GAIT_FREQUENCY + np.array([0.0, 0.5])) % 1
        stance = phases < 0.61
        swing = np.clip((phases - 0.61) / 0.39, 0, 1)
        step_extent = 0.61 * command_speed / (2 * GAIT_FREQUENCY)
        fore_aft = np.where(stance, 1 - 2 * phases / 0.61, -np.cos(np.pi * swing)) * step_extent
        height = 0.078 + np.where(stance, 0, 0.115 * np.sin(np.pi * swing))
        rotation = self.sim.data.xmat[self.torso].reshape(3, 3)
        feet = self.sim.data.geom_xpos[self.feet]
        actual_fore_aft = (feet[:, :2] - self.sim.data.qpos[:2]) @ rotation[:2, 0]
        error = (np.abs(feet[:, 2] - height) / 0.065
                 + np.abs(actual_fore_aft - fore_aft) / 0.22)
        return float(0.90 * np.exp(-error).mean() + 0.18 * (stance == grounded).mean())

    def step(self, action):
        sim = self.sim
        position = sim.data.qpos[:2].copy()
        waypoint = self.route[self.target]
        delta = waypoint - position
        old_distance = np.linalg.norm(delta)
        steer(sim, self.torso, delta, max_speed=self.speed,
              lower_lift=self.lower_lift, phase_gait=self.phase_support)
        _, _, fell, _, _ = sim.step(action)

        # Measure support and collisions using the simulator's actual contacts.
        contacts = set()
        collided = False
        for contact in sim.data.contact:
            geom1, geom2 = contact.geom1, contact.geom2
            if geom1 in self.obstacles or geom2 in self.obstacles:
                collided = True
            if geom1 == self.floor:
                contacts.add(geom2)
            elif geom2 == self.floor:
                contacts.add(geom1)
        grounded = np.asarray([foot in contacts for foot in self.feet])
        support = int(grounded.sum())
        bad_contact = bool(contacts.difference(self.feet))
        foot_positions = sim.data.geom_xpos[self.feet, :2].copy()
        foot_speeds = np.linalg.norm(foot_positions - self.previous_feet, axis=1) / sim.dt
        slip = float(np.clip(foot_speeds[grounded], 0, 3).sum()) / max(1, support)
        action_change = float(np.sqrt(np.mean(np.square(action - self.previous_action))))
        lean = 1 - float(sim.data.xmat[self.torso, 8])
        up_speed = abs(float(sim.data.qvel[2]))
        progress = float(old_distance - np.linalg.norm(waypoint - sim.data.qpos[:2]))
        displacement = sim.data.qpos[:2] - position
        distance = float(np.linalg.norm(displacement))
        forward = forward_distance(sim.data.xmat[self.torso].reshape(3, 3), displacement)
        self.previous_feet = foot_positions
        self.previous_action = action.copy()
        self.steps += 1

        self.sums["action_change"] += action_change
        self.sums["flight"] += support == 0
        self.sums["single_support"] += support == 1
        self.sums["lean"] += lean
        self.sums["up_speed"] += up_speed
        self.sums["slip"] += slip
        self.sums["left_support"] += grounded[0]
        self.sums["right_support"] += grounded[1]
        self.sums["distance"] += distance
        self.sums["forward_distance"] += forward

        switched = False
        if support == 1:
            single_side = int(np.argmax(grounded))
            switched = self.last_single_support is not None and self.last_single_support != single_side
            self.sums["switches"] += switched
            self.last_single_support = single_side

        reached = False
        if np.linalg.norm(waypoint - sim.data.qpos[:2]) < 0.18:
            if self.target == len(self.route) - 1:
                reached = True
            else:
                self.target += 1

        # Reward directed motion, upright posture and alternating planted feet.
        # Penalize visible skating, bouncing and abrupt actuator changes.
        reward = 0.18 + 28 * progress + 0.045 * (support == 1) + 0.07 * switched
        # Forward steps keep the full progress reward. Sideways and backward
        # travel lose credit in proportion to distance, even on a detour.
        reward -= self.heading_weight * max(0.0, distance - forward)
        reward -= (0.11 * (support == 0) + 0.11 * slip + 0.10 * up_speed
                   + 0.60 * lean + 0.24 * action_change + 0.25 * bad_contact
                   + 0.015 * float(np.mean(np.square(action))))
        if self.grounded_reward:
            # Favor a planted, alternating gait. Give no extra credit for
            # repeatedly toggling contacts within the same brief footstep.
            deliberate_switch = switched and self.steps - self.last_switch_step >= 10
            if deliberate_switch:
                self.last_switch_step = self.steps
            recent_imbalance = np.clip(
                (self.sums["left_support"] - self.sums["right_support"]) / 22, -1, 1)
            favor_other_foot = recent_imbalance * (int(grounded[1]) - int(grounded[0]))
            reward += 12 * progress + 0.40 * deliberate_switch + 0.08 * favor_other_foot
            reward -= 0.49 * (support == 0) + 0.18 * slip + 0.18 * action_change
        if self.phase_gait:
            reward += self.tracking_reward(grounded, min(self.speed, 2 * old_distance))

        status = ""
        if fell or collided:
            status = "collision" if collided else "fall"
            reward -= 16
        elif reached:
            status = "arrived"
            reward += 12
        elif self.steps >= 1800:
            status = "timeout"
            reward -= 6
        info = {"walk": self.metrics(status)} if status else {}
        return self.observation(), reward, bool(status and status != "timeout"), status == "timeout", info

    def metrics(self, status):
        elapsed = self.steps * self.sim.dt
        measurements = {}
        for key, value in self.sums.items():
            if key not in ("distance", "forward_distance", "switches"):
                measurements[key] = float(value / self.steps)
        measurements["support_imbalance"] = abs(measurements["left_support"] - measurements["right_support"])
        return dict(status=status, seconds=elapsed, speed=self.sums["distance"] / elapsed,
                    heading_alignment=self.sums["forward_distance"] / max(self.sums["distance"], 1e-8),
                    switches_sec=self.sums["switches"] / elapsed, **measurements)

    def close(self):
        self.course.close()


def evaluate(model, seed, episodes, lower_lift=False, phase_support=False, pace_scale=None):
    phase_gait = getattr(model, "gait_phase_inputs", False)
    if pace_scale is None:
        pace_scale = getattr(model, "gait_pace_scale", 1.0)
    results = []
    for index in range(episodes):
        episode_seed = seed + index
        env = ObstacleWalk(episode_seed, lower_lift=lower_lift,
                           phase_gait=phase_gait, phase_support=phase_support,
                           pace_scale=pace_scale)
        try:
            observation, _ = env.reset(seed=episode_seed)
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    results.append(dict(seed=episode_seed, **info["walk"]))
                    break
        finally:
            env.close()

    statuses = ("arrived", "collision", "fall", "timeout")
    finished = [result for result in results if result["status"] == "arrived"]
    successful = finished or results
    means = {}
    for key in ("seconds", "speed", "action_change", "flight", "single_support", "lean",
                "slip", "up_speed", "switches_sec", "support_imbalance", "heading_alignment"):
        means[key] = float(np.mean([result[key] for result in successful]))
    return dict(counts={status: sum(result["status"] == status for result in results)
                        for status in statuses}, successful_means=means, episodes=results)


def rank(summary, heading_weight=0.0):
    counts, means = summary["counts"], summary["successful_means"]
    # Compare gait only after reliability. Prefer less jitter, foot slip and
    # airtime, with a time penalty so standing still cannot win.
    cost = (3 * means["action_change"] + 3 * means["flight"] + means["slip"]
            + means["up_speed"] + 2 * means["lean"] + 0.06 * means["seconds"]
            + 0.65 * means["support_imbalance"]
            + heading_weight * (1 - means["heading_alignment"]))
    return counts["arrived"], -counts["collision"], -cost


def save_report(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_report(label, report):
    print(label, report["counts"], {key: round(value, 4)
          for key, value in report["successful_means"].items()}, flush=True)


def warm_start_phase(baseline, env, options):
    model = PPO("MlpPolicy", env, n_steps=256, **options)
    previous = baseline.policy.state_dict()
    expanded = model.policy.state_dict()
    # Initialize the extra input weights at zero so the old model's first
    # prediction is preserved. PPO can then learn how to use the new inputs.
    with no_grad():
        for name, weights in expanded.items():
            old = previous[name]
            if weights.shape == old.shape:
                weights.copy_(old)
            else:
                weights.zero_()
                weights[:, :old.shape[1]].copy_(old)
    model.policy.load_state_dict(expanded)
    return model


def train(args):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    baseline = PPO.load(args.baseline, device="cpu")
    pace_scale = getattr(baseline, "gait_pace_scale", 1.0)
    reference = evaluate(baseline, VALIDATION_SEED, args.validation,
                         lower_lift=getattr(baseline, "gait_lower_lift", False))
    save_report(args.run_dir / "baseline-validation.json", reference)
    print_report("Baseline validation", reference)
    if args.lower_lift or args.phase_gait:
        lower_lift_reference = evaluate(baseline, VALIDATION_SEED, args.validation, lower_lift=True,
                                        phase_support=args.phase_gait)
        save_report(args.run_dir / "baseline-lower-lift.json", lower_lift_reference)
        print_report("Baseline with lower lift", lower_lift_reference)

    set_num_threads(1)
    factories = [partial(ObstacleWalk, args.seed * 500 + index, args.grounded_reward,
                         args.lower_lift, args.phase_gait, pace_scale=pace_scale,
                         heading_weight=args.heading_weight)
                 for index in range(args.envs)]
    vec = SubprocVecEnv(factories, start_method="spawn") if args.parallel else DummyVecEnv(factories)
    env = VecMonitor(vec, str(args.run_dir / "training"))
    env.seed(args.seed)
    try:
        options = dict(device="cpu", seed=args.seed, learning_rate=args.learning_rate,
                       batch_size=256, n_epochs=5, clip_range=0.15, ent_coef=0.0,
                       target_kl=0.025, verbose=0)
        if args.phase_gait:
            model = warm_start_phase(baseline, env, options)
        else:
            model = PPO.load(args.baseline, env=env, **options)
        model.gait_lower_lift = args.lower_lift
        model.gait_phase_inputs = args.phase_gait
        model.gait_pace_scale = pace_scale
        # Start near the pretrained gait instead of exploring with large torque noise.
        with no_grad():
            model.policy.log_std.clamp_(max=math.log(args.exploration))
        best = None
        elapsed_steps = model.num_timesteps
        for _ in range(math.ceil(args.steps / args.interval)):
            remaining = args.steps - (model.num_timesteps - elapsed_steps)
            if remaining <= 0:
                break
            model.learn(total_timesteps=min(remaining, args.interval), reset_num_timesteps=False)
            steps = model.num_timesteps - elapsed_steps
            checkpoint = args.run_dir / f"policy-{steps:07d}.zip"
            model.save(checkpoint)
            validation = evaluate(model, VALIDATION_SEED, args.validation, args.lower_lift)
            save_report(checkpoint.with_suffix(".json"), validation)
            print_report(f"{steps:,} new steps", validation)
            if best is None or rank(validation, args.heading_weight) > rank(best, args.heading_weight):
                best = validation
                model.save(args.run_dir / "best.zip")
                save_report(args.run_dir / "selection.json",
                            dict(checkpoint=checkpoint.name, steps=steps, **validation))
    finally:
        env.close()


def compare(args):
    baseline = PPO.load(args.baseline, device="cpu")
    candidate = PPO.load(args.checkpoint, device="cpu")
    reference = evaluate(baseline, args.seed, args.episodes,
                         lower_lift=getattr(baseline, "gait_lower_lift", False))
    phase_gait = getattr(candidate, "gait_phase_inputs", False)
    lower_lift = args.lower_lift or getattr(candidate, "gait_lower_lift", False) or phase_gait
    pace_scale = getattr(candidate, "gait_pace_scale", 1.0)
    result = evaluate(candidate, args.seed, args.episodes, lower_lift)
    print_report("Baseline", reference)
    print_report("Fine-tuned", result)
    if args.output:
        report = dict(seed=args.seed, candidate_pace_scale=pace_scale,
                      baseline=reference, candidate=result)
        if lower_lift:
            control = evaluate(baseline, args.seed, args.episodes, lower_lift=True,
                               phase_support=phase_gait, pace_scale=pace_scale)
            print_report("Baseline with matched lift and pace", control)
            report["baseline_matched_assistance"] = control
        save_report(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--baseline", type=Path, default=BASELINE)
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--steps", type=int, default=200_000)
    training.add_argument("--interval", type=int, default=32_768)
    training.add_argument("--validation", type=int, default=12)
    training.add_argument("--envs", type=int, default=8)
    training.add_argument("--parallel", action="store_true")
    training.add_argument("--seed", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=4e-5)
    training.add_argument("--exploration", type=float, default=0.22)
    training.add_argument("--grounded-reward", action="store_true")
    training.add_argument("--lower-lift", action="store_true")
    training.add_argument("--phase-gait", action="store_true")
    training.add_argument("--heading-weight", type=float, default=0.0,
                          help="penalty per meter of travel that is not forward (default: 0)")
    training.set_defaults(execute=train)
    comparison = commands.add_parser("compare")
    comparison.add_argument("checkpoint", type=Path)
    comparison.add_argument("--baseline", type=Path, default=BASELINE)
    comparison.add_argument("--seed", type=int, default=TEST_SEED)
    comparison.add_argument("--episodes", type=int, default=30)
    comparison.add_argument("--output", type=Path)
    comparison.add_argument("--lower-lift", action="store_true")
    comparison.set_defaults(execute=compare)
    args = parser.parse_args()
    if args.command == "train" and (not math.isfinite(args.heading_weight) or args.heading_weight < 0):
        parser.error("--heading-weight must be a finite nonnegative number")
    set_num_threads(1)
    args.execute(args)


if __name__ == "__main__":
    main()

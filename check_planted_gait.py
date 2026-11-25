"""Measure pose foot tracking and check that the render data stays isolated."""
import numpy as np
from stable_baselines3 import PPO
from torch import set_num_threads

from avatar_gait import AvatarGait
from train_gait import ObstacleWalk


def compare_route(policy, seed):
    env = ObstacleWalk(seed, lower_lift=True, phase_gait=True,
                       pace_scale=policy.gait_pace_scale)
    try:
        observation, _ = env.reset(seed=seed)
        sim = env.sim
        gait = AvatarGait(sim.model)
        gait.step(sim.data.qpos[:2], 0.0, env.route[1])
        errors, raw_sliding, visible_sliding = [], [], []
        while True:
            before = sim.data.geom_xpos[gait.feet].copy()
            old_pose = gait.pose.geom_xpos[gait.feet].copy()
            planted = [gait.swing is None or gait.swing.side != side for side in range(2)]
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, stopped, timed_out, info = env.step(action)
            done = stopped or timed_out
            qpos = sim.data.qpos.copy()
            gait.step(qpos[:2], sim.dt, None if done else env.route[env.target])
            np.testing.assert_array_equal(qpos, sim.data.qpos)
            for side, foot in enumerate(gait.feet):
                shown = gait.pose.geom_xpos[foot]
                if gait.swing is None or gait.swing.side != side:
                    errors.append(np.linalg.norm(shown - [*gait.feet_xy[side], gait.foot_height]))
                    if planted[side]:
                        visible_sliding.append(np.linalg.norm(old_pose[side, :2] - shown[:2]) / sim.dt)
                if max(before[side, 2], sim.data.geom_xpos[foot, 2]) < gait.foot_height + 0.005:
                    raw_sliding.append(np.linalg.norm(
                        before[side, :2] - sim.data.geom_xpos[foot, :2]) / sim.dt)
            if done:
                return info["walk"]["status"], errors, raw_sliding, visible_sliding
    finally:
        env.close()


if __name__ == "__main__":
    set_num_threads(1)
    policy = PPO.load("runs/experimental-gait/policy.zip", device="cpu")
    totals = [[], [], []]
    for seed in (40003, 70000, 70003, 70008, 70009, 70012):
        status, *series = compare_route(policy, seed)
        print(seed, status, flush=True)
        for total, values in zip(totals, series):
            total.extend(values)
    for label, values in zip(("Foot position error (m)", "Raw grounded drift (m/s)",
                              "Visible planted drift (m/s)"), totals):
        print(label, dict(zip(("median", "p95", "max"), np.percentile(values, [50, 95, 100]))))

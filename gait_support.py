"""Experimental walking inputs and reduced vertical assistance."""

import math
import numpy as np

GAIT_FREQUENCY = 1.08
PHASE_FEATURES = 5


def limit_vertical_assistance(sim, torso, phase_gait=False):
    # Keep steering and balance torques. Lower the height at which the harness
    # helps and cap lift below the robot's own weight.
    height = 1.28 if phase_gait else 1.34
    height_error = height - sim.data.qpos[2]
    force = 4200 * height_error - 330 * sim.data.qvel[2]
    sim.data.xfrc_applied[torso, 2] = np.clip(force, 0, 400)


def walking_observation(sim, torso, delta, speed, phase_gait):
    observation = sim._get_obs()
    if not phase_gait:
        return observation
    angle = 2 * math.pi * GAIT_FREQUENCY * sim.data.time
    rotation = sim.data.xmat[torso].reshape(3, 3)
    distance = np.linalg.norm(delta)
    direction = delta / max(distance, 1e-8)
    command = direction @ rotation[:2, :2]
    return np.concatenate((observation,
                           [math.sin(angle), math.cos(angle), *command,
                            min(speed, 2 * distance)]))

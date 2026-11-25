"""Draw a planted walking pose using the simulated path and ground plane.

The pose lives in separate MuJoCo data. PPO, collisions and route decisions
continue to use the original simulated state.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np


ARM_POSES = np.array([
    [-0.824, 0.795, -1.570],
    [-0.652, 0.518, -1.570],
    [-0.534, 0.235, -1.545],
])


def smoothstep(x):
    return x * x * (3 - 2 * x)


@dataclass
class Swing:
    side: int
    start: np.ndarray
    end: np.ndarray
    duration: float
    elapsed: float = 0.0


class AvatarGait:
    def __init__(self, model):
        self.model = model
        self.pose = mujoco.MjData(model)
        self.qpos = self.pose.qpos
        self.feet = [model.geom(f"{side}_foot").id for side in ("left", "right")]
        self.foot_height = float(model.geom_size[self.feet[0], 0])
        self.joints = {}
        for name in ("abdomen_x", "abdomen_y", "abdomen_z"):
            self.joints[name] = model.joint(name).qposadr.item()
        self.legs = []
        self.arms = []
        for side in ("left", "right"):
            joints = [model.joint(f"{side}_{suffix}") for suffix in ("hip_x", "hip_y", "knee", "hip_z")]
            limits = np.array([joint.range.copy() for joint in joints])
            # Allow a little more extension and inward motion in the draw pose
            # so the narrow simulator rig can keep a foot down through a turn.
            limits[0, 1] = math.radians(17)
            limits[1, 1] = math.radians(34)
            limits[3] = np.radians([-26, 26])
            self.legs.append((
                [joint.qposadr.item() for joint in joints],
                [joint.dofadr.item() for joint in joints],
                limits,
            ))
            for suffix in ("shoulder1", "shoulder2", "elbow"):
                name = f"{side}_{suffix}"
                self.joints[name] = model.joint(name).qposadr.item()
            self.arms.append(side)
        self.jacobian = np.zeros((3, model.nv))
        self.previous_position = None

    def axes(self):
        return np.array([math.cos(self.yaw), math.sin(self.yaw)])

    def stance(self, position, heading, side):
        lateral = np.array([-heading[1], heading[0]])
        return position + (0.115 if side == 0 else -0.115) * lateral

    def reset(self, position, yaw=0.0):
        self.yaw = yaw
        self.previous_position = np.array(position)
        self.velocity = np.zeros(2)
        self.steps = 0
        self.next_side = 0
        self.swing = None
        self.arm_swing = 0.0
        self.feet_xy = [self.stance(position, self.axes(), side) for side in range(2)]
        self.qpos[:] = self.model.qpos0
        for qpos, _, _ in self.legs:
            self.qpos[qpos] = (0.0, -0.26, -0.52, 0.0)
        self.pose_inputs = (position, 0.0, 0.0)
        self.refresh_pose()

    def step(self, position, dt, target=None, *, update_pose=True):
        if self.previous_position is None or np.linalg.norm(position - self.previous_position) > 0.42:
            direction = np.asarray(target) - position if target is not None else None
            yaw = math.atan2(direction[1], direction[0]) if direction is not None else 0.0
            self.reset(position, yaw)
            return
        if dt <= 0:
            return

        displacement = position - self.previous_position
        self.previous_position = np.array(position)
        moving = target is not None
        blend = 1 - math.exp(-dt / 0.12)
        self.velocity += blend * ((displacement / dt if moving else 0) - self.velocity)
        speed = float(np.linalg.norm(self.velocity))
        if moving:
            direction = self.velocity if speed > 0.18 else np.asarray(target) - position
            if np.linalg.norm(direction) > 1e-8:
                yaw = math.atan2(direction[1], direction[0])
                error = math.remainder(yaw - self.yaw, 2 * math.pi)
                self.yaw += float(np.clip(error * 8 * dt, -4.2 * dt, 4.2 * dt))

        heading = self.axes()
        side = self.next_side
        if self.swing is None:
            neutral = self.stance(position, heading, side)
            trailing = float(np.dot(neutral - self.feet_xy[side], heading))
            if moving and speed > 0.10 and (
                    trailing > min(0.14, 0.034 + 0.095 * speed)
                    or np.linalg.norm(neutral - self.feet_xy[side]) > 0.23):
                duration = float(np.clip(0.39 - 0.13 * speed, 0.20, 0.36))
                # Aim ahead of the pelvis at touchdown, accounting for the
                # distance it will cover while the other foot stays planted.
                direction = self.velocity / max(speed, 1e-8)
                reach = min(0.53, speed * (duration + 0.125))
                landing = self.stance(position + reach * direction, heading, side)
                self.swing = Swing(side, self.feet_xy[side].copy(), landing, duration)
            elif not moving:
                side = max(range(2), key=lambda i: np.linalg.norm(
                    self.stance(position, heading, i) - self.feet_xy[i]))
                neutral = self.stance(position, heading, side)
                if np.linalg.norm(neutral - self.feet_xy[side]) > 0.08:
                    self.swing = Swing(side, self.feet_xy[side].copy(), neutral, 0.30)

        phase = lift = 0.0
        if self.swing is not None:
            swing = self.swing
            swing.elapsed = min(swing.elapsed + dt, swing.duration)
            phase = swing.elapsed / swing.duration
            self.feet_xy[swing.side] = swing.start + smoothstep(phase) * (swing.end - swing.start)
            step_length = np.linalg.norm(swing.end - swing.start)
            lift = min(0.125, 0.043 + 0.11 * step_length) * math.sin(math.pi * phase) ** 2
            if phase == 1.0:
                self.next_side = 1 - swing.side
                self.swing = None
                self.steps += 1

        lateral = np.array([-heading[1], heading[0]])
        if self.swing is not None:
            weight_shift = (1 if self.swing.side == 1 else -1) * 0.033 * math.sin(math.pi * phase)
            forward = np.dot(self.feet_xy[0] - self.feet_xy[1], heading)
        else:
            weight_shift = 0.0
            forward = 0.0
        self.arm_swing += (1 - math.exp(-dt / 0.13)) * (np.clip(forward, -0.60, 0.60) - self.arm_swing)
        self.pose_inputs = (position + lateral * weight_shift, lift, speed)
        if update_pose:
            self.refresh_pose()

    def refresh_pose(self):
        self.draw_pose(*self.pose_inputs)

    def draw_pose(self, position, lift, speed):
        heading = self.axes()
        reach = max(np.linalg.norm(foot - position) for foot in self.feet_xy)
        self.qpos[:3] = (*position, 1.285 - min(0.20, 0.75 * reach ** 2))
        self.qpos[3:7] = (math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2))
        self.qpos[self.joints["abdomen_y"]] = -min(0.055, speed * 0.038)
        self.qpos[self.joints["abdomen_x"]] = 0.042 * self.arm_swing
        self.qpos[self.joints["abdomen_z"]] = 0.17 * self.arm_swing

        for side, sign in zip(self.arms, (1, -1)):
            # Keep the rig's unusually angled shoulders down by interpolating
            # between calibrated back, neutral and forward arm poses.
            arm = -self.arm_swing * sign / 0.60
            neutral = ARM_POSES[1]
            extreme = ARM_POSES[0 if arm < 0 else 2]
            shoulder1, shoulder2, elbow = neutral + abs(arm) * (extreme - neutral)
            self.qpos[self.joints[f"{side}_shoulder1"]] = sign * shoulder1
            self.qpos[self.joints[f"{side}_shoulder2"]] = sign * shoulder2
            self.qpos[self.joints[f"{side}_elbow"]] = elbow

        targets = [np.array([*foot, self.foot_height]) for foot in self.feet_xy]
        if self.swing is not None:
            targets[self.swing.side][2] += lift

        # Warm-start a damped inverse kinematics solve at the preceding pose.
        # A stationary target remains at the exact same table coordinate.
        for _ in range(7):
            mujoco.mj_kinematics(self.model, self.pose)
            mujoco.mj_comPos(self.model, self.pose)
            if max(np.linalg.norm(target - self.pose.geom_xpos[foot])
                   for foot, target in zip(self.feet, targets)) < 0.001:
                break
            for foot, target, (qpos, dofs, limits) in zip(self.feet, targets, self.legs):
                error = target - self.pose.geom_xpos[foot]
                mujoco.mj_jacGeom(self.model, self.pose, self.jacobian, None, foot)
                jacobian = self.jacobian[:, dofs]
                # Favor the three primary leg axes over twisting the thigh.
                weighted = jacobian * [1.0, 1.0, 1.0, 0.22]
                change = weighted.T @ np.linalg.solve(
                    weighted @ jacobian.T + np.eye(3) * 0.0003, error)
                self.qpos[qpos] = np.clip(
                    self.qpos[qpos] + np.clip(change, -0.28, 0.28), limits[:, 0], limits[:, 1])
        mujoco.mj_fwdPosition(self.model, self.pose)


def project_ground(renderer, model, point):
    """Project the pelvis's ground point to the same pixels as the 3D rig."""
    first, second = renderer.scene.camera
    forward = np.array(first.forward)
    up = np.array(first.up)
    right = np.cross(forward, up)
    delta = np.array([*point, 0.0]) - (first.pos + second.pos) / 2
    depth = max(0.01, float(np.dot(delta, forward)))
    scale = renderer.height / (2 * depth * math.tan(math.radians(model.vis.global_.fovy) / 2))
    return (renderer.width / 2 + float(np.dot(delta, right)) * scale,
            renderer.height / 2 - float(np.dot(delta, up)) * scale)

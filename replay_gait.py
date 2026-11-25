"""Render both checkpoints on the same obstacle course at simulation speed."""

import argparse
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from torch import set_num_threads

from avatar_gait import AvatarGait
from train_gait import BASELINE, ObstacleWalk, TEST_SEED


WIDTH, HEIGHT = 600, 430
PANEL_TOP = 92
FRAME_SIZE = (1240, 620)
BACKGROUND = "#101822"
COLORS = ((0.87, 0.72, 0.41, 1.0), (0.32, 0.88, 0.76, 1.0))


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


class Replay:
    def __init__(self, checkpoint, seed, color, lower_lift=False, planted=False):
        self.model = PPO.load(checkpoint, device="cpu")
        self.phase_gait = getattr(self.model, "gait_phase_inputs", False)
        self.lower_lift = lower_lift or getattr(self.model, "gait_lower_lift", False) or self.phase_gait
        pace_scale = getattr(self.model, "gait_pace_scale", 1.0)
        self.env = ObstacleWalk(seed, lower_lift=self.lower_lift, phase_gait=self.phase_gait,
                                pace_scale=pace_scale)
        self.observation, _ = self.env.reset(seed=seed)
        self.renderer = mujoco.Renderer(self.env.sim.model, height=HEIGHT, width=WIDTH)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 4.6
        self.camera.azimuth = 115 if self.env.route[0, 0] < self.env.route[-1, 0] else 295
        self.camera.elevation = -26
        self.metrics = None
        self.color = np.asarray(color, dtype=np.float32)
        model = self.env.sim.model
        self.gait = AvatarGait(model) if planted else None
        if self.gait is not None:
            self.gait.step(self.env.sim.data.qpos[:2], 0.0, self.env.route[1])
        model.mat_reflectance[:] *= 0.12
        for index in range(model.ngeom):
            geom = model.geom(index)
            if index == self.env.floor:
                geom.rgba[:] = (0.31, 0.36, 0.42, 1)
                continue
            geom.matid = -1
            if index in self.env.obstacles:
                geom.rgba[:] = (0.23, 0.37, 0.53, 1)
            elif "foot" in geom.name or "hand" in geom.name:
                geom.rgba[:] = (0.20, 0.25, 0.31, 1)
            else:
                geom.rgba[:] = color

    def step(self):
        if self.metrics is None:
            action, _ = self.model.predict(self.observation, deterministic=True)
            self.observation, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                self.metrics = info["walk"]
        if self.gait is not None:
            sim = self.env.sim
            target = None if self.metrics else self.env.route[self.env.target]
            self.gait.step(sim.data.qpos[:2], sim.dt, target)

    def frame(self):
        sim = self.env.sim
        self.camera.lookat[:] = (*sim.data.qpos[:2], 0.70)
        pose = self.gait.pose if self.gait is not None else sim.data
        self.renderer.update_scene(pose, camera=self.camera)
        scene = self.renderer.scene
        for index, (x, y) in enumerate(self.env.route[1:], 1):
            geom = scene.geoms[scene.ngeom]
            final = index == len(self.env.route) - 1
            radius = 0.14 if final else 0.065
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                np.array([radius, radius, 0.012]),
                                np.array([x, y, 0.018]), np.eye(3).ravel(), self.color)
            scene.ngeom += 1
        return Image.fromarray(self.renderer.render())

    def close(self):
        self.renderer.close()
        self.env.close()


def render(args):
    set_num_threads(1)
    first = args.checkpoint if args.planted else args.baseline
    replays = [Replay(first, args.seed, COLORS[0]),
               Replay(args.checkpoint, args.seed, COLORS[1], args.lower_lift, args.planted)]
    heading, label, detail = font(34, True), font(23, True), font(17)
    size = f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", size,
               "-framerate", "100/3", "-i", "-", "-c:v", "libx264", "-crf", "19",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            hold = 0
            for _ in range(970):
                page = Image.new("RGB", FRAME_SIZE, BACKGROUND)
                draw = ImageDraw.Draw(page)
                draw.text((14, 14), "Humanoid obstacle course", font=heading, fill="#f1f6ff")
                subtitle = ("Same route   /   Fine-tune uses tuned assistance and pace   /   Real simulation speed"
                            if replays[1].lower_lift else
                            "Same route and steering settings   /   Real simulation speed")
                if args.planted:
                    subtitle = "Same trained policy and route   /   Real simulation speed"
                draw.text((15, 56), subtitle,
                          font=detail, fill="#acbdcd")

                for index, replay in enumerate(replays):
                    x = 15 + index * 610
                    page.paste(replay.frame(), (x, PANEL_TOP))
                    elapsed = replay.env.steps * replay.env.sim.dt
                    titles = ("Raw PPO pose", "Planted walking pose") if args.planted else (
                        "Original PPO", "Obstacle fine-tune")
                    title = titles[index]
                    color = ("#dfb769", "#52e0c2")[index]
                    draw.text((x, 535), title, font=label, fill=color)
                    status = replay.metrics["status"].capitalize() if replay.metrics else "Walking"
                    draw.text((x, 570), f"{status}   /   {elapsed:04.1f} s",
                              font=detail, fill="#c4d2e0")

                encoder.stdin.write(page.tobytes())
                if all(replay.metrics for replay in replays):
                    hold += 1
                    if hold == 40:
                        break
                for replay in replays:
                    replay.step()
                    replay.step()
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg could not write the comparison video.")
    finally:
        for replay in replays:
            replay.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--seed", type=int, default=TEST_SEED)
    parser.add_argument("--lower-lift", action="store_true")
    parser.add_argument("--planted", action="store_true", help="compare two renders of the same checkpoint")
    render(parser.parse_args())

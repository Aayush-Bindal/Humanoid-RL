"""Detect a white tabletop and anchor the simulated walker to it in live video."""

from collections import deque
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile
from threading import Thread
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from stable_baselines3 import PPO
from torch import set_num_threads

from table_vision import TABLE_SIZE, TablePlane, find_table
from tabletop_demo import GOAL, LENGTH, SPAN, START
from tabletop_demo import make_course, plan_route, steer


VIDEO_SIZE = TABLE_SIZE


class Webcam:
    def __init__(self, camera):
        self.frames = deque(maxlen=1)
        self.errors = TemporaryFile()
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "dshow", "-i", f"video={camera}", "-an",
            "-vf", "crop=w='min(iw,ih*1.6)':h='min(ih,iw/1.6)',scale=640:400,fps=24",
            "-pix_fmt", "rgb24", "-threads", "1", "-f", "rawvideo", "pipe:1",
        ]
        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=self.errors,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            self.errors.close()
            raise
        self.reader = Thread(target=self.read_frames, daemon=True)
        self.reader.start()

    def read_frames(self):
        size = VIDEO_SIZE[0] * VIDEO_SIZE[1] * 3
        with self.process.stdout as stream:
            while raw := stream.read(size):
                if len(raw) != size:
                    break
                self.frames.append(Image.frombytes("RGB", VIDEO_SIZE, raw))

    def error(self):
        self.errors.seek(0)
        return self.errors.read().decode(errors="replace").strip()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.reader.join()
        self.process.wait()
        self.errors.close()


class OverlayWalker:
    def __init__(self, checkpoint):
        set_num_threads(1)
        self.policy = PPO.load(checkpoint, device="cpu")
        with TemporaryDirectory() as directory:
            self.env = make_course([], [], directory, render=False, overlay=True)
        self.sim = self.env.unwrapped
        self.torso = self.sim.model.body("torso").id
        try:
            self.renderer = mujoco.Renderer(self.sim.model, height=320, width=320)
        except Exception:
            self.env.close()
            raise
        self.scene_options = mujoco.MjvOption()
        self.sim.model.geom("floor").group = 5
        self.scene_options.geomgroup[5] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = 90
        self.camera.elevation = -24
        self.camera.distance = 3.6
        vertical_fov = np.deg2rad(self.sim.model.vis.global_.fovy)
        self.sprite_pixels_per_unit = self.renderer.height / (
            2 * self.camera.distance * np.tan(vertical_fov / 2)
        )
        for index in range(1, self.sim.model.ngeom):
            geom = self.sim.model.geom(index)
            geom.matid = -1
            if any(part in geom.name for part in ("foot", "hand", "waist", "butt")):
                geom.rgba[:] = (0.17, 0.22, 0.26, 1)
            elif geom.name == "head" or "shin" in geom.name:
                geom.rgba[:] = (0.18, 0.74, 0.70, 1)
            else:
                geom.rgba[:] = (0.78, 0.83, 0.85, 1)
        self.reset()

    def reset(self):
        self.env.reset(seed=0)
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = START
        self.sim.set_state(qpos, np.zeros(self.sim.model.nv))
        self.route = deque()
        self.boxes = None
        self.goal_time = None

    def replan(self, boxes):
        self.boxes = boxes
        origin = tuple(self.sim.data.qpos[:2])
        try:
            self.route = deque(plan_route(boxes, origin=origin)[1:])
        except ValueError:
            self.route.clear()

    def advance(self, steps):
        if self.goal_time is not None:
            return "Reached the goal"
        if not self.route:
            return "Path blocked · move an object to make room"
        for _ in range(steps):
            position = self.sim.data.qpos[:2]
            if np.linalg.norm(np.asarray(self.route[0]) - position) < 0.18:
                self.route.popleft()
                if not self.route:
                    self.goal_time = time.monotonic()
                    return "Reached the goal"
            delta = np.asarray(self.route[0]) - position
            steer(self.sim, self.torso, delta)
            action, _ = self.policy.predict(self.sim._get_obs(), deterministic=True)
            _, _, fell, _, _ = self.sim.step(action)
            if fell:
                self.reset()
                return "Restarting"
        return "Walking · route updates as objects move"

    def render_avatar(self):
        """Render the pose as a transparent image with its natural proportions."""
        x, y = self.sim.data.qpos[:2]
        self.camera.lookat[:] = (x, y, 0.78)
        self.renderer.update_scene(self.sim.data, camera=self.camera,
                                   scene_option=self.scene_options)
        rgb = self.renderer.render()
        self.renderer.enable_segmentation_rendering()
        try:
            objects = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        alpha = (objects[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM).astype(np.uint8) * 255
        avatar = Image.fromarray(np.dstack((rgb, alpha)))
        bounds = avatar.getbbox()
        return avatar.crop(bounds) if bounds else None

    def composite(self, frame, table, guides=False):
        """Project ground marks, then place the upright avatar at its ground point."""
        width, height = VIDEO_SIZE
        x, y = self.sim.data.qpos[:2]
        u, v = x / LENGTH * width, y / SPAN * height
        layer = Image.new("RGBA", VIDEO_SIZE)
        draw = ImageDraw.Draw(layer)
        draw.ellipse((u - 18, v - 11, u + 18, v + 11), fill=(7, 16, 20, 105))
        layer = layer.filter(ImageFilter.GaussianBlur(5))
        draw = ImageDraw.Draw(layer)
        goal_x = GOAL[0] / LENGTH * width
        goal_y = GOAL[1] / SPAN * height
        draw.ellipse((goal_x - 13, goal_y - 11, goal_x + 13, goal_y + 11),
                     fill=(28, 168, 142, 90), outline="#77f5cb", width=3)
        if guides:
            for x1, y1, x2, y2 in self.boxes or ():
                draw.rectangle((x1 / LENGTH * width, y1 / SPAN * height,
                                x2 / LENGTH * width, y2 / SPAN * height),
                               fill=(243, 161, 76, 24), outline=(243, 161, 76, 215), width=3)
        result = frame.convert("RGBA")
        result.alpha_composite(table.project(layer, frame.size))
        if guides:
            boundary = [tuple(corner) for corner in table.corners]
            ImageDraw.Draw(result).line([*boundary, boundary[0]], fill="#71dfbe", width=2)
        avatar = self.render_avatar()
        if avatar is not None:
            ground = table.image_point(u, v)
            local_scale = np.linalg.norm(table.image_point(u + width / LENGTH, v) - ground)
            scale = np.clip(local_scale, 22, 85) / self.sprite_pixels_per_unit
            avatar_height = max(1, round(avatar.height * scale))
            avatar_width = max(1, round(avatar.width * scale))
            avatar = avatar.resize((avatar_width, avatar_height), Image.Resampling.LANCZOS)
            result.alpha_composite(avatar, (round(ground[0] - avatar_width / 2),
                                           round(ground[1] - avatar_height + 2)))
        return result

    def close(self):
        self.renderer.close()
        self.env.close()


def sample_feed(seconds):
    frame = Image.new("RGB", VIDEO_SIZE, "#4e5056")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((22, 58, 181, 139), fill="#d1d4cd")
    draw.rectangle((455, 79, 617, 184), fill="#ababae")
    draw.polygon(((238, 164), (507, 169), (639, 252), (639, 399), (38, 399)),
                 fill="#e3e4de")
    y = round(241 + 35 * np.sin(seconds * 0.8))
    draw.rounded_rectangle((338, y - 21, 380, y + 21),
                           radius=10, fill="#514446", outline="#363032", width=3)
    return frame


def show_live(args):
    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Tabletop Walker")
    root.configure(bg="#10171c")
    root.resizable(False, False)
    header = tk.Frame(root, bg="#10171c", padx=15, pady=10)
    header.pack(fill="x")
    tk.Label(header, text="Tabletop Walker", font=("Segoe UI", 14, "bold"),
             bg="#10171c", fg="#eef5f5").pack(side="left")
    guides = tk.BooleanVar(value=False)
    tk.Checkbutton(header, text="Show guides", variable=guides, font=("Segoe UI", 10),
                   bg="#10171c", fg="#c0d0d3", activebackground="#10171c",
                   activeforeground="#eef5f5", selectcolor="#263940",
                   highlightthickness=0).pack(side="right")
    root.bind("<d>", lambda _: guides.set(not guides.get()))
    picture = tk.Label(root, bg="#10171c", bd=0, highlightthickness=0)
    picture.pack()
    status = tk.StringVar(value="Opening webcam…")
    tk.Label(root, textvariable=status, bg="#10171c", fg="#b4c9cc",
             font=("Segoe UI", 10), anchor="w", justify="left",
             wraplength=610, padx=15, pady=10).pack(fill="x")
    root.update_idletasks()
    webcam = None
    if not args.live_sample:
        try:
            webcam = Webcam(args.camera)
        except OSError as error:
            status.set(f"Could not start FFmpeg: {error}. Use --live-sample to rehearse.")
            root.mainloop()
            return
    walker = None
    try:
        walker = OverlayWalker(args.checkpoint)
        started = time.monotonic()
        next_scan = 0
        table = None
        candidate = None

        def tick():
            nonlocal next_scan, table, candidate
            now = time.monotonic()
            if webcam is None:
                frame = sample_feed(now - started)
            elif webcam.process.poll() is not None:
                status.set(f"Could not stream {args.camera}. Close other camera apps and check Windows camera access.\n{webcam.error()}")
                return
            elif not webcam.frames:
                if now - started > 12:
                    status.set("No webcam frames received. Close other camera apps and restart.")
                    return
                root.after(40, tick)
                return
            else:
                frame = webcam.frames[-1]
            if now >= next_scan:
                corners = find_table(frame)
                if corners is None:
                    table = candidate = None
                elif table is not None and np.max(np.abs(corners - table.corners)) < 32:
                    table = TablePlane(0.70 * table.corners + 0.30 * corners)
                elif candidate is not None and np.max(np.abs(corners - candidate)) < 24:
                    table = TablePlane((corners + candidate) / 2)
                    walker.reset()
                    candidate = None
                else:
                    table, candidate = None, corners
                if table is not None:
                    if walker.goal_time is not None and now - walker.goal_time > 2:
                        walker.reset()
                    boxes = table.obstacles(frame, args.threshold)
                    boxes = [tuple(round(value, 1) for value in box) for box in boxes]
                    if boxes != walker.boxes and walker.goal_time is None:
                        walker.replan(boxes)
                next_scan = now + 0.18
            if table is None:
                status.set("Looking for the white tabletop. Keep its edges visible.")
                output = frame
            else:
                movement = walker.advance(2)
                status.set(f"Table locked · {len(walker.boxes or [])} obstacles · {movement}")
                output = walker.composite(frame, table, guides.get())
            picture.image = ImageTk.PhotoImage(output)
            picture.configure(image=picture.image)
            elapsed_ms = round((time.monotonic() - now) * 1000)
            root.after(max(1, 33 - elapsed_ms), tick)

        root.after(0, tick)
        root.mainloop()
    finally:
        if webcam is not None:
            webcam.close()
        if walker is not None:
            walker.close()

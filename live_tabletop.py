"""Detect a white tabletop and anchor the simulated walker to it in live video."""

from collections import deque
import math
import sqlite3
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile
from threading import Thread
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from stable_baselines3 import PPO
from torch import set_num_threads

from navigation_learning import NavigationLearning, SETTINGS
from table_vision import TABLE_SIZE, TableTracker
from tabletop_demo import LENGTH, SPAN, START
from tabletop_demo import make_course, plan_route, position_clear, segment_clear, steer


VIDEO_SIZE = TABLE_SIZE
CAMERA_FILTER = "scale=960:720:force_original_aspect_ratio=decrease:force_divisible_by=2,fps=24"
NEAR_EDGE_TARGET = 0.65
TIGHT_ROUTE_CLEARANCE = 0.55
MIN_ROUTE_CLEARANCE = 0.40
STOP_CLEARANCE = 0.30
TARGET_ADJUSTMENT = 0.60


def read_camera_frame(stream):
    """Read one RGB PAM frame, including its uncropped dimensions."""
    if stream.readline(80) != b"P7\n":
        return None
    header = {}
    for _ in range(12):
        line = stream.readline(80).strip()
        if line == b"ENDHDR":
            break
        key, separator, value = line.partition(b" ")
        if separator and value.isdigit():
            header[key] = int(value)
    else:
        return None
    width, height = header.get(b"WIDTH", 0), header.get(b"HEIGHT", 0)
    if not (0 < width <= 960 and 0 < height <= 720
            and header.get(b"DEPTH") == 3 and header.get(b"MAXVAL") == 255):
        return None
    size = width * height * 3
    data = stream.read(size)
    return Image.frombytes("RGB", (width, height), data) if len(data) == size else None


class Webcam:
    def __init__(self, camera):
        self.frames = deque(maxlen=1)
        self.errors = TemporaryFile()
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "dshow", "-i", f"video={camera}", "-an",
            "-vf", CAMERA_FILTER, "-pix_fmt", "rgb24",
            "-c:v", "pam", "-threads", "1", "-f", "image2pipe", "pipe:1",
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
        with self.process.stdout as stream:
            while (frame := read_camera_frame(stream)) is not None:
                self.frames.append(frame)

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
        self.route = deque()
        self.boxes = []
        self.destination = None
        self.spawned = False
        self.settings = SETTINGS[0]
        self.elapsed = 0
        self.distance = 0
        self.outcome = None
        self.status = "Click a clear spot on the table"

    def spawn(self):
        if self.spawned:
            return True
        candidates = [START, (1.4, 1.1), (1.4, 3.9), (4.0, 1.1), (4.0, 3.9), (7.0, 2.5)]
        position = next((point for point in candidates if position_clear(point, self.boxes)), None)
        if position is None:
            position = next(self.nearby_clear_points(START, TIGHT_ROUTE_CLEARANCE, LENGTH), None)
        if position is None:
            self.status = "Clear a spot so the robot can spawn"
            return False
        self.place_robot(position)
        return True

    def place_robot(self, position):
        self.env.reset(seed=0)
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = position
        self.sim.set_state(qpos, np.zeros(self.sim.model.nv))
        self.spawned = True
        self.status = "Click a clear spot on the table"

    def nearby_clear_points(self, point, clearance, radius):
        """Yield nearby positions in order of distance on the walking plane."""
        points = []
        for x in np.arange(NEAR_EDGE_TARGET, LENGTH - 0.5, 0.2):
            for y in np.arange(NEAR_EDGE_TARGET, SPAN - 0.5, 0.2):
                candidate = (float(x), float(y))
                distance = math.dist(point, candidate)
                if distance <= radius and position_clear(candidate, self.boxes, clearance):
                    points.append((distance, candidate))
        for _, candidate in sorted(points):
            yield candidate

    def choose_destination(self, destination, settings):
        if not self.spawned:
            raise ValueError("Clear a spot so the robot can spawn.")
        origin = tuple(self.sim.data.qpos[:2])
        # Keep the waiting point inside the robot's walking boundary even
        # when the user taps directly next to the visible edge.
        requested = destination
        destination = (min(max(destination[0], NEAR_EDGE_TARGET), LENGTH - NEAR_EDGE_TARGET),
                       min(max(destination[1], NEAR_EDGE_TARGET), SPAN - NEAR_EDGE_TARGET))
        if any(x1 <= requested[0] <= x2 and y1 <= requested[1] <= y2
               for x1, y1, x2, y2 in self.boxes):
            raise ValueError("An object covers that spot. Tap the clear table beside it.")
        if not position_clear(destination, self.boxes, MIN_ROUTE_CLEARANCE):
            destination = next(self.nearby_clear_points(destination, MIN_ROUTE_CLEARANCE,
                                                        TARGET_ADJUSTMENT), None)
            if destination is None:
                raise ValueError("There is no room to stand there. Tap farther from the object.")

        start = origin
        if not position_clear(origin, self.boxes, MIN_ROUTE_CLEARANCE):
            # Live detections can move an obstacle onto the current pose.
            # Recover only after finding a new starting point with a route.
            starts = self.nearby_clear_points(origin, TIGHT_ROUTE_CLEARANCE, 1.8)
            for start in starts:
                try:
                    route, clearance = self.route_to(start, destination, settings.clearance)
                except ValueError:
                    continue
                break
            else:
                raise ValueError("The robot is boxed in. Move the object beside it.")
            self.place_robot(start)
        else:
            route, clearance = self.route_to(start, destination, settings.clearance)
        self.route = deque(route[1:])
        self.destination = destination
        self.settings = settings
        self.route_clearance = clearance
        self.elapsed = 0
        self.distance = math.dist(start, destination)
        self.status = "Walking to your marker"
        if start != origin:
            return "Robot moved to a clear starting point and is walking to your marker."
        if destination != requested:
            return "Marker moved to a nearby clear spot."
        return None

    def route_to(self, origin, destination, clearance):
        # A learned preference may need more room than the robot actually
        # requires. Try tighter routes that still exceed the stop clearance.
        for margin in (clearance, TIGHT_ROUTE_CLEARANCE):
            try:
                return plan_route(self.boxes, origin, margin, destination), margin
            except ValueError:
                continue
        route = plan_route(self.boxes, origin, MIN_ROUTE_CLEARANCE, destination)
        return route, MIN_ROUTE_CLEARANCE

    def remap(self, previous_table, table):
        """Keep the robot and marker over the same spots as the plane grows."""
        if not self.spawned:
            return

        def translate(point):
            x, y = point
            pixel = previous_table.image_point(x / LENGTH * VIDEO_SIZE[0], y / SPAN * VIDEO_SIZE[1])
            return table.world_point(*pixel)

        qpos = self.sim.data.qpos.copy()
        try:
            qpos[:2] = translate(qpos[:2])
            destination = translate(self.destination) if self.destination is not None else None
        except ValueError:
            return
        qvel = self.sim.data.qvel.copy()
        qvel[:2] = 0
        self.sim.set_state(qpos, qvel)
        self.destination = destination
        self.route.clear()

    def replan(self, boxes, *, force=False):
        if not force and self.boxes == boxes and self.spawned:
            return
        self.boxes = boxes
        if not self.spawn() or self.destination is None or self.status == "Arrived":
            return
        origin = tuple(self.sim.data.qpos[:2])
        # Minor box jitter should not replace a route that still has clearance.
        if self.route:
            points = [origin, *self.route]
            if all(segment_clear(a, b, boxes, max(MIN_ROUTE_CLEARANCE, self.route_clearance - 0.12))
                   for a, b in zip(points, points[1:])):
                return
        try:
            route, self.route_clearance = self.route_to(origin, self.destination, self.settings.clearance)
            self.route = deque(route[1:])
            self.status = "Walking to your marker"
        except ValueError:
            self.route.clear()
            self.status = "Path blocked. Move the object or choose a new spot"

    def finish(self, reached, status):
        self.route.clear()
        self.status = status
        if self.distance >= 0.3:
            self.outcome = (self.settings, reached, self.elapsed, self.distance)
        if not reached:
            self.destination = None
        return self.status

    def advance(self, steps):
        if not self.route:
            return self.status
        for _ in range(steps):
            position = self.sim.data.qpos[:2]
            if np.linalg.norm(np.asarray(self.route[0]) - position) < 0.18:
                self.route.popleft()
                if not self.route:
                    return self.finish(True, "Arrived")
            delta = np.asarray(self.route[0]) - position
            steer(self.sim, self.torso, delta, max_speed=self.settings.speed)
            action, _ = self.policy.predict(self.sim._get_obs(), deterministic=True)
            previous_pose = self.sim.data.qpos.copy()
            _, _, fell, _, _ = self.sim.step(action)
            self.elapsed += self.sim.dt
            if fell:
                self.spawned = False
                self.spawn()
                return self.finish(False, "Lost balance. Choose another spot")
            if not position_clear(self.sim.data.qpos[:2], self.boxes, clearance=STOP_CLEARANCE):
                self.sim.set_state(previous_pose, np.zeros(self.sim.model.nv))
                return self.finish(False, "Stopped near an obstacle. Choose another spot")
            if self.elapsed >= 27:
                return self.finish(False, "Trip timed out. Choose another spot")
        return self.status

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
        layer = Image.new("RGBA", VIDEO_SIZE)
        draw = ImageDraw.Draw(layer)
        if self.spawned:
            x, y = self.sim.data.qpos[:2]
            u, v = x / LENGTH * width, y / SPAN * height
            draw.ellipse((u - 18, v - 11, u + 18, v + 11), fill=(7, 16, 20, 105))
        layer = layer.filter(ImageFilter.GaussianBlur(5))
        draw = ImageDraw.Draw(layer)
        if self.destination is not None:
            goal_x = self.destination[0] / LENGTH * width
            goal_y = self.destination[1] / SPAN * height
            draw.ellipse((goal_x - 16, goal_y - 16, goal_x + 16, goal_y + 16),
                         fill=(28, 168, 142, 70), outline="#77f5cb", width=3)
            draw.ellipse((goal_x - 4, goal_y - 4, goal_x + 4, goal_y + 4), fill="#c5ffeb")
        if guides:
            if self.route and self.spawned:
                draw.line([(u, v), *((x / LENGTH * width, y / SPAN * height) for x, y in self.route)],
                          fill="#77f5cb", width=3)
            for x1, y1, x2, y2 in self.boxes or ():
                draw.rectangle((x1 / LENGTH * width, y1 / SPAN * height,
                                x2 / LENGTH * width, y2 / SPAN * height),
                               fill=(243, 161, 76, 24), outline=(243, 161, 76, 215), width=3)
        result = frame.convert("RGBA")
        result.alpha_composite(table.project(layer, frame.size))
        if guides:
            boundary = [tuple(corner) for corner in table.corners]
            ImageDraw.Draw(result).line([*boundary, boundary[0]], fill="#71dfbe", width=2)
        avatar = self.render_avatar() if self.spawned else None
        if avatar is not None:
            ground = table.image_point(u, v)
            local_scale = np.linalg.norm(table.image_point(u + width / LENGTH, v) - ground)
            frame_scale = frame.width / width
            scale = np.clip(local_scale, 22 * frame_scale, 85 * frame_scale) / self.sprite_pixels_per_unit
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
    background, panel, ink, accent = "#10171c", "#172228", "#eef5f5", "#8af0cc"
    root.configure(bg=background)
    root.minsize(700, 500)
    width = min(1180, round(root.winfo_screenwidth() * 0.88))
    height = min(820, round(root.winfo_screenheight() * 0.86))
    root.geometry(f"{width}x{height}")
    header = tk.Frame(root, bg=background, padx=22, pady=15)
    header.pack(fill="x")
    tk.Label(header, text="Tabletop Walker", font=("Segoe UI", 19, "bold"),
             bg=background, fg=ink).pack(side="left")
    guides = tk.BooleanVar(value=False)
    fullscreen = tk.BooleanVar(value=False)

    def toggle_fullscreen(event=None):
        fullscreen.set(not fullscreen.get())
        root.attributes("-fullscreen", fullscreen.get())

    tk.Button(header, text="Fullscreen  F11", command=toggle_fullscreen,
              font=("Segoe UI", 10), bg=panel, fg=ink, relief="flat", padx=10,
              activebackground="#263940", activeforeground=ink).pack(side="right")
    tk.Checkbutton(header, text="Guides  D", variable=guides, font=("Segoe UI", 10),
                   bg=panel, fg=ink, activebackground="#263940",
                   activeforeground=ink, selectcolor=panel, highlightthickness=0,
                   padx=9).pack(side="right", padx=(0, 10))
    root.bind("<KeyPress-d>", lambda _: guides.set(not guides.get()))
    root.bind("<F11>", toggle_fullscreen)
    root.bind("<Escape>", lambda _: toggle_fullscreen() if fullscreen.get() else None)

    status = tk.StringVar(value="Loading walker")
    learning_status = tk.StringVar(value="Starting practice")
    footer = tk.Frame(root, bg=background, padx=22, pady=13)
    footer.pack(side="bottom", fill="x")
    tk.Label(footer, textvariable=learning_status, bg=background, fg=accent,
             font=("Segoe UI", 10), anchor="e").pack(side="right")
    status_label = tk.Label(footer, textvariable=status, bg=background, fg="#c2d7d8",
                            font=("Segoe UI", 11), anchor="w", justify="left", wraplength=500)
    status_label.pack(side="left", fill="x", expand=True, padx=(0, 10))
    picture = tk.Canvas(root, bg="#0a1013", bd=0, highlightthickness=1,
                        highlightbackground="#263940", cursor="crosshair")
    picture.pack(fill="both", expand=True, padx=20)
    frame_item = picture.create_image(0, 0, anchor="nw")
    footer.bind("<Configure>", lambda event: status_label.configure(wraplength=max(250, event.width - 325)))
    root.update_idletasks()

    webcam = walker = learner = None
    try:
        walker = OverlayWalker(args.checkpoint)
        settings = SETTINGS[0]
        try:
            learner = NavigationLearning(args.checkpoint)
            learner.start()
        except (OSError, sqlite3.Error):
            learning_status.set("Practice unavailable")

        def update_learning():
            nonlocal settings
            if learner is None:
                return
            try:
                settings, attempts = learner.summary()
            except sqlite3.Error:
                learning_status.set("Practice data unavailable")
            else:
                activity = "Practicing" if learner.worker.is_alive() else "Learned settings"
                if learner.error:
                    activity = "Practice interrupted"
                learning_status.set(f"{activity}  /  {settings.name}  /  {attempts} trials")
            root.after(1000, update_learning)

        update_learning()
        if not args.live_sample:
            status.set("Opening webcam")
            try:
                webcam = Webcam(args.camera)
            except OSError as error:
                status.set(f"Could not start FFmpeg. {error}")
                root.mainloop()
                return
        started = time.monotonic()
        last_tick = started
        remaining_time = 0.0
        next_scan = 0
        table = None
        tracker = TableTracker()
        video_view = None
        notice = None
        notice_until = 0
        tap_feedback = None

        def select_point(event):
            nonlocal notice, notice_until, tap_feedback
            tap_feedback = (event.x, event.y)
            notice_until = time.monotonic() + 3
            try:
                if table is None or video_view is None:
                    raise ValueError("Wait for the tabletop to be detected.")
                left, top, scale_x, scale_y = video_view
                destination = table.world_point((event.x - left) / scale_x, (event.y - top) / scale_y)
                notice = walker.choose_destination(destination, settings)
                tap_feedback = None
            except ValueError as error:
                notice = str(error)

        picture.bind("<ButtonRelease-1>", select_point)
        rejected_marker = picture.create_oval(0, 0, 0, 0, outline="#ffac72",
                                              width=3, state="hidden")

        def show_frame(output):
            nonlocal video_view
            canvas_width, canvas_height = picture.winfo_width(), picture.winfo_height()
            scale = min(canvas_width / output.width, canvas_height / output.height)
            size = (max(1, round(output.width * scale)), max(1, round(output.height * scale)))
            left, top = (canvas_width - size[0]) // 2, (canvas_height - size[1]) // 2
            picture.image = ImageTk.PhotoImage(output.resize(size, Image.Resampling.BILINEAR))
            picture.coords(frame_item, left, top)
            picture.itemconfigure(frame_item, image=picture.image)
            video_view = (left, top, size[0] / output.width, size[1] / output.height)

        def scan_table(frame):
            nonlocal table
            previous_table = tracker.plane
            table = tracker.update(frame)
            if table is not None:
                if tracker.refined:
                    walker.remap(previous_table, table)
                boxes = table.obstacles(frame, args.threshold)
                boxes = [tuple(round(value, 1) for value in box) for box in boxes]
                walker.replan(boxes, force=tracker.refined)

        def tick():
            nonlocal last_tick, next_scan, remaining_time
            now = time.monotonic()
            elapsed = min(now - last_tick, 0.12)
            last_tick = now
            if webcam is None:
                frame = sample_feed(now - started)
            elif webcam.process.poll() is not None:
                status.set(f"Could not stream {args.camera}. Check Windows camera access.\n{webcam.error()}")
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
                scan_table(frame)
                next_scan = now + 0.18
            if table is None:
                remaining_time = 0.0
                message = ("Table out of view. Trip paused" if walker.destination is not None else
                           "Looking for the white tabletop. Keep its edges visible.")
                output = frame
            else:
                remaining_time += elapsed
                steps = int(remaining_time / walker.sim.dt)
                remaining_time -= steps * walker.sim.dt
                movement = walker.advance(steps)
                if walker.outcome is not None:
                    setting, reached, elapsed, distance = walker.outcome
                    if learner is not None:
                        try:
                            learner.record(setting, "live", reached, elapsed, distance)
                        except sqlite3.Error:
                            learning_status.set("Could not save the last trip")
                    walker.outcome = None
                message = f"{movement}  /  {len(walker.boxes)} detected objects"
                output = walker.composite(frame, table, guides.get())
            status.set(notice if notice and now < notice_until else message)
            show_frame(output)
            if tap_feedback is not None and now < notice_until - 2:
                x, y = tap_feedback
                picture.coords(rejected_marker, x - 8, y - 8, x + 8, y + 8)
                picture.itemconfigure(rejected_marker, state="normal")
            else:
                picture.itemconfigure(rejected_marker, state="hidden")
            elapsed_ms = round((time.monotonic() - now) * 1000)
            root.after(max(1, 33 - elapsed_ms), tick)

        root.after(0, tick)
        root.mainloop()
    finally:
        if webcam is not None:
            webcam.close()
        if learner is not None:
            learner.close()
        if walker is not None:
            walker.close()

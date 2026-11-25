"""Map webcam obstacles onto a MuJoCo course and walk a PPO humanoid around them.

Register an empty tabletop with four clicks, subtract it from a second frame,
and plan around the resulting obstacle boxes. PPO drives the joint actuators.
A virtual harness assists balance and tracks the planned route.
"""

import argparse
from collections import deque
import heapq
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from stable_baselines3 import PPO


WIDTH, HEIGHT = 320, 200
LENGTH, SPAN = 8.0, 5.0
START, GOAL = (0.8, 2.5), (7.2, 2.5)
CLEARANCE = 0.85


def capture(camera):
    """Open the named Windows camera and take a frame after exposure settles."""
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "dshow", "-i", f"video={camera}", "-ss", "1",
        "-vf", "scale=800:-2", "-frames:v", "1", "-c:v", "png",
        "-f", "image2pipe", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=15, check=True)
        with Image.open(io.BytesIO(result.stdout)) as frame:
            return frame.convert("RGB")
    except FileNotFoundError as error:
        raise ValueError("FFmpeg is required for live Windows webcam capture.") from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode(errors="replace").strip()
        raise ValueError(f"Could not open {camera}. Check camera permissions and close other camera apps.\n{details}") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"{camera} did not deliver a frame within 15 seconds.") from error


def rectify(frame, corners):
    """Map the marked tabletop corners onto a top-down rectangle."""
    edges = np.roll(corners, -1, axis=0) - np.asarray(corners)
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(turns <= 0):
        raise ValueError("Mark the four corners in clockwise order, starting at the top left.")
    rows, values = [], []
    for (u, v), (x, y) in zip(((0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)), corners):
        rows.append([u, v, 1, 0, 0, 0, -x * u, -x * v])
        rows.append([0, 0, 0, u, v, 1, -y * u, -y * v])
        values.extend((x, y))
    try:
        weights = np.linalg.solve(rows, values)
    except np.linalg.LinAlgError as error:
        raise ValueError("Mark four distinct corners of the tabletop.") from error
    return frame.transform((WIDTH, HEIGHT), Image.Transform.PERSPECTIVE, weights,
                           resample=Image.Resampling.BILINEAR)


def detect_obstacles(empty, scene, threshold=32):
    """Find changed regions without depending on object classes or colors."""
    before = np.asarray(empty.filter(ImageFilter.GaussianBlur(1)), dtype=np.int16)
    after = np.asarray(scene.filter(ImageFilter.GaussianBlur(1)), dtype=np.int16)
    difference = after - before
    difference -= np.median(difference, axis=(0, 1)).astype(np.int16)
    mask = np.max(np.abs(difference), axis=2) >= threshold
    mask = Image.fromarray(mask.astype(np.uint8) * 255)
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask = np.array(mask, dtype=bool)
    mask[:3, :] = mask[-3:, :] = False
    mask[:, :3] = mask[:, -3:] = False
    return obstacle_boxes(mask)


def obstacle_boxes(mask):
    boxes = []
    # Flood-fill each foreground component and discard small changes.
    for iy, ix in np.argwhere(mask):
        if not mask[iy, ix]:
            continue
        pending = deque([(int(iy), int(ix))])
        mask[iy, ix] = False
        left = right = int(ix)
        top = bottom = int(iy)
        count = 0
        while pending:
            y, x = pending.popleft()
            count += 1
            left, right = min(left, x), max(right, x)
            top, bottom = min(top, y), max(bottom, y)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if mask[ny, nx]:
                    mask[ny, nx] = False
                    pending.append((ny, nx))
        if count >= 100 and right - left >= 5 and bottom - top >= 5:
            boxes.append((left / WIDTH * LENGTH, top / HEIGHT * SPAN,
                          (right + 1) / WIDTH * LENGTH, (bottom + 1) / HEIGHT * SPAN))
    return boxes


def position_clear(point, boxes, clearance=CLEARANCE):
    x, y = point
    return (0.5 <= x <= LENGTH - 0.5 and 0.5 <= y <= SPAN - 0.5 and
            all(not (x1 - clearance <= x <= x2 + clearance and
                     y1 - clearance <= y <= y2 + clearance)
                for x1, y1, x2, y2 in boxes))


def segment_clear(origin, destination, boxes, clearance):
    samples = max(1, math.ceil(math.dist(origin, destination) / 0.04))
    for index in range(samples + 1):
        fraction = index / samples
        point = (origin[0] + (destination[0] - origin[0]) * fraction,
                 origin[1] + (destination[1] - origin[1]) * fraction)
        if not position_clear(point, boxes, clearance):
            return False
    return True


def plan_route(boxes, origin=START, clearance=CLEARANCE, destination=GOAL):
    """Find a shortest collision-free route through a grid with body clearance."""
    spacing = 0.1

    def available(x, y):
        return position_clear((x, y), boxes, clearance)

    def clear(a, b):
        return segment_clear(a, b, boxes, clearance)

    if not available(*origin):
        raise ValueError("The robot needs more room around it.")
    if not available(*destination):
        raise ValueError("Choose a point farther from the table edge or an object.")
    if clear(origin, destination):
        return [origin, destination]

    def grid_node(point):
        column, row = (round(value / spacing) for value in point)
        candidates = [(x, y) for x in range(column - 1, column + 2)
                      for y in range(row - 1, row + 2)
                      if clear(point, (x * spacing, y * spacing))]
        if not candidates:
            raise ValueError("The route has insufficient clearance.")
        return min(candidates, key=lambda node: math.dist(point, (node[0] * spacing, node[1] * spacing)))

    start, goal = grid_node(origin), grid_node(destination)
    distance = {start: 0}
    parent = {start: None}
    queue = [(math.dist(start, goal), 0, start)]
    while queue:
        _, cost, current = heapq.heappop(queue)
        if current == goal:
            break
        if cost != distance[current]:
            continue
        x, y = current
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            neighbor = (x + dx, y + dy)
            if cost + 1 >= distance.get(neighbor, math.inf):
                continue
            if clear((x * spacing, y * spacing), (neighbor[0] * spacing, neighbor[1] * spacing)):
                distance[neighbor] = cost + 1
                parent[neighbor] = current
                priority = cost + 1 + math.dist(neighbor, goal)
                heapq.heappush(queue, (priority, cost + 1, neighbor))
    if goal not in parent:
        raise ValueError("No route fits. Leave more space between obstacles or around their sides.")
    points = []
    node = goal
    while node is not None:
        points.append(tuple(value * spacing for value in node))
        node = parent[node]
    points.reverse()
    points.insert(0, origin)
    points.append(destination)
    route = [origin]
    index = 0
    while index < len(points) - 1:
        next_index = len(points) - 1
        while next_index > index and not clear(points[index], points[next_index]):
            next_index -= 1
        if next_index == index:
            raise ValueError("The route has insufficient clearance. Move the obstacles.")
        route.append(points[next_index])
        index = next_index
    return route


def make_course(boxes, route, directory, render, overlay=False):
    """Use the benchmark model with collidable boxes and visible route markers."""
    template = gym.make("Humanoid-v5")
    try:
        tree = ET.parse(template.unwrapped.fullpath)
    finally:
        template.close()
    world = tree.getroot().find("worldbody")
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        ET.SubElement(world, "geom", name=f"obstacle_{index}", type="box",
                      pos=f"{(x1 + x2) / 2} {(y1 + y2) / 2} 0.36",
                      size=f"{(x2 - x1) / 2} {(y2 - y1) / 2} 0.36",
                      rgba="0.86 0.27 0.12 1", contype="1", conaffinity="1")
    for index, (a, b) in enumerate(zip(route, route[1:])):
        ET.SubElement(world, "geom", name=f"route_{index}", type="capsule",
                      fromto=f"{a[0]} {a[1]} 0.025 {b[0]} {b[1]} 0.025",
                      size="0.025", rgba="0.16 0.65 0.94 1", contype="0", conaffinity="0")
    if not overlay:
        for name, point, color in (("start", START, "0.16 0.65 0.94 1"),
                                   ("goal", GOAL, "0.27 0.82 0.37 1")):
            ET.SubElement(world, "geom", name=name, type="cylinder",
                          pos=f"{point[0]} {point[1]} 0.015", size="0.2 0.014",
                          rgba=color, contype="0", conaffinity="0")
    xml = Path(directory) / "course.xml"
    tree.write(xml, encoding="unicode")
    return gym.make("Humanoid-v5", xml_file=str(xml), max_episode_steps=1800,
                    render_mode="human" if render else None,
                    default_camera_config={"trackbodyid": -1, "distance": 10.5, "azimuth": 90,
                                           "elevation": -54, "lookat": np.array([4.0, 2.5, 0.2])})


def steer(sim, torso, delta, max_speed=1.0):
    """Apply the virtual harness forces for one policy step."""
    distance = np.linalg.norm(delta)
    if distance < 1e-8:
        sim.data.xfrc_applied[torso] = 0
        return
    desired_velocity = min(max_speed, 2.0 * distance) * delta / distance
    rotation = sim.data.xmat[torso].reshape(3, 3)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY, torso, velocity, 0)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    desired_yaw = math.atan2(delta[1], delta[0])
    yaw_error = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
    wrench = sim.data.xfrc_applied[torso]
    wrench[:2] = np.clip(180 * (desired_velocity - sim.data.qvel[:2]), -190, 190)
    wrench[2] = np.clip(4200 * (1.38 - sim.data.qpos[2]) - 330 * sim.data.qvel[2], 0, 450)
    wrench[3:] = np.cross(rotation[:, 2], [0, 0, 1]) * 310 - 27 * velocity[:3]
    wrench[5] += 160 * yaw_error
    wrench[3:] = np.clip(wrench[3:], -160, 160)


def run_course(boxes, route, checkpoint, render):
    """Use PPO joint actions and bounded simulated torso forces to follow waypoints."""
    from torch import set_num_threads

    set_num_threads(1)
    policy = PPO.load(checkpoint, device="cpu")
    with TemporaryDirectory() as directory:
        env = make_course(boxes, route, directory, render)
        try:
            return walk(env, boxes, route, policy, render)
        finally:
            env.close()


def walk(env, boxes, route, policy, render, *, max_speed=1.0, stop=None):
    sim = env.unwrapped
    # Set the starting pose before rendering the first frame.
    rng = np.random.default_rng(0)
    qpos = sim.init_qpos + rng.uniform(-0.005, 0.005, size=sim.model.nq)
    qpos[:2] = route[0]
    mujoco.mj_normalizeQuat(sim.model, qpos)
    sim.set_state(qpos, np.zeros(sim.model.nv))
    # Step the unwrapped environment after initializing its state here.
    observation = sim._get_obs()
    torso = sim.model.body("torso").id
    obstacle_ids = {sim.model.geom(f"obstacle_{i}").id for i in range(len(boxes))}
    target_index = 1
    for step in range(1800):
        if stop is not None and stop.is_set():
            return False, "Practice stopped."
        started = time.perf_counter()
        position = sim.data.qpos[:2]
        target = np.asarray(route[target_index])
        if np.linalg.norm(target - position) < 0.18:
            if target_index == len(route) - 1:
                if render:
                    time.sleep(1)
                return True, f"Reached goal in {step * sim.dt:.1f} simulated seconds with zero obstacle contacts."
            target_index += 1
            target = np.asarray(route[target_index])
        steer(sim, torso, target - position, max_speed=max_speed)
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, fell, _, _ = sim.step(action)
        for contact in sim.data.contact:
            if contact.geom1 in obstacle_ids or contact.geom2 in obstacle_ids:
                return False, "The humanoid contacted an obstacle."
        if fell:
            return False, "The humanoid fell before it reached the goal."
        if render:
            time.sleep(max(0, sim.dt - (time.perf_counter() - started)))
    return False, "The humanoid did not reach the goal before the time limit."


def sample_frames():
    """Provide an offline scene for rehearsal without opening the webcam."""
    empty = Image.new("RGB", (WIDTH, HEIGHT), "#d4c8b3")
    scene = empty.copy()
    draw = ImageDraw.Draw(scene)
    draw.rounded_rectangle((136, 76, 169, 119), radius=10, fill="#373c44")
    draw.ellipse((206, 144, 230, 169), fill="#426f99")
    return empty, scene


def annotate(frame, boxes, route):
    result = frame.copy()
    draw = ImageDraw.Draw(result)

    def pixels(point):
        return point[0] / LENGTH * WIDTH, point[1] / SPAN * HEIGHT

    for x1, y1, x2, y2 in boxes:
        draw.rectangle((*pixels((x1, y1)), *pixels((x2, y2))), outline="#df521e", width=3)
    if route:
        draw.line([pixels(point) for point in route], fill="#1784e4", width=3)
    for point, label, color in ((START, "START", "#1067b4"), (GOAL, "GOAL", "#137c37")):
        x, y = pixels(point)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x - 15, y + 7), label, fill=color)
    return result


def show_demo(args, frames, photo_pair=False):
    """Capture a registered scene, preview the route, and supervise the viewer."""
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Webcam Humanoid Navigation")
    panel = ttk.Frame(root, padding=12)
    panel.pack(fill="both", expand=True)
    message = tk.StringVar()
    ttk.Label(panel, textvariable=message, wraplength=700).pack(fill="x", pady=(0, 8))
    canvas = tk.Canvas(panel, width=640, height=400, highlightthickness=0)
    canvas.pack()
    controls = ttk.Frame(panel)
    controls.pack(fill="x", pady=(8, 0))
    corners = []
    reference = None
    display_scale = 1
    photo = None
    course = None
    child = None

    def display(frame):
        nonlocal display_scale, photo
        display_scale = min(700 / frame.width, 490 / frame.height)
        scaled = frame.resize((round(frame.width * display_scale), round(frame.height * display_scale)))
        photo = ImageTk.PhotoImage(scaled)
        canvas.configure(width=scaled.width, height=scaled.height)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)

    def prepare(empty, scene):
        nonlocal course
        boxes = detect_obstacles(empty, scene, args.threshold)
        course = None
        start_button.state(["disabled"])
        try:
            route = plan_route(boxes)
        except ValueError as error:
            display(annotate(scene, boxes, []))
            message.set(str(error))
            return
        display(annotate(scene, boxes, route))
        if not boxes:
            message.set("No obstacles found. Add a contrasting object near the middle and capture again.")
            return
        course = (boxes, route)
        start_button.state(["!disabled"])
        message.set(f"Detected {len(boxes)} obstacle(s). Blue shows the planned route. Run simulation to follow it.")

    def read_frame():
        message.set("Capturing webcam frame…")
        root.update_idletasks()
        try:
            return capture(args.camera)
        except ValueError as error:
            message.set(str(error))
            return None

    def capture_empty():
        nonlocal reference, course
        frame = frames[0] if photo_pair else read_frame()
        if frame is None:
            return
        reference, course = frame, None
        corners.clear()
        display(frame)
        start_button.state(["disabled"])
        scene_button.state(["disabled"])
        message.set("Click the corners in order: top left, top right, bottom right, bottom left.")

    def mark(event):
        if reference is None or len(corners) == 4:
            return
        corners.append((event.x / display_scale, event.y / display_scale))
        canvas.create_oval(event.x - 4, event.y - 4, event.x + 4, event.y + 4, fill="#1784e4")
        if len(corners) == 4:
            if photo_pair:
                capture_scene()
            else:
                scene_button.state(["!disabled"])
                message.set("Keep the webcam still. Place objects on the surface and click Capture obstacles.")

    def capture_scene():
        frame = frames[1] if photo_pair else read_frame()
        if frame is None:
            return
        try:
            if frame.size != reference.size:
                raise ValueError("The camera resolution changed. Capture the empty surface again.")
            prepare(rectify(reference, corners), rectify(frame, corners))
        except ValueError as error:
            message.set(str(error))

    def poll():
        nonlocal child
        if child.poll() is None:
            root.after(200, poll)
            return
        result = child.stdout.read().strip()
        child.stdout.close()
        child = None
        message.set(result if result else "The simulator closed before reporting a result.")
        start_button.state(["!disabled"])
        if frames is None:
            empty_button.state(["!disabled"])
            scene_button.state(["!disabled"])
        elif photo_pair:
            empty_button.state(["!disabled"])

    def launch():
        nonlocal child
        command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint",
                   str(args.checkpoint.resolve()), "--run-course"]
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
        child.stdin.write(json.dumps(course))
        child.stdin.close()
        for button in (start_button, empty_button, scene_button):
            button.state(["disabled"])
        message.set("Simulation running in the MuJoCo viewer…")
        root.after(200, poll)

    def close():
        if child is not None and child.poll() is None:
            child.terminate()
        root.destroy()

    empty_button = ttk.Button(controls, text="Reset corners" if photo_pair else "Capture empty surface",
                              command=capture_empty)
    scene_button = ttk.Button(controls, text="Capture obstacles", command=capture_scene)
    start_button = ttk.Button(controls, text="Run simulation", command=launch)
    for button in (empty_button, scene_button, start_button):
        button.pack(side="left", padx=(0, 8))
        button.state(["disabled"])
    canvas.bind("<Button-1>", mark)
    root.protocol("WM_DELETE_WINDOW", close)
    if frames is None:
        empty_button.state(["!disabled"])
        message.set("Aim the webcam at a flat, empty tabletop or floor and click Capture empty surface.")
    elif photo_pair:
        empty_button.state(["!disabled"])
        capture_empty()
    else:
        prepare(*frames)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/baseline-seed0/policy.zip"))
    parser.add_argument("--camera", default="Integrated Camera", help="Windows DirectShow camera name")
    parser.add_argument("--sample", action="store_true", help="rehearse with a generated scene")
    parser.add_argument("--live-sample", action="store_true", help="rehearse the live overlay without a webcam")
    parser.add_argument("--snapshot", action="store_true", help="open the original capture workflow")
    parser.add_argument("--images", type=Path, nargs=2, metavar=("EMPTY", "SCENE"),
                        help="use two matching photos of the same surface instead of live capture")
    parser.add_argument("--headless", action="store_true", help="evaluate sample or images without windows")
    parser.add_argument("--threshold", type=int, choices=range(1, 256), metavar="1-255", default=32)
    parser.add_argument("--run-course", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.run_course:
        boxes, route = json.load(sys.stdin)
        success, result = run_course(boxes, route, args.checkpoint, render=True)
        print(result)
        return 0 if success else 1
    if sum(bool(mode) for mode in (args.sample, args.images, args.snapshot, args.live_sample)) > 1:
        parser.error("choose one demo mode")
    if args.live_sample and args.headless:
        parser.error("--live-sample opens the live overlay window")
    if args.live_sample or not (args.sample or args.images or args.snapshot or args.headless):
        from live_tabletop import show_live
        show_live(args)
        return 0
    frames = None
    if args.sample:
        frames = sample_frames()
    if args.images:
        empty_path, scene_path = args.images
        try:
            with Image.open(empty_path) as empty, Image.open(scene_path) as scene:
                if empty.size != scene.size:
                    parser.error("the two images must have the same dimensions")
                frames = (empty.convert("RGB"), scene.convert("RGB"))
        except OSError as error:
            parser.error(str(error))
    if args.headless:
        if frames is None:
            parser.error("--headless requires --sample or --images")
        boxes = detect_obstacles(*(frame.resize((WIDTH, HEIGHT)) for frame in frames), args.threshold)
        try:
            route = plan_route(boxes)
        except ValueError as error:
            parser.error(str(error))
        print(f"Detected {len(boxes)} obstacle(s); planned {len(route) - 1} segment(s).")
        success, result = run_course(boxes, route, args.checkpoint, render=False)
        print(result)
        return 0 if success else 1
    show_demo(args, frames, photo_pair=args.images is not None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

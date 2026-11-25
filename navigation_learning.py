"""Learn which walking settings work best from practice and live trips.

Each launch rehearses the same generated obstacle course with each setting.
SQLite keeps compact outcome statistics so later launches build on prior runs.
"""

from contextlib import closing
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event, Thread

from stable_baselines3 import PPO

from tabletop_demo import GOAL, START, make_course, plan_route, walk


@dataclass(frozen=True)
class WalkingSettings:
    name: str
    speed: float
    clearance: float


SETTINGS = (
    WalkingSettings("Balanced", 1.0, 0.85),
    WalkingSettings("Careful", 0.78, 0.95),
    WalkingSettings("Agile", 1.28, 0.85),
)


class NavigationLearning:
    def __init__(self, checkpoint, database=None):
        self.checkpoint = checkpoint
        self.database = Path(database or Path(__file__).resolve().parent / "runs" / "navigation-learning.sqlite3")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.policy_key = f"v1:{digest}"
        self.stop = Event()
        self.worker = None
        self.error = None
        with closing(self.connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS attempts (
                    policy TEXT NOT NULL, setting TEXT NOT NULL, source TEXT NOT NULL,
                    reached INTEGER NOT NULL, cost REAL NOT NULL
                )
            """)

    def connect(self):
        return sqlite3.connect(self.database, timeout=2)

    def summary(self):
        with closing(self.connect()) as db:
            rows = db.execute("""
                SELECT setting, source, COUNT(*), AVG(cost)
                FROM attempts WHERE policy = ? GROUP BY setting, source
            """, (self.policy_key,)).fetchall()
        results = {(name, source): (count, cost) for name, source, count, cost in rows}

        def preference(setting):
            count, practice = results.get((setting.name, "practice"), (0, 2.0))
            live_count, live = results.get((setting.name, "live"), (0, 2.0))
            # Start from an equal prior and keep practice dominant because each
            # setting gets the same practice course and live trips vary.
            practice_cost = (4.0 + count * practice) / (2 + count)
            live_cost = (4.0 + live_count * live) / (2 + live_count)
            return 0.75 * practice_cost + 0.25 * live_cost

        best = min(SETTINGS, key=preference)
        return best, sum(count for _, _, count, _ in rows)

    def record(self, setting, source, reached, elapsed, distance):
        row = self.attempt_row(setting, source, reached, elapsed, distance)
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, ?)", row)

    def attempt_row(self, setting, source, reached, elapsed, distance):
        # Penalize a collision, fall, or timeout more heavily than a slow trip.
        cost = min(7.0, elapsed / max(distance, 0.5)) if reached else 10.0
        return self.policy_key, setting.name, source, reached, cost

    def start(self):
        self.worker = Thread(target=self.practice, name="navigation-practice", daemon=True)
        self.worker.start()

    def practice(self):
        try:
            _, attempts = self.summary()
            rng = random.Random(attempts)
            center_y = rng.uniform(1.65, 3.15)
            boxes = [(3.1, center_y - 0.55, 4.0, center_y + 0.55)]
            # Each run varies the target to rehearse more than straight crossings.
            goal = (GOAL[0], rng.uniform(1.5, 3.5))
            policy = PPO.load(self.checkpoint, device="cpu")
            rows = []
            for setting in SETTINGS:
                if self.stop.is_set():
                    return
                reached, elapsed = self.rehearse(policy, boxes, goal, setting)
                rows.append(self.attempt_row(setting, "practice", reached, elapsed, math.dist(START, goal)))
            # Save complete comparisons so closing early cannot favor a setting.
            if not self.stop.is_set():
                with closing(self.connect()) as db, db:
                    db.executemany("INSERT INTO attempts VALUES (?, ?, ?, ?, ?)", rows)
        except Exception as error:
            self.error = str(error)

    def rehearse(self, policy, boxes, goal, setting):
        try:
            route = plan_route(boxes, clearance=setting.clearance, destination=goal)
        except ValueError:
            return False, 0
        with TemporaryDirectory() as directory:
            env = make_course(boxes, [], directory, render=False, overlay=True)
            try:
                reached, _ = walk(env, boxes, route, policy, render=False,
                                  max_speed=setting.speed, stop=self.stop)
                return reached, env.unwrapped.data.time
            finally:
                env.close()

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join()

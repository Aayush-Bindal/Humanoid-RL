"""Check candidate checkpoint loading, arrivals and replanning in the demo."""

import argparse
import json
from pathlib import Path

from live_tabletop import OverlayWalker
from navigation_learning import SETTINGS
from train_gait import BASELINE, obstacle_layout


def exercise(checkpoint, first_seed, count):
    walker = OverlayWalker(checkpoint)
    results = []
    try:
        for index in range(count):
            seed = first_seed + index
            start, goal = (0.9, 1.0 + 0.9 * (index % 4)), (7.2, 3.8 - 0.8 * (index % 4))
            if index % 2:
                start, goal = goal, start
            walker.reset()
            walker.boxes = obstacle_layout(seed)
            walker.place_robot(start)
            setting = SETTINGS[(index // 2) % len(SETTINGS)]
            walker.choose_destination(goal, setting)
            moved = False
            while walker.route:
                if index % 4 >= 2 and not moved and walker.elapsed >= 2.0:
                    # Add an obstacle ahead, at a midpoint of the remaining path.
                    x, y = walker.sim.data.qpos[:2]
                    remaining = [(px, py) for px, py in walker.route if abs(px - x) >= 2]
                    if remaining:
                        tx, ty = remaining[0]
                        center_x = (x + tx) / 2
                        center_y = (y + ty) / 2
                        added = (center_x - 0.19, center_y - 0.21,
                                 center_x + 0.19, center_y + 0.21)
                        walker.replan([*walker.boxes, added])
                    moved = True
                walker.advance(1)
            results.append(dict(seed=seed, setting=setting.name, moved_obstacle=moved,
                                status=walker.status, seconds=walker.elapsed))
        return dict(lower_lift=walker.lower_lift, phase_gait=walker.phase_gait, results=results)
    finally:
        walker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--cases", type=int, default=18)
    args = parser.parse_args()
    report = {"baseline": exercise(BASELINE, args.seed, args.cases),
              "candidate": exercise(args.checkpoint, args.seed, args.cases)}
    for label, item in report.items():
        outcomes = [case["status"] for case in item["results"]]
        print(label, {value: outcomes.count(value) for value in sorted(set(outcomes))})
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

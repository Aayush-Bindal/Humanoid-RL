# Humanoid walking demo

Realistic walk is the default policy in this copy of the tabletop demo. Both checkpoints are included. The benchmark uses collidable MuJoCo boxes, seeded routes that turn around the obstacles, and the demo's waypoint controller.

## Try the walking policy

From this folder on Windows, with Python 3.12, uv and FFmpeg available:

```powershell
uv sync --locked
uv run python tabletop_demo.py --live-sample
```

Click a spot on the generated table to make it walk. Drop `--live-sample` to use the webcam. Use the **Policy** selector at the top to switch between **Realistic walk** and **Original**. Switching keeps the robot's current position and clears its destination so the new policy can start from a standing pose. Click a destination to resume walking.

**Realistic walk** displays a planted walking pose. It moves weight onto the supporting leg, keeps that foot at a fixed ground point while the other foot swings, and brings the feet back under the body at stops. **Original** uses the original policy and raw simulated joints. Pass `--raw-gait` to see the raw joints for the fine-tuned policy too. The walking pose is a rendering layer. It does not change the physics, path planning or collision decisions for the selected policy.

The experimental checkpoint uses the code in this package. Its PPO policy observes the same MuJoCo state as the original, plus a stride clock and a steering command. It was initialized with the original policy's weights. Training rewards foot placement, route completion and balance. The simulator still assists steering and balance. The experimental checkpoint uses less vertical lift and a pace selected on validation routes.

## Review and reproduce

`realistic-walk-comparison.mp4` compares the raw simulated pose and the new pose on the same run at simulation speed. `evaluation.json` contains the held-out results from policy training. `overlay-check.json` checks the live demo's walker across simulated trips with changing obstacle positions. `validation-selection.json` shows how the checkpoint and pace were selected before the held-out evaluation.

```powershell
uv run python train_gait.py compare runs/experimental-gait/policy.zip --episodes 40 --output evaluation-rerun.json
uv run python train_gait.py compare runs/experimental-gait/policy.zip --seed 81000 --episodes 60 --output policy-benchmark-rerun.json
uv run python check_overlay.py runs/experimental-gait/policy.zip overlay-rerun.json
uv run python check_planted_gait.py
uv run python replay_gait.py runs/experimental-gait/policy.zip comparison.mp4 --seed 40003 --planted
```

To reproduce the stride training run:

```powershell
uv run python train_gait.py train --run-dir runs/retrain-gait --steps 491520 --interval 40960 --validation 16 --parallel --phase-gait --lower-lift --seed 12 --learning-rate 0.00020 --exploration 0.32
```

To fine-tune Realistic walk for forward-facing movement:

```powershell
uv run python train_gait.py train --baseline runs/experimental-gait/policy.zip --run-dir runs/heading-gait --steps 32768 --interval 8192 --validation 8 --phase-gait --lower-lift --heading-weight 12 --learning-rate 0.00004 --exploration 0.18 --seed 16
uv run python tabletop_demo.py --checkpoint runs/heading-gait/best.zip
```

The heading reward keeps full progress credit when the torso faces the direction it travels and reduces credit for sideways or backward motion. Validation reports `heading_alignment`, the fraction of traveled distance projected onto the torso's forward axis. A score of 1 means forward travel, 0 means sideways travel, and −1 means backward travel. Checkpoint selection prioritizes arrivals and collision avoidance before scoring gait and heading.

## Policy benchmark

Both policies used the same 60 fresh obstacle courses (seeds 81000–81059). The matched-settings control ran the original policy at the new policy's pace and lift.

| Configuration | Finished | Falls | Obstacle contacts | Mean time on finishes | Steps with a foot on the ground |
|---|---:|---:|---:|---:|---:|
| Original | 60/60 | 0 | 0 | 9.54 s | 9.6% |
| Realistic walk | 60/60 | 0 | 0 | 8.81 s | 38.8% |
| Original at matched pace and lift | 57/60 | 3 | 0 | 8.35 s | 28.1% |

The foot contact and timing figures describe the physics. The live avatar's foot planting is a rendering layer.

In 24 separate simulated overlay trips, including cases with changing obstacles, Original finished 24 and Realistic walk finished 23. The latter stopped near an obstacle once.

Raw results are in `policy-benchmark.json` and `policy-overlay-benchmark.json`.

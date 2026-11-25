# Webcam-guided humanoid navigation

The webcam demo detects a walking area, plans routes around obstacles, and draws a simulated humanoid over the live video. Both PPO checkpoints are included. See the [gait experiment](EXPERIMENT.md) for training and benchmark commands.

Run from this project folder. On a fresh Windows machine, use Python 3.12 and uv to install the locked dependencies. The webcam stream uses FFmpeg on PATH with DirectShow input and scales the full camera view while preserving its aspect ratio.

```powershell
uv sync --locked
uv run python tabletop_demo.py
```

Point the webcam toward a white tabletop with its far edge and at least one side in view. When the robot appears, tap or click a clear spot on the table to send it there. A mint marker shows the destination. Taps near the edge or an object can move the marker to a nearby clear spot. Blocked taps flash an orange ring with a reason in the footer. Tap another spot to change the destination. The walker replans when objects block its route. If a new detection traps the robot, a valid tap can restart it from nearby clear space. If the table leaves view, the trip pauses and resumes from the same position when the tabletop is detected again.

Resize the window or press **F11** for fullscreen. Press **Esc** to exit fullscreen. Turn on **Guides** with **D** to see the detected tabletop, obstacle boxes, and planned route.

Use the **Policy** dropdown to switch between **Realistic walk**, the default fine-tuned PPO checkpoint with a foot-planting animation, and **Original**, the baseline PPO checkpoint with the raw simulated pose. Switching keeps the robot in place and clears its destination. Click a new destination to resume walking. Pass `--raw-gait` to see the raw simulated pose with Realistic walk.

For a different webcam, list Windows camera names with `ffmpeg -hide_banner -list_devices true -f dshow -i dummy` and pass its name with `--camera "Camera Name"`.

Rehearse the live overlay with a generated feed and a moving obstacle.

```powershell
uv run python tabletop_demo.py --live-sample
```

## Surfaces and calibration

Live mode looks for a bright, mostly neutral patch with visible edges and maps it to a top-down rectangle. A white tabletop works best. To try it on the floor, lay down white poster board and keep its far edge and one side in view. Dark or patterned surfaces and a floor that fills the entire frame are unreliable. Strong shadows can register as obstacles, and pale objects can blend into the surface.

For other flat surfaces, use `--snapshot` to mark four corners on an empty capture, then capture the same scene with obstacles. This mode opens a separate MuJoCo viewer. Use `--images empty.jpg scene.jpg` for two photos from the same camera position or `--sample --headless` for an automated demonstration.

## How it works

The vision system maps your click to the walking plane and detects obstacles by contrast with the surface. The route planner steers around their estimated footprints. The selected PPO policy runs on the CPU and drives the joints from MuJoCo state observations. A virtual harness assists balance and waypoint tracking, and MuJoCo draws the avatar through OpenGL.

Background MuJoCo practice trials compare walking speeds and obstacle margins. The app stores trial results and completed live trips per checkpoint in `runs/navigation-learning.sqlite3`. The footer shows the chosen settings and the trial count.

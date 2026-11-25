# Webcam-guided humanoid navigation

Run from this project folder. On a fresh Windows machine, use Python 3.12 and uv to install the locked dependencies. The webcam stream uses FFmpeg on PATH with DirectShow input and scales the full camera view while preserving its aspect ratio.

```powershell
uv sync --locked
uv run python tabletop_demo.py
```

Point the webcam toward a white tabletop with its far edge and at least one side in view. When the robot appears, tap or click a clear spot on the table to send it there. A mint marker shows the destination. Taps at the edge place the marker slightly inward, and blocked taps flash an orange ring with a reason in the footer. Tap another spot to change the destination. The walker replans when objects block its route. If the table leaves view, the trip pauses and resumes from the same position when the tabletop is detected again.

Resize the window or press **F11** for fullscreen. Press **Esc** to exit fullscreen. Turn on **Guides** with **D** to see the detected tabletop, obstacle boxes, and planned route.

Each launch runs a background MuJoCo practice course. The app compares walking speeds and obstacle margins on the same course, then picks settings using the saved scores and completed live trips. Results persist in `runs/navigation-learning.sqlite3`. The footer shows the selected settings and trial count. This adapts navigation while the existing PPO checkpoint continues to control the joints. Allow the practice trials to finish before closing if you want that run to count.

For a different webcam, list Windows camera names with `ffmpeg -hide_banner -list_devices true -f dshow -i dummy` and pass its name with `--camera "Camera Name"`.

Rehearse the live overlay with a generated feed and a moving obstacle.

```powershell
uv run python tabletop_demo.py --live-sample
```

## How it works

Live mode traces the visible table edges and maps them to a rectangular walking area. It searches outside the initial bright patch for a larger fit supported by the table edges. If several frames show more of the same table, it expands the map and keeps the robot and target anchored in the camera view. It detects objects within the map, plans around their footprints, then projects the humanoid back onto the table. Strong shadows, pale objects, or fully hidden edges can affect detection. The trained PPO policy drives the joints using MuJoCo state observations. A separate virtual harness assists balance and waypoint tracking.

The original capture workflow remains available via `--snapshot`. It supports manual four-corner calibration and a separate MuJoCo viewer. Use `--images empty.jpg scene.jpg` for two photos or `--sample --headless` for its automated demonstration.

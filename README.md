# Webcam-guided humanoid navigation

Run from this project folder. On a fresh Windows machine, use Python 3.12 and uv to install the locked dependencies. The webcam stream uses FFmpeg on PATH with DirectShow input.

```powershell
uv sync --locked
uv run python tabletop_demo.py
```

Point the webcam toward a white tabletop with its far edge and at least one side in view. The app outlines the tabletop in mint and marks detected objects in orange. The humanoid starts walking once it finds a route across the table. Move an object and the app replans. The walker pauses if the route is blocked or the table leaves view.

For a different webcam, list Windows camera names with `ffmpeg -hide_banner -list_devices true -f dshow -i dummy` and pass its name with `--camera "Camera Name"`.

Rehearse the live overlay with a generated feed and a moving obstacle.

```powershell
uv run python tabletop_demo.py --live-sample
```

## How it works

Live mode finds the white tabletop in the camera view and maps its detected boundaries to a rectangular walking area. It detects objects within that area, plans around their footprints, then projects the humanoid back onto the table. Strong shadows, pale objects, or hidden edges can affect detection. The trained PPO policy drives the joints using MuJoCo state observations. A separate virtual harness assists balance and waypoint tracking.

The original capture workflow remains available via `--snapshot`. It supports manual four-corner calibration and a separate MuJoCo viewer. Use `--images empty.jpg scene.jpg` for two photos or `--sample --headless` for its automated demonstration.

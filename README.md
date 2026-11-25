# Tabletop Walker

Click a spot on a tabletop and watch a simulated humanoid walk there. Tabletop Walker uses a webcam to locate the walking surface, estimate obstacle positions, and display the MuJoCo humanoid over the video. It replans when an object blocks the route.

## Quick start

From the project folder on Windows, with Python 3.12 and uv installed:

```powershell
uv sync --locked
uv run python tabletop_demo.py --live-sample
```

Click a clear spot on the generated table. To use a real webcam or OBS Virtual Camera, install FFmpeg, put it on your PATH, and run:

```powershell
uv run python tabletop_demo.py
```

The app opens **cam0**. Use **Camera** to switch feeds. For OBS, add the tabletop image as an Image source, fit it to the scene, and start OBS Virtual Camera before opening Tabletop Walker. The app lists cameras at startup.

| Control | What it does |
| --- | --- |
| Click the table | Set a destination. The mint marker shows the selected point. |
| **Camera** | Switch feeds and detect the tabletop again. |
| **Policy** | Switch between **Realistic walk** and **Original**. Choose a new destination after switching. |
| **Guides** or **D** | Show the detected table outline, obstacle boxes, and planned route. |
| **F11** / **Esc** | Enter fullscreen / exit fullscreen. |

## Get a reliable tabletop lock

Use a bright white tabletop or a sheet of white poster board. Frame the far edge and at least one side. Keeping all four corners visible and leaving clear strips of white around the obstacles gives the detector more to work with. Start with a few small, dark objects. Shadows can be picked up as obstacles, and pale or transparent objects can blend into the surface.

If the app loses the table, the trip pauses and resumes when the surface is detected again. For a dark or patterned flat surface, try snapshot mode with a fixed camera. Mark the four corners on the empty capture, then capture the same surface with objects in place.

## Run modes

| Command arguments after `tabletop_demo.py` | Mode |
| --- | --- |
| `--live-sample` | Try the live overlay with a generated feed and a moving obstacle. |
| `--list-cameras` | List DirectShow devices with their camera numbers. |
| `--camera cam1` | Override the starting camera. A number or an exact device name also works. |
| `--snapshot` | Calibrate from webcam snapshots and open a separate MuJoCo viewer. |
| `--images empty.jpg scene.jpg` | Use two photos of the same surface from a fixed camera angle. |
| `--sample --headless` | Run the snapshot workflow in simulation without a window. |
| `--raw-gait` | Show the simulated joint pose in the live overlay. |

## How the demo works

1. **Vision.** The detector traces the bright tabletop against its surroundings, fits four corners, and maps the camera image to a top-down rectangle. Color and brightness differences inside that rectangle become obstacle boxes.
2. **Navigation.** The planner turns clicks into destinations on the walking plane and routes around the boxes. Background practice trials evaluate walking speeds and obstacle margins. The app saves results per checkpoint in `runs/navigation-learning.sqlite3`.
3. **Locomotion.** The PPO policy reads simulated body state and predicts joint actions on the CPU. A virtual harness assists steering and balance. The renderer draws the humanoid in MuJoCo and composites it over the webcam feed.

## Included policies

| Policy | Checkpoint | What you see |
| --- | --- | --- |
| **Realistic walk** (default) | [Fine-tuned checkpoint](runs/experimental-gait/policy.zip) | A planted-foot animation follows the simulated trajectory. |
| **Original** | [Baseline checkpoint](runs/baseline-seed0/policy.zip) | The original PPO policy with its raw simulated joint pose. |

The planted-foot animation is applied at render time. Both policies use the same tabletop detector and route planner. See the [gait experiment](EXPERIMENT.md) for benchmark results, replay commands, and training instructions.

## Tests

```powershell
uv run python -m unittest discover -s tests
```

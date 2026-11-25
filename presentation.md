# Vision-Guided Reinforcement Learning for Humanoid Locomotion
## Presenter guide for the final review

### Macro perspective — the whole project in 45 seconds

“At the proposal stage, we had shown that our reinforcement-learning setup could train an agent, and that we could capture camera and screen frames. Our next question was whether a real scene could guide a simulated humanoid. Today we can point a camera at a tabletop, click a destination, and watch the humanoid follow a route around objects. The camera tells us where it can walk. A planner chooses the route. A PPO policy controls the joints inside MuJoCo. We trained two walking policies, tested them on the same courses, and separately worked on how the walk looks in the camera view.”

Slides 2–5 follow a click from the camera to a route. Slides 6–10 cover training, the policy replay, checkpoint selection and the held-out benchmark. Slide 11 compares early and current footage. Slide 12 hands over to the live demo.

### Micro perspective — slide by slide

#### 1. Vision-Guided Reinforcement Learning for Humanoid Locomotion

Introduce the project with the click-to-walk task before naming the algorithms.

The project title comes from our pitch. We use a camera to guide navigation and train the joint controller in simulation. The real tabletop supplies the scene; the humanoid is a simulated body.

#### 2. From the original pitch to the final demo

Explain how the team connected its pitch prototypes into one working demo.

The pitch milestones were capture and RL prototypes. Our final deliverable connects a physical scene to a simulated walking task and lets the user choose between the two checkpoints.

#### 3. What runs after you click?

Trace one click from the image to the waypoint and finally to a joint action.

The app rescans the scene at a nominal interval of 0.18 seconds. PPO runs on CPU and receives simulator observations. The camera and A* are engineered components. Navigation-learning trials tune speed and clearance separately.

#### 4. One camera view, two coordinate systems

Point to the same object in the camera view and in the rectangular map.

This controlled test image is generated. In a 960 × 540 synthetic boundary test, a uniform light frame produced no plane. Adding a 17-pixel dark strip across the top produced a fit. These are fixture results. We have no measured real-camera accuracy or calibration error. Snapshot mode can use four manually marked corners.

#### 5. Finding a route with room for the body

Follow the line across the three panels. The square is the obstacle and the dots mark the endpoints.

All three panels come from the same call to the project planner. The first is the rejected direct segment, the second is the reconstructed search path and the third is the simplified route. The clearance is in modeled table units. The live planner can retry tighter margins when the preferred clearance blocks every route.

#### 6. Two stages of PPO training

The two figures count simulator transitions. The second training stage starts from the first policy’s weights.

Original PPO logged 1,000,448 transitions. The obstacle fine-tune logged 491,520. Its reward also includes upright posture, action smoothness and a phase-dependent foot placement target. The +28 term rewards moving toward the current waypoint. Training and validation use a steering and balance harness.

#### 7. How the policies changed during training

Read each column vertically: reward above, episode length below. Compare trends within a stage because the reward function changes between stages.

Source: four baseline monitor CSVs and the obstacle-gait-phase training monitor. We sorted completed episodes by their logged end time and split each stage into ten equal-count groups. The x-axis is training order by episode decile; axes use each stage’s own units. An obstacle episode can end early on arrival as well as on failure.

#### 8. How training changed the walk

Let the clip play. Original PPO is on the left and Realistic walk is on the right. Ask the audience to watch how each body turns.

Both policies were trained with PPO. Original learned on Humanoid-v5. Realistic walk was fine-tuned on obstacle routes with stride-phase and steering inputs. The replay shows the physical simulation for each configuration, including its pace and assistance settings.

#### 9. Which checkpoint kept a foot on the ground?

Read the percentage as a share of sampled simulation steps. The remaining steps had neither foot touching the ground.

We compared four checkpoints from one training run. Pace multiplies the speed command. This is a selection set, so the 39.1% value is descriptive. The held-out benchmark uses different course seeds. Contact is calculated from physical foot-floor collisions at policy steps, before any display pose is applied.

#### 10. Both default policies reached every goal

Start with the two default configurations, then explain the three falls in the matched-settings control. SD is the spread of completion times among successful runs.

Seeds 81000–81059. Sample SD describes variation across finished courses. For the two defaults, we resampled 60 paired course time differences 20,000 times with seed 7 and used percentile endpoints. This interval describes variation across courses for these fixed policies. We did not repeat training across seeds. The matched baseline’s mean excludes its three falls. Support switches per second on finishes were 0.60 for Original and 5.28 for the fine-tune; rapid switches can include brief contact flicker.

#### 11. From the early trial to the trained walk

Play the two clips together. The phone recording is the earlier tabletop build and the right panel shows the current simulation on a different route.

Left: the requested 00:05–00:15 excerpt of the early phone recording. Right: a simulation replay from a different run. PPO learns the joint actions. At display time the renderer aligns the body with travel and holds the supporting foot steady. Heading-reward fine-tuning did not improve the held-out score.

#### 12. Live demo

Launch from the project directory with `uv run python tabletop_demo.py`. Turn on Guides, wait for the table outline and invite someone to choose a destination. After switching policies, click a new destination.

Invite someone to choose a clear spot beyond an obstacle. Open the live app, select the camera, enable Guides and wait for a tabletop outline. For a camera-free rehearsal use tabletop_demo.py --live-sample. The demo uses a simulated balance harness; the physical camera supplies the scene geometry.

### Questions to rehearse

- **Does PPO read the webcam?** The vision and planning pipeline consumes webcam pixels. The joint policy consumes MuJoCo state; the fine-tune also receives phase and steering inputs.
- **Why use a planner as well as RL?** The planner can explicitly enforce obstacle clearance in the measured map. PPO solves the continuous-action part of moving the body along the resulting route.
- **What does “realistic” mean here?** It is the name of the fine-tuned policy configuration and the display that steadies the supporting foot. The deck evaluates physical contact and shows the animation separately. No human motion-capture similarity or hardware transfer result was measured.
- **Can we use any floor?** Automatic live detection works best on a bright neutral plane with clear edges. Snapshot mode supports manual corner calibration and comparison of an empty reference with a later scene.
- **What evidence should we show if challenged?** Use the per-episode benchmark and overlay JSON, rather than drawing a conclusion from one successful clip.

### Evidence and reading

- Implementation in the project: `table_vision.py` (detection and projection), `tabletop_demo.py` (A* and clearance), `humanoid_ppo.py` and `train_gait.py` (training and evaluation), `avatar_gait.py` (display pose), `live_tabletop.py` (live integration), `navigation_learning.py` (settings trials).
- [Gait experiments and reproduction commands](EXPERIMENT.md)

# Vision-Guided Reinforcement Learning for Humanoid Locomotion
## Presenter guide for the final review

### Macro perspective — the whole project in 45 seconds

“At the proposal stage, we had shown that our reinforcement-learning setup could train an agent, and that we could capture camera and screen frames. Our next question was whether a real scene could guide a simulated humanoid. Today we can point a camera at a tabletop, click a destination, and watch the humanoid follow a route around objects. The camera tells us where it can walk. A planner chooses the route. A PPO policy controls the joints inside MuJoCo. We trained two walking policies, tested them on the same courses, and separately worked on how the walk looks in the camera view.”

Keep three comparisons straight. Slide 7 changes the trained policy and some assistance settings. Slide 8 keeps the policy and route fixed and changes how the pose is drawn. Slide 10 measures outcomes in the physics simulator. These answer three different questions.

**Suggested delivery:** Allow about nine minutes including the three short clips. One presenter can introduce the original question and explain the camera and planner (slides 1–5). The other can take training and evaluation (6–10). Hand back for the early recording and live demo. For the first handoff, ask “Once we have waypoints, how does the humanoid learn which joints to move?”

### Micro perspective — slide by slide

#### 1. Vision-Guided Reinforcement Learning for Humanoid Locomotion
**Say:** “This is the same project we pitched earlier. Here is what vision-guided locomotion looks like in our implementation.” Point to the cover figure. It is the project’s own simulated humanoid in mid-stride. The real webcam trial appears on slide 11. Leave algorithm names for the architecture slide.
**Handoff:** “At the first review, those components were still separate.”

#### 2. From capturing pixels to walking a route
The earlier CartPole run exercised the SB3 training pipeline at roughly a million steps across 128 environments. It established that the team could collect rollouts, train, save a checkpoint and evaluate it. Camera and screen capture were exploratory prototypes then. CartPole weights did not initialize the humanoid; the humanoid baseline was trained in its own environment.
**Say:** “The pitch asked how camera observations could guide locomotion. The architecture we could build and evaluate uses the camera for navigation and simulated state for joint control.” If asked about the original stand–balance–walk curriculum or Isaac GPU simulation, identify those as proposal directions. The completed runs used MuJoCo Humanoid-v5 and PPO.
**Handoff:** “Here is the boundary between the camera and the controller.”

#### 3. Where vision ends and joint control begins
Trace a single click. A pixel location on the screen becomes a tabletop coordinate. The route planner checks whether it can connect that position to the robot’s position. The next waypoint gives a direction and desired speed. MuJoCo supplies the body state. Stable-Baselines3 uses the policy to predict continuous joint commands, and the simulator advances the body. A separate steering and balance harness applies forces and torques. This is assisted locomotion in simulation.
**If asked what learns:** PPO learns joint control during training. Live navigation also compares a few fixed speed/clearance settings through practice and records outcomes per checkpoint in SQLite. Those practice trials do not update the PPO weights.
**Handoff:** “First we need a coordinate system that makes a click meaningful.”

#### 4. A tabletop becomes a walking map
The two views use the same controlled tabletop image. The green outline and orange boxes come from the project detector. Show the far edge first, then the red block in both views. The detector finds bright, fairly neutral spans, fits edges, estimates four corners and waits for repeated evidence before locking the plane. A homography maps that quadrilateral to a rectangle. After rectification, the detector estimates local background brightness by row and groups contrasting pixels into boxes.
**Qualification:** The method approximates each object’s footprint from appearance. It has no depth camera or object-class labels. White objects, deep shadows, clipped edges and a large box covering the near surface can all confuse detection. For the demo, use a white board with visible boundaries and a few dark objects. The project includes manual corner selection in snapshot mode.
**Handoff:** “Once the scene is on a plane, obstacle avoidance becomes a geometric problem.”

#### 5. A click behind an obstacle needs a detour
Follow the green path from left to right. This path was computed on the previous slide’s obstacle boxes. A straight line would pass too close to an object once body clearance is included. The planner tests direct visibility first, searches a four-neighbor grid at spacing 0.1 in course coordinates if needed, then removes intermediate grid points when a longer segment is clear. Wider clearance gives the body more room but can eliminate narrow passages.
**If asked about changes mid-trip:** Live detections are refreshed. If boxes change, the app can replan. A lost table pauses the trip. If a new obstacle blocks the walker and the algorithm cannot find safe clearance, it can stop and request another destination.
**Handoff:** “These waypoints say where the humanoid should go. They still don’t say which joints to move.”

#### 6. How we trained the two walking policies
PPO means Proximal Policy Optimization. The agent collects experience by acting in simulation. Each transition supplies an observation, action, reward and next observation. PPO uses batches of this experience to update a policy while limiting large changes in action probabilities. The data is generated by interaction, so there is no set of labeled webcam pictures teaching joint angles.
The original Humanoid-v5 checkpoint records 1,000,448 steps. The fine-tune started with baseline weights. Its observation includes a periodic stride clock and steering information alongside simulator state. During warm start, the added input weights begin at zero. Training then gives the policy a chance to use those features. Rewards favor progress, upright posture and consistent foot placement, with penalties for slipping, bouncing and abrupt action changes. Falling or hitting an obstacle ends an episode with a penalty.
**Explain the numbers:** The second figure is 491,520 fine-tuning steps. The million-step CartPole run mentioned on slide 2 is a different experiment. Both bundled humanoid policies use CPU inference.

#### 7. Original PPO and the obstacle fine-tune
**Cue:** In desktop PowerPoint Slide Show, the embedded eight-second excerpt is set to start when this slide appears. “Watch the timing of the steps as the two humanoids round the obstacle.” Original PPO is on the left. The obstacle fine-tune is on the right. Both follow the same course, but the fine-tune uses its selected pace and reduced vertical assistance. A single replay illustrates behavior and cannot isolate the effect of training by itself.
**If asked why the harness matters:** Changing lift can directly change how often feet contact the ground. That is why slide 10 includes the original policy running with the new assistance and pace.

#### 8. Watch the supporting foot through the turn
**Cue:** Let the eight-second embedded excerpt run in Slide Show. “This time the trajectory and the trained policy are the same. Look at the supporting foot as the body moves over it.” The raw pose is on the left. On the right a rendering layer selects a stance foot, holds its target at a ground coordinate, and moves the swing foot toward the next landing point. A damped inverse-kinematics solve adjusts the display rig’s leg joints to reach those targets.
The visual pose lives in separate MuJoCo data. Collision decisions, falls, and the benchmark’s contact measurements use the physical simulation. In the webcam overlay the sprite’s camera orientation also has to agree with the tabletop projection, especially for motion toward the viewer.
**Handoff:** “We evaluated the trained controller in physics as a separate question.”

#### 9. Which checkpoint should run in the demo?
These bars are validation results at four saved training checkpoints, all at a common 1.30× pace. The chart starts at zero. The percentages show how often at least one foot contacted the floor. All four finished the 16 validation routes at this pace. The 327,680-step value dips before later checkpoints improve; a longer run does not guarantee a better gait on every metric.
The full selection file also includes a 1.22× pace. Candidate selection considered completion and gait quality. The selected 491,520-step checkpoint at 1.30× had 39.1% contact on this validation batch. The 38.8% on the next slide is a separate result on fresh courses.

#### 10. Both default policies finished 60 of 60 courses
Read the first two rows together. On seeds 81000–81059 both default configurations finished all 60 courses with zero falls or obstacle contacts. Realistic walk averaged 8.81 seconds versus 9.54 for Original. Ground contact increased by 29.2 percentage points. “Foot contact” means at least one simulated foot was touching the floor on a physics step. The displayed animation does not enter that calculation.
Now read the third row. At matched lift and pace, Original finished 57 of 60 and fell three times. Its mean time of 8.35 seconds excludes the three failures. Comparing times alone would hide that tradeoff. Earlier testing on another 40 courses gave Original 40/40 and the fine-tune 39/40. In 24 simulated overlay trips per policy, including changing obstacles on 12, Original finished 24 and the fine-tune finished 23, stopping near an obstacle once.
**If asked about moving backward:** A separate 32,768-step heading-reward run penalized movement away from the torso’s forward direction. Both models finished 24/24 additional test routes. Average heading alignment was about 0.951 for the existing checkpoint and 0.949 for the candidate. The existing checkpoint stayed the default.
**Handoff:** “Those results are from controlled simulation. Here’s how a physical tabletop enters the system.”

#### 11. Before we fixed the walking pose
**Cue:** Play the ten-second excerpt (00:05 to 00:15 in the phone recording). Point out which way the torso faces and which way the humanoid travels. Give the audience a few seconds to watch before speaking.

**Say:** "This is an early run on our laptop. The character could move across the tabletop, but sometimes it faced sideways or backwards as it walked. That was the walking pose we wanted to improve."

#### 12. Live demo
**Handoff:** "Let's try the current version live. Pick a clear spot on the other side of the obstacle and we'll send the humanoid there."

For a live demonstration, run these commands from the humanoid-rl project directory:

```powershell
uv run python tabletop_demo.py
```

1. Connect the camera before launching. On Windows, FFmpeg with DirectShow support must be available. If using OBS, start its virtual camera first and frame the full tabletop.
2. Select the feed, leave Realistic walk selected, enable Guides, and wait for the outline.
3. Click a clear destination across an obstacle. Follow the route line before pointing at the gait.
4. To compare policies, switch to Original and click a destination again. Changing the policy clears the previous destination.
5. If the table fails to lock, launch the generated live feed with `uv run python tabletop_demo.py --live-sample`. The generated feed is a rehearsal mode.

### Questions to rehearse

- **Does PPO read the webcam?** The vision and planning pipeline consumes webcam pixels. The joint policy consumes MuJoCo state; the fine-tune also receives phase and steering inputs.
- **Why use a planner as well as RL?** The planner can explicitly enforce obstacle clearance in the measured map. PPO solves the continuous-action part of moving the body along the resulting route.
- **What does “realistic” mean here?** It is the name of the fine-tuned policy configuration and the default planted-foot display. The deck evaluates physical contact and shows the animation separately. No human motion-capture similarity or hardware transfer result was measured.
- **Can we use any floor?** Automatic live detection works best on a bright neutral plane with clear edges. Snapshot mode supports manual corner calibration and comparison of an empty reference with a later scene.
- **What evidence should we show if challenged?** Use the per-episode benchmark and overlay JSON, rather than drawing a conclusion from one successful clip.

### Evidence and reading

- Implementation in the project: `table_vision.py` (detection and projection), `tabletop_demo.py` (A* and clearance), `humanoid_ppo.py` and `train_gait.py` (training and evaluation), `avatar_gait.py` (planted pose), `live_tabletop.py` (live integration), `navigation_learning.py` (settings trials).
- Experiment folder: `C:\Users\Lenovo\Documents\ChatGPT\ML\gait-experiment`. Slide 7 uses `humanoid-gait-comparison.mp4`; slide 8 uses `realistic-walk-comparison.mp4`. Slides 9–10 use `validation-selection.json`, `policy-benchmark.json`, `evaluation.json` and `policy-overlay-benchmark.json`.
- Slide 11: `IMG_3040.MOV`, 00:05 to 00:15, cropped to the laptop display. Slides 4–5 use the controlled image `Three objects on a white tabletop.png` processed by the current project code.
- Methods referenced in the pitch: Schulman et al., *Proximal Policy Optimization Algorithms* (2017); Raffin et al., *Stable-Baselines3: Reliable Reinforcement Learning Implementations* (2021); Gymnasium, Farama Foundation; Todorov, Erez and Tassa, *MuJoCo: A Physics Engine for Model-Based Control* (2012).


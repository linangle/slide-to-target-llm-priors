# Arm controller (`arm.py`)

The iiwa now pushes the object itself. You give it a **target point** (where the object's centre should stop, world x in metres) and a friction value. It works out the paddle speed, plans the motion of each joint, and drives the arm in Drake.

## Try it

```bash
source .venv/bin/activate
python3 arm.py --target 0.75 --mu 0.3                  # plan with the true friction (the "reference" method)
python3 arm.py --target 0.75 --mu 0.3 --mu-est 0.25    # plan with a wrong estimate: watch it miss
python3 arm.py --target 0.75 --mu 0.3 --meshcat        # open http://localhost:7000, press play in the Animation panel
python3 arm.py --target 0.75 --mu 0.3 --plate          # the same plan, pushed by the old plate (for comparison)
python3 arm.py --reach                                 # where the arm can reach, and its fastest allowed push
python3 -m pytest tests/test_arm.py                    # 7 tests, ~10 s
```

The output starts with two lines that matter most:

```
target x = 0.750 m, true mu = 0.3, planned with mu = 0.3, J = 100 m/s^3 -> peak paddle speed u = 0.6633 m/s, release speed (model) 0.6200 m/s
  landed at x = 0.7498 m, y = -0.02 mm: miss -0.17 mm along the push
```

Below them is `push.py`'s usual report: the release read three ways, slide versus `v²/(2μg)`, and tracking. After that come the joint-limit lines (see *Limits*).

Flags: `--target`, `--mu` (the simulator's true friction), `--mu-est` (what the planner believes), `--J` (braking jerk), `--a-push`, `--x0` (object start), `--shape`, `--plate`, `--meshcat`, `--reach`.

## How it works

The code has three layers. Each one is a function you can call on its own.

**1. Target → peak paddle speed** (`plan_peak_speed`)

The object rides on the paddle until the paddle brakes harder than friction. It then slides `d = v_r² / (2 μ g)`. So the object's rest position is

```
x_rest(u) = x0 − gap + release_push_distance(u, μ, J)  +  v_r(u)² / (2 μ g),      v_r = u − (μg)² / (2J)
            └── release point (from physics.py) ──┘     └── the proposal's equation ──┘
```

The planner solves `x_rest(u) = x_target` for the peak speed `u` with a root finder. You can also pass `mu` as a belief: a dict with `grid` (values of ln μ) and `pmf` (probabilities), which is what `belief.posterior` returns. The planner then picks the `u` that minimises the expected squared miss, `E[(x_rest − x_target)²]`. This is the hook the LLM, default and lookup methods will use.

**2. Paddle line → joint trajectory** (`solve_ik`, `joint_trajectory`)

- `push.profile(u, J)` gives the paddle speed over time: accelerate at 2 m/s², then brake with jerk J.
- The paddle face has to follow that profile along +x. The face stays square to the push, its lower edge stays 3 mm above the table, and it stays centred on the object.
- Inverse kinematics (`InverseKinematics`, SNOPT) finds the start pose.
- From there, differential kinematics moves the face along the line, 0.1 ms per step: `q̇ = J⁺ (V_desired + K·error)`, with K = 200 /s. The planned face stays within about 3 µm of the line.
- Inverse dynamics (`CalcInverseDynamics`, minus gravity) turns `q, q̇, q̈` into a feed-forward torque for each joint.

**3. Driving the arm** (`ArmPusher`)

- Each of the 7 joints gets Drake's implicit PD (`set_controller_gains`). The PD tracks the planned `[q(t), q̇(t)]`, and the feed-forward torque goes on the actuation port.
- `ArmPusher` plugs in the same way as `push.Pusher`. Any script that calls `push_slide(..., pusher=...)` can use the arm instead of the plate. The one change in `push.py` (4 lines) lets a pusher wire up its own controller.
- `slide_to_target(x_target, mu, mu_est=...)` does all three layers and returns the result dict, with `miss`, `u` and `joint_limits` added.

## How well it works (μ 0.3, J 100, box)

| | plate (`push.py`) | arm |
|---|---|---|
| speed at release vs model | +0.13% | +0.14% |
| slide vs `v²/(2μg)` | −0.38% | −0.37% |
| paddle tracking, rms | 0.03 mm | 0.06 mm |
| landing miss, target 0.75 m, true μ | n/a | −0.2 mm |

The arm passes every bar in `tests/test_push.py`'s acceptance test (`tests/test_arm.py::test_arm_push_meets_the_plate_bars`). The full suite runs 49 tests, and all pass.

## Limits: read this before running the experiments

Drake does **not** enforce joint velocity limits, so the simulation "works" even when the real iiwa couldn't make the push. `arm.py` checks every plan against the iiwa 14's limits and prints a `WARNING` line when a plan goes over.

- **The elbow (joint 4, 75°/s) is the bottleneck.** The paddle has to move along a line through the robot's base, low over the table. Almost all of that motion comes from the elbow. Within the velocity limits the fastest push is **about 0.48 m/s** peak speed, at any J from 30 to 250.
- **Braking jerk (decision 7):** at 0.48 m/s the feed-forward torques stay within limits up to **J ≈ 250 m/s³**. At J = 300 the speed limit drops to 0.43 m/s, and at J = 1000 to 0.13 m/s. E0-4 found that losing less than 2% of the slide needs J ≈ 500 at μ 0.3, which this arm can't do in this layout.
- **What that means for targets:** at 0.48 m/s the object stops at most about 11 cm past its start. That's x ≈ 0.66 m at μ 0.3 and 0.73 m at μ 0.1, on a board that runs to 1.52 m. The E0 pushes ran at 0.6–1.2 m/s.
- **Reach:** the face can travel from x ≈ 0.50 to 0.92 m. At the start pose (x0 = 0.55 m) the elbow is at 98% of its joint limit.

**One thing worth testing:** moving the robot base so it pushes from the side instead of straight out from its base, e.g. base at (0.3, −0.5) turned 57°. Then joint 1 helps the motion. In a quick IK check, peak joint use at 0.6 m/s fell from 1.29× to 0.48× the limit. That changes `scene.add_arm`, so it's a team decision. The proposal's fallback also applies: plan over a small set of validated speeds.

## Validation runs: repeatability and targeting (`e1_arm_validation.py`)

This is the evidence for the Nov 6 milestone ("repeatable robot pushes, achievable release speeds, targeting accuracy with known friction"):

```bash
python3 e1_arm_validation.py --quick     # 5 pushes, ~5 s: check it runs
python3 e1_arm_validation.py             # 85 pushes, ~40 s on 4 workers -> e1_arm_results.csv, e1_arm_summary.txt, e1_arm_summary.png
python3 e1_arm_validation.py --summarize e1_arm_results.csv   # reprint the summary from a saved run
```

It has three parts.

- **A, arm vs plate.** 12 pushes spread over μ 0.1–0.5, v_peak 0.6–1.0 m/s, J 30–300, box and disc, as the team decided on 5 Nov. Each one runs with the arm and with the plate. **All 12 pass:** the speed at release is within 0.5% of the model, the slide is within 1% of v²/(2μg), and the arm's landing is within 0.5 mm of the plate's. Tracking is 0.03–0.10 mm rms, except 0.26 mm at J = 300.
- **B, targeting with the true μ.** Targets from x = 0.60 to 0.90 m, at μ 0.1, 0.3 and 0.5. **All 21 land within 3.3 mm, with an RMS miss of 0.9 mm.** The 6 targets the iiwa can reach within its joint limits land within 1 mm.
- **C, repeatability under disturbances the planner doesn't know about.** 5 cases × 8 trials, each with the object 0.5–3 mm off its assumed start, the paddle 3 mm off-centre, and the mass ±10%. **Every miss is within 2.6 mm, with a standard deviation of 0.6–1.0 mm.** Lateral drift stays under 0.1 mm. Most of the miss comes from the unknown start gap, which shifts the release point millimetre for millimetre.

Every row records `v_lim_frac`, the peak joint speed divided by the limit. Only 22 of the 73 arm pushes are within the real iiwa's limits; see *Limits* above. `tests/test_e1_arm.py` checks the job list and runs one push end to end.

## Where to change things

- Gains: `ArmPusher(kp=..., kd=...)`
- Start pose seed: `Q_SEED`
- Line-following gain: `joint_trajectory(K=...)`
- Reach used for planning: `REACH_X`
- Turn off the feed-forward torque: `ArmPusher(feedforward=False)`

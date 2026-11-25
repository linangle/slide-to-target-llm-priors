# Arm controller (`arm.py`)

The iiwa pushes the object itself. You give it a **target point** (where the object's centre should stop, world x in metres) and a friction value. It works out the paddle speed, plans the motion of each joint, and drives the arm in Drake.

> **16 Nov: two controllers merged into one.** Diego and Angelina each wrote an `arm.py` on 6 Nov. This file now describes the merged one:
> - **Kept from Diego:** the target planner (`rest_x`, `plan_peak_speed`, `max_peak_speed`, `limited_peak_speed`), `slide_to_target`, `report`, `joint_limit_report`, the `--target` / `--mu-est` / `--plate` / `--reach` commands, and the validation runs in `e1_arm_validation.py`. Names and arguments are unchanged, so scripts written against his `arm.py` run as they are.
> - **Replaced:** his differential-IK trajectory (`solve_ik`, `joint_trajectory`, `Q_SEED`, `REACH_X`) and his `ArmPusher`. The joints are now driven by Angelina's controller (multi-start IK posture, a spline over the push line, the same implicit PD with inverse-dynamics feed-forward). It gives the same answers: reach to x = 0.92 m, 0.49 m/s within the joint speed limits, the same joint-speed ratios on every E1 row.
> - **Changed in the planner:** with a *belief* over μ, `plan_peak_speed` now calls `decide.choose_speed` (see *Planning with a belief*). With a single μ it returns exactly what it did before.
>
> The controller itself is described in the README (*The arm controller*).

## Try it

```bash
source .venv/bin/activate
python3 src/arm.py --target 0.75 --mu 0.3                  # plan with the true friction (the "reference" method)
python3 src/arm.py --target 0.75 --mu 0.3 --mu-est 0.25    # plan with a wrong estimate: watch it miss
python3 src/arm.py --target 0.75 --mu 0.3 --meshcat        # open the printed URL, press play in the Animation panel
python3 src/arm.py --target 0.75 --mu 0.3 --plate          # the same plan, pushed by the old plate (for comparison)
python3 src/arm.py --reach                                 # how far the arm can push, and its fastest allowed push
python3 src/arm.py --mu 0.3 --v 0.8 --J 100                # no target: one push at a given peak speed
python3 src/arm.py --feasibility                           # joint-speed and torque ratios over speed and jerk (--scan: over base positions)
python3 -m pytest tests/test_arm.py                    # 10 tests, ~15 s
```

With `--target` the output starts with the two lines that matter most:

```
target x = 0.750 m, true mu = 0.3, planned with mu = 0.3, J = 100 m/s^3 -> peak paddle speed u = 0.6633 m/s, release speed (model) 0.6200 m/s
  landed at x = 0.7499 m, y = -0.00 mm: miss -0.11 mm along the push (model with the true mu predicted 0.7500 m)
```

Below them is `push.py`'s usual report: the release read three ways, slide versus `v²/(2μg)`, and tracking. After that come the joint-limit lines (see *Limits*) and the controller's plan.

Flags: `--target`, `--mu` (the simulator's true friction), `--mu-est` (what the planner believes), `--v` (peak speed when there is no target), `--J` (braking jerk), `--a-push`, `--x0` (object start), `--shape`, `--plate`, `--meshcat`, `--reach`, `--base-x` / `--base-y` (where the iiwa's base stands), `--a-max` (brake cap), `--posture` (how the start posture is chosen: `speed`, the default, or `limits`).

## How it works

The code has three layers. Each one is a function you can call on its own.

**1. Target → peak paddle speed** (`plan_peak_speed`)

The object rides on the paddle until the paddle brakes harder than friction. It then slides `d = v_r² / (2 μ g)`. So the object's rest position is

```
x_rest(u) = x0 − gap + release_push_distance(u, μ, J)  +  v_r(u)² / (2 μ g),      v_r = u − (μg)² / (2J)
            └── release point (from physics.py) ──┘     └── the proposal's equation ──┘
```

With a single friction value, the planner solves `x_rest(u) = x_target` for the peak speed `u` with a root finder. It raises `ValueError` if the target is behind the object or needs more than `u_max`.

**Planning with a belief.** You can also pass `mu` as a belief: a dict with `grid` (values of ln μ) and `pmf` (probabilities), which is what `belief.posterior` returns. This is the hook the LLM, default and lookup methods use.

`decide.py` plans a *slide length* d\*, measured from the release point. A target point is the same problem with a d\* that depends on the speed, because a faster push carries the object further before it lets go:

```
d*(u, μ) = x_target − x_r(u, μ)          (decide.point_target)
```

So `plan_peak_speed` on a belief is `decide.choose_speed(belief, Release(J), point_target(x_target, x0, gap), loss, u_max)`. There is one decision rule and one loss for slide targets and point targets.

- The default `loss` is `decide.loss_sq_rel`, the specification's.
- `loss=decide.loss_sq_abs` is the expected squared miss `E[(x_rest − x_target)²]`, which is what this planner minimised before the merge. It reproduces the old speeds to four digits.
- The two differ by 0.5–6 mm/s on the cases tried (targets 0.60–0.64 m, belief widths 0.2–0.5 in ln μ). The relative loss is the slower one for a wide belief.
- Do not pass a point target to `choose_speed` as a fixed d\* taken from some nominal speed's release point. The rule then picks a slower push, which releases earlier, and lands about 2 cm short (μ 0.3, target 0.75 m, belief width 0.3). `tests/test_arm.py` pins this.

**`u_max`.** The default is `max_peak_speed`: the fastest push whose whole stroke stays inside the arm's reach. That is a reach limit only. `limited_peak_speed` is the fastest push within the joint *speed* limits (add `which=("v", "tau")` for torque too). Pass it as `u_max` to keep plans physically deliverable. Both take `base=`.

**2. Paddle line → joint trajectory** (`ArmModel`, `ArmPusher.make_plan`)

- `push.profile(u, J, a_max=12)` gives the paddle speed over time: accelerate at 2 m/s², then brake with jerk J down to a 12 m/s² cap.
- The paddle face has to follow that profile along +x. The face stays square to the push, its lower edge stays 3 mm above the table, and it stays centred on the object.
- Inverse kinematics from 42 seeds gives the candidate start postures (8 distinct ones at the default start). Each is integrated along the whole line. The planner keeps the one with the lowest peak joint speed that stays inside the joint limits and away from singularities. With `posture="limits"` (19 Nov, off by default) it plans the push with each of them and keeps the one whose worst ratio of joint speed, torque and acceleration to its limit is lowest; the posture then differs between slow and fast pushes.
- The path `q(s)` is a spline over the paddle's displacement. The profile turns it into `q(t)`, `q̇(t)`, `q̈(t)`.
- Inverse dynamics (gravity, Coriolis, inertia including the rotors) gives the feed-forward torque for each joint.

**3. Driving the arm** (`ArmPusher`)

- Each of the 7 joints gets Drake's implicit PD (`set_controller_gains`). The PD tracks the planned `[q(t), q̇(t)]`, and the feed-forward torque goes on the actuation port. The plant clamps the torque at the iiwa's effort limits.
- `ArmPusher` plugs in the same way as `push.Pusher`. Any script that calls `push_slide(..., pusher=...)` can use the arm instead of the plate.
- `pusher.traj` is the planned joint trajectory after a push, and `joint_limit_report(pusher.traj)` gives each joint's peak position, speed and torque as a fraction of its limit.
- `slide_to_target(x_target, mu, mu_est=...)` does all three layers and returns the result dict, with `miss`, `u`, `joint_limits` and `over_limits` added.

## How well it works (μ 0.3, J 100, box)

| | plate (`push.py`) | arm |
|---|---|---|
| speed at release vs model, 0.8 m/s | +0.13% | +0.11% |
| slide vs `v²/(2μg)`, 0.8 m/s | −0.38% | −0.20% |
| paddle tracking, rms | 0.03 mm | 0.05 mm |
| landing miss, target 0.75 m, true μ | n/a | −0.2 mm |

The arm passes every bar in `tests/test_push.py`'s acceptance test (`tests/test_arm.py::test_one_push_with_the_arm_meets_the_plate_bar`). The full suite runs 55 tests, and all pass.

## Limits: read this before running the experiments

Drake does **not** enforce joint velocity limits, so the simulation "works" even when the real iiwa couldn't make the push. `slide_to_target` checks every plan against the iiwa 14's limits and `report` prints a `WARNING` line when a plan goes over.

- **The elbow (joint 4, 75°/s) is the bottleneck.** With the base at the origin the paddle moves along a line through the robot's base, low over the table. Almost all of that motion comes from the elbow. Within the velocity limits the fastest push is **0.49 m/s** peak speed (0.48–0.49 for every J from 30 to 2000).
- **Braking jerk (decision 7):** with the brake capped at 12 m/s² the torque stays within limits at that speed for every J up to 2000 m/s³. The cap does not change the release, because the object has already left. Jerk is not what limits this arm; joint speed is.
- **What that means for targets:** at 0.49 m/s the object stops at most about 11 cm past its start with μ 0.3 (x = 0.66 m), 19 cm with μ 0.1 (0.74 m) and 10 cm with μ 0.5 (0.65 m), on a board that runs to 1.52 m. The E0 pushes ran at 0.6–1.2 m/s.
- **Reach:** the face can travel 0.42 m, from x = 0.50 to 0.92 m. `python3 src/arm.py --reach` prints it for any base.
- **Moving the base fixes the speed.** With the base beside the board at (0.6, 0.65) m, joints 1 and 2 sweep the paddle instead of the elbow. The reach grows to 0.68 m and the joint limits allow **1.02 m/s**, which puts the object at x = 1.01 m with μ 0.3. Where the base stands is still a team decision (README, *Decisions*). Every function here takes `base=`.

## The side base: layout, clearance, speed cap, targets, acceleration (default since 16 Nov)

The default base now stands beside the board at (0.6, 0.65) m. The layout before that, with the base at the origin, is still there as `--layout origin` (`base=LAYOUTS["origin"]` in code), and its saved results are untouched in `results/e1/`, `results/arm_rerun/` and `results/evaluation_v3/`. Results at the default layout are in `results/side_base/`.

```bash
python3 src/arm.py --reach                        # the default (side) base: reach, speed bounds, what they afford
python3 src/arm.py --reach --layout origin        # the same for the layout before 16 Nov
python3 src/arm.py --target 0.94 --mu 0.26 --a-max 4      # one push inside speed, torque and acceleration limits
python3 experiments/arm_clearance.py              # do the links, the base and the camera stay clear? (--layout origin to compare)
python3 experiments/e1_arm_validation.py          # -> results/side_base/e1_arm_*   (--layout origin -> results/e1/)
python3 experiments/arm_rerun.py                  # -> results/side_base/arm_rerun_* (--layout origin -> results/arm_rerun/)
```

**Clearance** (`experiments/arm_clearance.py`, from the convex hulls of the links' visual meshes along the planned push at the speed cap; `results/side_base/arm_clearance.txt`). The scene loads the iiwa without collision geometry, so the simulation itself would not show a link passing through the board.

| | side base | origin |
|---|---|---|
| lowest moving link above the board | link 5, 51 mm | link 5, 55 mm |
| nearest link to the object (the paddle aside) | link 7, 53 mm | link 7, 53 mm |
| nearest any link comes to the camera | 0.7 m | 0.7 m |
| base footprint against the board | reaches y = 0.53 m | 179 mm before the near edge |

The footprint is the catch: a 1.2 m wide board centred on the push line has its edge at y = 0.60 m, 71 mm under the base. Moving the base outward does not work (at y = 0.75 m the stroke drops from 0.68 to 0.48 m and the arm is near the edge of its workspace; at 0.80 m it cannot reach the line). So the board moves instead: with the arm at the side base the scene shifts the board 10 cm away from it (`scene.BOARD_Y`: centre line at y = −0.10 m, edges at −0.70 and +0.50 m), which clears the base by 29 mm. The push line stays at y = 0. Scenes without an arm, with the prescribed plate, or with the arm at the origin keep the board centred, so their saved results reproduce exactly.

**Layout.** `scene.LAYOUTS` (also `arm.LAYOUTS`) names two base positions: `side` (0.6, 0.65) m, the default, and `origin` (0, 0). With the base beside the board the paddle sweeps across the arm's front, so joint 1 does the work instead of the elbow.

| | origin | side |
|---|---|---|
| stroke along the push line | 0.42 m (to x = 0.92 m) | 0.68 m (to x = 1.18 m) |
| fastest push within joint speed and torque limits | 0.49 m/s | 1.02 m/s |
| longest slide at μ 0.1 / 0.3 / 0.5 | 12 / 3 / 1 cm | 53 / 16 / 8 cm |
| E1: arm pushes within joint speed limits | 22 of 73 | 73 of 73 |
| E1: targeting RMS miss, worst (part B) | 0.74, 2.4 mm | 0.83, 2.7 mm |
| arm re-run: pushes within speed and torque limits | 0 of 12 | 12 of 12 |
| arm re-run: release speed vs the plate's | −0.13..−0.01% | −0.13..+0.01% |
| E0-5 through the camera: worst bias of A/B/C, missed frames | 0.90%, 0 (plate, original camera) | 2.2%, 0 (arm in view, full-coverage camera, height gate; 0.45% without the first 30 ms) |

The arm is in the camera's view at the side base, always behind the paddle face. What matters is the paddle's own front face, which a camera ahead of the paddle sees; the README (*The side base*) and `experiments/camera_study.py` cover the camera position and the height gate that deals with it.


**Speed cap.** `limited_peak_speed(J, x0, base=..., which=("v", "tau"))` is the fastest push inside the checked limits. `plan_peak_speed(..., u_max=cap)` clips a belief's plan at it; `clip=True` does the same for a single friction value, so a target the arm cannot reach is pushed at the cap and scored as a miss.

**Targets.** `affordable(u_max, mu)` gives the longest slide and farthest rest point a cap allows. `target_grid(u_max, mu_hi, n)` spaces n targets from x0 + 5 cm to what the highest-friction pair reaches at 90% of the cap. For Y01–Y03 at the side base that is x = 0.60, 0.77 and 0.94 m; at the origin's 0.45 m/s cap the farthest is 0.64 m.

**Acceleration.** `joint_limit_report` now has an `"a"` entry: planned joint acceleration over the SDF's `drake:acceleration` limits (8.6–15.7 rad/s²). KUKA publishes no acceleration limits, so these are Drake's values, and the simulator does not enforce them. What sets the acceleration is the brake cap (`ArmPusher.a_brake_max`, 12 m/s² by default), not the speed:

| brake cap | releases friction below | fastest push within speed, torque and acceleration: origin | side |
|---|---|---|---|
| 4 m/s² | 0.41 | 0.49 m/s | 1.02 m/s |
| 6 m/s² | 0.61 | 0.11 m/s | 0.84 m/s |
| 8 m/s² | 0.82 | 0.10 m/s | 0.59 m/s |
| 12 m/s² (default) | 1.22 | 0.10 m/s | 0.36 m/s |

- The cap has to stay above μg, or the object never leaves the paddle. So a soft brake only works when the friction is known to be low enough.
- With the default 12 m/s² every push so far has been over the acceleration limits, at either base (E1 at the side base: 5 of 73 within; with a 6 m/s² cap: 65 of 73).
- A softer brake does not change the release, so accuracy is the same (E1 with a 6 m/s² cap: same misses to 0.01 mm).
- The table is for the default posture choice. With `--posture limits` the 6 m/s² row at the side base becomes 1.16 m/s (every speed from 0.15 to 1.14 m/s checked on a 0.02–0.05 m/s grid), and E1 has 73 of 73 pushes within all three limits (19 Nov, `results/side_base/base_trial/`).

## Validation runs: repeatability and targeting (`e1_arm_validation.py`)

This is the evidence for the Nov 6 milestone ("repeatable robot pushes, achievable release speeds, targeting accuracy with known friction"). The numbers below are from the 16 Nov rerun on the merged controller, with the base at the origin.

```bash
python3 experiments/e1_arm_validation.py --quick     # 5 pushes, ~10 s: check it runs
python3 experiments/e1_arm_validation.py             # 85 pushes, ~40 s on 4 workers -> results/e1/e1_arm_results.csv, results/e1/e1_arm_summary.txt, results/e1/e1_arm_summary.png
python3 experiments/e1_arm_validation.py --summarize results/e1/e1_arm_results.csv   # reprint the summary from a saved run
python3 experiments/e1_arm_validation.py --base-x 0.6 --base-y 0.65 --out results/e1/e1_arm_side   # the same run with the base beside the board
```

It has three parts.

- **A, arm vs plate.** 12 pushes spread over μ 0.1–0.5, v_peak 0.6–1.0 m/s, J 30–300, box and disc, as the team decided on 5 Nov. Each one runs with the arm and with the plate. **All 12 pass:** the speed at release is within 0.6% of the model, the slide is within 1.7% of v²/(2μg), and the arm's landing is within 1.3 mm of the plate's. Tracking is 0.04–0.09 mm rms.
- **B, targeting with the true μ.** Targets from x = 0.60 to 0.90 m, at μ 0.1, 0.3 and 0.5. **All 21 land within 2.4 mm, with an RMS miss of 0.7 mm.** The 6 targets the iiwa can reach within its joint limits land within 0.4 mm.
- **C, repeatability under disturbances the planner doesn't know about.** 5 cases × 8 trials, each with the object 0.5–3 mm off its assumed start, the paddle 3 mm off-centre, and the mass ±10%. **Every miss is within 2.8 mm, with a standard deviation of 0.6–1.1 mm.** Lateral drift stays under 0.1 mm. Most of the miss comes from the unknown start gap, which shifts the release point millimetre for millimetre. If the evaluation scores the slide from the release point that forward kinematics reports, instead of the rest position on the board, that part cancels.

Every row records `v_lim_frac`, the peak joint speed divided by the limit. Only 22 of the 73 arm pushes are within the real iiwa's limits; see *Limits* above. `tests/test_e1_arm.py` checks the job list and runs one push end to end.

## Where to change things

- Base position: `ArmPusher(base=(x, y, 0))`, and `base=` on the planner functions
- Gains: `ArmPusher(kp=..., zeta=...)`
- Brake cap: `ArmPusher(a_brake_max=...)`
- Start posture search: `ArmPusher(posture_seed=...)`
- Start posture choice: `ArmPusher(posture="speed")` (default) or `"limits"`. With `a_brake_max=6.0` and `"limits"` every push from 0.15 to 1.14 m/s at the side base is inside speed, torque and acceleration limits (`results/side_base/base_trial/`)
- Loss for a belief: `plan_peak_speed(..., loss=decide.loss_sq_abs)`
- Speed cap: `plan_peak_speed(..., u_max=limited_peak_speed(J, x0))`

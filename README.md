# Slide-to-Target

MIT 6.4210 (Robotic Manipulation) team project, Fall 2026.

A simulated KUKA iiwa pushes a flat-bottomed object across a table and releases it so it slides to a target distance, `d = v² / (2 μ g)`. Friction `μ` is unknown. We ask a language model for a prior over `μ` from a text description of the object and the table, update it with a few practice slides (Bayesian, in `ln μ`), and choose the release speed that minimises expected landing error. The main question is how much the language-model prior is worth compared with a published friction table, practice slides alone, and the true friction.

## Status

Phase 1, Oct 25 – Nov 1: measuring ground-truth kinetic friction for 27 object–surface pairs (M01–M27) and 3 validation pairs (V01–V03). Output: `measured_friction.csv` (columns in `measured_friction_interface.md`). On 31 Oct the model specification, estimators, grid posterior, decision rule, Drake configuration and E0-1/E0-2 checks landed in response to the instructor's feedback; `model_specification.tex` is the reference note. On 5 Nov the team settled the open decisions (see *Decisions* below); decisions 1, 2, 3, 5, 6 and 8 are applied in the code and data, the rest are recorded there. On 5 Nov a prescribed-paddle push harness (`push.py`) stood in for the arm controller so that E0-3, E0-4 and E0-5 could run; all three pass, and decisions 7 and 8 now have their evidence. The same day the decision-10 SimReady audit (`simready_audit.md`) found that NVIDIA's library stores one friction pair per material class, so a C2-SimReady baseline is possible but coarse. On 6 Nov the arm controller landed (`arm.py`): the iiwa tracks the plate's push profile under Drake's implicit PD with inverse-dynamics feed-forward, passes the plate's acceptance test, and matches the plate's release speed within 0.13% on the 12-push re-run (`arm_rerun.py`). The decision-7 numbers came with it: the braking jerk is not the constraint, the joint speed is. With the base at the origin every grid push runs joint 4 at 1.3–3× its 75°/s limit (which the simulator does not enforce), so the real iiwa would deliver 0.35–0.47 m/s on that line; with the base beside the board, level with the middle of the push, it delivers 1.0 m/s within its limits. Where the base stands is the team's call. On 16 Nov the two arm controllers written on 6 Nov (Angelina's and Diego's, both `arm.py`) were merged into one file: Angelina's joint controller drives the arm, Diego's target planner and E1 validation run on top of it, and point targets now go through `decide.py`'s rule (see *The 16 Nov merge*). The same day Alejandro's evaluation branch was merged: the LLM query and cache, the generic / lookup / LLM / oracle comparison after 0, 1 and 2 practice slides, and the saved pilot results (see *The evaluation pipeline*). The files were then grouped into folders (see *Repository layout*): after pulling, run `pip install -r requirements.txt` once more. The side base then became the default layout, with a computed speed cap, derived targets, an acceleration check and a camera that covers the whole board; the origin layout and all its results are kept (see *The side base*).

## The arm controller (6 Nov)

> **The numbers in this section were measured with the base at the origin.** That layout is now `--layout origin`; since 16 Nov the default base stands beside the board (*The side base*), where the same checks were rerun. The saved origin results are unchanged in `results/e1/`, `results/arm_rerun/` and `results/evaluation_v3/`.

The iiwa now pushes. `arm.py`'s `ArmPusher` has the plate's hook (`add`, `connect`, `q`, `qd`, `face_x`, `set_displacement`, `record`, `summary`), so `push_slide(..., pusher=ArmPusher())` runs the same settle → push → brake → slide with the arm, reads the release the same three ways and reports the same scalars, plus what the push asked of the iiwa. `python3 src/arm.py --mu 0.3 --v 0.8 --J 100` prints one push (`--meshcat` to watch; `--base-x`, `--base-y` move the base).

**How it works.**
- The push is a straight line, so the paddle pose fixes link 7's pose (the plate hangs 5 cm below the flange with its face along link-7 +z, as `scene.add_arm` welds it) and the task is a 6-DoF frame path with one redundant joint. `ArmModel` is a kinematics-and-dynamics copy of the arm (no contact). The start posture is picked among multi-start IK solutions by integrating each along the whole path with the minimum-norm joint tangent `dq/ds = J⁺ eₓ` and keeping the one with the smallest peak joint-speed ratio that stays inside the joint limits and away from singularities (6 of 8 solutions cover the acceptance push). `q(s)` is a Hermite spline over the displacement; the profile gives `q_d = q(x)`, `q̇_d = q'(x) v`, `q̈_d = q''(x) v² + q'(x) a`.
- Each joint runs Drake's implicit PD (`JointActuator.set_controller_gains`, kp = 4 × 10⁴ N m/rad, critically damped against the mass-matrix diagonal) on `(q_d, q̇_d)` fed one time step ahead, as the plate is, with the arm's inverse dynamics along the desired trajectory (gravity, Coriolis, inertia including the reflected rotor inertia, 160² × 1.3 × 10⁻⁴ = 3.4 kg m² per joint) as the feed-forward torque. The object's contact force (≤ 42 N) is left to the PD. The plant clamps the total torque at the SDF effort limits (320, 320, 176, 176, 110, 40, 40 N m), so a brake the iiwa cannot deliver appears as a tracking error, not as an unphysical push.
- `push.profile` gained `a_max`: the arm's brake ramps at jerk J down to −12 m/s² (`ArmPusher.a_brake_max`) and then holds that deceleration to the stop. The release is unchanged because 12 m/s² > μg for every pair (the object has left before the cap is reached; `tests/test_arm.py` checks `release_from_profile` and `release_push_distance` on the capped profile). The plate still brakes uncapped, as in E0-3/E0-4.
- `q(ctx)` and `qd(ctx)` are the paddle body's x displacement and speed from forward kinematics, so the release model is checked against the arm's actual motion, and `x_fk` is the face from FK as for the plate. `record` adds the joint positions, speeds and net torques to the trajectory; `summary` adds the peak joint-speed and torque ratios, the effective mass along the push, and the paddle's drift off its line while in contact.

**The bar, met.** The plate's acceptance test passes with the arm (`tests/test_arm.py`, same assertions as `tests/test_push.py`): at μ 0.3, 0.8 m/s, J 100 the object leaves at +0.11% of the release model (+0.06% on the arm's recorded motion), the force vanishes 1.6 ms after t*, the slide is within −0.20% of `v²/(2μg)`, release heading 0.001°, FK release point exact to 0.01 mm, paddle tracking 0.05 mm rms (max 0.22 mm, the first step of the acceleration), z drift 0.05 mm and tilt 0.008° while in contact, no torque clamping.

**The 12-push re-run** (`python3 experiments/arm_rerun.py`, 24 pushes, ~15 s on 6 workers → `results/arm_rerun/arm_rerun_results.csv`, `results/arm_rerun/arm_rerun_summary.txt`, `results/arm_rerun/arm_rerun_summary.png`): eight cases from the E0-4 grid spanning μ 0.1–0.5, v 0.6–1.0 and J 30–2000, and four from E0-3 (disc, COM at w/4, plate yaw 2°, paddle friction 0.6), each run with the plate and with the arm. The arm's speed at t* is within −0.13..−0.01% of the plate's in every case and within 1% of the release model in 11 of 12; the twelfth is E0-4's known worst case (μ 0.5, 0.6 m/s, J 30), +1.13% with the arm and +1.18% with the plate. Tracking is 0.045–0.086 mm rms (plate 0.029–0.050), the rest position agrees with the plate's within −0.4..+1.9 mm, the FK release point is exact except for the yawed plate (+2.1 mm, the same geometric offset the plate has). The one new number the plate could not supply is the arm's tracking error against the profile: ≤ 0.09 mm rms, ≤ 0.34 mm peak.

**What the iiwa can deliver (decision 7).** `python3 src/arm.py --feasibility` plans the pushes without contact and reports the peak joint-speed and torque ratios to the SDF limits; `--scan` repeats it for bases beside the board (y = 0.65 m is just off the 1.2 m wide board). Peak joint speed / limit, J = 100 (J = 2000 is within 0.05 of it):

| iiwa base (x, y), m | 0.6 m/s | 0.8 m/s | 1.0 m/s | paddle speed the joint limits allow |
|---|---|---|---|---|
| (0, 0), the scene's | 1.29 (joint 4) | 1.95 (joint 4) | 3.10 (joint 4) | 0.32–0.47 m/s |
| (0.3, 0.65) | 0.51 | 1.23 (joint 1) | 2.50 (joint 4) | 0.40–1.27 m/s |
| (0.45, 0.65) | 0.49 (joint 1) | 0.68 (joint 1) | 1.26 (joint 1) | 0.79–1.22 m/s |
| (0.6, 0.65) | 0.46 (joint 4) | 0.63 (joint 1) | 0.96 (joint 1) | 1.04–1.31 m/s |

- **Speed.** With the base on the push line the wrist moves radially from the shoulder, which only the elbow can produce (`d ḋ = l₁ l₂ sin θₑ θ̇ₑ`, with joint 4's limit 75°/s), so the paddle is capped near 0.45 m/s whatever the posture or lateral offset of the base (y = 0.2–0.5 m at the scene's x tried too: no better). Beside the board, level with the middle of the push, joints 1 and 2 sweep the paddle across the arm's front: at (0.6, 0.65) m the iiwa delivers 1.0 m/s at 0.96 of joint 1's limit, tracking 0.07 mm rms in simulation (release +0.10%), with the effective mass at the paddle 13 kg instead of 24–34 kg. The simulator enforces torque limits but not speed limits, so the frontal pushes in the re-run are physically consistent but run joint 4 at 1.3–3× its rated speed; they stand as the comparison with the plate, not as something the real iiwa would do.
- **Jerk.** The release needs the ramp to reach −μg, which takes μg/J (49 ms at J = 100 and μ 0.5, 2.5 ms at J = 2000) and costs torque set by the deceleration reached, not by J: with the 12 m/s² cap the torque ratio is 0.83–0.89 at 0.8 m/s frontal for every J in the E0-4 table, 0.43–0.66 beside the board. So the iiwa delivers J = 2000 m/s³ (ramp loss ≤ 2% of the distance at μ ≤ 0.5, `e0_4_summary.txt`), and the release model holds there (−0.16..−0.34% in the re-run and the beside-the-board check). The 5 Nov reading (brake hard) is available: `decide.py` still carries J = 100 m/s³, a one-line change once the team confirms the speed and the base. At 1.0 m/s frontal the brake needs 1.08–1.37× joint 4's torque; the simulator clamps it after the release, which the release does not see (+0.06%, −0.19% in cases 5 and 7).
- **Acceleration (checked since 16 Nov).** The SDF's `drake:acceleration` limits (8.6–15.7 rad/s²) are exceeded 2–4× in every push made with the default 12 m/s² brake cap, at either base. The overshoot comes from that cap, not from the speed: with a 4 m/s² cap the plan is inside them (see *The side base*). They are Drake's model values (KUKA publishes none) and the simulator does not enforce them. **Not modelled:** the command runs at the 1 ms plant step; a real iiwa's 4 ms FRI cycle, joint friction and sensor noise are not in the model.

**Next steps for whoever picks this up.**
1. ~~Decide where the base stands.~~ Done 16 Nov: the default base is beside the board at (0.6, 0.65) m (`scene.LAYOUTS`, *The side base*); `--layout origin` and `base=LAYOUTS["origin"]` keep the old layout runnable.
2. Then rerun `arm_rerun.py` at that base (set the base in `run()`), and E0-5 with the arm: `push_slide(..., ArmPusher())` for the pushes and `pusher.set_displacement(ctx, q)` for the camera replay both exist, but E0-5 has only been run with the plate.
3. Decision 7 in `decide.py`: J = 100 → the value the team picks (2000 is deliverable).
4. If the frontal layout stays, say so in the report: the simulated iiwa then exceeds its joint-4 speed rating by up to 3×.

## The 16 Nov merge: one `arm.py`

Diego pushed his own `arm.py` on 6 Nov, a few hours after Angelina's was committed locally, and Alejandro's evaluation branch (`alejandro/bayesian-integration`) is built on Diego's. The merge keeps both pieces of work in one file. `ARM_CONTROLLER.md` (Diego's guide, updated) is the how-to.

**What is where.**
- **Controller: Angelina's.** `ArmModel`, `ArmPusher`, `feasibility`, `--feasibility`, `--scan`, `arm_rerun.py`, as described above. Diego's differential-IK trajectory (`solve_ik`, `joint_trajectory`) and his `ArmPusher` are gone, since two classes cannot both be `arm.ArmPusher`. They agreed where they overlapped: reach to x = 0.92 m, 0.49 m/s within the joint speed limits (Diego: 0.48), and the same joint-speed ratio on every E1 row.
- **Target planner: Diego's, kept with its names and arguments.** `rest_x`, `plan_peak_speed`, `max_peak_speed`, `limited_peak_speed`, `slide_to_target`, `report`, `joint_limit_report(pusher.traj)`, and the `--target`, `--mu-est`, `--plate`, `--reach` commands. The reach and the joint-limit bound now come from the merged controller, and every planner function takes `base=`.
- **E1 validation: Diego's** `e1_arm_validation.py` and `tests/test_e1_arm.py`, with a `--base-x` / `--base-y` option added. Rerun on the merged controller (85 pushes): all 12 arm-vs-plate cases pass, the 21 targets land within 2.4 mm (RMS 0.7 mm), repeatability misses stay within 2.8 mm, and 22 of 73 arm pushes are within the joint speed limits, the same count as Diego's run. `e1_arm_results.csv`, `e1_arm_summary.txt` and `e1_arm_summary.png` are from this rerun.
- **`push.py`:** Angelina's version. Diego's four-line change (let a pusher wire its own controller) is covered by the pusher hook.

**Point targets and slide targets, one rule.** `decide.py` planned a slide length d\* from the release point; Diego's planner aimed at a point x\* on the board. They differ by the release point, which moves with the speed being chosen (5.5 cm of paddle travel at 0.40 m/s, 13.4 cm at 0.66 m/s, 28.9 cm at 1.0 m/s, for μ 0.3 and J 100). Now:
- `decide.point_target(x_target, x0, gap)` returns d\*(u, μ) = x\* − x_r(u, μ), and `expected_loss` / `choose_speed` accept it in place of a number.
- `arm.plan_peak_speed` with a belief calls `choose_speed` with that d\*. With a single μ it is unchanged (identical speeds to Diego's original).
- `decide.loss_sq_abs` (new) is the squared miss in metres, Diego's original objective. `plan_peak_speed(..., loss=decide.loss_sq_abs)` reproduces his speeds to four digits.
- The default loss is `loss_sq_rel`, the specification's. Against the squared miss it shifts a belief's speed by 0.5–6 mm/s (targets 0.60–0.64 m, belief widths 0.2–0.5 in ln μ, `u_max` 0.45).
- Feeding a point target as a *fixed* d\* (from a nominal speed's release point) lands about 2 cm short at μ 0.3, x\* 0.75 m, belief width 0.3. `tests/test_arm.py` pins it.
- `physics.release_push_distance` takes arrays, so the search stays vectorised.

**Checked against Alejandro's branch.** With the merged files copied over a checkout of `alejandro/bayesian-integration`, its 58 tests pass and `run_arm_comparison_v2.execute(0.3, 0.45)` runs and is valid (peak joint speed 0.91 of the limit).

**Next steps for the team.**
1. *Alejandro:* your branch is merged into `main` (16 Nov), with this README kept and extended instead of the branch's shorter one. Speeds planned from a belief move by up to 6 mm/s under the default loss, so the saved pilot results will not reproduce bit for bit; pass `loss=decide.loss_sq_abs` to get the old numbers, or rerun. Each run's `settings.json` hashes `arm.py`, `push.py` and `physics.py`, all of which changed in the arm merge.
2. *Team:* choose the loss for point targets (the specification's relative loss, now the default, or the squared miss); it moves the zero-practice generic baseline from 1.9 to 7.4 mm on Y01 (*The evaluation pipeline*). Then say in the report whether the evaluation scores the slide from the FK release point or the rest position on the board. The start-gap sensitivity in E1 part C exists only under the second.
3. *Team:* the base position is still open, now with a trial behind it (*The side base*): beside the board the joint limits allow 1.02 m/s instead of 0.49, and every E1 push is within them. The evaluators still pass `u_max=.45` by hand unless run with `--peak-cap auto`.
4. *Everyone:* after pulling the folder reorganisation, run `pip install -r requirements.txt` once (it adds the editable install). Work in progress on files at their old root paths merges normally, because the files were moved without being renamed; new scripts go in the matching folder and should take their default paths from `src/paths.py`.
5. *Diego:* please read `docs/ARM_CONTROLLER.md` and the planner section of `arm.py` and say if anything you relied on is missing. `solve_ik`, `joint_trajectory`, `Q_SEED`, `REACH_X` and `ArmPusher(kd=, feedforward=)` no longer exist.

## The evaluation pipeline (Alejandro, merged 16 Nov)

The comparison the proposal is about now runs end to end on three published pairs (Y01–Y03, stainless steel on ABS, acetal and plywood). `docs/ALEJANDRO_CONTRIBUTION.md` is Alejandro's audit of what it adds and what is still open; `EVALUATION_V3.md` and `PILOT_V2_PROTOCOL.md` are the protocols; `results/README.md` indexes the saved runs.

**What runs.**
- **LLM estimate.** `llm_prior.py` sends the text-only prompt to OpenAI (`gpt-5.4-mini-2026-03-17`, structured JSON: median, low, high, reasoning) and validates the numbers. Responses are cached with the model and request settings in the key (`llm_cache_published_pilot.json` for Y01–Y03, `llm_cache_pilot.json` for the earlier pilot). A live request needs `--query`; the evaluators never query on a cache miss.
- **Four methods.** Generic prior (ln μ normal, 5th/95th percentiles 0.08/1.1), lookup (the generic prior times `belief.loglik_lookup`), raw LLM (a lognormal from the returned median and interval, not calibrated), and an oracle that plans with the true friction.
- **Practice, then a scored push.** Each pair gets two practice pushes at 0.45 m/s. Every method sees the same noisy observations (frames at 60 Hz, rest position, FK release point; observation model C) and updates its belief after 0, 1 or 2 of them. The scored push is planned by `arm.plan_peak_speed` on the belief, capped at 0.45 m/s, and never feeds back into the belief.
- **Three versions, not to be pooled.** v1 (`run_arm_comparison.py`): an early narrow generic prior, kept as history. v2 (`run_arm_comparison_v2.py`): the proposal-width prior, target x = 0.64 m, ten noise seeds, 360 scored rows. v3 (`run_arm_evaluation.py`): targets 0.60, 0.62 and 0.64 m, low / nominal / high noise plus a no-added-noise check, 432 scored rows at one seed.

```bash
python3 -m pytest tests/test_evaluation_v3.py -q                           # offline checks, no Drake, no API
python3 evaluation/run_arm_evaluation.py --pair all --seeds 0 1 2 --check-only        # configuration and cache check (the side-base default; needs Drake)
python3 evaluation/run_arm_evaluation.py --v3 --pair all --seeds 0 1 2 --check-only   # the saved v3 protocol: origin base, 0.45 m/s, targets 0.60 0.62 0.64
python3 evaluation/run_arm_evaluation.py --pair Y01 --seeds 8 --out results/evaluation_v3/Y01_seed8
python3 evaluation/summarize_arm_evaluation.py results/evaluation_v3/Y01_seed8
```

**Read before using the saved results.**
- They were produced on Alejandro's branch, before the arm merge: with Diego's original controller and the squared-miss loss. The folders under `results/` have not been regenerated.
- **The merged controller reproduces them.** The v3 Y01 check (seed 7, 144 rows) was rerun on 16 Nov on the merged, reorganised code with the squared-miss loss (`loss=decide.loss_sq_abs`): every landing RMSE is within 0.2 mm of the saved one (zero practice slides: generic 1.89 vs 1.87 mm, lookup 12.50 vs 12.33, LLM 22.01 vs 22.01, oracle 0.18 vs 0.21).
- **The loss matters before any practice slide.** The same rerun with `plan_peak_speed`'s current default, the specification's squared relative loss, gives generic 7.40 mm (not 1.89) and lookup 10.53 mm (not 12.50) at zero practice slides; LLM and oracle are unchanged, and after one or two practice slides every method is within 0.02 mm of the squared-miss run. The wide generic prior is where the two losses disagree. The evaluators do not pass a loss, so a fresh run today uses the relative loss.
- The evaluation is not frozen. It accepts only Y01–Y03; `measured_friction.csv` does not exist yet; the material descriptions for the measured pairs are not written.
- The joint checks in the evaluators cover the planned position, speed and inverse-dynamics torque (and the acceleration, with `--check-accel`). The saved v1–v3 runs were at the 0.45 m/s cap with the base at the origin; `--v3` still runs that protocol, and the runner's default is now the side-base setup (*The side base*).

## The side base (default since 16 Nov)

The iiwa's base now stands beside the board at (0.6, 0.65) m by default, the evaluator computes its speed cap and targets from that layout, and the camera covers the whole board. The layout before, with the base at the origin, is kept in full: `--layout origin` on the scripts (`--v3` on the evaluator, `base=LAYOUTS["origin"]` in code), and its saved results are untouched in `results/e1/`, `results/arm_rerun/`, `results/evaluation_v3/` and `results/e0/`. Results at the new default are in `results/side_base/` (index: `results/side_base/README.md`); `docs/ARM_CONTROLLER.md` has the full tables. The team signed off on these defaults on 16 Nov; `scene.DEFAULT_LAYOUT = "origin"` would switch back.

**What changed.**
1. **Base beside the board** (`scene.LAYOUTS`, `scene.DEFAULT_LAYOUT = "side"`). Joint 1 sweeps the paddle instead of the elbow.
2. **Board shifted 10 cm.** The clearance check (`experiments/arm_clearance.py`) found the moving links fine but the base's footprint reaching y = 0.53 m, which is 71 mm onto a 1.2 m board centred on the push line. Moving the base outward costs stroke and conditioning (it cannot reach the line from y = 0.80 m), so with the arm at the side base the board is shifted away from it instead (`scene.BOARD_Y`: centre at y = −0.10 m, edges at −0.70 and +0.50 m). The push line stays at y = 0. Scenes without the side-mounted arm keep the board centred, so their saved results reproduce exactly.
3. **Speed cap and targets computed** in the evaluator by default (`--peak-cap auto`, `--practice-peak auto`, `--targets auto`): 1.019 m/s, practice 1.01 m/s, targets 0.60 / 0.77 / 0.94 m for Y01–Y03. `--clip-unreachable` pushes a target beyond the cap at the cap and scores the miss.
4. **Joint acceleration checked** alongside speed and torque (`joint_limit_report(...)["a"]`, `--check-accel`, `--a-brake`).
5. **Camera moved for full coverage**, from (0.75, 0, 0.8) to (1.0, 0, 1.0) m (`scene.CAMERAS`), with a height gate in the perception window (below).

**Old layout against new.**

| | origin | side base |
|---|---|---|
| Fastest push within joint speed and torque limits | 0.49 m/s | 1.02 m/s |
| Longest slide at μ 0.1 / 0.3 / 0.5 | 12 / 3 / 1 cm | 53 / 16 / 8 cm |
| Lowest moving link above the board | 55 mm | 51 mm |
| E1: arm pushes within joint speed limits | 22 of 73 | 73 of 73 |
| E1: targeting with the true friction, RMS and worst miss | 0.74, 2.4 mm | 0.83, 2.7 mm |
| Arm re-run: pushes within speed and torque limits | 0 of 12 | 12 of 12 |
| E0-2: hydroelastic contact only in the pushed scene | PASS | PASS |
| Targets for Y01–Y03 | 0.60, 0.62, 0.64 m (hand-picked) | 0.60, 0.77, 0.94 m (derived) |
| Camera sees, along the push | x = 0.30–1.21 m | x = 0.43–1.57 m (the board ends at 1.52) |

**Acceleration is set by the brake, not the speed.** With the default 12 m/s² brake cap every push is over the SDF's acceleration limits, at either base. With a 4 m/s² cap the side base still reaches 1.02 m/s inside speed, torque and acceleration limits. The catch: the cap has to exceed μg or the object never leaves the paddle, so 4 m/s² only works for friction below about 0.4 (Y01–Y03 are 0.14, 0.14, 0.26). At 6 m/s² (friction below 0.6) the bound is 0.84 m/s. The limits are Drake's model values; KUKA publishes none.

**The camera, and what the position study found** (`experiments/camera_study.py`, `results/camera_study/`). The original camera lost a fast slide on low friction: on a slide resting at x = 1.17 m its friction estimate is 23% off. Moving it to cover the board fixed that but raised E0-5's worst bias from 0.9% to 6.9%. The cause is the viewing direction, not the distance: a camera ahead of the paddle sees the paddle's front face, depth noise puts some of those pixels just inside the perception window, and they attach to the object while the gap is a few pixels wide. The original camera only looked clean because E0-5's 0.8 m/s pushes release right under it.

Worst friction bias of observation model C over four pushes (a short high-friction slide, E0-5's middle case, a slow push, the longest practice slide), 60 Hz, arm in view:

| camera (x, z), m | sees x, m | plain window | with the height gate |
|---|---|---|---|
| (0.75, 0.80), original | 0.31–1.19 | 23.2% | 23.2% |
| (0.80, 0.85) | 0.33–1.27 | 2.7% | 1.4% |
| (0.85, 0.90) | 0.35–1.35 | 3.7% | 1.5% |
| (0.90, 0.95) | 0.38–1.42 | 4.5% | 1.8% |
| **(1.00, 1.00), default** | 0.45–1.55 | 6.3% | **2.3%** |
| (0.95, 1.05) | 0.37–1.53 | 5.6% | 2.3% |
| (0.90, 1.15) | 0.26–1.54 | 5.9% | 2.5% |
| (0.75, 1.40) | −0.02–1.52 | 2.3% | 0.8% |
| (0.60, 1.20) | −0.06–1.26 | 150% | 150% |

- **The height gate** (`perception.segment(face_gate=...)`, on by default for arm runs of E0-5): within 4 mm of the paddle face, keep only pixels at the object's top height. The paddle face's stray pixels sit at every height. It removes most of the effect at every position and costs nothing elsewhere.
- **What is left is the first frame.** With the gate, dropping the first 30 ms of frames (model C1) brings the default camera to 0.3% or less on all four pushes. The same residual was already there with the original camera.
- **A wider window margin (3 mm) is not the answer:** it evens the bias out at 1.5–2.5% but triples the frame noise.
- **Behind the paddle is worse, not better:** from (0.60, 1.20) the arm itself blocks the view.
- **The alternatives.** (0.85, 0.90) has the lowest bias among the small moves and sees to x = 1.35 m, enough for every slide Y01–Y03 make (the longest rests at 1.20 m) but not the board's last 17 cm. (0.75, 1.40) is the most even but 60 cm higher, with three times the frame noise. Both are one line in `scene.CAMERAS`.
- **E0-5 at the default** (arm in view, full-coverage camera, gate): worst bias 2.2% (0.45% for C1), frame noise 0.16 mm against 0.09 mm before, no frames lost. The perception study's rest-pose error SD is 0.11 mm against 0.05 mm.

**How far the arm is from the release model** (`experiments/e1_release_error.py`, `results/side_base/e1_release_error.csv`; 35 pushes over μ 0.1–0.5 and 0.3–1.0 m/s, all within joint speed and torque limits). This is the execution model a harness without the arm needs (Tier B): actual release speed = the model's × (1 + ε).

| | mean | range |
|---|---|---|
| Release speed against the model, ε | +0.21% | −0.02% to +1.23% |
| Release point | −0.04 mm | −0.07 to −0.01 mm |
| Landing against the model's rest point | −0.56 mm (RMS 1.6 mm) | worst −8.0 mm |

- ε grows with friction and falls with speed: +1.2% at μ 0.5 and 0.3 m/s, under 0.25% everywhere at 1.0 m/s, near zero at μ 0.1.
- The landing error is the free slide, not the arm: it is largest on the longest slides (−4.6 and −8.0 mm for μ 0.1 and 0.14 at 1.0 m/s, slides of 37–50 cm) and under 0.6 mm elsewhere.
- These are deterministic biases, not noise: the simulator has no actuator noise. A harness should add its stated spread on top.

**The evaluation at the side base** (`results/side_base/evaluation/`; Y01–Y03, seed 7, 432 scored rows per run). Landing RMSE at nominal noise:

| practice slides | generic | lookup | LLM | oracle |
|---|---|---|---|---|
| 0 | 54.8 mm | 50.8 mm | 65.2 mm | 0.4 mm |
| 1 | 1.5 mm | 1.5 mm | 1.5 mm | 0.4 mm |
| 2 | 0.5 mm | 0.6 mm | 0.6 mm | 0.4 mm |

- **Every push is valid**, with either brake cap. With a 4 m/s² brake the scored pushes peak at 0.80 of the joint speed limit, 0.25 of the torque limit and 0.75 of the acceleration limit; with 12 m/s² the acceleration reaches 2.1×. The two runs agree within 0.1 mm.
- **No push reached the cap** with the derived targets, which are in reach by construction. A separate run with a target placed beyond reach (`clip_beyond_reach/`: targets 0.60 / 0.94 / 1.10 m, `--clip-unreachable`) exercises the rule: Y03 cannot pass x = 1.03 m, so at 1.10 m the oracle and most methods push at the cap, land 66 mm short and are scored; 13 of 108 nominal rows are at the cap and the run completes.
- **The loss matters much more at this range.** With the squared miss instead of the default relative loss (`loss_sq_abs/`, via `evaluation/run_with_loss.py`), the zero-practice generic baseline is 34.6 mm, not 54.8; lookup 54.3 against 50.8, LLM 67.1 against 65.2. After one practice slide the two losses agree to 0.01 mm. At the origin the same gap was 8.9 against 14.1 mm.
- **Farther targets make the starting belief matter much more.** Before any practice the methods miss by 51–65 mm, against 9–17 mm in the saved origin runs whose targets stop at 0.64 m. Not like for like: different targets, and the saved runs used the squared-miss loss (*The evaluation pipeline*).
- **Ten seeds** (`ten_seeds_Y01/`, `_Y02/`, `_Y03/`; 3,348 scored rows, all valid, none at the cap). Landing RMSE over the three pairs and three targets, with the range of the per-seed RMSE:

  | noise | practice slides | generic | lookup | LLM | oracle |
  |---|---|---|---|---|---|
  | any | 0 | 54.8 mm | 50.8 mm | 65.2 mm | 0.4 mm |
  | low | 1 | 1.05 (0.26–1.51) | 1.05 (0.27–1.51) | 1.11 (0.27–1.58) | 0.4 |
  | nominal | 1 | 1.34 (0.29–2.45) | 1.34 (0.32–2.43) | 1.37 (0.34–2.40) | 0.4 |
  | high | 1 | 2.74 (0.81–5.12) | 2.77 (0.65–5.15) | 2.69 (0.25–5.01) | 0.4 |
  | nominal | 2 | 0.90 (0.29–2.12) | 0.93 (0.33–2.08) | 0.95 (0.32–2.17) | 0.4 |

  - **One practice slide erases the prior.** After a single practice push at 1.01 m/s the three methods are within 0.1 mm of each other at every noise level; what is left is observation noise. The friction estimate is then within 0.002 of the truth (0.0008 after two slides), and the 90% interval covers it in 80% of beliefs (93% after two).
  - **Before practice the ranking depends on the pair.** Nominal zero-practice RMSE, generic / lookup / LLM: Y01 28 / 37 / 87 mm, Y02 28 / 67 / 70 mm, Y03 86 / 43 / 16 mm. The LLM prior is the best of the three on Y03 and the worst on Y01 and Y02. Three pairs do not rank the methods.
- The evaluation reads simulator state, so the camera does not affect it.

**Checked that the old work is intact.** `run_arm_evaluation.py --v3` gives the same 144 rows as before, cell for cell (Y01, seed 7). The v3 protocol rerun on the merged code for all three pairs (`results/evaluation_v3/merged_code_seed7_sq_abs/`, squared-miss loss as the saved runs used) matches the saved checks: zero-practice RMSE 8.88 / 13.47 / 16.66 mm against 8.88 / 13.46 / 16.63. A plate push is bit-identical to before. No file under `results/` outside `side_base/` and `camera_study/` changed.

**Still open for the team** (the layout, the computed cap, the derived targets, the board shift and the camera are signed off).
- Which brake cap: a soft one meets the acceleration limits but needs a stated upper bound on friction.
- Whether to drop the first 30 ms of frames (C1) in the observation model, now that the gate has removed the rest of the camera bias.
- For the real rig: the board's edge on the arm's side must be within 0.50 m of the push line, or the board needs a cut-out for the base.

## Repository layout

Since 16 Nov the files are grouped into folders. Nothing was renamed, only moved, so `git log --follow <file>` still shows each file's history.

```
src/           the library: physics, scene, push, arm, decide, belief, observe, perception, llm_prior
experiments/   E0 and E1 validation scripts, the arm re-run
evaluation/    the method comparison: runners, summarisers, demonstrations
friction/      rig analysis, lookup table, SimReady audit
figures/       figure and video scripts with their outputs
data/          inputs: ground truth, lookup table, LLM caches
results/       saved outputs of experiments/ and evaluation/
paper/         LaTeX and its generated rows
docs/          guides and protocols
tests/         pytest suite
```

**How things find each other.**
- `pip install -r requirements.txt` installs this repository in editable mode (`pyproject.toml`), which puts `src/` on Python's import path. **If you set up before 16 Nov, run it once more.** Without it, scripts outside `src/` fail with `ModuleNotFoundError: No module named 'physics'`.
- Scripts run by path from any directory: `python3 experiments/e0_4_release.py`, `python3 src/arm.py --target 0.75 --mu 0.3`.
- Default input and output locations come from `src/paths.py`, not from the current directory. Paths you pass on the command line are still relative to where you are.
- The tests need no install: `pyproject.toml` and `tests/conftest.py` add the code folders to the path.

### `src/`

The library. Every other folder imports from here (`import physics`, `from arm import ...`). Each module also runs as a script, e.g. `python3 src/arm.py --reach`.

| File | Purpose |
|---|---|
| `scene.py` | The Drake scene: patchy table, object, iiwa 14 with paddle. Also runs passive slides against the predicted stopping distance. |
| `perception.py` | Landing pose from the scene's overhead depth camera: depth noise, plane removal, segmentation, outline ICP. Also measures the perception error. |
| `physics.py` | Analytic model: release model `v_r = v_peak − (μg)²/2J`, free slide, slides across friction strips, uniform-pressure disk spin integrator, tipping check (A5). |
| `observe.py` | Observation models A/B/C (frames, + rest pose, + FK release point) and the O1 ablation; least-squares fit; Cramér–Rao error budget; residual signatures. `python3 src/observe.py` prints the budget table and writes `error_budget_rows.tex`. |
| `belief.py` | Grid posterior over `ln μ` (500 points on [ln 0.05, ln 1]): prior from measured pairs, LLM observation model fitted leave-one-object-out, slide likelihood with `(x_r, v_r)` profiled out. |
| `decide.py` | Release-speed decision by expected loss with the release model inside `d(μ, u)`; five losses; closed-form limit; practice speed; `point_target` turns a target point on the board into a target slide that moves with the speed. |
| `llm_prior.py` | Prompt builder that can read only the `description_*` and `photo_*` columns (test 7); the text-only OpenAI request; response validation; a cache keyed on the prompt, model and request settings. `--query` makes a live request. |
| `sim_config.yaml`, `drake_config.py` | The one Drake contact configuration every experiment loads (frozen by E0-1), and contact-result checks (`assert_hydro_only`). |
| `push.py` | Prescribed-paddle push harness: the plate on a prismatic joint under Drake's implicit PD, tracking a constant-jerk profile (`profile(..., a_max)` caps the brake); one push-and-slide with the release read three ways. Stood in for the arm in E0-3 to E0-5; `push_slide` drives any pusher with the same hook. |
| `arm.py` | The iiwa 14 as the pusher: IK path for the straight push with the posture chosen by joint-speed ratio, implicit PD per joint with inverse-dynamics feed-forward, torque clamped at the SDF effort limits; `--feasibility` / `--scan` tabulate the joint-speed and torque ratios over v_peak, J and the base position (decision 7). Also Diego's target planner (16 Nov merge): `plan_peak_speed`, `slide_to_target`, the reach and joint-limit speed bounds, `--target` and `--reach`. |
| `paths.py` | Where each folder is (`DATA`, `RESULTS`, `PAPER`, `FIGURES`, ...), so no script depends on the directory it is run from; `locate(name)` finds a source or input file by name. |

### `experiments/`

Validation runs. Each writes its CSV, summary and figure to `results/` by default.

| File | Purpose |
|---|---|
| `check_drake.py` | Smoke test for the Drake install: loads the iiwa 14 and checks a hydroelastic box slide against `d = v² / (2 μ g)`. |
| `e0_sweep.py` | E0 contact-model study: passive slides across shapes and contact settings, checked against `d = v² / (2 μ g)`. |
| `e0_2_hydro_check.py` | E0-2: every contact hydroelastic at every step, in the passive and the push scene, with negative controls. |
| `e0_1_free_slide.py` | E0-1: free slide vs `d = v²/(2μg)` over approximation × damping × step × margin, plus spin and heading launches. Freezes A6. |
| `e1_arm_validation.py` | Diego's E1 arm milestone: 12 arm-vs-plate pushes, 21 targets with the true μ, 40 repeatability trials; writes `e1_arm_results.csv`, `e1_arm_summary.txt`, `e1_arm_summary.png`. |
| `arm_rerun.py` | The 5 Nov arm re-run: 12 pushes from the E0-3/E0-4 grids with the arm beside the plate; writes `arm_rerun_results.csv`, `arm_rerun_summary.txt`, `arm_rerun_summary.png`. |
| `e1_release_error.py` | The arm's execution error against the release model over friction × commanded speed (35 pushes): release speed, release point and landing. Writes `results/side_base/e1_release_error.csv`, the Tier B execution model's input. |
| `arm_clearance.py` | Do the iiwa's links, its base and the camera stay clear during a push? Convex hulls of the links' visual meshes along the planned push: clearance above the board, to the object and to the camera, and the base's footprint against the board's edge. |
| `camera_study.py` | Camera position and perception window against coverage, frame noise and friction-estimate bias, with the arm in view; writes `results/camera_study/`. |
| `e0_3_contact_geometry.py` | E0-3: box (line contact) vs disc (point), COM offset, plate yaw and paddle friction: release heading, spin, lateral miss, slide error. |
| `e0_4_release.py` | E0-4: object speed at separation vs `v_peak − (μg)²/(2J)` over J, μ, v_peak and each contact approximation; the decision-7 table of what braking costs. |
| `e0_5_observation.py` | E0-5: pushes replayed through the depth camera at 30, 60 and 120 Hz; models A, B, C and O1 fitted; bias and SE vs the Cramér–Rao bound; Tier B noise values (`e0_5_noise.csv`). |

### `evaluation/`

The method comparison (generic, lookup, LLM, oracle). Pass `--out results/<version>/<name>` for a new run.

| File | Purpose |
|---|---|
| `run_arm_evaluation.py`, `summarize_arm_evaluation.py` | Evaluation v3: methods × practice counts × targets × noise levels × seeds, with preflight feasibility checks; the summariser rejects partial or invalid runs. |
| `run_with_loss.py` | Runs `run_arm_evaluation.py` with another planner loss from `decide.LOSSES` (e.g. `sq_abs`) without changing the runner; writes `loss_override.json` in the run folder. |
| `run_arm_comparison_v2.py`, `summarize_arm_v2.py` | Pilot v2 (one target, ten seeds). v3 imports its execution and priors, so it is a live dependency. |
| `run_arm_comparison.py`, `summarize_arm_pilot.py` | Pilot v1 (narrow generic prior), kept as history. |
| `demo_published_comparison.py`, `demo_bayesian_integration.py`, `demo_compare_priors.py` | Small demonstrations of priors → update → plan. `demo_published_comparison.py` holds the Y01–Y03 descriptions and the raw-LLM prior, which v2 and v3 import. |

### `friction/`

Rig data processing, the published-value lookup table and the SimReady audit.

| File | Purpose |
|---|---|
| `friction_analysis.py` | Computes `μ_k` from phone-accelerometer (phyphox) recordings of a sled sliding on a flat board. Also does the six-position accelerometer calibration. |
| `lookup_table.py` | Builds the published-value friction lookup table (baseline C2): `friction_lookup.csv` plus LaTeX rows `rows_*.tex`. |
| `aggregate_pairs.py` | Per-pair aggregation of rig runs and friction profiles into `measured_friction.csv`. |
| `make_mock_measured.py` | Regenerates `ground_truth_placeholder.csv` with the measured-file columns (values preserved, new columns mock). |
| `simready_audit.py` | Decision-10 audit of NVIDIA's SimReady Central library (`nvidia/simready-assets` on Hugging Face): downloads each asset's small physics layers, reads every physics material's static/dynamic friction, restitution and density with `usd-core`, matches the rig's materials by name, and plots `simready_friction.png`. Read-only; nothing in the pipeline uses it. |

### `figures/`

Scripts that draw figures or render video, with their outputs.

| File | Purpose |
|---|---|
| `demo_video.py` | Renders a demo video offscreen (VTK + ffmpeg): the scene, a passive slide, the four shapes, landing perception. |
| `simready_friction.png` | Distribution of the static and dynamic friction values authored on the library's physics materials. |

### `data/`

Inputs. `measured_friction.csv` will live here once it exists.

| File | Purpose |
|---|---|
| `ground_truth_placeholder.csv` | Synthetic friction values in the `measured_friction.csv` columns, used until the measured file exists. **Not friction estimates.** |
| `ground_truth_published.csv` | Yu et al. 2016's measured pairs (Y01–Y03, stainless steel on ABS, acetal, plywood; the polyurethane pair Y04 was dropped from the simulated set on 5 Nov, decision 6). Simulator friction comes from these plus our measured pairs. |
| `friction_lookup.csv` | The published-value lookup table (baseline C2), written by `friction/lookup_table.py`. |
| `llm_cache_published_pilot.json`, `llm_cache_pilot.json` | Cached LLM responses for Y01–Y03 (used by the evaluators) and for the earlier pilot. |
| `run_log_template.csv` | Run-log template: session, pair, file, lane, orientation, speed setting, `rest_mm`, accepted, reject reason. |

### `results/`

Saved outputs. `results/README.md` indexes the evaluation runs.

| File | Purpose |
|---|---|
| `e0/` | E0 outputs: `e0_1_*` to `e0_5_*` (results, summaries, figures), the sweep's `e0_results.csv` and `e0_summary.txt`, and `e0_5_noise.csv`, the Tier B noise values. |
| `e1/` | `e1_arm_results.csv`, `e1_arm_summary.txt`, `e1_arm_summary.png` from `e1_arm_validation.py`. |
| `arm_rerun/` | `arm_rerun_results.csv`, `arm_rerun_summary.txt`, `arm_rerun_summary.png` from `arm_rerun.py`. |
| `perception_results.csv` | Landing-perception error study from `perception.py --study`. |
| `side_base/` | Results at the default layout since 16 Nov (base beside the board): link clearance, E1, arm re-run, E0-2 and E0-5 with the arm, the perception study, and the evaluation with a computed cap and derived targets. Indexed in `results/side_base/README.md`. The other folders here are the origin layout (`e0/` used no arm). |
| `camera_study/` | The camera position study: `camera_study.csv` and `camera_study_summary.txt`. |

### `paper/`

LaTeX. The `rows_*.tex` and `error_budget_rows.tex` files are generated; compile from inside `paper/`.

| File | Purpose |
|---|---|
| `model_specification.tex` | Reference note: state and assumptions, contact table, generative model, observation models and error budget, inference, decision rule, fidelity ladder, E0 status. |
| `friction_lookup_table.tex` | Lookup table v2 document; its tables are the generated `rows_*.tex`. |

### `docs/`

Guides and protocols.

| File | Purpose |
|---|---|
| `EVALUATION_V3.md`, `PILOT_V2_PROTOCOL.md` | Settings, commands and caveats for v3 and v2. |
| `ALEJANDRO_CONTRIBUTION.md`, `REPOSITORY_MAP.md` | Alejandro's contribution audit with the open items, and his map of the evaluation pipeline. |
| `ARM_CONTROLLER.md` | Diego's guide to the arm: how to run it, the three layers, limits, and the E1 validation results; updated for the merged controller. |
| `measured_friction_interface.md` | Column contract and interface rules for the measured file, the run log, per-run results and `profiles.csv`. |
| `simready_audit.md` | The audit's finding: what the library stores, the friction distribution, the rig-material match table, timing, and the conclusion for decision 10. |

### `tests/`

pytest suite (66 tests): spec tests 1–7, the rig self-checks, the push harness, the arm, the target planner, the E1 script, the offline evaluation checks and the layouts (the side-base default, board and camera, link clearance, acceleration, speed cap, derived targets, the height gate). `python3 -m pytest tests` (`-m "not slow"` skips three simulated pushes).

## Setup

You need Python 3.12 or newer. Drake 1.58 ships wheels for macOS 26 on Apple silicon and for Linux with glibc 2.34 or newer (Ubuntu 22.04+). On Windows, use WSL2.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 friction/friction_analysis.py --selftest
python3 experiments/check_drake.py
python3 -m pytest tests
```

`requirements.txt` pins Drake and `manipulation`, the 6.4210 course package, and installs this repository in editable mode so that the modules in `src/` import from anywhere (see *Repository layout*).

**`friction_analysis.py --selftest`** runs the analysis on synthetic slides with `μ = 0.30`. With a 2° board tilt and a 1° mounting tilt, the fwd/rev mean should come out close to 0.300.

**`check_drake.py`** gives a box a speed of 0.8 m/s on a table with `μ = 0.30` and prints PASS if:
- it stops within 2% of `v² / (2 μ g)`
- it turns by less than 2°

Options:
- `--mu`, `--v0`: friction and launch speed
- `--dt`: plant time step
- `--approx lagged|similar|sap`: Drake's discrete contact approximation
- `--meshcat`: replay the slide in the browser

The first run downloads Drake's model files, so it needs internet.

## Friction measurement

**Method.** During a free slide, the phone on the sled measures only contact forces, so `μ_k = −f_along / f_normal`. This holds whatever the board's tilt, launch speed or `g`. Each pair is slid in both sled orientations (`fwd` and `rev`), and the two results are averaged to cancel phone mounting tilt and accelerometer bias. The full protocol, including apparatus, procedure and uncertainty budget, is in the friction technical approach document (v3).

**phyphox settings.**
- Use the experiment **Acceleration with g**. Do not use "without g".
- Turn on remote access, so you can start and stop runs from a laptop.
- Export as **CSV (comma, decimal point)**.
- Keep the `Raw Data.csv` from each export and rename it `<pairID>_<lane>_<orient>_<speed>_<n>.csv`, for example `M07_L_fwd_s1_1.csv`.

**At the start of each session, calibrate.** Record about 10 s with each phone axis pointing straight up, one file per axis direction. Then run:

```bash
python3 friction/friction_analysis.py --make-calib px.csv mx.csv py.csv my.csv pz.csv mz.csv
```

This writes `calib.json` to the current directory. Scales should be within 1–2% of 1, and biases below 0.1 m/s².

**For each pair, analyse each orientation separately:**

```bash
python3 friction/friction_analysis.py M07_*fwd*.csv --pair M07 --direction fwd --calib calib.json --out results.csv
python3 friction/friction_analysis.py M07_*rev*.csv --pair M07 --direction rev --calib calib.json --out results.csv
```

The script prints `μ_k` for each run and flags rejects: a short slide, lateral force, or yaw. `--out` appends to the CSV; rejected runs are written too, with `accepted = 0` and a `reject_reason`. If more than 3 of a pair's 12 runs are rejected, fix the cause and redo them. Runs where the sled touched the paddle again after release are marked in the run log (`reject_reason`), which `--runlog` reads.

**Since 31 Oct** the script detects the true stop (along-force back at rest level for 50 ms) and integrates the untrimmed signal from there, so `v_start` no longer reads low by `μg × 0.03 s`. It also integrates twice for the slide distance `d_acc_m`, computes the endpoint estimate `mu_v2d = v_r²/(2 g d_tape)` from the run log's `rest_mm`, and bins the friction ratio by board position into `profiles.csv` (`--profiles`; 20 mm bins, features shorter than the 100 mm coupon are not resolvable). `--pair` and `--direction` default to the run log, then the file name. Then per pair:

```bash
python3 friction/friction_analysis.py data/raw/S1/M07_*.csv --runlog run_log.csv --calib calib.json --out results.csv --profiles profiles.csv
python3 friction/aggregate_pairs.py --results results.csv --profiles profiles.csv --runlog run_log.csv --meta data/ground_truth_placeholder.csv --out data/measured_friction.csv
```

`aggregate_pairs.py` averages forward and reverse within each lane and speed first, then writes the pair's `mu_k_nominal`, `s_tab_log`, `speed_effect`, `mu_slope_per_mps`, `strip_scale_mm` (profile correlation length, floored at 100 mm: the simulated strip width), `profile_repeatability` and the reject counts. Column meanings are in `measured_friction_interface.md`.

**Where data goes.** Raw recordings are small, so commit them, with each session's `calib.json` and run log beside them. Suggested layout: `data/raw/<session>/`.

## Simulation scene

`scene.py` builds one `MultibodyPlant` for a given pair from the ground-truth file.

**What's in the scene**
- **Table:** the 48 in board (decision 1; `physics.BOARD_LENGTH`, `BOARD_X_MIN`, `BOARD_X_MAX`): its top is at z = 0 and it runs from x = 0.3 to 1.52 m along the travel direction (+x), tiled with equal strips across it (`Table.edges()`: 24 strips of 2 in, 5.08 cm, for the nominal 5 cm). Strip draws are indexed from the board's near edge, so a patchy-table run (patchiness > 0) from before 5 Nov, when the table started at x = −0.3 m with 5 cm strips, does not reproduce from its seed.
  - Each strip's friction is the pair's `mu_k_nominal × exp(patchiness · s_tab_log · z)`, sampled per seed.
  - `--bimodal` uses two levels instead.
- **Object:** compliant hydroelastic.
  - Shapes: box, disc, ring, or block on 3 or 4 feet. The ring's volume and surface meshes are generated on the fly.
  - For robustness cases: an optional centre-of-mass offset, or one longer foot.
- **iiwa 14:** base beside the board at (0.6, 0.65) m by default (`scene.LAYOUTS["side"]`; `"origin"` is the layout before 16 Nov, 0.3 m behind the board's near edge), with a flat rigid paddle welded to the flange (`add_arm(plant, contact, X_WB, mu_p)` places the base and sets the paddle friction; `build_scene(..., base=...)` passes it on). With the arm at the side base the board is shifted 10 cm away from it (centre line at y = −0.10 m) so the base's footprint clears its edge; the push line stays at y = 0.
  - `arm.py` drives it (6 Nov); `build_scene(..., arm=True)` alone adds no controller.

**Getting the friction right.** Drake combines two geometries' friction as `2 μa μb / (μa + μb)`. The object therefore carries a fixed μ, and each strip and the paddle are given the value that makes the *pair* coefficient equal to its target.

```bash
python3 src/scene.py --pair M07 --meshcat                      # view the scene
python3 src/scene.py --pair M07 --slide 0.8                    # passive slide vs prediction
python3 src/scene.py --pair M07 --slide 0.8 --shape footed --foot-error 0.001 --patchiness 0
```

**What `--slide` does:** it gives the object a speed and lets it slide to rest. It compares the stopping distance with the prediction `g ∫ μ̄ dx = v₀²/2`, where `μ̄` is the strip friction averaged over the object's footprint. It prints PASS within 2% in distance and 2° in yaw.

**From another script:** use `build_scene(builder, pair, obj, table, contact, arm=True, seed=...)`, which returns the plant, the object body, the iiwa model instance and the paddle body.

### E0 contact-model sweep

```bash
python3 experiments/e0_sweep.py                            # about 2,700 slides, ~7 min on 10 cores
python3 experiments/e0_sweep.py --quick                    # a few slides, to check it runs
python3 experiments/e0_sweep.py --summarize results/e0/e0_results.csv # reprint the summary from a saved run
```

The sweep has three stages:
- **grid:** shape × approximation × time step (1, 2, 5 ms) × modulus (10⁵–10⁷ Pa) × damping. Damping is Hunt–Crossley dissipation 0–100 s/m, or relaxation time 0–1 s for SAP, which ignores Hunt–Crossley.
- **resolution:** mesh resolution 0.5–2 cm.
- **robust:** uneven feet and centre-of-mass offsets.

Every slide is on a uniform table and is checked against `d = v² / (2 μ g)` with the 2% / 2° bar.

The scene's defaults passed on every shape: `lagged`, 1 ms, 10⁷ Pa, 1 s/m, 1 cm. Avoid:
- 5 ms steps
- Similar with stiff objects
- SAP relaxation times of 0.1 s or more
- dissipation of 100 s/m

### E0-1 and E0-2 (31 Oct)

```bash
python3 experiments/e0_2_hydro_check.py                 # every contact hydroelastic at every step; ~1 min
python3 experiments/e0_1_free_slide.py                  # 396 slides, ~1 min on 10 cores; writes results/e0/e0_1_results.csv, results/e0/e0_1_summary.txt, results/e0/e0_1_summary.png
python3 experiments/e0_1_free_slide.py --summarize results/e0/e0_1_results.csv
```

Both build the plant from `sim_config.yaml` through `drake_config.py`, with the strict `hydroelastic` contact model: a rigid–rigid pair throws instead of silently falling back to point contact (`Contact.contact_model`; the scene's default stays `hydroelastic_with_fallback`). `Contact.margin` sets the hydroelastic margin.

- **E0-2 passes:** box, disc, ring and footed, passive and pushed (arm posed by IK, joints locked, object against the paddle), 300 steps each: object–table and paddle–object contacts are hydroelastic at every step and there are no point-pair contacts. Negative controls: the paddle lowered 1 mm into the table makes strict mode throw; a free rigid box on the rigid table runs on point contact in fallback mode. (A contact between a *locked* arm and the world has no participating degree of freedom, so Drake drops it from the contact results even when the geometries overlap.)
- **E0-1 freezes A6** at the config values: `lagged`, 1 ms, 10⁷ Pa, dissipation 1 s/m, margin 10⁻⁴ m, which stop within 0.96% of `d = v²/(2μg)` with no yaw over μ ∈ {0.1, 0.3, 0.5} and v ∈ {0.6, 1.0} m/s. Spinning discs land within 2% of the uniform-pressure disk model (`physics.disk_spin_slide`; d/d₀ ≈ 1.06 at Rω/v = 0.5 and 1.27 at 1), heading launches within 1%. **Finding:** with the 10⁻⁴ m margin, Hunt–Crossley dissipation ≥ 10 s/m under `lagged` (≥ 1 s/m under `similar`) shortens slides by 4–85%, while without margin `lagged` passes up to 50 s/m; keep dissipation ≤ 1 s/m with this margin. 2 ms steps pass only without margin, 5 ms never; SAP passes at 1 ms with relaxation ≤ 0.01 s. 19 of 66 settings pass.

### E0-3, E0-4 and E0-5 (5 Nov)

The arm had no controller until 6 Nov, so `push.py` drives the paddle plate directly: a rigid plate on a prismatic joint along the push, under Drake's implicit PD actuator (`JointActuator.set_controller_gains`, which needs the SAP-family solver: `lagged`, `similar` or `sap`) tracking a constant-jerk speed profile with a feed-forward force. The profile accelerates at 2 m/s² and then runs one jerk segment −J through the peak into the brake, so braking starts from zero deceleration at `v_peak`, as the release model assumes. The command is fed one time step ahead (the implicit scheme reaches a step's desired state at its end); the plate then tracks to 0.05 mm. `build_scene(..., pusher=Pusher())` puts the plate in place of the arm. The release is read three ways: the simulator's object state at the model's instant t* = t_peak + μg/J (what the decision rule predicts), the model on the plate's recorded motion, and the instant the paddle–object force actually vanishes. Near separation both bodies decelerate at about μg, so a one-step shift moves the speed by μg·Δt, 1.4% of a 0.36 m/s release at 1 ms.

```bash
python3 src/push.py --mu 0.3 --v 0.8 --J 100            # one push-and-slide, printed; --shape disc --com 0.02 --yaw 2 --mu-p 0.6 --meshcat
python3 experiments/e0_3_contact_geometry.py                    # 132 pushes, ~1 min on 8 workers -> results/e0/e0_3_results.csv, results/e0/e0_3_summary.txt, results/e0/e0_3_summary.png
python3 experiments/e0_4_release.py                             # 66 pushes -> results/e0/e0_4_results.csv, results/e0/e0_4_summary.txt, results/e0/e0_4_summary.png
python3 experiments/e0_5_observation.py                         # 3 pushes replayed 180 times through the camera, ~3 min -> results/e0/e0_5_results.csv, results/e0/e0_5_frames.csv, results/e0/e0_5_noise.csv, e0_5_summary.*
```

- **E0-3 (A2) passes for the box:** with the plate covering the whole face, the box leaves with |ψ_r| ≤ 0.004° and a lateral miss ≤ 2 mm for every COM offset up to w/4, every plate yaw (0, 0.5, 2°) and every paddle friction (0.1, 0.3, 0.6); a yawed plate turns the box to its own angle (rest yaw = plate yaw) without giving it a heading. A disc with an offset COM spins off (heading 5–27°, miss 23–116 mm at e = R/4 and R/2) and follows a yawed plate at about two thirds of its angle. The box with its COM 5 mm inside the footprint edge rocks onto that edge during the push (tilt 4–5°, heading up to 6°); every disc with the COM 5 mm inside the rim topples. Pushed by a plate slid sideways so the covered segment no longer straddles the COM, the box turns 90° and leaves 33–86% below the release model. The slide error stays within 0.8% wherever the object does not tilt.
- **E0-4 (A3) passes; decision 7 has its table:** at the config contact the object's speed at t* is within 1% of `v_peak − (μg)²/(2J)` in 53 of 54 cases (worst +1.2% at μ 0.5, v 0.6, J 30, a 0.2 m/s release); on the plate's recorded motion within 0.08% everywhere. The paddle–object force vanishes 0–2 ms after t*. Under `similar` and `sap`, with any damping, the box tilts 0.4–1.2° during the push and leaves 1–7% off the model with spin; the release model needs `lagged` (0 or 1 s/m), which supports A6. What braking costs: at μ 0.3 and 0.8 m/s the brake must reach J ≈ 500 m/s³ to lose under 2% of the distance, at μ 0.5 about 2,000 (`e0_4_summary.txt`). Either option in decision 7 is consistent with the data; the choice is the team's.
- **E0-5 passes; decision 8 and Tier B have their numbers:** models A, B and C are within 1% of the simulator's μ at 30, 60 and 120 Hz for μ ∈ {0.1, 0.3, 0.5}, with SE at or below the Cramér–Rao bound at the measured noise (ratio ≤ 0.73). Their residual −0.6 to −0.9% bias at μ 0.3–0.5 is the first frame after release, taken with the object within a millimetre of the plate: dropping the first 30 ms of frames leaves ≤ 0.3%. O1's bias matches the release-ramp prediction (+1.2%, +12%, +39%). Measured noise for Tier B: σ_f 0.09 mm (later frames 0.03–0.09 mm, the first frame +0.25 mm), σ_rest 0.03 mm near the camera axis and 0.1 mm at the far end of the table (x ≈ 1.1 m); σ_FK assumed 0.5 mm, since the simulator's forward kinematics are exact. The error-budget table assumed 2 mm frames, so its predictions are 10–20× pessimistic; the measured values are lower bounds, from a rendered camera with exact pose and intrinsics and no motion blur. Frames are perceived in a window starting 0.5 mm ahead of the plate face (known from FK): without it the plate's visible face joins the object's blob and biases the first frames by up to 1 mm. Model C's SE per slide is 0.42% at 30 Hz and 0.25% at 120 Hz on the shortest slide (μ 0.5), so the camera rate is not set by μ's precision but by what else the frames must resolve.

### Landing perception

`build_scene(..., camera=Camera())` adds an overhead depth camera. It looks straight down from (1.0, 0, 1.0) m, where it sees the board from the start of the push to its far end (`scene.CAMERAS["full"]`; `"original"` is the (0.75, 0, 0.8) m setting used until 16 Nov, which sees to x = 1.21 m), at 640 × 480, and is turned 3° about its axis. `perception.py` turns its depth image into the object's pose:
1. Add stereo-like depth noise.
2. Build a point cloud.
3. Fit and remove the table plane with RANSAC.
4. Keep the largest blob above the table.
5. Fit the object's top-view outline to the blob's edge with 2-D ICP, solving for x, y and yaw.

```bash
python3 src/perception.py --study 50                 # 50 random landing poses per shape -> results/perception_results.csv
python3 src/perception.py --study 50 --no-noise      # noise-free depth: the algorithm's own error
python3 src/perception.py --pair M07 --slide 0.8     # passive slide, then perceive where it stopped
```

From another script: `perceive(depth, camera, obj, rng, x_window=...)` returns the estimated pose. With the arm in the scene, use `x_window` to leave out the paddle.

### Demo video

```bash
python3 figures/demo_video.py                          # demo.mp4: pair Y03, v0 = 1.4 m/s, 1920 x 1080, ~30 s to render
python3 figures/demo_video.py --no-text                # no captions, to add your own in an editor
python3 figures/demo_video.py --pair M07 --v0 1.2 --size 1280 720 --out m07.mp4
```

You need `ffmpeg` on the PATH (`brew install ffmpeg` or `apt install ffmpeg`). The video has four parts: the scene, a slow-motion passive slide with the predicted stop drawn on the table, the same slide for each shape, and the perception steps on the overhead depth image. The arm is posed by IK with the paddle behind the object but is not simulated; the controller (`arm.py`, 6 Nov) is not in the video yet. As with `--slide`, the object starts at `--v0`.

## Lookup table

```bash
python3 friction/lookup_table.py
```

This writes `friction_lookup.csv` (v2) and the LaTeX row files that `friction_lookup_table.tex` inputs. v2 keeps v1's 11 columns and appends `role` (secondary L01–L20, yu_pair Y01–Y03, rig M01–M27 and V01–V03), `tier` (A exact pair, B same object on a similar surface class, C different counterface or reversed geometry, none), `static_upper`, `source_speed_mps`, `source_method`, `verified`. Rules:
- **Kinetic only.** A static value never becomes an entry; a close static value goes in `static_upper`.
- **Spread by tier:** 90% interval μ/f to μ·f with f = 2 / 2.5 / 3 for A / B / C (team decision 3, 5 Nov; a modelling choice). The value enters `belief.py` as one Gaussian observation of ln μ on top of the population prior (`loglik_lookup`, `lookup_observation`). L01–L20 keep v1's spreads; L06 and Y02 share the pooled acetal-on-steel entry (0.25, f = 1.8, decision 5).
- **No Yu et al.** in any row or source list. Yu's three simulated pairs are simulator truth only (`ground_truth_published.csv`, Y01–Y03; Y04 dropped on 5 Nov); the lookup gives the same pair IDs independent values, which miss Yu's by about 2× on two of three.
- **Tier none** falls back to the default prior C0; C2's coverage is reported. v2 covers 17 of the 27 rig pairs (felt, acrylic and glazed tile have no kinetic source).
- L01–L20's simulator truth is sampled from the table's own interval, so report them as a memorisation probe.

`verified` is `n` everywhere until someone checks the value against the source PDF (the checklist is in the document).

## Model, estimation and decision (31 Oct)

`model_specification.tex` is the reference note; the code follows it section by section.

- **Release model** (`physics.py`): the object rides on the paddle until the paddle decelerates faster than μg, so `v_r = v_peak − (μg)²/(2J)` for constant-jerk braking. At J = 100 m/s³ and 0.8 m/s this costs 11% of the distance at μ = 0.3.
- **Observation models** (`observe.py`): frames at `fps` (60 Hz, `CAMERA_FPS`, decision 8) from the arm's separation time, the rest pose and the FK release point (models A, B, C), fitted by least squares; the O1 endpoint estimator is an ablation that carries the release-ramp bias. `crlb` gives the error budget: at 60 Hz and 2 mm frames, model C's relative error runs from 0.3% (μ = 0.1, 1.1 m/s) to 19% (μ = 0.5, 0.5 m/s); at the 31 Oct budget's 30 Hz it was 0.4% to 26%. `residual_signature` labels a fit's residuals clean / step / smooth trend by BIC; a strip crossing is only separable from spin in one slide at sub-millimetre frames or 60 Hz and up.
- **Belief** (`belief.py`): 500-point grid over ln μ; prior fitted on training folds; the LLM median enters through `m = α + βθ + ε` fitted leave-one-object-out; the lookup value (C2) enters as one Gaussian observation of ln μ with the row's `s_tab_log` (`loglik_lookup`); each slide's `(x_r, v_r)` is profiled out at every grid point. `n0 = σ_obs²/s0²` runs from 5 × 10⁻⁵ slides (slippery, fast) to 0.17 (μ = 0.5 at 0.5 m/s) at 60 Hz.
- **Decision** (`decide.py`): bounded 1-D search on the expected loss with the release model inside `d(μ, u)`; losses: squared relative, absolute relative, miss outside max(0.1 d*, 1 cm), table edge at 1.3 d*. The closed form `v² = 2 g d* exp(m − 1.5 S²)` is the J → ∞ limit. Practice slides run at the fastest speed that keeps the slide on the board for the slipperiest plausible μ (`slide_room(x0)`: the 48 in board's far edge less the release point and the object's half length; the release point is the start x0 plus the push's travel to release, `physics.release_push_distance`, so the room shrinks as the speed grows).
- **Prompt hygiene** (`llm_prior.py`): only `description_*` and `photo_*` columns can reach a prompt; test 7 fails if any other column does.

## Decisions (5 Nov)

What the team decided on 5 Nov, against what the code and data still carry. "Applied" means nothing is left to change; everything else is a placeholder until someone changes the file named.

| # | Topic | Decided 5 Nov | In the code and data |
|---|---|---|---|
| 1 | Board length | 48 in (1.22 m) | Applied 5 Nov: `physics.BOARD_LENGTH` = 48 × 0.0254 m, with the board at x = 0.3 to 1.52 m (`BOARD_X_MIN`, `BOARD_X_MAX`); `scene.Table` defaults to it (24 equal strips of 2 in, 5.08 cm) and `decide.slide_room` gives `practice_speed` its length limit from the release point, which moves with the push speed (`physics.release_push_distance`). Against the old table (x = −0.3 to 1.5 m) the near edge moves in by 0.6 m and the far edge out by 1.9 cm, so the E0 runs (starts at x = 0.55 m, rest points up to 1.35 m) stay on the board (`tests/test_scene.py`). The rig protocol's use of the board is the team's. |
| 2 | V03 reference value | 0.33 ("perpendicular", 0.32–0.34); list 0.19 ("crossed") as the alternative | Applied 5 Nov: V03 = 0.33 (source 0.32–0.34, static upper 0.54) with 0.19 in the notes; `friction_lookup.csv` and the `rows_*.tex` files regenerated. |
| 3 | Lookup spread factors | 2 / 2.5 / 3 for tiers A / B / C, entered as an observation on top of the population prior | Applied 5 Nov: `TIER_F` = 2 / 2.5 / 3 (L01–L20 keep v1's spreads); `belief.loglik_lookup` adds the row's value as one Gaussian observation of ln μ and `belief.lookup_observation` reads it by column name: `None` for a tier-none row, `KeyError` for a pair ID that is not in the table (tested in `tests/test_belief.py`). |
| 4 | Rig validation against Yu (V01, M18) | Keep as an external comparison; validate the rig by same-run cross-method checks | Applied as far as the data goes: rows stay labelled; the cross-method check is the accelerometer-vs-endpoint ratio (`mu_v2d`, `v2d_ratio_*`) that `aggregate_pairs.py` already writes. |
| 5 | L06 acetal on steel | One pooled entry, 0.25 ×/÷ 1.8, for L06 and Y02; if it must be binary, keep 0.20 | Applied 5 Nov: L06 and Y02 both carry 0.25 with f = 1.8 (90% 0.14–0.45) and the four pooled sources; Y02 is now 1.8× Yu's value instead of 2.2×. |
| 6 | Polyurethane (Y04) as simulator truth | Drop from the simulated set | Applied 5 Nov: the Y04 row is removed from `ground_truth_published.csv` (git history keeps it: Yu's 0.285, rising to 1.0 at 0.5 m/s), the lookup's Y04 candidate (0.61) is in the excluded list, and `--pair Y04` no longer resolves in `scene.py`. |
| 7 | Braking strategy (Diego) | The E0-4 evidence favours braking hard | Numbers in hand 6 Nov: the iiwa delivers every J in the E0-4 table at the release (the torque cost is set by the 12 m/s² brake cap, 0.83–0.89 of joint 4's limit at 0.8 m/s frontal, not by J), and the release model holds at J = 2000. `decide.py` still carries J = 100 m/s³: a one-line change once the team confirms the speed and the base (see *The arm controller*). |
| 8 | Camera rate | 60 Hz | Applied 5 Nov: `observe.CAMERA_FPS` = 60, the one place the rate is set (`scene.Camera` renders single poses and has no rate of its own); the error budget (`error_budget_rows.tex`, test 2) is regenerated at 60 Hz (model C 0.3–19% instead of 0.4–26%). `sim_config.yaml` stays contact-only; E0-5's 30/60/120 Hz study is unchanged. |
| 9 | Strip orientation | Across the travel direction, confirmed | Applied (`scene.py` tiles strips across +x). |
| 10 | Simulator-comparison branch | Audit done; default no-go | Applied: `simready_audit.md` (5 Nov) read 9,597 physics materials in 2,669 assets of NVIDIA's `simready-assets`. Friction there is one fixed (static, dynamic) pair per material class (plastic 0.4/0.4, steel 0.3/0.3, cardboard 0.7/0.7, rubber 0.8/0.8, wood 0.5/0.5, ceramic 0.35/0.3), not measured; every rig material has a class, none a pair value; the engines' combination rules disagree by up to 0.27 on one pair. A C2-SimReady baseline is possible but coarse, which is why the default is no-go. |
| 12 | Repository layout | Flat, Y01–Y04 as the Yu rows, confirmed; add `data/raw/` only | Applied (no `sim/` package; `yu_sim_truth.csv` not created); the Yu rows are Y01–Y03 since decision 6. `data/raw/<session>/` appears with the first recordings. |
| new | Scene changes (`Contact` fields, `pusher`) | Keep; then make `sim_config.yaml` the only contact setup | Kept. `scene.py`'s `Contact` defaults (`hydroelastic_with_fallback`) are still separate from `sim_config.yaml`, which only the E0 scripts load through `drake_config.py`. |
| new | Rig scope | Split: S1 (white melamine) with the nine objects first, as a development set | Measurement plan; nothing in the code. |
| new | First-frame rule | A gap rule: use frames once the object is 0.5 mm clear of the plate face (31 ms after release at J = 100) | E0-5 perceives in a 0.5 mm window and showed the 30 ms effect, but `observe.py`'s fits do not apply the rule. |
| new | Tier B noise | Inflate to a stated sensor model; keep the as-measured values as a lower-bound arm | `e0_5_noise.csv` holds the as-measured values only; the sensor model is not written. |
| new | Arm re-run | Re-run 12 pushes with the arm, not the E0-3/E0-4 grids | Done 6 Nov (`arm_rerun.py`): the arm's release speed within 0.13% of the plate's in all 12, 11 of 12 within 1% of the model (the twelfth is the plate's known +1.18% case). |
| new | iiwa base position | Not discussed (the scene has it at the origin) | Open. At the origin the push is radial from the shoulder and joint 4's 75°/s caps the paddle at 0.32–0.47 m/s; beside the board at (0.6, 0.65) m the iiwa delivers 1.0 m/s within its limits (`arm.py --scan`). `ArmPusher(base=...)` and `scene.add_arm(X_WB=...)` take any base; the scene, demo and `BOARD_X_MIN` keep the origin. **16 Nov trial** at (0.6, 0.65) m: 1.02 m/s within speed and torque limits, all 73 E1 pushes within them, same accuracy, camera study unchanged with the arm in view (`results/side_base/`). **Default since 16 Nov** (`scene.DEFAULT_LAYOUT = "side"`), with the board shifted 10 cm to clear the base's footprint; the origin layout is kept as `--layout origin`. Signed off by the team 16 Nov. |
| new | Extra published pairs | Last resort, as a labelled subset | Nothing in the code. |

`model_specification.tex` and `friction_lookup_table.tex` record decisions 1, 2, 3, 5, 6 and 8 as applied (5 Nov). Still open in the code: decision 7's J in `decide.py` and the base position (both one-line changes once decided); `model_specification.tex` does not yet describe the arm controller.

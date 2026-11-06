# Slide-to-Target

MIT 6.4210 (Robotic Manipulation) team project, Fall 2026.

A simulated KUKA iiwa pushes a flat-bottomed object across a table and releases it so it slides to a target distance, `d = v² / (2 μ g)`. Friction `μ` is unknown. We ask a language model for a prior over `μ` from a text description of the object and the table, update it with a few practice slides (Bayesian, in `ln μ`), and choose the release speed that minimises expected landing error. The main question is how much the language-model prior is worth compared with a published friction table, practice slides alone, and the true friction.

## Status

Phase 1, Oct 25 – Nov 1: measuring ground-truth kinetic friction for 27 object–surface pairs (M01–M27) and 3 validation pairs (V01–V03). Output: `measured_friction.csv` (columns in `measured_friction_interface.md`). On 31 Oct the model specification, estimators, grid posterior, decision rule, Drake configuration and E0-1/E0-2 checks landed in response to the instructor's feedback; `model_specification.tex` is the reference note. On 5 Nov the team settled the open decisions (see *Decisions* below); decisions 1, 2, 3, 5, 6 and 8 are applied in the code and data, the rest are recorded there. On 5 Nov a prescribed-paddle push harness (`push.py`) stood in for the arm controller so that E0-3, E0-4 and E0-5 could run; all three pass, and decisions 7 and 8 now have their evidence. The same day the decision-10 SimReady audit (`simready_audit.md`) found that NVIDIA's library stores one friction pair per material class, so a C2-SimReady baseline is possible but coarse.

## Next: the arm controller (Diego)

The iiwa still has no controller. Every push so far (E0-3, E0-4, E0-5, the demo video) uses `push.py`'s prescribed plate as a stand-in, so the arm is the open piece of the pipeline. What exists, what we need, and the bar it has to meet:

**What exists.**
- `scene.py` builds the iiwa 14 at the origin with the paddle plate (`PADDLE`, 0.12 × 0.18 × 0.01 m) welded to the flange, rigid hydroelastic, pair friction `PADDLE_MU` = 0.3. `build_scene(builder, pair, ..., arm=True)` returns the plant, the object body, the iiwa model instance and the paddle body. Nothing drives the arm: a controller has to be connected to the plant's actuation input.
- `push.py` is the reference for what the paddle must do. `profile(v_peak, J)` is the speed profile (accelerate at 2 m/s², then one constant-jerk segment −J through the peak into the brake, so braking starts from zero deceleration at `v_peak`); `Pusher` is the plate on a prismatic joint under Drake's implicit PD; `push_slide(mu, v_peak, J, ...)` runs settle → push → brake → free slide and reads the release three ways (object state at t* = t_peak + μg/J, the model on the paddle's recorded motion, and the instant the paddle–object force vanishes). `build_scene(..., pusher=Pusher())` is how the plate replaces the arm.
- The contact configuration is frozen (`sim_config.yaml`: `lagged`, 1 ms, 10⁷ Pa, dissipation 1 s/m, margin 10⁻⁴ m). E0-4 showed the release model only holds under `lagged`; `similar` and `sap` tilt the box during the push. Drake's implicit PD actuator needs a SAP-family solver, which `lagged` is.

**What we need from the arm.**
1. A controller that moves the paddle face along +x following `profile(v_peak, J)`, with the face normal along the push (E0-3 tested plate yaw of 0, 0.5 and 2°: a yawed plate turns the box to its own angle at rest without giving it a heading, so the distance survives but the landing yaw does not), the plate's lower edge a few millimetres above the table (`Pusher.clearance` = 3 mm), and the plate covering the object's whole back face so the contact segment straddles the centre of mass (assumption A2). Joint-space or task-space is Diego's call; the output that matters is the paddle's recorded motion.
2. The same hook as the plate, so the E0 scripts can be rerun unchanged: either an object with `add(plant, contact, obj, x0)` that `build_scene` accepts as `pusher`, or an `arm=True` path that `push_slide` can drive. The release must be reported the same three ways, because the decision rule predicts the object's state at t*.
3. Numbers for decision 7 (braking strategy): the peak speed and the braking jerk J the iiwa can actually deliver with the paddle, given its joint velocity and torque limits. `e0_4_summary.txt` tabulates what the ramp costs: losing under 2% of the distance needs J ≈ 500 m/s³ at μ = 0.3 and about 2,000 at μ = 0.5. The team's 5 Nov reading of E0-4 is that the evidence favours braking hard, so the number that matters most is the largest J the iiwa can deliver. If it cannot reach the table's values, `decide.py` keeps the release model `v_r(μ)` (J = 100 m/s³ by default) and the ramp is modelled rather than removed.
4. The paddle's forward kinematics at release (the FK release point) and the plate face's position per step, which observation model C and the E0-5 camera replay use (frames are perceived in a window 0.5 mm ahead of the plate face).

**The bar.** `tests/test_push.py` is the acceptance test for the plate and should pass for the arm too: the profile reproduces `physics.release_speed`, and one push at μ 0.3, 0.8 m/s, J 100 lands within 1% of the release model and 2% of the free slide. Then re-run 12 pushes with the arm in place of the plate, not the E0-3/E0-4 grids (5 Nov decision): pick them from the grids so they span μ, `v_peak`, J and the box/disc cases, and compare each with the plate's result for the same case (E0-4's plate bar: speed at t* within 1% of `v_peak − (μg)²/(2J)`, tracking to 0.05 mm). The tracking error of the arm against the profile is the one new number the rest of the pipeline cannot supply.

## Contents

| File | Purpose |
|---|---|
| `friction_analysis.py` | Computes `μ_k` from phone-accelerometer (phyphox) recordings of a sled sliding on a flat board. Also does the six-position accelerometer calibration. |
| `lookup_table.py` | Builds the published-value friction lookup table (baseline C2): `friction_lookup.csv` plus LaTeX rows `rows_*.tex`. |
| `check_drake.py` | Smoke test for the Drake install: loads the iiwa 14 and checks a hydroelastic box slide against `d = v² / (2 μ g)`. |
| `scene.py` | The Drake scene: patchy table, object, iiwa 14 with paddle. Also runs passive slides against the predicted stopping distance. |
| `e0_sweep.py` | E0 contact-model study: passive slides across shapes and contact settings, checked against `d = v² / (2 μ g)`. |
| `perception.py` | Landing pose from the scene's overhead depth camera: depth noise, plane removal, segmentation, outline ICP. Also measures the perception error. |
| `demo_video.py` | Renders a demo video offscreen (VTK + ffmpeg): the scene, a passive slide, the four shapes, landing perception. |
| `ground_truth_placeholder.csv` | Synthetic friction values in the `measured_friction.csv` columns, used until the measured file exists. **Not friction estimates.** |
| `ground_truth_published.csv` | Yu et al. 2016's measured pairs (Y01–Y03, stainless steel on ABS, acetal, plywood; the polyurethane pair Y04 was dropped from the simulated set on 5 Nov, decision 6). Simulator friction comes from these plus our measured pairs. |
| `physics.py` | Analytic model: release model `v_r = v_peak − (μg)²/2J`, free slide, slides across friction strips, uniform-pressure disk spin integrator, tipping check (A5). |
| `observe.py` | Observation models A/B/C (frames, + rest pose, + FK release point) and the O1 ablation; least-squares fit; Cramér–Rao error budget; residual signatures. `python3 observe.py` prints the budget table and writes `error_budget_rows.tex`. |
| `belief.py` | Grid posterior over `ln μ` (500 points on [ln 0.05, ln 1]): prior from measured pairs, LLM observation model fitted leave-one-object-out, slide likelihood with `(x_r, v_r)` profiled out. |
| `decide.py` | Release-speed decision by expected loss with the release model inside `d(μ, u)`; four losses; closed-form limit; practice speed. |
| `llm_prior.py` | Prompt builder that can read only the `description_*` and `photo_*` columns (test 7), plus a frozen-prompt response cache. The model call is a stub for the LLM owner. |
| `sim_config.yaml`, `drake_config.py` | The one Drake contact configuration every experiment loads (frozen by E0-1), and contact-result checks (`assert_hydro_only`). |
| `e0_2_hydro_check.py` | E0-2: every contact hydroelastic at every step, in the passive and the push scene, with negative controls. |
| `e0_1_free_slide.py` | E0-1: free slide vs `d = v²/(2μg)` over approximation × damping × step × margin, plus spin and heading launches. Freezes A6. |
| `push.py` | Prescribed-paddle push harness: the plate on a prismatic joint under Drake's implicit PD, tracking a constant-jerk profile; one push-and-slide with the release read three ways. Stands in for the arm controller in E0-3 to E0-5. |
| `e0_3_contact_geometry.py` | E0-3: box (line contact) vs disc (point), COM offset, plate yaw and paddle friction: release heading, spin, lateral miss, slide error. |
| `e0_4_release.py` | E0-4: object speed at separation vs `v_peak − (μg)²/(2J)` over J, μ, v_peak and each contact approximation; the decision-7 table of what braking costs. |
| `e0_5_observation.py` | E0-5: pushes replayed through the depth camera at 30, 60 and 120 Hz; models A, B, C and O1 fitted; bias and SE vs the Cramér–Rao bound; Tier B noise values (`e0_5_noise.csv`). |
| `aggregate_pairs.py` | Per-pair aggregation of rig runs and friction profiles into `measured_friction.csv`. |
| `make_mock_measured.py` | Regenerates `ground_truth_placeholder.csv` with the measured-file columns (values preserved, new columns mock). |
| `measured_friction_interface.md` | Column contract and interface rules for the measured file, the run log, per-run results and `profiles.csv`. |
| `run_log_template.csv` | Run-log template: session, pair, file, lane, orientation, speed setting, `rest_mm`, accepted, reject reason. |
| `model_specification.tex` | Reference note: state and assumptions, contact table, generative model, observation models and error budget, inference, decision rule, fidelity ladder, E0 status. |
| `friction_lookup_table.tex` | Lookup table v2 document; its tables are the generated `rows_*.tex`. |
| `simready_audit.py` | Decision-10 audit of NVIDIA's SimReady Central library (`nvidia/simready-assets` on Hugging Face): downloads each asset's small physics layers, reads every physics material's static/dynamic friction, restitution and density with `usd-core`, matches the rig's materials by name, and plots `simready_friction.png`. Read-only; nothing in the pipeline uses it. |
| `simready_audit.md` | The audit's finding: what the library stores, the friction distribution, the rig-material match table, timing, and the conclusion for decision 10. |
| `simready_friction.png` | Distribution of the static and dynamic friction values authored on the library's physics materials. |
| `tests/` | pytest suite (42 tests): spec tests 1–7, the rig self-checks and the push harness. `python3 -m pytest tests`. |

## Setup

You need Python 3.12 or newer. Drake 1.58 ships wheels for macOS 26 on Apple silicon and for Linux with glibc 2.34 or newer (Ubuntu 22.04+). On Windows, use WSL2.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 friction_analysis.py --selftest
python3 check_drake.py
python3 -m pytest tests
```

`requirements.txt` pins Drake and `manipulation`, the 6.4210 course package.

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
python3 friction_analysis.py --make-calib px.csv mx.csv py.csv my.csv pz.csv mz.csv
```

This writes `calib.json` to the current directory. Scales should be within 1–2% of 1, and biases below 0.1 m/s².

**For each pair, analyse each orientation separately:**

```bash
python3 friction_analysis.py M07_*fwd*.csv --pair M07 --direction fwd --calib calib.json --out results.csv
python3 friction_analysis.py M07_*rev*.csv --pair M07 --direction rev --calib calib.json --out results.csv
```

The script prints `μ_k` for each run and flags rejects: a short slide, lateral force, or yaw. `--out` appends to the CSV; rejected runs are written too, with `accepted = 0` and a `reject_reason`. If more than 3 of a pair's 12 runs are rejected, fix the cause and redo them. Runs where the sled touched the paddle again after release are marked in the run log (`reject_reason`), which `--runlog` reads.

**Since 31 Oct** the script detects the true stop (along-force back at rest level for 50 ms) and integrates the untrimmed signal from there, so `v_start` no longer reads low by `μg × 0.03 s`. It also integrates twice for the slide distance `d_acc_m`, computes the endpoint estimate `mu_v2d = v_r²/(2 g d_tape)` from the run log's `rest_mm`, and bins the friction ratio by board position into `profiles.csv` (`--profiles`; 20 mm bins, features shorter than the 100 mm coupon are not resolvable). `--pair` and `--direction` default to the run log, then the file name. Then per pair:

```bash
python3 friction_analysis.py data/raw/S1/M07_*.csv --runlog run_log.csv --calib calib.json --out results.csv --profiles profiles.csv
python3 aggregate_pairs.py --results results.csv --profiles profiles.csv --runlog run_log.csv --meta ground_truth_placeholder.csv --out measured_friction.csv
```

`aggregate_pairs.py` averages forward and reverse within each lane and speed first, then writes the pair's `mu_k_nominal`, `s_tab_log`, `speed_effect`, `mu_slope_per_mps`, `strip_scale_mm` (profile correlation length, floored at 100 mm: the simulated strip width), `profile_repeatability` and the reject counts. Column meanings are in `measured_friction_interface.md`.

**Where data goes.** Raw recordings are small, so commit them, with each session's `calib.json` and run log beside them. Suggested layout: `data/raw/<session>/`.

## Simulation scene

`scene.py` builds one `MultibodyPlant` for a given pair from the ground-truth file.

**What's in the scene**
- **Table:** the 48 in board (decision 1; `physics.BOARD_LENGTH`, `BOARD_X_MIN`, `BOARD_X_MAX`): its top is at z = 0 and it runs from x = 0.3 to 1.52 m along the travel direction (+x), tiled with 5 cm strips across it (`Table.edges()`; the last strip absorbs the remainder).
  - Each strip's friction is the pair's `mu_k_nominal × exp(patchiness · s_tab_log · z)`, sampled per seed.
  - `--bimodal` uses two levels instead.
- **Object:** compliant hydroelastic.
  - Shapes: box, disc, ring, or block on 3 or 4 feet. The ring's volume and surface meshes are generated on the fly.
  - For robustness cases: an optional centre-of-mass offset, or one longer foot.
- **iiwa 14:** base at the origin, 0.3 m behind the board's near edge, with a flat rigid paddle welded to the flange.
  - No controller is included: connect one to the plant's actuation input.

**Getting the friction right.** Drake combines two geometries' friction as `2 μa μb / (μa + μb)`. The object therefore carries a fixed μ, and each strip and the paddle are given the value that makes the *pair* coefficient equal to its target.

```bash
python3 scene.py --pair M07 --meshcat                      # view the scene
python3 scene.py --pair M07 --slide 0.8                    # passive slide vs prediction
python3 scene.py --pair M07 --slide 0.8 --shape footed --foot-error 0.001 --patchiness 0
```

**What `--slide` does:** it gives the object a speed and lets it slide to rest. It compares the stopping distance with the prediction `g ∫ μ̄ dx = v₀²/2`, where `μ̄` is the strip friction averaged over the object's footprint. It prints PASS within 2% in distance and 2° in yaw.

**From another script:** use `build_scene(builder, pair, obj, table, contact, arm=True, seed=...)`, which returns the plant, the object body, the iiwa model instance and the paddle body.

### E0 contact-model sweep

```bash
python3 e0_sweep.py                            # about 2,700 slides, ~7 min on 10 cores
python3 e0_sweep.py --quick                    # a few slides, to check it runs
python3 e0_sweep.py --summarize e0_results.csv # reprint the summary from a saved run
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
python3 e0_2_hydro_check.py                 # every contact hydroelastic at every step; ~1 min
python3 e0_1_free_slide.py                  # 396 slides, ~1 min on 10 cores; writes e0_1_results.csv, e0_1_summary.txt, e0_1_summary.png
python3 e0_1_free_slide.py --summarize e0_1_results.csv
```

Both build the plant from `sim_config.yaml` through `drake_config.py`, with the strict `hydroelastic` contact model: a rigid–rigid pair throws instead of silently falling back to point contact (`Contact.contact_model`; the scene's default stays `hydroelastic_with_fallback`). `Contact.margin` sets the hydroelastic margin.

- **E0-2 passes:** box, disc, ring and footed, passive and pushed (arm posed by IK, joints locked, object against the paddle), 300 steps each: object–table and paddle–object contacts are hydroelastic at every step and there are no point-pair contacts. Negative controls: the paddle lowered 1 mm into the table makes strict mode throw; a free rigid box on the rigid table runs on point contact in fallback mode. (A contact between a *locked* arm and the world has no participating degree of freedom, so Drake drops it from the contact results even when the geometries overlap.)
- **E0-1 freezes A6** at the config values: `lagged`, 1 ms, 10⁷ Pa, dissipation 1 s/m, margin 10⁻⁴ m, which stop within 0.96% of `d = v²/(2μg)` with no yaw over μ ∈ {0.1, 0.3, 0.5} and v ∈ {0.6, 1.0} m/s. Spinning discs land within 2% of the uniform-pressure disk model (`physics.disk_spin_slide`; d/d₀ ≈ 1.06 at Rω/v = 0.5 and 1.27 at 1), heading launches within 1%. **Finding:** with the 10⁻⁴ m margin, Hunt–Crossley dissipation ≥ 10 s/m under `lagged` (≥ 1 s/m under `similar`) shortens slides by 4–85%, while without margin `lagged` passes up to 50 s/m; keep dissipation ≤ 1 s/m with this margin. 2 ms steps pass only without margin, 5 ms never; SAP passes at 1 ms with relaxation ≤ 0.01 s. 19 of 66 settings pass.

### E0-3, E0-4 and E0-5 (5 Nov)

The arm has no controller yet, so `push.py` drives the paddle plate directly: a rigid plate on a prismatic joint along the push, under Drake's implicit PD actuator (`JointActuator.set_controller_gains`, which needs the SAP-family solver: `lagged`, `similar` or `sap`) tracking a constant-jerk speed profile with a feed-forward force. The profile accelerates at 2 m/s² and then runs one jerk segment −J through the peak into the brake, so braking starts from zero deceleration at `v_peak`, as the release model assumes. The command is fed one time step ahead (the implicit scheme reaches a step's desired state at its end); the plate then tracks to 0.05 mm. `build_scene(..., pusher=Pusher())` puts the plate in place of the arm. The release is read three ways: the simulator's object state at the model's instant t* = t_peak + μg/J (what the decision rule predicts), the model on the plate's recorded motion, and the instant the paddle–object force actually vanishes. Near separation both bodies decelerate at about μg, so a one-step shift moves the speed by μg·Δt, 1.4% of a 0.36 m/s release at 1 ms.

```bash
python3 push.py --mu 0.3 --v 0.8 --J 100            # one push-and-slide, printed; --shape disc --com 0.02 --yaw 2 --mu-p 0.6 --meshcat
python3 e0_3_contact_geometry.py                    # 132 pushes, ~1 min on 8 workers -> e0_3_results.csv, e0_3_summary.txt, e0_3_summary.png
python3 e0_4_release.py                             # 66 pushes -> e0_4_results.csv, e0_4_summary.txt, e0_4_summary.png
python3 e0_5_observation.py                         # 3 pushes replayed 180 times through the camera, ~3 min -> e0_5_results.csv, e0_5_frames.csv, e0_5_noise.csv, e0_5_summary.*
```

- **E0-3 (A2) passes for the box:** with the plate covering the whole face, the box leaves with |ψ_r| ≤ 0.004° and a lateral miss ≤ 2 mm for every COM offset up to w/4, every plate yaw (0, 0.5, 2°) and every paddle friction (0.1, 0.3, 0.6); a yawed plate turns the box to its own angle (rest yaw = plate yaw) without giving it a heading. A disc with an offset COM spins off (heading 5–27°, miss 23–116 mm at e = R/4 and R/2) and follows a yawed plate at about two thirds of its angle. The box with its COM 5 mm inside the footprint edge rocks onto that edge during the push (tilt 4–5°, heading up to 6°); every disc with the COM 5 mm inside the rim topples. Pushed by a plate slid sideways so the covered segment no longer straddles the COM, the box turns 90° and leaves 33–86% below the release model. The slide error stays within 0.8% wherever the object does not tilt.
- **E0-4 (A3) passes; decision 7 has its table:** at the config contact the object's speed at t* is within 1% of `v_peak − (μg)²/(2J)` in 53 of 54 cases (worst +1.2% at μ 0.5, v 0.6, J 30, a 0.2 m/s release); on the plate's recorded motion within 0.08% everywhere. The paddle–object force vanishes 0–2 ms after t*. Under `similar` and `sap`, with any damping, the box tilts 0.4–1.2° during the push and leaves 1–7% off the model with spin; the release model needs `lagged` (0 or 1 s/m), which supports A6. What braking costs: at μ 0.3 and 0.8 m/s the brake must reach J ≈ 500 m/s³ to lose under 2% of the distance, at μ 0.5 about 2,000 (`e0_4_summary.txt`). Either option in decision 7 is consistent with the data; the choice is the team's.
- **E0-5 passes; decision 8 and Tier B have their numbers:** models A, B and C are within 1% of the simulator's μ at 30, 60 and 120 Hz for μ ∈ {0.1, 0.3, 0.5}, with SE at or below the Cramér–Rao bound at the measured noise (ratio ≤ 0.73). Their residual −0.6 to −0.9% bias at μ 0.3–0.5 is the first frame after release, taken with the object within a millimetre of the plate: dropping the first 30 ms of frames leaves ≤ 0.3%. O1's bias matches the release-ramp prediction (+1.2%, +12%, +39%). Measured noise for Tier B: σ_f 0.09 mm (later frames 0.03–0.09 mm, the first frame +0.25 mm), σ_rest 0.03 mm near the camera axis and 0.1 mm at the far end of the table (x ≈ 1.1 m); σ_FK assumed 0.5 mm, since the simulator's forward kinematics are exact. The error-budget table assumed 2 mm frames, so its predictions are 10–20× pessimistic; the measured values are lower bounds, from a rendered camera with exact pose and intrinsics and no motion blur. Frames are perceived in a window starting 0.5 mm ahead of the plate face (known from FK): without it the plate's visible face joins the object's blob and biases the first frames by up to 1 mm. Model C's SE per slide is 0.42% at 30 Hz and 0.25% at 120 Hz on the shortest slide (μ 0.5), so the camera rate is not set by μ's precision but by what else the frames must resolve.

### Landing perception

`build_scene(..., camera=Camera())` adds an overhead depth camera. It looks straight down from 0.8 m over the landing zone, at 640 × 480, and is turned 3° about its axis. `perception.py` turns its depth image into the object's pose:
1. Add stereo-like depth noise.
2. Build a point cloud.
3. Fit and remove the table plane with RANSAC.
4. Keep the largest blob above the table.
5. Fit the object's top-view outline to the blob's edge with 2-D ICP, solving for x, y and yaw.

```bash
python3 perception.py --study 50                 # 50 random landing poses per shape -> perception_results.csv
python3 perception.py --study 50 --no-noise      # noise-free depth: the algorithm's own error
python3 perception.py --pair M07 --slide 0.8     # passive slide, then perceive where it stopped
```

From another script: `perceive(depth, camera, obj, rng, x_window=...)` returns the estimated pose. With the arm in the scene, use `x_window` to leave out the paddle.

### Demo video

```bash
python3 demo_video.py                          # demo.mp4: pair Y03, v0 = 1.4 m/s, 1920 x 1080, ~30 s to render
python3 demo_video.py --no-text                # no captions, to add your own in an editor
python3 demo_video.py --pair M07 --v0 1.2 --size 1280 720 --out m07.mp4
```

You need `ffmpeg` on the PATH (`brew install ffmpeg` or `apt install ffmpeg`). The video has four parts: the scene, a slow-motion passive slide with the predicted stop drawn on the table, the same slide for each shape, and the perception steps on the overhead depth image. The arm is posed by IK with the paddle behind the object but is not simulated, because it has no controller yet. As with `--slide`, the object starts at `--v0`.

## Lookup table

```bash
python3 lookup_table.py
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
| 1 | Board length | 48 in (1.22 m) | Applied 5 Nov: `physics.BOARD_LENGTH` = 48 × 0.0254 m, with the board at x = 0.3 to 1.52 m (`BOARD_X_MIN`, `BOARD_X_MAX`); `scene.Table` defaults to it (24 five-cm strips and a 6.9 cm last strip) and `decide.slide_room` gives `practice_speed` its length limit from the release point, which moves with the push speed (`physics.release_push_distance`). The far edge is where the old 1.8 m table ended, so the E0 runs stay on the board (`tests/test_scene.py`). The rig protocol's use of the board is the team's. |
| 2 | V03 reference value | 0.33 ("perpendicular", 0.32–0.34); list 0.19 ("crossed") as the alternative | Applied 5 Nov: V03 = 0.33 (source 0.32–0.34, static upper 0.54) with 0.19 in the notes; `friction_lookup.csv` and the `rows_*.tex` files regenerated. |
| 3 | Lookup spread factors | 2 / 2.5 / 3 for tiers A / B / C, entered as an observation on top of the population prior | Applied 5 Nov: `TIER_F` = 2 / 2.5 / 3 (L01–L20 keep v1's spreads); `belief.loglik_lookup` adds the row's value as one Gaussian observation of ln μ and `belief.lookup_observation` reads it by column name: `None` for a tier-none row, `KeyError` for a pair ID that is not in the table (tested in `tests/test_belief.py`). |
| 4 | Rig validation against Yu (V01, M18) | Keep as an external comparison; validate the rig by same-run cross-method checks | Applied as far as the data goes: rows stay labelled; the cross-method check is the accelerometer-vs-endpoint ratio (`mu_v2d`, `v2d_ratio_*`) that `aggregate_pairs.py` already writes. |
| 5 | L06 acetal on steel | One pooled entry, 0.25 ×/÷ 1.8, for L06 and Y02; if it must be binary, keep 0.20 | Applied 5 Nov: L06 and Y02 both carry 0.25 with f = 1.8 (90% 0.14–0.45) and the four pooled sources; Y02 is now 1.8× Yu's value instead of 2.2×. |
| 6 | Polyurethane (Y04) as simulator truth | Drop from the simulated set | Applied 5 Nov: the Y04 row is removed from `ground_truth_published.csv` (git history keeps it: Yu's 0.285, rising to 1.0 at 0.5 m/s), the lookup's Y04 candidate (0.61) is in the excluded list, and `--pair Y04` no longer resolves in `scene.py`. |
| 7 | Braking strategy (Diego) | The E0-4 evidence favours braking hard | `decide.py` keeps `v_r(μ)` at J = 100 m/s³; the switch waits on the J the iiwa can deliver (see *Next: the arm controller*). |
| 8 | Camera rate | 60 Hz | Applied 5 Nov: `observe.CAMERA_FPS` = 60, the one place the rate is set (`scene.Camera` renders single poses and has no rate of its own); the error budget (`error_budget_rows.tex`, test 2) is regenerated at 60 Hz (model C 0.3–19% instead of 0.4–26%). `sim_config.yaml` stays contact-only; E0-5's 30/60/120 Hz study is unchanged. |
| 9 | Strip orientation | Across the travel direction, confirmed | Applied (`scene.py` tiles strips across +x). |
| 10 | Simulator-comparison branch | Audit done; default no-go | Applied: `simready_audit.md` (5 Nov) read 9,597 physics materials in 2,669 assets of NVIDIA's `simready-assets`. Friction there is one fixed (static, dynamic) pair per material class (plastic 0.4/0.4, steel 0.3/0.3, cardboard 0.7/0.7, rubber 0.8/0.8, wood 0.5/0.5, ceramic 0.35/0.3), not measured; every rig material has a class, none a pair value; the engines' combination rules disagree by up to 0.27 on one pair. A C2-SimReady baseline is possible but coarse, which is why the default is no-go. |
| 12 | Repository layout | Flat, Y01–Y04 as the Yu rows, confirmed; add `data/raw/` only | Applied (no `sim/` package; `yu_sim_truth.csv` not created); the Yu rows are Y01–Y03 since decision 6. `data/raw/<session>/` appears with the first recordings. |
| new | Scene changes (`Contact` fields, `pusher`) | Keep; then make `sim_config.yaml` the only contact setup | Kept. `scene.py`'s `Contact` defaults (`hydroelastic_with_fallback`) are still separate from `sim_config.yaml`, which only the E0 scripts load through `drake_config.py`. |
| new | Rig scope | Split: S1 (white melamine) with the nine objects first, as a development set | Measurement plan; nothing in the code. |
| new | First-frame rule | A gap rule: use frames once the object is 0.5 mm clear of the plate face (31 ms after release at J = 100) | E0-5 perceives in a 0.5 mm window and showed the 30 ms effect, but `observe.py`'s fits do not apply the rule. |
| new | Tier B noise | Inflate to a stated sensor model; keep the as-measured values as a lower-bound arm | `e0_5_noise.csv` holds the as-measured values only; the sensor model is not written. |
| new | Arm re-run | Re-run 12 pushes with the arm, not the E0-3/E0-4 grids | Waits on the arm controller (see *Next: the arm controller*). |
| new | Extra published pairs | Last resort, as a labelled subset | Nothing in the code. |

`model_specification.tex` and `friction_lookup_table.tex` record decisions 1, 2, 3, 5, 6 and 8 as applied (5 Nov). Still open in the code: decision 7 (braking, waiting on the arm).

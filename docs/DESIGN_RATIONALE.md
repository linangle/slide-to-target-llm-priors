# Design rationale: what we chose and why

Written 25 Nov 2026. One place for the reasoning behind every design choice in this repository: what was chosen, the chain of reasoning, the evidence in the repository, and the literature it rests on. It also sets the proposal's plan against what exists (section 11) and records the logic checks of 25 Nov (section 12).

**Status of this text.** A reference for the team, like `paper/model_specification.tex`. The report's prose is the team's to write. Every number below is from the simulator or the analytic model; there is no rig data yet.

**How each entry is laid out.** *Choice*, then *Why* as numbered steps, then *Evidence* (the script or result folder), then *Limits* (what the choice does not cover). Numbers in square brackets are the references at the end. References marked † were checked against publisher records on 25 Nov; those marked ‡ come from the proposal's source-checked list of 17 Oct and were not rechecked.

---

## 1. The question and the task

**Choice.** One task, sliding a pushed object to a target distance, with friction as the single unknown; a language model supplies a prior over friction, practice slides update it, and the release speed minimises expected landing error.

**Why.**
1. After release nothing can be corrected, and the stopping distance is `d = v² / (2 μ g)`: mass cancels, friction does not. A 30% error in μ is a 30% error in distance.
2. Push-and-slide positioning has a long history with friction assumed known or learned from many trials: tapping [11], hitting with an iiwa [12], sliding unknown objects after a few identification pushes [13], curling with calibration throws [14], throwing with a learned residual [39]. None of them measures what a prior is worth before the first touch.
3. Language models are already used to seed physical parameters, friction among them [19, 20], and to elicit priors that data then update [17]. In those systems the prior sits inside a larger pipeline, so its own contribution cannot be separated. Stated confidence from language models tends to be too high [18].
4. A task whose physics is one equation lets every millimetre of miss be attributed to the prior, the execution or the model. That is the reason for choosing something this simple.

**Limits.** The proposal's hypotheses H1 to H4 assume the prior is worth a measurable fraction of a practice slide. The simulator says it is worth about 1/2000 of one (section 7), so the comparison between priors lives in the shot made before any practice. Section 10 says what the study became.

## 2. Physics of the slide

**Choice.** Coulomb kinetic friction, a rigid flat-bottomed object in pure translation, friction constant with speed. Spin, heading and tipping are diagnostics, not states.

**Why.**
1. For a translating body every support point moves with the same velocity, so the friction force is `μ m g` opposite to it whatever the pressure distribution; the pressure distribution decides only the torque [37, 4].
2. Spin is not harmless but its sign is known: sliding and spinning share the friction, so a spinning disk's sliding friction is reduced and it stops sliding and spinning at the same instant [5]. Integrating that model gives a slide 6.5% longer at `Rω/v = 0.5` and 27% longer at 1 (`physics.disk_spin_slide`). An earlier draft said off-centre friction makes the path "curve and shorten"; that was wrong and is corrected in the specification.
3. Speed dependence is excluded by choosing materials: Yu et al. found little change with speed for ABS, acetal and plywood, and a rise to about 1.0 for polyurethane [1]. Polyurethane was dropped as a simulated pair (decision 6, 5 Nov).
4. Tipping is excluded by geometry: no tip during the slide if `μ h_c < b`, and none during the push if `(a + μg) h_p ≤ g b + a h_c` (`physics.no_tip`).

**Evidence.** E0-1: boxes within 0.96% of `d = v²/(2μg)` with no yaw; spinning discs within 2% of the disk integrator; heading launches within 1% (`results/e0/e0_1_summary.txt`).

**Limits.** Static friction plays no part: Drake's discrete solver cannot resolve it and ignores the static coefficient [34, Drake friction-model page], and the lookup table is kinetic only for the same reason (section 5).

## 3. Contact between paddle and object

**Choice.** A flat paddle on a flat face (line contact in the plane), centre of mass inside the contact segment, push along the face normal.

**Why.**
1. With a line contact the pushing force can act anywhere along the contact segment, so the object translates with the pusher while the required force passes through the segment and stays inside the friction cone; this is the stable-push condition of Lynch and Mason [2], building on Mason's pushing mechanics [3]. (The condition was checked against a secondary summary; the primary text was not opened on 25 Nov.)
2. Under that condition the stopping distance does not depend on the centre of mass, so the centre of mass need not be a state. It is treated as known for boxes and tested as a sensitivity.
3. A disc gives point contact on purpose, to show where the assumption fails.

**Evidence.** E0-3: the box leaves with heading within 0.004° and a lateral miss within 2 mm for centre-of-mass offsets up to a quarter width, plate yaw up to 2° and paddle friction 0.1 to 0.6. A disc with an offset centre of mass spins off (heading 5 to 27°). With the covered segment not straddling the centre of mass the box turns 90° and leaves 33 to 86% below the release model (`results/e0/e0_3_summary.txt`).

**Limits.** The centre of mass is not in the belief. Real pushes are stochastic with input-dependent variance [15]; the simulated pushes repeat exactly.

## 4. Simulator settings

**Choice.** Drake [34], strict hydroelastic contact (object compliant, paddle and table rigid), the Lagged approximation, 1 ms step, modulus 10⁷ Pa, Hunt–Crossley dissipation 1 s/m, margin 10⁻⁴ m (`src/sim_config.yaml`, frozen by E0-1).

**Why.**
1. Hydroelastic contact gives a pressure field over a patch instead of point forces [6, 7], which is what a flat base on a table needs: point contact on a face lets the object rock and yaw.
2. The default `hydroelastic_with_fallback` silently switches a rigid–rigid pair to point contact, so experiments use the strict model, which throws instead (E0-2).
3. Lagged and Similar use Hunt–Crossley dissipation [10] and ignore `relaxation_time`; SAP [8] uses a linear model with `relaxation_time` and ignores Hunt–Crossley dissipation. Each parameter is swept only with the approximation that reads it. Castro et al. describe the Lagged and Similar approximations [9].
4. The choice among the three was made by the release, not the slide: under Similar and SAP the box tilts 0.4 to 1.2° during the push and leaves 1 to 7% off the release model; under Lagged it leaves within 1% (E0-4).
5. Drake combines two surfaces' coefficients as `2 μa μb / (μa + μb)` [34, CoulombFriction page], so the object carries a fixed value and each strip and the paddle get the partner value that makes the pair's coefficient the target (`scene.partner_mu`).

**Evidence.** E0 sweep (about 2,700 slides), E0-1 (19 of 66 settings pass; with the margin, dissipation of 10 s/m or more shortens slides by 4 to 85%), E0-2 (every contact hydroelastic at every step, with two negative controls).

**Limits.** The uniform board is tiled in 24 strips, and at each seam the contact patch vanishes for a step; about a third of slides on low friction land short, by up to 7.5 mm. This is a simulator artefact (D15). The rerun uses a one-piece board.

## 5. Friction truth, the lookup table and the rig

**Choice.** Three truth sets, kept apart: three published pairs from Yu et al. (Y01 to Y03, simulator truth only), twenty handbook pairs (L01 to L20, interim), and the team's own measurements (M01 to M27, V01 to V03, not yet available). The lookup baseline is kinetic only, with a spread set by how close the source is to the pair, and never uses Yu's values.

**Why.**
1. A friction coefficient belongs to a pair of surfaces under stated conditions, and tabulated values transfer poorly between setups [16, 42‡]. So the truth has to be measured for the pair, and a table is a baseline with a stated spread, not truth.
2. A language model may have read the tables. Measured pairs are the only truth it cannot have seen, which is why the rig exists, and why L01 to L20 always carry two caveats: the lookup holds the answer by construction, and the model may recall the same handbooks [17 reports the same contamination risk].
3. Yu et al. is the one published source with spatial spread, wear and speed dependence for the same surfaces [1]. Using it as simulator truth and giving the lookup independent values for those pairs keeps the baseline from being handed the answer; the independent values miss Yu's by about 2× on two of three pairs.
4. The lookup's spread is a 90% interval of μ/f to μ·f with f = 2, 2.5, 3 for an exact pair, the same object on a similar surface, and a different counterface (decision 3, 5 Nov). It enters the belief as one observation of ln μ on top of the generic prior (`belief.loglik_lookup`), the same way an expert opinion is combined with a prior [32].
5. The rig measures friction during a free slide with a phone accelerometer [30, 43‡]: the ratio of along-track to normal force is μ whatever the tilt or launch speed, and averaging the two sled orientations cancels mounting tilt and bias (`friction/friction_analysis.py --selftest`).

**Limits.** Every lookup row still has `verified = n`. **An error found and fixed on 25 Nov:** the Y03 (plywood) row of `data/ground_truth_published.csv` carried a spread of 0.25 with the note "spatial std 0.064 / mu". Yu et al. give 0.024 for plywood and 0.064 for polyurethane [1], so the row now has 0.09 (90% interval 0.224–0.301 in place of 0.172–0.392). No evaluation reads this column for the Y pairs (the strip spread is set explicitly), so no saved number changes; `scene.py --pair Y03` with its default patchiness now tiles a less patchy board. The specification and two READMEs had quoted "0.11–0.25" as Yu's range from it and are corrected. The file is hashed by the evaluators: saved runs record its sha256 before the fix (`f438d40b…`), the file is now `33ae4086…`.

## 6. The push and the release

**Choice.** The paddle accelerates at 2 m/s², then one constant-jerk segment takes it through the speed peak into the brake. Jerk J = 100 m/s³. The arm's brake is capped (12 m/s² by default; 6 m/s² for the rerun).

**Why.**
1. The object rides the paddle until the paddle decelerates faster than friction would slow the free object, so separation is at deceleration `μ g` and the release speed is `v_r = u − (μ g)² / (2J)`. The release speed therefore depends on the unknown friction, and the decision rule has to carry the release model inside it (section 8).
2. Starting the brake from zero deceleration at the peak avoids an acceleration jump, which made the plate dip and the object leave early at low friction.
3. J was reopened twice. A larger J makes the release less friction-dependent, which looked like a gain. The trial said otherwise: at J = 500 the first shot is 2 to 5 mm worse, the object leaves 0.15 to 0.54% slower than the model, and the oracle goes from 0.27 to 0.92 mm on a one-piece board. J stays 100 (D10, 20 Nov).
4. The brake cap is set by the arm's joint acceleration, not by the release (section 7). A cap of `a` releases only friction below `a / g`: 0.61 for 6 m/s².

**Evidence.** E0-4: the object's speed at separation is within 1% of the model in 53 of 54 cases over J from 30 to 2000, μ from 0.1 to 0.5 and peak speed from 0.6 to 1.0 m/s. Release-error grid with the arm: −0.02 to +1.23% (`results/side_base/e1_release_error_summary.txt`).

**Limits (corrected 25 Nov).** The commanded deceleration peaks at `min(a_brake, sqrt(2 J u))`, so a slow push cannot release a high-friction object. In Drake the bound is lower still, because a barely released object is re-touched by the settling paddle: a 0.33 m/s push releases friction up to about 0.64 with the default brake. The planner does not check this bound. Details in section 12.

## 7. The arm

**Choice.** A joint-space controller: the start posture is chosen among multi-start inverse-kinematics solutions, the path follows the minimum-norm joint tangent, and each joint runs Drake's implicit PD with the arm's inverse dynamics as feed-forward. The base stands beside the board at (0.6, 0.65) m. The speed cap and the targets are computed from the layout.

**Why.**
1. The push is a straight line, so the paddle pose fixes link 7's pose and leaves one redundant joint. Following the minimum-norm tangent `dq/ds = J⁺ eₓ` is resolved-rate control [29, 33]; choosing the start posture decides everything after it.
2. Feed-forward from the inverse dynamics [38] with PD on the error is the standard way to track a fast trajectory without high-gain lag [40, 33]. The plant clamps torque at the model's effort limits, so a brake the iiwa cannot deliver shows as tracking error instead of an unphysical push.
3. Where the base stands decides the speed. On the push line the wrist moves radially from the shoulder and only the elbow can produce that; joint 4's rated 75°/s [41] caps the paddle near 0.45 m/s. Beside the board joint 1 sweeps the paddle and the bound is 1.02 m/s inside the speed and torque limits. The proposal's own estimate (median 0.43 m/s over random configurations) had the same cause.
4. The cap and targets are computed, not hand-set, so that a change of layout or brake cannot leave a stale number in the protocol (D7, D8). A request above the cap is pushed at the cap and scored as a miss, as the proposal said.
5. Joint acceleration is set by the brake cap, not by the speed. At 12 m/s² the model's acceleration limits are exceeded 2 to 4× in every push. With a 6 m/s² cap and a posture chosen for the lowest worst limit ratio, 73 of 73 validation pushes are inside the speed, torque and acceleration limits (D9, decided 20 Nov for the rerun).

**Evidence.** `tests/test_arm.py` (the plate's acceptance test passes with the arm: release +0.11% of the model, tracking 0.05 mm rms); E1 (`results/side_base/e1_arm_summary.txt`); `results/side_base/base_trial/`; `results/side_base/posture_search/`.

**Limits.**
- The acceleration limits are Drake's model values. KUKA's specification gives rated speeds (85, 85, 100, 75, 130, 135, 135 °/s) and stopping distances but no acceleration [41]; the model's acceleration values are attributed in Drake's model repository to experiments in a 2019 master's thesis, not to KUKA.
- The simulator enforces torque limits but not speed or acceleration limits; the checks are on the planned trajectory.
- **The proposal's E1 comparison, run 25 Nov** (`experiments/e1_motion_generators.py`, `results/side_base/e1_motion_generators/`). What `arm.py` does is neither of the two: it is a pseudoinverse (differential-IK) path computed off line, with a search over the one free choice, the start posture. Against it, Drake's differential IK in a 1 ms loop and Drake's kinematic trajectory optimisation, given the same commanded paddle motion and executed by the same joint controller. With the base beside the board and the 6 m/s² brake, all three make every push from 0.2 to 1.1 m/s. Differential IK from arm.py's start posture gives arm.py's joint motion (release +0.1 to +0.5% of the model, landing within 0.4 mm, acceleration at most 0.93 of its limit, no clipping). Trajectory optimisation with velocity and acceleration bounds stays inside the limits but sits at the acceleration bound and reproduces the brake less exactly: release −0.4 to +6.9%, landing −1.5 to +4.6 mm. Without acceleration bounds it plans up to 3.2 times the acceleration limit. With the base on the push line, differential IK is clipped at 0.5 m/s and above and says nothing: the paddle is 2 to 39% slow, the release up to 44% low and the landing up to 118 mm short. Trajectory optimisation refuses those pushes instead, and with acceleration bounds refuses every push at that base.
  - *Why trajectory optimisation releases less exactly.* The object leaves when the paddle's deceleration passes μg, so the release depends on the paddle's acceleration profile. Pinning the pose at 70 times fixes positions to 0.02 to 0.07 mm, but a deviation that small over a few milliseconds is a large acceleration, and the optimiser spends its freedom up to the acceleration bound. `arm.py` plans the path over displacement and takes the timing from the profile exactly, so the brake is reproduced by construction.
  - *What decides the outcome is the base and the start posture, not the generator.* Where a posture exists that keeps the push inside the limits, the local method follows it without clipping; where none exists, no generator delivers the push, and they differ only in whether they say so.
  - *Not done:* the proposal's own trajectory-optimisation formulation (free timing, minimum duration, a release-velocity constraint), which would need its own release model; an optimisation of the path over displacement with the profile's timing kept exact; other brake caps.
- Not modelled: a real iiwa's 4 ms control cycle, joint friction, sensor noise.

## 8. Observation, belief and decision

### 8.1 What a practice slide measures

**Choice.** Model C: frames of the free slide at 60 Hz, the rest pose, and the release point from forward kinematics. The release position and speed are per-slide unknowns estimated from the data, never taken from the command.

**Why.**
1. The endpoint estimator with the commanded speed, `μ = u² / (2 g d)`, is biased by the release ramp by `(u / v_r)² − 1`, which grows with friction (+1.2%, +12%, +39% at μ = 0.1, 0.3, 0.5 in E0-5). It is kept only as an ablation.
2. Fitting the trajectory removes that bias. The Cramér–Rao bound [35] gives the price: the error scales roughly as `σ_f / (√N T²)`, so short slides are noisy (`observe.crlb`).
3. The same bound says the standard error falls as about `v^−2.5`, so practice slides run as fast as the board allows (`decide.practice_speed`). This is experiment design in its simplest form [26].
4. The landing pose comes from a depth image by plane removal with RANSAC [27] and outline fitting with ICP [28], as in the course notes [33].

**Evidence.** E0-5: models A, B, C within 1% of the simulator's friction at 30, 60 and 120 Hz, standard errors at or below the bound. Measured frame noise 0.09 mm, against the 2 mm the budget assumed.

**Limits.** The camera is not in the comparison loop: the evaluators add Gaussian noise to simulator positions at the nominal 2 mm, about 20× the rendered camera's. The first frame after release is biased by the plate's face; the gap rule decided on 5 Nov is not applied in `observe.py`'s fits.

### 8.2 The belief

**Choice.** A grid posterior over θ = ln μ. Generic prior: lognormal with 5th and 95th percentiles 0.08 and 1.1. Lookup: the generic prior times one observation. Language model: a lognormal from its median and stated interval, uncalibrated. The per-slide unknowns are profiled out. Friction does not change between slides (identity transition).

**Why.**
1. ln μ keeps friction positive and makes errors multiplicative, which is how both tables and language models err.
2. The slide likelihood is Gaussian in the measurements while the priors are Gaussian in ln μ, so a grid is exact where a conjugate update is not, and it can hold a multimodal belief (`tests/test_belief.py` checks it against the conjugate form).
3. The language model's interval is used as stated because the question is what the model knows before any correction. The specification's alternative, the model's median as an observation `m = α + βθ + ε` fitted leave-one-object-out (`belief.fit_llm_likelihood_loo`), needs measured pairs to fit and is written but unused.
4. The identity transition was a decision, not an omission: Yu et al. saw plywood fall from 0.28 to 0.24 over 100 passes [1], negligible over two slides.

**Evidence.** Calibration check of 25 Nov (section 12): with the truth drawn from the prior, the 90% interval holds it 90 to 91% of the time.

**Limits.** The identity transition also assumes friction is the same along the board, and that part fails. On a board whose strips vary with spread s, the friction the shot meets differs from the practice estimate by about 0.8 s, and the belief's 90% interval holds it 8 to 19% of the time (D18). A wider belief is honest but no more accurate; practising on the stretch the shot will cross is what reduces the miss. A model of friction as a field observed through slide integrals [25] would replace both fixes; it is not built.

### 8.3 The decision

**Choice.** The commanded peak speed minimises the expected loss under the belief, by a one-dimensional search with the release model inside the distance. A target point is the same rule with a slide length that moves with the speed.

**Why.**
1. This is the Bayes action for the stated loss [36]. With abrupt braking and squared relative loss it has the closed form `v² = 2 g d* exp(m − 1.5 S²)`: aim short, because distance goes as 1/μ and overestimating friction costs more than underestimating it by the same factor. The closed form is kept as a unit-test limit.
2. The release point moves with the speed (5.5 cm of paddle travel at 0.40 m/s, 28.9 cm at 1.0 m/s), so a point target fed as a fixed slide length lands about 2 cm short (D2).
3. The loss is not settled, on purpose. The relative loss is the specification's; the squared miss is what is scored in millimetres. They agree to 0.01 mm after one practice slide and differ only for the wide generic prior before practice (91.8 against 67.0 mm on 23 pairs). Both are run and reported (D3).

**Limits.** The proposal's H2 (uncertainty-aware against plug-in release, and a table-edge variant) is not run: `decide.loss_table_edge` exists, the plug-in policy and the experiment do not.

## 9. How the comparison is run

**Choice.** Every method sees the same practice observations; the scored push never feeds back into the belief; an oracle plans with the true friction; runs are hashed, never overwritten, and versions are not pooled.

**Why.**
1. Shared observations make every comparison paired, which removes the practice slide's noise from the difference between methods.
2. The oracle is the floor set by execution and the board, so a method's miss can be read against it.
3. One seed says nothing after practice: the seed is the practice slide's observation noise and the per-seed error varies tenfold. Ten seeds, and the 23-pair set where possible.
4. A prior can win by being sharp rather than by being right. Sharpness is only worth rewarding subject to calibration [21, 22], so the generic prior is also run at the language model's own interval width, pair by pair (D17). That control is the reason the headline changed (section 10).
5. Each run records the hashes of its sources and inputs, because a change to `arm.py` or the truth file silently changes every number. New versions go in as options whose default reproduces the old behaviour.

**Limits.** No bootstrap intervals or signed-rank tests yet, which the proposal named; paired differences with standard errors are reported in the patchy-board study only. The language-model answers for the 23 pairs are a stand-in (one fresh subagent per pair), not API calls with fixed settings.

## 10. What the results say so far

A map of regimes (D14), no single headline:

| Regime | What happens | Source |
|---|---|---|
| No practice | The prior is everything. 23 pairs, ten seeds: generic 91.8, lookup 20.5, language model 41.7 mm. On the three Yu pairs, which no table lists, neither language model beats the generic prior. | `results/side_base/trials_nov21/` |
| Confidence against knowledge | Handbook pairs, first shot: wide generic 61 mm, generic at the model's width 20 mm, language model 14 mm. Most of the gain is the width. | `results/side_base/pair_sets/` |
| Free practice, uniform board | One fast slide gives ln μ to 0.006; the priors' spreads are 0.26 to 0.80. All methods land within 0.02 mm of each other (about 1.7 mm after one slide, 1.2 after two; oracle 0.17 on a one-piece board). | `results/side_base/practice_study/` |
| Patchy board | The floor is the board, not the sensor. Fast practice on a different stretch is worse than planning with nominal friction. | `results/side_base/patchy_table/` |
| Practice aimed at the target | The first setting where the prior matters after practice: shots 5 to 19 mm apart. After one aimed slide at spread 0.12 the lookup beats the generic prior by 4.3 ± 0.8 mm, the language model by 3.6 ± 0.8, the width-matched control by 3.2 ± 0.7. | `results/side_base/patchy_table/pair_set/` |

The proposal expected a prior worth about 0.03 of a clean slide. It is worth about 1/2000, because the trajectory fit is far more informative than the endpoint observation the proposal's model assumed. Ways to check and sharpen the "confidence, not knowledge" result: proper scoring of the priors [21], a prior–data conflict check [23], and a robust mixture prior that discounts a confident wrong prior once data disagree [24]. None is implemented.

## 11. The proposal's plan against the repository

Against the long-form draft of 19 Oct (goals G1 to G4, conditions C0 to C7, experiments E0 to E5).

| Planned | State | Where, or what is missing |
|---|---|---|
| G1: friction measured for ≥ 24 pairs | Blocked on the rig | Analysis code and file contract exist (`friction/`, `docs/measured_friction_interface.md`) |
| E0: contact-model and assumption study | Done | `experiments/e0_*.py`, `results/e0/`. E0-6 (strip scale) waits on the rig |
| E1: execution, two motion generators | Done | `arm.py`'s controller validated against a plate and over a release-error grid; differential IK against trajectory optimisation compared on 85 pushes at two bases (25 Nov, section 7) |
| C0 default prior, C5 practice only | Done | `generic` at 0, 1, 2 slides |
| C2 table, C4 language model with interval, C6 fusion | Done as beliefs | `lookup`, `llm_raw` at 0, 1, 2 slides. C2 was specified as a plug-in; it runs with its spread |
| C7 oracle | Done | `oracle` |
| C1 language model asked for the speed directly | Not built | |
| C3 plug-in median, E3 uncertainty-aware release, table edge | Not built | The loss exists (`decide.loss_table_edge`) |
| C6 variants: robust prior, recalibration | Not built | `belief.fit_llm_likelihood_loo` exists, unused |
| E2 first shot on all pairs | Done on 23 simulated pairs | Stand-in language-model answers; GPT cached for Y01 to Y03 only |
| E4 practice curves, k = 0 to 5, three patchiness levels | Done differently | k = 0, 1, 2; practice study and patchy-board study in place of the overlay on the closed form |
| E5 calibration study | Mostly not done | Width-matched control done. Not done: interval score, two wordings × five samples, hidden-material and counterfactual prompts, memorisation probe, open-weight model, photographs |
| Tier A (arm in the loop) and Tier B (object only) | Tier B not built | Everything runs with the arm in Drake, which is the stronger version; the camera is in E0-5 only |
| ≥ 48 pairs, four targets, five seeds | 23 pairs, three targets, ten seeds | |
| Statistics: bootstrap intervals, signed-rank tests | Not built | |

## 12. Logic checks of 25 Nov

`python3 -m pytest tests` passes (100 tests), and the setup checks of the README pass. Three further checks, `experiments/logic_checks.py` → `results/logic_checks/summary.txt`:

1. **Release bound: a statement in the README and the decision log was wrong.** It said a push of peak speed u brakes at most at `sqrt(J u)`. The profile has no ramp back down, so the peak is `sqrt(2 J u)` (now pinned in `tests/test_push.py`). Consequences: the 12 m/s² cap is reached from 0.72 m/s, not "never"; the 6 m/s² cap from 0.18 m/s, not 0.36, so every push of the 6 m/s² runs brakes at the cap. In Drake the largest friction released is below the profile's bound: 0.50 / 0.59 / 0.67 / 0.80 for the plate at 0.20 / 0.26 / 0.33 / 0.45 m/s, and 0.56 / 0.64 / 0.76 for the arm with the default brake at 0.26 / 0.33 / 0.45 m/s. Close to the bound the object does leave at the model's speed, but the gap it opens before the paddle stops is tens of micrometres and the settling paddle closes it. The neoprene rig pairs (lookup 0.83) need about 0.5 m/s, not 0.66.
2. **The belief is calibrated.** With friction drawn from the prior and one simulated slide, the 90% interval holds the truth 90.2 to 90.9% of the time in three regimes (2,408 draws each). The belief sits slightly low in friction (mean posterior CDF at the truth 0.51 to 0.53), most where the slide is weakly informative. A marginal likelihood with a flat prior on the per-slide unknowns is better in the middle regime and worse in the prior-dominated one, so profiling is kept. The cause of the tilt was not isolated.
3. **The 2,000-point grid is fine at the camera's measured noise.** At 0.05× nominal noise the posterior is narrower than the grid spacing, and the landing differs from a 32,000-point grid by 0.05 mm on average and 0.15 mm at most.

Also found while checking references: the Y03 spread (section 5, fixed the same day), and that Yu et al.'s spatial spread for the three simulated surfaces is about 0.09 to 0.12, so the patchy-board column at 0.25 is about twice the realistic value. The finding that the 6 m/s² brake fails on a patchy board at 0.25 therefore concerns a board rougher than any of the three surfaces.

## 13. What is still open

Leaving the rig data aside:

1. **The single-protocol rerun.** The settings are decided (6 m/s² brake with the limits posture and the acceleration check, J = 100, one-piece board, three baselines, both losses, both miss measures, ten seeds, 23 pairs) but the run has not been made. Two pieces are missing: the one-piece board as an option in `scene.py` / `push.py` (today it is patched from outside), and the population prior (D17).
2. **Language-model answers by API**, with fixed settings and several samples per pair, for Claude and GPT, replacing the stand-in (D16). The 30 rig pairs still need their text descriptions, and the prompts must be frozen before anyone sees measured friction.
3. **Proposal items to build or to drop with a sentence in the report:** C1, C3 and the table edge (H2), the robust and recalibrated variants (H4), the calibration probes of E5, Tier B, bootstrap intervals.
4. **Decisions not taken:** which loss and which miss measure lead (D3, D4); the transition model and where to practise (D18); Tier B (D13); whether the rerun requires `--check-release`; a single fixed posture as an option in `arm.py`.
5. **Fixes waiting for a go** (hashed files): a release-bound check in the planner; the first-frame gap rule in `observe.py`. (The Y03 spread was fixed on 25 Nov.)
6. **Deliverables:** result figures (only the intro figure and per-experiment summaries exist), the report, the video (`figures/demo_side_base.mp4` shows a passive slide with the arm posed beside the board; the arm's push is not in it), the in-class summary.

---

## References

† checked against publisher records on 25 Nov 2026. ‡ from the proposal's source-checked list of 17 Oct, not rechecked.

1. † K.-T. Yu, M. Bauza, N. Fazeli, A. Rodriguez. More than a million ways to be pushed: a high-fidelity experimental dataset of planar pushing. IEEE/RSJ IROS 2016, pp. 30–37. arXiv:1604.04038. *As read on 25 Nov: spatial SD of the dynamic coefficient 0.016 (acetal), 0.017 (ABS), 0.024 (plywood), 0.064 (polyurethane); after 100 passes ABS 0.15→0.13, acetal 0.16→0.12, plywood 0.28→0.24, polyurethane 0.29→0.28; little speed dependence except polyurethane.*
2. † K. M. Lynch, M. T. Mason. Stable pushing: mechanics, controllability, and planning. Int. J. Robotics Research 15(6):533–556, 1996.
3. † M. T. Mason. Mechanics and planning of manipulator pushing operations. Int. J. Robotics Research 5(3):53–71, 1986.
4. † S. Goyal, A. Ruina, J. Papadopoulos. Planar sliding with dry friction. Part 1. Limit surface and moment function. Wear 143(2):307–330, 1991.
5. † Z. Farkas, G. Bartels, T. Unger, D. E. Wolf. Frictional coupling between sliding and spinning motion. Physical Review Letters 90(24):248302, 2003. arXiv:physics/0210024. *They show reduced sliding friction with spin and a common stopping instant; "slides farther" and the 1.065 and 1.27 ratios are our integration of their model.*
6. † R. Elandt, E. Drumwright, M. Sherman, A. Ruina. A pressure field model for fast, robust approximation of net contact force and moment between nominally rigid objects. IEEE/RSJ IROS 2019, pp. 8238–8245.
7. † J. Masterjohn, D. Guoy, J. Shepherd, A. Castro. Velocity level approximation of pressure field contact patches. IEEE Robotics and Automation Letters 7(4):11593–11600, 2022.
8. † A. M. Castro, F. N. Permenter, X. Han. An unconstrained convex formulation of compliant contact. IEEE Trans. Robotics 39(2):1301–1320, 2023.
9. † A. M. Castro, X. Han, J. Masterjohn. Irrotational contact fields. IEEE Trans. Robotics 41:6176–6192, 2025. arXiv:2312.03908.
10. † K. H. Hunt, F. R. E. Crossley. Coefficient of restitution interpreted as damping in vibroimpact. J. Applied Mechanics 42(2):440–445, 1975.
11. † W. H. Huang, M. T. Mason. Mechanics, planning, and control for tapping. Int. J. Robotics Research 19(10):883–894, 2000.
12. † H. Khurana, M. Bombile, A. Billard. Learning to hit: a statistical dynamical system based approach. IEEE/RSJ IROS 2021, pp. 9415–9421. Journal follow-up: H. Khurana, A. Billard. Motion planning and inertia-based control for impact aware manipulation. IEEE Trans. Robotics 40:2201–2216, 2024.
13. † C. Song, A. Boularias. Learning to slide unknown objects with differentiable physics simulations. Robotics: Science and Systems XVI, 2020. arXiv:2005.05456.
14. † D.-O. Won, K.-R. Müller, S.-W. Lee. An adaptive deep reinforcement learning framework enables curling robots with human-like performance in real-world conditions. Science Robotics 5(46):eabb9764, 2020.
15. † M. Bauza, A. Rodriguez. A probabilistic data-driven model for planar pushing. IEEE ICRA 2017, pp. 3008–3015. arXiv:1704.03033.
16. † P. J. Blau. The significance and use of the friction coefficient. Tribology International 34(9):585–591, 2001.
17. † A. Capstick, R. G. Krishnan, P. Barnaghi. AutoElicit: using large language models for expert prior elicitation in predictive modelling. Proc. 42nd ICML, PMLR 267:6746–6777, 2025. arXiv:2411.17284.
18. † M. Xiong, Z. Hu, X. Lu, Y. Li, J. Fu, J. He, B. Hooi. Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. ICLR 2024. arXiv:2306.13063.
19. † A. J. Zhai, Y. Shen, E. Y. Chen, G. X. Wang, X. Wang, S. Wang, K. Guan, S. Wang. Physical property understanding from language-embedded feature fields. IEEE/CVF CVPR 2024, pp. 28296–28305. arXiv:2404.04242.
20. † Y. J. Ma, W. Liang, H.-J. Wang, Y. Zhu, L. Fan, O. Bastani, D. Jayaraman. DrEureka: language model guided sim-to-real transfer. Robotics: Science and Systems XX, 2024. arXiv:2406.01967. *The arXiv version lists an eighth author, S. Wang.*
21. † T. Gneiting, A. E. Raftery. Strictly proper scoring rules, prediction, and estimation. J. American Statistical Association 102(477):359–378, 2007.
22. † T. Gneiting, F. Balabdaoui, A. E. Raftery. Probabilistic forecasts, calibration and sharpness. J. Royal Statistical Society B 69(2):243–268, 2007.
23. † M. Evans, H. Moshonov. Checking for prior-data conflict. Bayesian Analysis 1(4):893–914, 2006.
24. † H. Schmidli, S. Gsteiger, S. Roychoudhury, A. O'Hagan, D. Spiegelhalter, B. Neuenschwander. Robust meta-analytic-predictive priors in clinical trials with historical control information. Biometrics 70(4):1023–1032, 2014.
25. † J. N. Hendriks, C. Jidling, A. Wills, T. B. Schön. Evaluating the squared-exponential covariance function in Gaussian processes with integral observations. arXiv:1812.07319, 2018. *Preprint only.*
26. † K. Chaloner, I. Verdinelli. Bayesian experimental design: a review. Statistical Science 10(3):273–304, 1995.
27. † M. A. Fischler, R. C. Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM 24(6):381–395, 1981.
28. † P. J. Besl, N. D. McKay. A method for registration of 3-D shapes. IEEE Trans. Pattern Analysis and Machine Intelligence 14(2):239–256, 1992.
29. † D. E. Whitney. Resolved motion rate control of manipulators and human prostheses. IEEE Trans. Man-Machine Systems 10(2):47–53, 1969.
30. † S. Staacks, S. Hütz, H. Heinke, C. Stampfer. Advanced tools for smartphone-based experiments: phyphox. Physics Education 53(4):045009, 2018.
31. † G. Schwarz. Estimating the dimension of a model. Annals of Statistics 6(2):461–464, 1978. *The criterion `observe.residual_signature` uses to label a fit's residuals.*
32. † R. T. Clemen, R. L. Winkler. Combining probability distributions from experts in risk analysis. Risk Analysis 19(2):187–203, 1999.
33. † R. Tedrake. Robotic Manipulation: Perception, Planning, and Control. Course notes for MIT 6.421, 2026. http://manipulation.mit.edu
34. † R. Tedrake and the Drake Development Team. Drake: model-based design and verification for robotics, 2019. https://drake.mit.edu. *Documentation pages used: the friction model (static coefficient ignored by discrete solvers), CoulombFriction (the combination rule), the hydroelastic user guide (which dissipation parameter each approximation reads).*
35. † S. M. Kay. Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory. Prentice Hall, 1993.
36. † J. O. Berger. Statistical Decision Theory and Bayesian Analysis, 2nd ed. Springer, 1985.
37. † M. T. Mason. Mechanics of Robotic Manipulation. MIT Press, 2001.
38. † J. Y. S. Luh, M. W. Walker, R. P. C. Paul. On-line computational scheme for mechanical manipulators. ASME J. Dynamic Systems, Measurement, and Control 102(2):69–76, 1980. *The recursive Newton–Euler inverse dynamics.*
39. † A. Zeng, S. Song, J. Lee, A. Rodriguez, T. Funkhouser. TossingBot: learning to throw arbitrary objects with residual physics. Robotics: Science and Systems XV, 2019; IEEE Trans. Robotics 36(4):1307–1319, 2020.
40. † P. K. Khosla, T. Kanade. Real-time implementation and evaluation of the computed-torque scheme. IEEE Trans. Robotics and Automation 5(2):245–253, 1989.
41. † KUKA Deutschland GmbH. LBR iiwa specification ("Spez LBR iiwa V8"), 2019, section 4.3.2, rated axis speeds. *Read from a third-party-hosted copy; it contains no acceleration values. Drake's iiwa14 acceleration limits: README of `RobotLocomotion/models`, `iiwa_description`.*
42. ‡ ASTM International. Standard guide for measuring and reporting friction coefficients. ASTM G115-24, 2024.
43. ‡ C. Baldock, R. Johnson. Investigation of kinetic friction using an iPhone. Physics Education 51(6):065005, 2016.

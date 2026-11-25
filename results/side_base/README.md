# Results at the default layout: the side base (16 Nov)

Everything here was produced with the iiwa's base beside the board at (0.6, 0.65) m (`scene.LAYOUTS["side"]`), the default since 16 Nov. The layout before that, with the base at the origin, is kept: its results are in the folders next to this one (`../e1/`, `../arm_rerun/`, `../evaluation_v3/`, `../pilot_v1/`, `../pilot_v2/`; `../e0/` used no arm) and `--layout origin` or `--v3` reruns it. Do not pool the two layouts.

Commands are written from the repository root. Each is the script's default, so no output paths are needed.

| Files | What | Made with |
|---|---|---|
| `arm_clearance.txt`, `arm_clearance_brake4.txt` | Clearance of every link from the board, the object and the camera along the push at the speed cap; the base's footprint against the board | `python3 experiments/arm_clearance.py --out ...` (`--a-brake 4`) |
| `arm_clearance_origin_for_comparison.txt` | The same at the origin | `... --layout origin --out ...` |
| `e1_arm_*` | E1 arm validation, same 85 pushes as `../e1/` | `python3 experiments/e1_arm_validation.py` |
| `e1_arm_brake6_*` | The same with a 6 m/s² brake cap (releases friction below 0.61) | `python3 experiments/e1_arm_validation.py --a-brake 6` |
| `arm_rerun_*` | The 12-push arm re-run, arm beside the plate, as `../arm_rerun/` | `python3 experiments/arm_rerun.py` |
| `e0_2_*` | E0-2 contact check with the arm of the pushed scene at the side base (`../e0/e0_2_*`: at the origin) | `python3 experiments/e0_2_hydro_check.py` |
| `e0_5_*` | E0-5 camera study with the arm pushing and in view (`../e0/e0_5_*`: the plate and the original camera) | `python3 experiments/e0_5_observation.py` |
| `perception_results.csv` | Landing-perception error study with the default camera (`../perception_results.csv`: the original camera) | `python3 src/perception.py --study 50` |
| `evaluation/brake12_speed_torque/` | Method comparison, Y01–Y03, seed 7, the evaluator's default: computed cap, derived targets, 12 m/s² brake; speed and torque required, acceleration recorded | `python3 evaluation/run_arm_evaluation.py --pair all --seeds 7 --out ...` |
| `evaluation/brake4_all_limits/` | The same with a 4 m/s² brake cap and acceleration required as well | `... --a-brake 4 --check-accel --out ...` |
| `evaluation/brake6_posture_limits_same_targets/` | 19 Nov trial: the default run's cap and targets (1.019 m/s; 0.60 / 0.77 / 0.94 m) with a 6 m/s² brake, the posture chosen for all three limits, acceleration required. Like for like with `brake12_speed_torque/` | `... --pair all --seeds 7 --a-brake 6 --check-accel --posture limits --peak-cap 1.019 --targets 0.60 0.77 0.94 --out ...` |
| `evaluation/brake6_posture_limits/` | The same with the cap and targets that setup computes for itself: 1.158 m/s; 0.60 / 0.825 / 1.05 m | `... --pair all --seeds 7 --a-brake 6 --check-accel --posture limits --out ...` |
| `trials_nov20/` | 20 Nov trials through `evaluation/run_trial.py`, nine runs (Y01–Y03, seed 7, 6 m/s² brake, `--posture limits`, cap 1.019, targets 0.60 / 0.77 / 0.94 m): `striped_` / `onepiece_` × `rel_` / `sqabs_` × `J100` / `J500`, and `onepiece_rel_J100_gap` (start gap 0.5–3 mm per push). Each folder also has `trial_override.json`, `release_log.csv`, `both_measures.csv`. `summary.txt` compares them; `striped_rel_J100` equals `evaluation/brake6_posture_limits_same_targets` byte for byte | `python3 evaluation/run_trial.py [--loss sq_abs] [--jerk 500] [--one-piece] [--gap-error 7] -- --pair all --seeds 7 --a-brake 6 --check-accel --posture limits --peak-cap 1.019 --practice-peak 1.01 --targets 0.60 0.77 0.94 --out ...` |
| `trials_nov21/yu_10seeds/` | 21 Nov: the 20 Nov trials on ten seeds (0–9), Y01–Y03, with the rerun's setup (6 m/s² brake, `--posture limits`, acceleration checked, cap 1.158, targets 0.60 / 0.825 / 1.05 m): `onepiece_rel`, `onepiece_sqabs`, `onepiece_rel_gap`, `striped_rel`, `onepiece_rel_J500`, 1,080 rows each; `summary.txt` | `python3 evaluation/run_trial.py [--loss sq_abs] [--jerk 500] [--one-piece] [--gap-error 7] -- --pair all --seeds 0 1 2 3 4 5 6 7 8 9 --noise nominal --a-brake 6 --check-accel --posture limits --out ...`; `python3 evaluation/trial_summary.py ...` |
| `trials_nov21/pairs23_10seeds/` | 21 Nov: the same five settings on the 23-pair set (L01–L20, Y01–Y03; Claude Haiku stand-in), seeds 0–9, targets 0.60 / 0.78 / 0.96 m, 5,796 rows each. Each folder has the runner's files plus `trial_override.json`, `release_log.csv` and `both_measures.csv`; `summary.txt` | `python3 evaluation/run_pair_set_trial.py --a-brake 6 --check-accel --posture limits [--loss sq_abs] [--jerk 500] [--one-piece] [--gap-error 7] -- --llm claude=data/llm_standin_claude_haiku.json --seeds 0 1 2 3 4 5 6 7 8 9 --out ...`; `... --summarize RUN [RUN ...] --out summary.txt` |
| `posture_search/` | 21 Nov: the search for one start posture inside the joint speed, torque and acceleration limits at every speed, 6 m/s² brake (`brake6.txt`, `.json`), and its Drake check through E1 and the release-error grid (`brake6_single_*`) | `python3 experiments/posture_search.py`; `... --validate` |
| `evaluation/ten_seeds_Y01/`, `_Y02/`, `_Y03/` | The default setup over ten noise seeds, one folder per pair (1,116 scored rows each) | `python3 evaluation/run_arm_evaluation.py --pair Y01 --seeds 0 1 2 3 4 5 6 7 8 9 --out ...` |
| `evaluation/clip_beyond_reach/` | Targets 0.60 / 0.94 / 1.10 m with `--clip-unreachable`: the last is beyond Y03's reach, to exercise the clip-and-score rule | `... --pair all --seeds 7 --targets 0.60 0.94 1.10 --clip-unreachable --out ...` |
| `evaluation/loss_sq_abs/` | The default setup planned with the squared-miss loss instead of the relative loss | `python3 evaluation/run_with_loss.py sq_abs --pair all --seeds 7 --out ...` |
| `e1_release_error.csv`, `_summary.txt` | The arm's execution error against the release model over friction × speed (35 pushes): the Tier B execution model's input | `python3 experiments/e1_release_error.py` |
| `practice_study/` | How informative a practice slide has to be before the prior matters (17 Nov): `summary.txt`, `summary.png`, `results.csv` (per practice speed, noise, slides and method), `cells.csv.gz` and `transfer_cells.csv.gz` (every scored shot), `landing_map.csv` and `map_check.json` (the Drake landing map and the board's seam losses), `grid_check.json`, `verify.csv`, `settings.json` | `python3 evaluation/practice_study.py` |
| `patchy_table/` | The transition model tested on the scene's patchy board (strip friction varying by 0–25%), Y01–Y03, 16 boards per pair and spread, 12,186 scored shots: `summary.txt`, `results.csv`, `transfer_sd.json`. Practice aimed at the shot's own target, same boards, 5,292 shots: `aimed_summary.txt`, `aimed.csv`, `aimed_settings.json`. The aimed rule on the 23-pair set with a matched-width control, 22,287 shots: `pair_set/summary.txt`, `pair_set/aimed.csv`, `pair_set/settings.json` | `python3 evaluation/patchy_table_study.py`, then `--aimed`, then `--aimed-pairs` |
| `patchy_table_brake6/` | 21 Nov: the patchy board under the rerun's arm setup (6 m/s² brake, `--posture limits`, acceleration checked). `strip_census.txt`: saved boards with a strip above the brake's release bound (friction 0.61). `pair_set/`: the aimed-practice study on the nine pairs that have one plus L10 and Y03, same boards, cap 1.019 and targets as `patchy_table/pair_set/` (187 boards, 10,641 shots). `compare.txt`: row by row against the saved run. `release_audit_s025.csv`, `_s012.csv`, `_brake12_s025.csv` with `_summary.txt` and `.log`: the distinct pushes behind the saved rows replayed, with whether the object left the paddle before it stopped (6 m/s² run at spread 0.25, all 11 pairs; at 0.12, L02 L17 L16; the saved 12 m/s² run at 0.25, five pairs) | `python3 evaluation/run_patchy_trial.py --census --out ...`; `python3 evaluation/run_patchy_trial.py --a-brake 6 --check-accel --posture limits --peak-cap 1.019 --pairs L18 L05 L03 L19 L01 L04 L02 L17 L16 L10 Y03 -- --aimed-pairs --set-out .../pair_set`; `... --compare OLD.csv NEW.csv --out ...`; `... --release-audit AIMED.csv --strip-sd 0.25 --a-brake 6 --check-accel --posture limits --out .../release_audit_s025.csv` |
| `base_trial/` | The 19 Nov trial of the brake cap and the base position (decision 9): planning scans over every speed with either posture choice (`scan_speed_posture.txt`, `scan_limit_aware.txt`); E1, release error, clearance and camera replay at the default base with a 6 m/s² brake and the limit-aware posture (`base060_brake6_limit_aware_*`); the camera replay with `arm.py`'s posture for comparison (`base060_brake6_speed_posture_e0_5_*`); E1 and release error at a base 8 cm further along (`*base068_brake6*`), which fails on slow pushes; E1 and release error rerun through the option that replaced the patch (`option_brake6_limits_*`, identical to the patch's in every cell) | `python3 experiments/base_trial.py --scan`, `--validate`, `--camera`; `python3 experiments/e1_arm_validation.py --a-brake 6 --posture limits --out ...`; `python3 experiments/e1_arm_validation.py --base-x 0.68 --base-y 0.65 --a-brake 6 --out ...` |
| `pair_sets/handbook_claude/` | The comparison on 23 pairs: the 20 handbook pairs (L01–L20) and Y01–Y03, with Claude Haiku as the LLM (stand-in), seeds 0–2, 1,932 scored shots | `python3 evaluation/run_pair_set.py --llm claude=data/llm_standin_claude_haiku.json --out ...` |
| `pair_sets/yu_gpt_claude/` | Y01–Y03 with GPT and Claude side by side | `... --llm gpt=data/llm_cache_published_pilot.json claude=data/llm_standin_claude_haiku.json --pairs Y01 Y02 Y03 --out ...` |
| `pair_sets/llm_comparison.txt` | The LLMs against the truth and each other from their cached answers (no simulation), with the matched-width control | `python3 evaluation/compare_llms.py --llm claude=... gpt=... --out ...` |

## What they show

| | origin (saved folders) | side base (here) |
|---|---|---|
| Fastest push within joint speed and torque limits | 0.49 m/s | 1.02 m/s |
| Lowest moving link above the board | 55 mm | 51 mm |
| Base footprint | 179 mm before the board's near edge | clears the shifted board's edge by 29 mm (a centred board would be 71 mm under it) |
| E1: arm pushes within joint speed limits | 22 of 73 | 73 of 73 |
| E1: targeting with the true friction, RMS and worst miss | 0.74, 2.4 mm | 0.83, 2.7 mm |
| E1: repeatability, worst miss | 2.8 mm | 2.6 mm |
| Arm re-run: pushes within speed and torque limits | 0 of 12 | 12 of 12 |
| Arm re-run: release speed against the plate's | −0.13..−0.01% | −0.13..+0.02% |
| E0-2: hydroelastic contact only, pushed scene | PASS | PASS |
| E1: pushes within the acceleration limits, 12 m/s² brake cap | not recorded | 5 of 73 |
| E1: pushes within the acceleration limits, 6 m/s² brake cap | not run | 65 of 73 |

## The camera

`e0_5_*` and `perception_results.csv` use the full-coverage camera at (1.0, 0, 1.0) m, which sees x = 0.43–1.57 m (the original, at (0.75, 0, 0.8) m, saw to 1.21 m and lost the longest slide). E0-5 here also uses the height gate in the perception window. The study behind both is in `../camera_study/`.

| | original camera, plate (`../e0/`) | full-coverage camera, arm in view (here) |
|---|---|---|
| E0-5: worst bias of models A/B/C | 0.90% | 2.2% |
| E0-5: worst bias without the first 30 ms of frames (C1) | 0.31% | 0.45% |
| E0-5: frame noise at 60 Hz | 0.089 mm | 0.16 mm |
| E0-5: frames lost | 0 | 0 |
| Perception study: rest-pose x error, SD (box) | 0.05 mm | 0.12 mm |
| A slide resting at x = 1.17 m | 23% friction bias (out of view) | seen |

## The evaluation

Y01–Y03, seed 7, 432 scored rows per run; each run folder has its own `summary/`. The computed cap is 1.019 m/s with the default 12 m/s² brake (speed and torque required) and 1.017 m/s with a 4 m/s² brake (acceleration required too); practice pushes run at 1.01 m/s; the derived targets are x = 0.60, 0.77 and 0.94 m. Landing RMSE at nominal noise:

| practice slides | generic | lookup | LLM | oracle |
|---|---|---|---|---|
| 0 | 54.8 mm | 50.8 mm | 65.2 mm | 0.4 mm |
| 1 | 1.5 mm | 1.5 mm | 1.5 mm | 0.4 mm |
| 2 | 0.5 mm | 0.6 mm | 0.6 mm | 0.4 mm |

- **Every push is valid in both runs.** With the 4 m/s² brake the scored pushes peak at 0.80 of the joint speed limit, 0.25 of the torque limit and 0.75 of the acceleration limit; the practice pushes at 0.98, 0.25 and 0.85. With the 12 m/s² brake the acceleration reaches 2.1× (scored) and 2.3× (practice).
- **The brake cap does not change the outcome.** The two runs agree within 0.1 mm.
- **No push reached the cap** with the derived targets, which are in reach by construction. `clip_beyond_reach/` puts one target (1.10 m) beyond Y03's reach (1.03 m): the oracle and most methods push at the cap, land 66 mm short and are scored; 13 of 108 nominal rows are at the cap and the run completes.
- **The loss matters much more at this range.** `loss_sq_abs/` against the default: zero-practice generic 34.6 mm against 54.8, lookup 54.3 against 50.8, LLM 67.1 against 65.2; identical after one practice slide.
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
- **Farther targets make the starting belief matter much more.** Before any practice the methods miss by 51–65 mm, against 9–17 mm in the saved origin runs whose targets stop at 0.64 m. One practice slide brings all three to 1.5 mm. The two sets are not like for like: different targets, and the saved runs used the squared-miss loss (see the main README).
- **The evaluation reads simulator state, not the camera**, so the camera setting does not affect it.

## The practice study

`practice_study/summary.txt` has every table; `summary.png` the figure. Same setup as the evaluation above; the practice slide is weakened by a slower practice push and by more observation noise.

- One slide at the default setting gives ln μ to 0.006 on its own; the priors' SDs are 0.26–0.80. The three methods' shots then land 0.03 mm apart.
- Their shots land more than 1 mm apart once a slide alone leaves ln μ uncertain by about 0.035: a practice push of 0.45 m/s or slower at nominal noise, or 8× the nominal noise at the default speed.
- Where the shots differ the average miss still does not: the LLM prior helps on Y03 and hurts on Y01 and Y02.
- Friction that differs from slide to slide by 0.12 in ln μ puts the miss after one slide at 15.6 mm for all three methods (release-model world).
- The belief grid does not limit the result (2,000 to 32,000 points agree to 0.001 mm).
- The board's seams: `push_slide`'s uniform table is 24 equal strips and a third of the Y01/Y02 map pushes land short, by up to 7.5 mm (`map_check.json`). The study scores without those losses; the evaluation runs above include them (RMSE after one slide at nominal noise 1.34 mm with them, 1.08 mm without).

## More pairs: the handbook set

`pair_sets/*/summary.txt` and `pair_sets/llm_comparison.txt`. On L01–L20 the truth is the lookup's nominal, so the lookup baseline holds the answer, and the LLM may recall the same handbooks. The LLM for the 23 pairs is a Claude Haiku stand-in; GPT is cached for Y01–Y03 only.

| first shot, mean miss | generic | lookup | Claude Haiku | GPT-5.4-mini |
|---|---|---|---|---|
| handbook pairs (20), targets 0.60 / 0.735 / 0.87 m | 61.3 mm | 3.9 mm | 13.7 mm | not asked |
| Yu pairs (3), same targets | 33.5 | 34.2 | 41.3 | not run |
| Yu pairs (3), their own targets 0.60 / 0.77 / 0.94 m | 39.8 | 41.3 | 49.4 | 48.4 |

- The LLM is closer than the generic prior on 19 of 20 handbook pairs; the truth is inside its 90% interval on 19 of 20 (21 of 23 with the Yu pairs).
- A generic prior at the LLM's interval width misses by 19.7 mm on the handbook pairs (release model): most of the gain from 61 mm is width, the step from 19.7 to 13.8 mm is the LLM knowing the pair.
- After one practice slide all methods are within 0.15 mm of each other on all 23 pairs (mean miss 0.94–0.97 mm).

## The transition model on a patchy board

`patchy_table/summary.txt`. Mean miss, mm; s is the SD of ln μ from strip to strip (Yu et al.: about 0.09–0.12 for these three surfaces; the 0.25 quoted here until 25 Nov was polyurethane's spatial SD, entered on the plywood row of `data/ground_truth_published.csv` and fixed to 0.09 that day; the saved `summary.txt` files keep the old header. So s = 0.25 is about twice the largest spread of the simulated surfaces).

| s | no practice | nominal friction | practice at 1.01 m/s | practice at 0.7 m/s | shot friction against practice estimate, SD of ln | shots inside the belief's 90% interval |
|---|---|---|---|---|---|---|
| 0 | about 40 | 0.3 | 0.8 | 0.6 | 0.008 | 100% |
| 0.06 | about 40 | 2.7 | 4.0 | 3.5 | 0.047 | 19% |
| 0.12 | about 40 | 5.5 | 8.0 | 5.9 | 0.093 | 12% |
| 0.25 | about 42 | 11.9 | 16.5 | 11.8 | 0.197 | 8% |

- The friction a shot meets differs from the practice estimate by about 0.8 s; the belief after practice claims 0.009.
- A practice slide at full speed runs on a different stretch from the shots and ends worse than planning with the nominal friction; a slower one on the shots' stretch matches it.
- A belief told the measured spread is honest (88–92% inside its 90% interval) and no more accurate (7.8–7.9 against 8.0 mm at s = 0.12).
- After fixed-speed practice the three methods make the same shot.

Practice aimed at the shot's own target (`patchy_table/aimed_summary.txt`): the method's first shot at the target is its practice slide. Mean miss, mm, generic / lookup / LLM.

| s | nominal friction | fixed practice at 0.7 m/s | one aimed slide | two aimed slides | two aimed slides, latest only |
|---|---|---|---|---|---|
| 0 | 0.3 | 0.6 | 3.4 / 4.2 / 2.8 | 2.1 / 2.6 / 1.9 | 2.7 / 2.5 / 2.1 |
| 0.06 | 2.7 | 3.5 | 3.9 / 3.8 / 3.8 | 2.4 / 2.6 / 3.0 | 3.0 / 3.1 / 2.9 |
| 0.12 | 5.5 | 5.9 | 5.1 / 5.6 / 6.0 | 3.0 / 3.9 / 4.5 | 4.0 / 3.7 / 3.9 |
| 0.25 | 11.9 | 11.8 | 9.4 / 10.0 / 11.0 | 5.5 / 7.3 / 8.4 | 7.3 / 6.2 / 6.4 |

- One aimed slide equals the slower fixed slide (−0.2 ± 0.4 mm at s = 0.12, paired by board); two beat the nominal-friction plan by 1.7 ± 0.5 mm (s = 0.12) and 4.8 ± 0.9 mm (s = 0.25).
- At the 0.77 and 0.94 m targets the practice-to-shot mismatch falls to about a quarter of the strip spread after two aimed slides (0.8 after fixed practice). At the 0.60 m target the aimed slide is too short to say much.
- On a uniform board the aimed rule is worse (2–4 mm against 0.8 mm).
- The methods' shots land 5–19 mm apart: the prior matters after practice. A narrow prior near the truth helps (Y03), one far from it hurts (Y01, Y02).

The aimed rule on the 23-pair set (`patchy_table/pair_set/summary.txt`), generic / lookup / LLM (Claude Haiku stand-in) / control (generic median at the LLM's width). Mean miss, mm.

| s | fixed practice at 1.01 m/s (generic) | nominal friction | one aimed slide | two aimed slides |
|---|---|---|---|---|
| 0 | 0.9 | 0.2 | 6.3 / 2.6 / 2.2 / 2.9 | 4.6 / 1.8 / 1.6 / 2.4 |
| 0.12 | 5.2 | 3.6 | 7.3 / 3.0 / 3.8 / 4.1 | 4.4 / 2.4 / 3.0 / 3.1 |
| 0.25 | 10.1 | 7.3 | 10.1 / 4.4 / 5.5 / 6.1 | 6.0 / 3.8 / 4.6 / 4.8 |

- After one aimed slide at s = 0.12 (paired over pairs): lookup better than generic by 4.3 ± 0.8 mm, LLM by 3.6 ± 0.8 mm, control by 3.2 ± 0.7 mm. LLM against control 0.4 ± 0.3 mm; lookup against control 1.1 ± 0.3 mm.
- Most of the gain is confidence: the set's frictions straddle the generic median, which is nearly as close to a pair's friction as the LLM's answer (0.34 against 0.29 in |ln|).
- The generic prior loses at the 0.60 m target (8.8 against 1.6–2.5 mm); at the farther targets after two slides the methods are level.

The same study under a 6 m/s² brake (`patchy_table_brake6/compare.txt`, 21 Nov; 11 of the 23 pairs, same boards). The brake releases friction below 0.61 only.

| s | invalid pushes, 12 → 6 m/s² brake | scored rows moved over 5 mm | unobservable practice slides | one aimed slide, mean miss mm, generic / lookup / LLM / control |
|---|---|---|---|---|
| 0 | 0 → 0 of 627 | 0 | 9 → 9 | 8.7 / 2.1 / 2.1 / 3.9 → 8.7 / 2.2 / 2.1 / 3.9 |
| 0.12 | 0 → 0 of 5,016 | 1 | 99 → 99 | 9.7 / 2.8 / 2.9 / 4.0 → 9.8 / 2.8 / 2.9 / 4.0 |
| 0.25 | 35 → 142 of 4,998 | 85 | 156 → 240 | 12.9 / 4.1 / 3.9 / 4.9 → 15.7 / 4.2 / 4.0 / 5.0 |

- At s = 0.25 the pairs at friction 0.40 and above (L19, L04, L02, L17, L16) have strips the brake cannot release on: the object rides the paddle to its stop, nothing is observed, and the next attempt repeats the push.
- The push check flags only some of these pushes; the rest are scored, mostly as large short misses. Release audit (`release_audit_s025_summary.txt`): 103 of 4,697 pushes at s = 0.25 never released, 44 of them passed the check, 140 of the 4,856 scored rows. At s = 0.12 every push released. `--check-release` (21 Nov, off by default) makes such a push invalid.

## Execution error (Tier B)

`e1_release_error.csv`: 35 pushes over μ 0.1–0.5 and 0.3–1.0 m/s, all within joint speed and torque limits. Release speed against the model: mean +0.21%, range −0.02% to +1.23% (largest at high friction and low speed). Release point within 0.07 mm. Landing against the model's rest point: RMS 1.6 mm, worst −8.0 mm on the longest slide. Deterministic biases, not noise.

## Motion generators (the proposal's E1 comparison, 25 Nov)

`e1_motion_generators/` (`experiments/e1_motion_generators.py`): `results.csv`, `summary.txt`, `summary.png`. 85 pushes in Drake at friction 0.26, J = 100, brake cap 6 m/s²: five generators (arm.py's, differential IK with and without acceleration limits, trajectory optimisation with and without acceleration bounds) × 10 speeds at the side base and 7 at the origin base. With the base beside the board and the 6 m/s² brake, all three make every push from 0.2 to 1.1 m/s. Differential IK from arm.py's start posture gives arm.py's joint motion (release +0.1 to +0.5% of the model, landing within 0.4 mm, acceleration at most 0.93 of its limit, no clipping). Trajectory optimisation with velocity and acceleration bounds stays inside the limits but sits at the acceleration bound and reproduces the brake less exactly: release −0.4 to +6.9%, landing −1.5 to +4.6 mm. Without acceleration bounds it plans up to 3.2 times the acceleration limit. With the base on the push line, differential IK is clipped at 0.5 m/s and above and says nothing: the paddle is 2 to 39% slow, the release up to 44% low and the landing up to 118 mm short. Trajectory optimisation refuses those pushes instead, and with acceleration bounds refuses every push at that base.


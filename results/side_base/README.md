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
| `evaluation/ten_seeds_Y01/`, `_Y02/`, `_Y03/` | The default setup over ten noise seeds, one folder per pair (1,116 scored rows each) | `python3 evaluation/run_arm_evaluation.py --pair Y01 --seeds 0 1 2 3 4 5 6 7 8 9 --out ...` |
| `evaluation/clip_beyond_reach/` | Targets 0.60 / 0.94 / 1.10 m with `--clip-unreachable`: the last is beyond Y03's reach, to exercise the clip-and-score rule | `... --pair all --seeds 7 --targets 0.60 0.94 1.10 --clip-unreachable --out ...` |
| `evaluation/loss_sq_abs/` | The default setup planned with the squared-miss loss instead of the relative loss | `python3 evaluation/run_with_loss.py sq_abs --pair all --seeds 7 --out ...` |
| `e1_release_error.csv`, `_summary.txt` | The arm's execution error against the release model over friction × speed (35 pushes): the Tier B execution model's input | `python3 experiments/e1_release_error.py` |

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

## Execution error (Tier B)

`e1_release_error.csv`: 35 pushes over μ 0.1–0.5 and 0.3–1.0 m/s, all within joint speed and torque limits. Release speed against the model: mean +0.21%, range −0.02% to +1.23% (largest at high friction and low speed). Release point within 0.07 mm. Landing against the model's rest point: RMS 1.6 mm, worst −8.0 mm on the longest slide. Deterministic biases, not noise.

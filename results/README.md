# Saved experiment results

Commands below are written from the repository root. Do not pool v1, v2, and v3 as independent evidence.

| Folder | Meaning | Start with |
|---|---|---|
| `pilot_v1/` | Early narrow-prior pilot; superseded configuration, retained as history | [360-row summary](pilot_v1/pilot_summary/summary.txt) |
| `pilot_v2/` | Proposal-width generic prior; one target; ten noise seeds | [360-row summary](pilot_v2/arm_v2_summary/summary.txt) |
| `evaluation_v3/` | Same v2 priors; three targets; low/nominal/high noise and deterministic no-added-noise check | [Y01](evaluation_v3/arm_eval_v3_Y01_check/summary/summary.txt), [Y02](evaluation_v3/arm_eval_v3_Y02_check/summary/summary.txt), [Y03](evaluation_v3/arm_eval_v3_Y03_check/summary/summary.txt) |
| `e0/` | E0 contact, release and observation checks (`experiments/e0_*.py`): results, summaries, figures, and `e0_5_noise.csv` | `e0/e0_4_summary.txt`, `e0/e0_5_summary.txt` |
| `e1/` | Arm validation with known friction (`experiments/e1_arm_validation.py`), rerun 16 Nov on the merged controller | `e1/e1_arm_summary.txt` |
| `arm_rerun/` | 12 pushes with the arm beside the plate (`experiments/arm_rerun.py`) | `arm_rerun/arm_rerun_summary.txt` |
| `perception_results.csv` | Landing-perception error study (`src/perception.py --study`) | |
| `side_base/` | Results at the default layout since 16 Nov, with the iiwa beside the board at (0.6, 0.65) m: link clearance, E1, arm re-run, E0-2 and E0-5 with the arm, the perception study, and the evaluation with a computed speed cap and derived targets. A different layout from the folders above (which are the origin layout): never pool them. Also the practice study (`side_base/practice_study/`: how informative a practice slide has to be before the prior matters), the pair sets (`side_base/pair_sets/`: the comparison on the 20 handbook pairs plus Y01–Y03, and the LLM comparison) and the patchy-table study (`side_base/patchy_table/`: the transition model tested on a board whose friction varies) | [index](side_base/README.md) |
| `camera_study/` | Camera position and perception window against coverage, frame noise and friction-estimate bias, with the arm in view (`experiments/camera_study.py`) | `camera_study/camera_study_summary.txt` |
| `logic_checks/` | Checks of 25 Nov (`experiments/logic_checks.py`): release bound per push speed, calibration of the belief, belief grid at low noise | `logic_checks/summary.txt` |

The three evaluation folders above (`pilot_v1/`, `pilot_v2/`, `evaluation_v3/`) are the origin layout, produced before the 16 Nov arm merge (Diego's original controller, squared-miss loss), and have not been rerun since. `e1/` and `arm_rerun/` are also the origin layout; `e0/` used no arm. `python3 evaluation/run_arm_evaluation.py --v3` and `--layout origin` on the experiment scripts rerun that layout.

`evaluation_v3/merged_code_seed7/` and `evaluation_v3/merged_code_seed7_sq_abs/` are the v3 protocol rerun on 16 Nov on the merged arm code (all three pairs, seed 7), under the runner's default relative loss and under the squared-miss loss the saved checks used. The second reproduces the saved checks (zero-practice RMSE 8.88 / 13.47 / 16.66 mm against 8.88 / 13.46 / 16.63); the first shows what the loss change does at the origin (generic 14.1 mm).

V1 and v2 also retain single-pair checks. V2's `arm_v2_Y01_grid4000` is a grid-resolution diagnostic, not an additional independent experiment.

## Inside a v3 run

- `settings.json`: configuration, package versions, source and input hashes.
- `preflight.json`: oracle target-feasibility checks.
- `*_practice_validity.json`: practice-rollout checks.
- `*_observations.json`: shared noisy practice data.
- `results.csv`: separate scored attempts, with deterministic reuse identified.
- `status.json`: completion/failure and counts.
- `summary/`: aggregate, per-pair, per-target tables and readable report.

Each saved v3 check has 144 scored rows and 84 unique scored rollouts. The three folders total 432 rows. Repeated oracle/zero-practice entries, reused trajectories, and noise seeds do not create new materials. The ten-seed full v3 matrix is not in this checkpoint.

Existing artifacts were moved without changing their contents. New runner commands should use explicit `--out results/<version>/<new-name>` paths. The v1/v2 summarizers now default to the corresponding grouped folders and still accept `--root` for older layouts.

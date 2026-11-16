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

The three evaluation folders were produced before the 16 Nov arm merge (Diego's original controller, squared-miss loss) and have not been rerun since.

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

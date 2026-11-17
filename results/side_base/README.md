# Side-base trial (16 Nov)

Everything here was produced with the iiwa's base beside the board at (0.6, 0.65) m (`arm.LAYOUTS["side"]`) instead of the origin. It is a trial: the code's defaults and the folders next to this one (`e0/`, `e1/`, `arm_rerun/`, `evaluation_v3/`, ...) are the origin layout and were not changed. Do not pool these results with them.

Commands are written from the repository root.

| Files | What | Made with |
|---|---|---|
| `e1_arm_*` | E1 arm validation, same 85 pushes as `../e1/`, default 12 m/s² brake cap | `python3 experiments/e1_arm_validation.py --layout side --out results/side_base/e1_arm` |
| `e1_arm_brake6_*` | The same with a 6 m/s² brake cap (releases friction below 0.61) | `... --layout side --a-brake 6 --out results/side_base/e1_arm_brake6` |
| `arm_rerun_*` | The 12-push arm re-run, arm beside the plate, as `../arm_rerun/` | `python3 experiments/arm_rerun.py --layout side --out ... --summary ... --fig ...` |
| `e0_5_*` | E0-5 camera study with the arm pushing and in the camera's view (the saved `../e0/e0_5_*` used the plate) | `python3 experiments/e0_5_observation.py --layout side --out ... --frames ... --noise ... --summary ... --fig ...` |
| `evaluation/brake4_all_limits/` | Method comparison, Y01–Y03, seed 7: 4 m/s² brake cap, joint speed, torque and acceleration all required | `python3 evaluation/run_arm_evaluation.py --pair all --seeds 7 --layout side --a-brake 4 --check-accel --peak-cap auto --practice-peak auto --targets auto --out ...` |
| `evaluation/brake12_speed_torque/` | The same with the default 12 m/s² brake cap; speed and torque required, acceleration recorded | `... --layout side --peak-cap auto --practice-peak auto --targets auto --out ...` |

## What they show

| | origin (saved folders) | side base (here) |
|---|---|---|
| Fastest push within joint speed and torque limits | 0.49 m/s | 1.02 m/s |
| E1: arm pushes within joint speed limits | 22 of 73 | 73 of 73 |
| E1: targeting with the true friction, RMS and worst miss | 0.74, 2.4 mm | 0.77, 2.3 mm |
| E1: repeatability, worst miss | 2.8 mm | 2.6 mm |
| Arm re-run: pushes within speed and torque limits | 0 of 12 | 12 of 12 |
| Arm re-run: release speed against the plate's | −0.13..−0.01% | −0.13..+0.01% |
| E0-5: worst bias of models A/B/C, missed frames | 0.90%, 0 (plate) | 0.92%, 0 (arm in view) |
| E1: pushes within the acceleration limits, 12 m/s² brake cap | not recorded | 5 of 73 |
| E1: pushes within the acceleration limits, 6 m/s² brake cap | not run | 65 of 73 |

## The evaluation

Y01–Y03, seed 7, 432 scored rows per run; each run folder has its own `summary/`. The computed cap is 1.017 m/s with a 4 m/s² brake (speed, torque and acceleration required) and 1.019 m/s with the default 12 m/s² (speed and torque required); practice pushes run at 1.01 m/s; the derived targets are x = 0.60, 0.77 and 0.94 m. Landing RMSE at nominal noise, 4 m/s² brake:

| practice slides | generic | lookup | LLM | oracle |
|---|---|---|---|---|
| 0 | 54.8 mm | 50.7 mm | 65.1 mm | 0.4 mm |
| 1 | 1.5 mm | 1.5 mm | 1.5 mm | 0.4 mm |
| 2 | 0.5 mm | 0.6 mm | 0.6 mm | 0.4 mm |

- **Every push is valid in both runs.** With the 4 m/s² brake the scored pushes peak at 0.80 of the joint speed limit, 0.25 of the torque limit and 0.75 of the acceleration limit; the practice pushes at 0.98, 0.25 and 0.85. With the 12 m/s² brake the acceleration reaches 2.1× (scored) and 2.3× (practice).
- **The brake cap does not change the outcome.** Every RMSE in the 12 m/s² run is within 0.1 mm of the table above.
- **No push reached the cap**, so the clip-and-score rule had nothing to do here: the derived targets are in reach by construction. It will matter for higher-friction pairs or targets placed beyond reach.
- **Farther targets make the starting belief matter much more.** Before any practice the methods miss by 51–65 mm, against 9–17 mm in the saved origin runs whose targets stop at 0.64 m. One practice slide brings all three to 1.5 mm. The two sets are not like for like: different targets, and the saved runs used the squared-miss loss (see the main README).

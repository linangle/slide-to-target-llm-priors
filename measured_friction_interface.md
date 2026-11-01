# Measured-friction interface

The friction rig produces three CSV files and reads one. This document is the contract for
all four, so that the simulator (`scene.py`), the baselines, the LLM prompt builder and the
evaluation can be written against the mock (`ground_truth_placeholder.csv`) and run unchanged
on the real `measured_friction.csv` when it arrives (8 Nov).

| File | Written by | Read by |
|---|---|---|
| run log (`run_log_template.csv` columns) | the operator, one row per recording | `friction_analysis.py --runlog`, `aggregate_pairs.py --runlog` |
| per-run results (`results.csv`) | `friction_analysis.py --out` | `aggregate_pairs.py --results` |
| profiles (`profiles.csv`) | `friction_analysis.py --profiles` | `aggregate_pairs.py --profiles` |
| `measured_friction.csv` (mock: `ground_truth_placeholder.csv`) | `aggregate_pairs.py` (mock: `make_mock_measured.py`) | `scene.py`, baselines, evaluation, prompt builder (restricted, see rules) |

## Interface rules

1. **Read CSVs by column name** (`csv.DictReader`, `pandas` by header). Never hard-code the
   number of pairs, the row order, or any friction value. The real file may have fewer or more
   rows than the mock (pairs can be dropped or added), and the pair set is whatever `pair_id`
   says.
2. **The first 21 columns of the measured-friction file are fixed**, in the order below. New
   columns are only ever appended after them, so one reader handles the mock, the published
   file and the measured file. A reader that needs an appended column must tolerate its absence
   (`row.get("strip_scale_mm", "")`), since `ground_truth_published.csv` has only the 21.
3. **The LLM prompt builder may read only `description_object`, `description_surface`,
   `photo_object` and `photo_surface`.** It must not read `mu_k_nominal`, the interval, `s_tab_log`,
   `sources`, `object_material`, `surface_material`, the run log, the per-run results, the
   profiles, or any note that names a material or a number. The material columns are labels
   for us, not prompt text; the descriptions are written for the prompt (what a person would
   say about the object and the table, with no coefficient and no material datasheet value).
4. **Prompts are frozen and LLM responses are cached before the measured file exists.** The
   prompt text for every pair is generated from the mock's `description_*`/`photo_*` columns
   (which are the real descriptions, filled in before the measurement) and committed; responses
   are cached to disk, keyed by the prompt hash; the measured file then only changes the
   evaluation, never the prompts or the cache.
5. Empty cell = unknown. Readers must treat `""` as missing, not as zero. Numbers are plain
   decimals, no units in cells; units are in this document.

## Pair IDs

- `Mnn`, measured pairs: `nn = 9 (surface − 1) + object`, objects O1–O9 on surfaces S1–S3,
  so M01–M09 are on S1, M10–M18 on S2, M19–M27 on S3.
  - Objects: O1 PLA (3D-printed, bed face), O2 ABS sheet (extruded), O3 furniture felt,
    O4 neoprene rubber 60A, O5 corrugated cardboard, O6 cast acrylic, O7 oak with grain along
    travel, O8 glazed ceramic tile, O9 304 stainless steel.
  - Surfaces: S1 white melamine shelf board, S2 birch plywood (sanded), S3 painted MDF (satin).
- `Vnn`, validation pairs: V01 = O9 on S5 (ABS sheet, smooth face), V02 = O7 on S4 (oak board,
  grain along travel), V03 = O7b (oak, grain across travel) on S4.
- `Ynn`, published pairs (Yu et al. 2016) live in `ground_truth_published.csv`, same 21 columns.
- Object and surface IDs (`object_id`, `surface_id`) are the keys for descriptions and photos;
  a pair row is the only place the two meet.

## Measured-friction file: `measured_friction.csv` and its mock

Columns 1–21 (fixed; the mock and `ground_truth_published.csv` have exactly these):

| # | Column | Meaning, units | Filled by |
|---|---|---|---|
| 1 | `pair_id` | see above | scheme |
| 2 | `object_id` | O1–O9, O7b | scheme |
| 3 | `surface_id` | S1–S5 | scheme |
| 4 | `object_material` | short material label (internal, never in a prompt) | team |
| 5 | `surface_material` | idem | team |
| 6 | `mu_k_nominal` | kinetic friction coefficient, pair mean, 3 decimals | `aggregate_pairs.py`: mean of the lane/speed means after the fwd/rev average |
| 7 | `s_tab_log` | log-spread of friction, dimensionless; the simulated table draws strips as `mu·exp(patchiness·s_tab_log·z)` | sample SD of ln(lane/speed means); with one group, the spatial log-SD of the profile |
| 8, 9 | `interval90_low`, `interval90_high` | `mu·exp(∓1.645·s_tab_log)` | `aggregate_pairs.py` |
| 10 | `sources` | free text; first comma-separated token is the short label shown in figures; the mock says `PLACEHOLDER ...`, the real file says `friction rig (...)` | script |
| 11 | `n_runs` | accepted runs in the pair mean | script |
| 12 | `se` | standard error of `mu_k_nominal`: SD of the lane/speed means / √groups (SD of runs / √runs with one group) | script |
| 13 | `speed_effect` | `(m_fastest − m_slowest)/mu_k_nominal`, speed settings averaged over lanes; blank with one setting | script |
| 14 | `load_effect` | blank: the rig does not vary the load | — |
| 15, 16 | `temp_C`, `RH_pct` | room temperature (°C) and relative humidity (%), mean over the pair's run-log rows | run log |
| 17 | `in_sim_range` | 1 if `0.10 ≤ mu_k_nominal ≤ 0.50` (the range the simulator and lookup table cover), else 0 | script |
| 18, 19 | `description_object`, `description_surface` | prompt text (see rule 3) | team, before measurement |
| 20, 21 | `photo_object`, `photo_surface` | relative path of the photo used in the prompt, or blank | team |

Appended columns (22 on), in this order, present in the mock and the real file:

| Column | Meaning, units |
|---|---|
| `mu_v2d` | mean endpoint estimate `v_start² / (2 g d_tape)` over accepted runs with a tape reading, i.e. the friction the robot would infer from a release speed and a stopping distance; `g = 9.80665`. Level-board formula: a board tilt φ biases it by about `sin φ / mu` |
| `v2d_ratio_mean`, `v2d_ratio_sd` | mean and SD over runs of `mu_v2d / mu_k` per run. The mean checks the stopping law `d = v²/(2 μ g)`; the SD is the real-world noise of one endpoint observation (it includes the ±δ(1+μ²) mounting-tilt bias of a single orientation, which the pair mean cancels) |
| `mu_slope_per_mps` | within-slide change of μ with speed, per m/s: mean over runs of the per-run least-squares slope of the per-sample ratio against the sled speed |
| `strip_scale_mm` | correlation length of the μ(x) profile, mm: the lag at which the pass-averaged autocorrelation of the mean-removed profile first falls to 1/e, floored at the 100 mm coupon length and capped at half the profile length. Sets the strip width of the simulated patchy table |
| `profile_repeatability` | mean Pearson correlation between the μ(x) profiles of passes on the same lane over their common bins (≥ 5 bins), averaged over lanes; blank with fewer than two passes per lane |
| `n_rejected` | rejected runs of the pair (not in `n_runs`) |
| `reject_reasons` | `reason:count;reason:count`, most frequent first; reasons are the script's flags (`short slide`, `lateral force`, `yaw`, `no slide segment found`) and the run log's manual reasons (`touched paddle`, ...) |
| `s_lane` | sample SD of ln(lane means) across lanes (lane mean = mean over speeds of the lane/speed means); blank with one lane |

The mock is regenerated with `python3 make_mock_measured.py`, which keeps the 21 existing
columns of `ground_truth_placeholder.csv` byte for byte and refills the appended columns from a
seeded RNG; `sources` marks every mock row `PLACEHOLDER`. Check with
`git diff ground_truth_placeholder.csv` that only trailing columns changed.

## Run log

Template: `run_log_template.csv` (header plus two rows whose `session` is `EXAMPLE`; the
readers skip those, delete them anyway). One row per recording, rejected runs included (6.1 item
3): lateral force and rotation are evidence about how stable line contact is on real hardware.

| Column | Meaning | Who |
|---|---|---|
| `session` | session label, e.g. `S1`; raw files and `calib.json` live in `data/raw/<session>/` | operator |
| `date` | ISO date | operator |
| `operator` | initials | operator |
| `temp_C`, `RH_pct` | room temperature and humidity at the start of the session | operator |
| `calib_file` | the session's `calib.json` (from `--make-calib`) | operator |
| `pair_id` | `Mnn` / `Vnn` | operator |
| `file` | the phyphox export, `<pairID>_<lane>_<orient>_<speed>_<n>.csv`; matched to the analysed file by basename | operator |
| `lane` | lane on the board, e.g. `L`, `R` | operator |
| `orientation` | `fwd` (phone top, +y, along the slide) or `rev` (sled turned 180°, same travel direction on the board) | operator |
| `speed_setting` | carriage setting, `s1`, `s2`, ... (ordered by the number); the fast setting should reach about 1.1 m/s | operator |
| `rest_mm` | tape reading of the sled's leading edge at rest, to 1 mm, with the tape zeroed at the leading edge when the sled sits against the paddle at the carriage stop; it is the slide distance `d_tape` | operator, every run |
| `accepted` | `1`, or `0` for a manual reject | operator; the script adds its own flags |
| `reject_reason` | why, e.g. `touched paddle` (the sled touched the paddle again after release, which the accelerometer cannot see); any non-empty reason rejects the run | operator |
| `notes` | free text | operator |

If a file is not in the run log, `friction_analysis.py` takes lane, orientation and speed from
the file name and leaves `rest_mm` (and so `mu_v2d`) blank. `--pair` and `--direction` on the
command line override both the log and the file name.

## Per-run results (`friction_analysis.py --out results.csv`)

One row per analysed file, rejected ones included (`accepted = 0`). The first 16 columns are
the original ones; the rest are appended. An older `results.csv` with 16 columns is upgraded in
place on the next append (old rows padded with blanks).

| Column | Meaning |
|---|---|
| `pair_id`, `object`, `surface`, `method`, `direction`, `load_g`, `file` | identity; `direction` equals `orientation` |
| `mu_k` | `−mean(f_along)/mean(f_normal)` over the trimmed slide window |
| `dur_s`, `n_samples` | trimmed window length |
| `v_start` | release speed, m/s: `−∫ a dt` from the release to the true stop, `a = f_along − f_along(rest)` |
| `lat_rms` | lateral specific force RMS about its rest level, m/s² (reject > 0.5) |
| `yaw_max_degps` | peak yaw rate if a gyroscope column exists (reject > 20), else `nan` |
| `delta_rest_deg` | tilt + misalignment seen at rest, degrees |
| `accepted` | 1 if no script flag and no manual reject |
| `flags` | script flags only: `short slide`, `lateral force`, `yaw`, `no slide segment found` |
| `lane`, `orientation`, `speed_setting`, `rest_mm`, `session` | from the run log / file name |
| `d_acc_m` | slide distance from the accelerometer, m: second integral from the release to the true stop |
| `mu_v2d` | `v_start² / (2 g rest_mm/1000)`; blank without `rest_mm` |
| `v2d_ratio` | `mu_v2d / mu_k` |
| `mu_slope_per_mps` | slope of the per-sample ratio against speed over the trimmed window |
| `stop_ms` | how long after the half-return point the true stop was found (the stop transient; was the bug of spec 6.2a) |
| `warnings` | non-rejecting notes: `rest band widened`, `stop near record end`, `no stop found` (then `mu_v2d` is blank) |
| `reject_reason` | all reasons, script flags then manual, `; `-separated; empty when accepted |

## Profiles (`friction_analysis.py --profiles profiles.csv`)

First line is a `#` comment stating the resolution limit; readers skip lines starting with `#`.
Then one row per 20 mm bin per accepted run:

| Column | Meaning |
|---|---|
| `pair_id`, `lane`, `orientation`, `speed_setting`, `file` | the pass |
| `bin_mm` | lower edge of the bin of board position: the sled's leading edge, mm from the release point (the same coordinate as the tape, 0 at the carriage stop) |
| `mu` | mean per-sample ratio in the bin |
| `n` | samples in the bin |

The coupon covers `[x − 100, x]` mm, so the profile is the local friction smoothed over one
coupon length: a step shows as a 100 mm ramp whose midpoint lies 50 mm past the step, and
features shorter than about 10 cm are not resolvable.

## Running it

```bash
python3 friction_analysis.py --selftest                      # synthetic checks; exit code 1 on failure
python3 friction_analysis.py --make-calib px.csv mx.csv py.csv my.csv pz.csv mz.csv
python3 friction_analysis.py data/raw/S1/M07_*.csv --runlog data/raw/S1/run_log.csv \
    --calib data/raw/S1/calib.json --out results.csv --profiles profiles.csv
python3 aggregate_pairs.py --results results.csv --profiles profiles.csv \
    --runlog data/raw/S1/run_log.csv --meta ground_truth_placeholder.csv --out measured_friction.csv
python3 make_mock_measured.py                                # refresh the mock's appended columns
python3 -m pytest tests/test_friction_analysis.py
```

`--meta` copies the identity columns (2–5, 18–21) from the mock, so descriptions and photo
paths written there before the measurement carry over to the measured file unchanged.

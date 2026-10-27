# Slide-to-Target

MIT 6.4210 (Robotic Manipulation) team project, Fall 2026.

A simulated KUKA iiwa pushes a flat-bottomed object across a table and releases it so it slides to a target distance, `d = v² / (2 μ g)`. Friction `μ` is unknown. We ask a language model for a prior over `μ` from a text description of the object and the table, update it with a few practice slides (Bayesian, in `ln μ`), and choose the release speed that minimises expected landing error. The main question is how much the language-model prior is worth compared with a published friction table, practice slides alone, and the true friction.

## Status

Phase 1, Oct 25 – Nov 1: measuring ground-truth kinetic friction for 27 object–surface pairs (M01–M27) and 3 validation pairs (V01–V03). Output: `measured_friction.csv`, using the same columns as `friction_lookup.csv`.

## Contents

| File | Purpose |
|---|---|
| `friction_analysis.py` | Computes `μ_k` from phone-accelerometer (phyphox) recordings of a sled sliding on a flat board. Also does the six-position accelerometer calibration. |
| `lookup_table.py` | Builds the published-value friction lookup table (baseline C2): `friction_lookup.csv` plus LaTeX rows `rows_*.tex`. |
| `check_drake.py` | Smoke test for the Drake install: loads the iiwa 14 and checks a hydroelastic box slide against `d = v² / (2 μ g)`. |
| `scene.py` | The Drake scene: patchy table, object, iiwa 14 with paddle. Also runs passive slides against the predicted stopping distance. |
| `ground_truth_placeholder.csv` | Synthetic friction values in the `measured_friction.csv` columns, used until the measured file exists. **Not friction estimates.** |

## Setup

You need Python 3.12 or newer. Drake 1.58 ships wheels for macOS 26 on Apple silicon and for Linux with glibc 2.34 or newer (Ubuntu 22.04+). On Windows, use WSL2.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 friction_analysis.py --selftest
python3 check_drake.py
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

The script prints `μ_k` for each run and flags rejects: a short slide, lateral force, or yaw. `--out` appends to the CSV. If more than 3 of a pair's 12 runs are rejected, fix the cause and redo them. Runs where the sled touched the paddle again after release must be marked by hand in the run log.

**Where data goes.** Raw recordings are small, so commit them, with each session's `calib.json` and run log beside them. Suggested layout: `data/raw/<session>/`.

## Simulation scene

`scene.py` builds one `MultibodyPlant` for a given pair from the ground-truth file.

**What's in the scene**
- **Table:** its top is at z = 0 and is tiled with 5 cm strips across the travel direction (+x).
  - Each strip's friction is the pair's `mu_k_nominal × exp(patchiness · s_tab_log · z)`, sampled per seed.
  - `--bimodal` uses two levels instead.
- **Object:** compliant hydroelastic.
  - Shapes: box, disc, or block on 3 or 4 feet.
  - For robustness cases: an optional centre-of-mass offset, or one longer foot.
- **iiwa 14:** base at the origin on the table, with a flat rigid paddle welded to the flange.
  - No controller is included: connect one to the plant's actuation input.

**Getting the friction right.** Drake combines two geometries' friction as `2 μa μb / (μa + μb)`. The object therefore carries a fixed μ, and each strip and the paddle are given the value that makes the *pair* coefficient equal to its target.

```bash
python3 scene.py --pair M07 --meshcat                      # view the scene
python3 scene.py --pair M07 --slide 0.8                    # passive slide vs prediction
python3 scene.py --pair M07 --slide 0.8 --shape footed --foot-error 0.001 --patchiness 0
```

**What `--slide` does:** it gives the object a speed and lets it slide to rest. It compares the stopping distance with the prediction `g ∫ μ̄ dx = v₀²/2`, where `μ̄` is the strip friction averaged over the object's footprint. It prints PASS within 2% in distance and 2° in yaw.

**From another script:** use `build_scene(builder, pair, obj, table, contact, arm=True, seed=...)`, which returns the plant, the object body, the iiwa model instance and the paddle body.

## Lookup table

```bash
python3 lookup_table.py
```

This writes `friction_lookup.csv` and `rows_main.tex`, `rows_excluded.tex`, `rows_primary.tex`. Entries are kinetic coefficients only, with values between 0.10 and 0.50. Static-only sources are listed as excluded.

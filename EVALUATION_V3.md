# Multi-target/noise evaluation infrastructure (v3)

This extends the v2 pilot without replacing it. The current materials are
three previously inspected published pairs, not a finalized broader dataset.
The raw LLM method remains uncalibrated.

## Files

- run_arm_evaluation.py: evaluation runner
- summarize_arm_evaluation.py: validation and summaries
- tests/test_evaluation_v3.py: offline tests

Existing v2 modules and the published LLM response cache are required.
No new dependencies or API requests are needed.

## Evaluation settings

- Targets: world x = 0.60, 0.62, 0.64 m; y = 0.
- Start x = 0.55 m; J = 100.
- Peak speed cap = 0.45 m/s; practice peak = 0.45 m/s.
- Methods: generic, lookup, raw LLM, and true-friction oracle.
- Practice counts: 0, 1, 2 slides.
- Existing v2 priors; 2,000-point friction grid over [0.005, 5].
- Nominal measurement SDs: frame 2 mm, rest 0.5 mm, FK 0.5 mm.
- Low/nominal/high conditions scale these SDs by 0.5/1/2.
- Inference uses the matching SDs for these noisy conditions.
- No-added-noise diagnostic injects no noise but retains nominal positive
  likelihood SDs. It runs once with seed label -1.
- Standard-normal draws are shared across low/nominal/high conditions.
- Practice observations are shared across methods and targets.
- Independent observation-noise draws are used between practice slides.
- Two deterministic practice pushes per pair use the existing push_slide
  state-reset implementation.
- Sensing uses noisy simulator state and ideal contact-event release timing.
  No rendered camera is used.

## Commands

Run the tests:

    python -m pytest tests/test_evaluation_v3.py -q

Check configuration without simulation:

    python run_arm_evaluation.py --pair Y01 --seeds 7 --check-only

Run and summarize one pair:

    python run_arm_evaluation.py --pair Y01 --seeds 7 --out results/evaluation_v3/arm_eval_v3_Y01_check
    python summarize_arm_evaluation.py results/evaluation_v3/arm_eval_v3_Y01_check

Supported controls:
--pair Y01/Y02/Y03/all
--targets followed by world-x positions
--noise followed by low nominal high no_added
--seeds followed by nonnegative integers
--grid-points (default 2000)

## Validation and outputs

The configuration-only check does not establish physical feasibility.
Real runs perform oracle feasibility checks for every pair-target.
Preflight failures are recorded and execution stops. Invalid scored shots
remain in results.csv; summaries reject partial or failed runs.
Practice validity is recorded separately.

Outputs include settings.json, preflight.json, results.csv, status.json,
observations, and practice validity records. Settings record configuration,
input/source hashes, and package versions.

Existing output directories are not overwritten. Preserve failed runs for
diagnosis and use a new output directory after a fix.

Summaries contain aggregate.csv, per_pair.csv, per_target.csv, and summary.txt.
Conditions are reported separately. Friction metrics count each shared
belief once across targets.

Exact duplicate deterministic scored trajectories are reused by pair and
commanded speed without rounding. This assumes a deterministic simulator
with fixed scene configuration and no randomized execution disturbances.

## Completed development checks

Saved local development results (validated against the uploaded checkpoint):

- All three v3 offline tests passed.
- Y01, Y02, and Y03 each completed the seed-7 check.
- Each pair produced 144 scored rows and 84 unique scored rollouts.
- Each summary validated its complete selected matrix and all validity flags.
- Across the three runs: 432 scored rows.
- No reported speed-cap hits.

These checks cover three targets and low, nominal, high, and no-added-noise
conditions. They are infrastructure checks, not a final comparative study.

## Optional repeated-seed development evaluation

    python run_arm_evaluation.py --pair all --seeds 0 1 2 3 4 5 6 7 8 9 --out results/evaluation_v3/arm_eval_v3_matrix
    python summarize_arm_evaluation.py results/evaluation_v3/arm_eval_v3_matrix

This schedules 3,348 scored rows, including 108 no-added-noise diagnostic
rows, plus six actual practice pushes. No API requests are made.
This larger v3 run has not been established as completed by the checks above.

## Interpretation and remaining decisions

- Noise seeds do not create new material pairs.
- Repeated zero-practice/oracle rows and reused rollouts are not independent
  evidence.
- Coverage concerns fitted grid posteriors, not original LLM intervals.
- No-added-noise uses positive likelihood SDs, not a zero-variance posterior.
- Joint checks cover planned position/velocity and feed-forward torque,
  not actual total actuator torque.
- Previously inspected development pairs are not an unseen test set.
- Final protocol freezing still requires material provenance/manifest review,
  fixed descriptions for new pairs, and team agreement on raw versus
  calibrated LLM scope.

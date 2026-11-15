# Fixed three-pair pilot — version 2

Checkpoint established after inspecting the v1 pilot and v2 validation runs on November 10, 2026. This is not a preregistration or a previously unseen test set. Settings are fixed for the next repeated-noise run; later changes require a new version and separate results.

## Scope and data

- Pairs Y01/Y02/Y03, using ground_truth_published.csv: simulator kinetic friction 0.14/0.14/0.26.
- These values were accepted following user confirmation of verification. This checkpoint does not independently establish their provenance.
- Three pairs, all with stainless-steel contacting faces. Noise seeds are not independent material samples.
- The existing lookup rows retain their source and verification caveats. They are read unchanged from friction_lookup.csv.

## Priors and updates

- Generic: ln(mu) normal, median sqrt(0.08*1.1), log SD ln(1.1/0.08)/(2*1.6448536269514722). Before numerical truncation this has 5th/95th percentiles 0.08/1.1 and median about 0.29665 (proposal's rounded 0.3).
- All methods use 2,000 uniformly spaced ln(mu) grid points over mu in [0.005,5]. The normalized generic interval is about [0.07999,1.09844].
- Lookup: generic log density plus existing loglik_lookup; missing lookup falls back to generic.
- LLM raw: median-preserving lognormal approximation, log SD ln(high/low)/(2*1.6448536269514722). It matches reported log-interval width, not both endpoints. No generic-prior multiplication and no empirical calibration.
- Oracle: true simulator friction supplied explicitly to the planner.
- Model C updates from the original initial belief plus each practice observation once. Scored shots never become practice data.

## LLM and prompts

- Model gpt-5.4-mini-2026-03-17; reasoning none; maximum output 600 tokens; no tools; structured JSON.
- Existing template in llm_prior.py. Object text: A block with a stainless-steel contacting face.
- Surface texts: An ABS plastic sheet; An acetal (Delrin) sheet; A plywood sheet.
- Reuse llm_cache_published_pilot.json. No new queries, prompt changes, or selected replacement responses for this batch.
- Existing cache contains model/settings/prompt identity; each run stores its file hash.

## Arm, sensing, and evaluation

- Drake arm, default box, initial x=0.55 m, fixed target x=0.64 m/y=0, jerk=100 m/s^3.
- Target commands capped at peak paddle speed 0.45 m/s; use arm.plan_peak_speed with the full belief and moving release location.
- Two independent practice pushes per pair/seed, each at 0.45 m/s. Every method receives the same observations.
- Sample simulator object positions at 60 Hz after the contact-force release event; use moving frames under the runner's fixed rule.
- Independent Gaussian SDs: frames 2 mm, rest position 0.5 mm, FK point 0.5 mm. No camera rendering. Ideal simulator contact-event timing is assumed.
- Existing sim_config.yaml and source files are hashed in settings.json.
- Reject and stop on failed separation or reported planned position/velocity/feed-forward torque limit violations. This is not a certification of actual total actuator torque including feedback.
- Seeds 0 through 9, shared consistently across methods. 360 scored shots plus 60 practice shots.
- Primary metric: equal-weight planar landing RMSE in mm, per method and practice count. Also report per-pair results, friction MAE, fitted-posterior 90% coverage, interval width, and speed-cap frequency.
- Oracle and zero-practice repetitions are deterministic duplicates. Coverage is not coverage of the original LLM text intervals. No broad model ranking or statistical significance claim from three pairs.

## Validation and remaining work

User runs: all three pairs passed validity checks at seed 7; Y01 at 2,000 vs 4,000 grid points agreed at displayed speed/error precision. This is a limited resolution check, not proof of convergence for every posterior.

The runner still labels its configuration v2_candidate_not_final: this is intentional for the overall project. This document fixes the three-pair pilot only. The broader material dataset, calibrated-model alternative, final sensor assumptions, additional targets, and final project claims remain unresolved.

## Commands

Saved folders were subsequently grouped under `results/pilot_v2/`; commands below use that layout. The experimental settings above are unchanged.

```bash
for seed in {0..9}; do
  python run_arm_comparison_v2.py --pair all --seed "$seed" --out "results/pilot_v2/arm_v2_seed_${seed}" || break
done
python summarize_arm_v2.py
```

Keep the old arm_pilot_* outputs. Store this document with the source, cache, and versioned outputs. Do not commit API secrets or the virtual environment.

# Alejandro's contribution and completion audit

Audit date: November 15, 2026. Source checkpoint: `0317183`, following integration commit `e07cf40`. Baseline before these commits: `a4ec542`. Assignment source: section 5 of the uploaded Team 3.4 Final Proposal.

## Direct answer

The implementation and development checks are complete for the current three-pair setup. The full November 13 milestone is **not completely closed**, because it explicitly includes freezing prompts and evaluation settings for the final comparison. The current checkpoint freezes a three-pair pilot and implements a broader evaluation runner, but still labels the final configuration exploratory.

This is a specific remaining deliverable, not evidence that the integration failed. The November 23 comparisons and November 30 report/figures/video are shared team milestones, not work that had to be finished by November 13.

| Proposal milestone | Lead | Evidence and status |
|---|---|---|
| Oct 30: collect friction data, build scene, validate passive sliding | Angelina | Scene and validation artifacts exist; team measured dataset is absent from this checkpoint |
| Nov 6: repeatable pushes, achievable speeds, known-friction targeting | Diego | `arm.py`, `ARM_CONTROLLER.md`, E1 tests/results predate Alejandro's commits |
| Nov 13: integrate all estimate methods with Bayesian updating | Alejandro | Implemented for Y01–Y03 through shared observations, posterior updates, arm planning, and scored attempts |
| Nov 13: freeze prompts and evaluation settings | Alejandro | Pilot prompts/cache/settings recorded; final material descriptions, dataset and protocol are not frozen |
| Nov 23: all comparisons including noise sensitivity | All | Infrastructure and development checks exist; final dataset evaluation is pending |
| Nov 30: report, figures and video draft | All | Not established as complete by this repository audit |

## What existed before your work

At `a4ec542`, the repository already had the main mathematical and physical building blocks:

- `physics.py`: predicts release and sliding motion.
- `observe.py`: observation models and estimation from slide measurements.
- `belief.py`: distributions over friction and Bayesian updates, including an older calibrated-LLM observation model.
- `decide.py`: expected-loss speed selection.
- `arm.py`: IK, joint trajectories, arm control, and a target planner that already accepts a friction distribution.
- `scene.py`, `push.py`, configuration and E0/E1 tests: simulation and physical validation infrastructure.
- `llm_prior.py`: restricted prompt construction and a basic cache, but `query_llm` raised `NotImplementedError`.

The old README lagged behind the code: it still said the arm had no controller. That sentence was not an accurate description of the actual baseline commit.

## Exactly what your two commits added

| Contribution | Where | Why it accomplishes the assignment |
|---|---|---|
| Working text-only LLM request and validated response cache | `llm_prior.py`; two cache JSON files; OpenAI dependency | Supplies usable initial friction estimates and records prompts/model/settings/responses |
| Settings-aware cache identity and numeric validation | `prompt_key`, `parse_response`, `cached_response` | Prevents reuse under changed settings and rejects malformed estimates |
| Raw LLM distribution construction | `demo_published_comparison.py::raw_llm_logprior` | Converts the returned median/interval into a friction distribution accepted by existing Bayesian code |
| Small integration demonstrations | `demo_bayesian_integration.py`, `demo_compare_priors.py`, `demo_published_comparison.py` | Checks the connection from estimates through updates to planning before arm evaluation |
| Arm comparisons after 0/1/2 practice slides | `run_arm_comparison.py`, `run_arm_comparison_v2.py` | Uses common practice evidence for generic/lookup/LLM methods; oracle receives true friction; scored attempts are separate |
| Proposal-width generic prior | V2 prior constants and common expanded grid | Replaces the early narrow generic pilot prior with approximately median 0.3 and interval 0.08–1.1 |
| Multiple targets and noise levels | `run_arm_evaluation.py` | Makes the planned noise-sensitivity comparison executable, with shared random draws and a deterministic diagnostic |
| Reproducible, checked outputs | Summary scripts, v3 tests, settings/hashes/status/observations | Makes runs inspectable, rejects invalid/partial results, and avoids counting one belief multiple times across targets |
| Saved development evidence and protocol notes | `results/`, `PILOT_V2_PROTOCOL.md`, `EVALUATION_V3.md` | Demonstrates the connected pipeline works under the stated development conditions |

You did not replace the team's Bayesian solver or arm controller. You connected them, supplied usable priors, made the experimental choices explicit, and built the repeatable comparisons around them.

## One example of the complete flow

1. For stainless steel on ABS, load its fixed text description and cached LLM response. The response is not queried again after practice.
2. Construct three initial distributions: broad generic, lookup-informed, and raw LLM. Each represents uncertainty about the same unknown friction.
3. Generate two predetermined practice pushes. Extract noisy simulator-state measurements; every method receives the same measurements.
4. For zero practice slides, use each initial distribution. For one slide, combine it with the first observation. For two, combine the initial distribution with both observations once each.
5. Pass the resulting distribution to the existing arm planner. It chooses a feasible peak speed to minimize expected squared landing error, accounting for where release occurs.
6. Execute a separate scored attempt and measure its planar distance from the target. Do not use this score to update the belief.
7. Repeat over targets, conditions and seeds, then validate and summarize.

Bayesian updating changes the friction distribution, not the LLM's weights or cached answer. Calibration would instead fit a correction from other known-friction examples before these steps; it is optional and is not required by the submitted proposal.

## Remaining work to close the November 13 milestone

1. Obtain the final material manifest and actual measurement file, or document a team-approved scope revision. The proposal says at least 24 measured object–surface pairs plus published data; currently only Y01–Y03 are evaluated.
2. Extend the runner's hard-coded published-pair interface to that manifest. Today `--pair` accepts only Y01/Y02/Y03/all, loads only `ground_truth_published.csv`, and uses descriptions in `SURFACES`. Simply adding `measured_friction.csv` will not make the runner use it.
3. Fix each new material description before examining its comparison results; cache one response per pair under fixed model/settings. Retain Y01–Y03 as already-inspected development pairs.
4. Verify lookup source independence and document matching/fallback rules. Existing pilot protocol notes explicitly retain provenance caveats.
5. Record the agreed final targets, noise levels, seeds, prior construction, feasibility handling, sensing assumptions and metrics in one final protocol. Current settings are a concrete candidate, not a completed freeze.

The existing protocol already supplies most numeric settings, so this is primarily a final-data integration and documentation step. Calibration need not delay it unless the team elects to add calibration.

## Team-level proposal gaps to track separately

- The proposal mentions comparing differential IK with trajectory optimization and respecting acceleration limits. Existing controller documentation establishes differential IK and planned position/velocity/feed-forward torque checks; this audit does not establish the requested alternative comparison or acceleration-limit validation.
- V3 reports landing RMSE, friction MAE, fitted-posterior coverage/width and speed-cap frequency. It does not save a per-scored-shot release-speed tracking metric, although the proposal lists that metric and earlier arm validation studies tracking. Decide whether to add it to final runs or explicitly reference separate controller validation.
- No rendered camera is used by the evaluator. The proposal permits simulated measurement noise; camera implementation is not automatically an additional requirement. Document the actual noisy-state/ideal-timing observation model and its difference from simplified speed-and-distance wording.
- Observation noise does not simulate varying physical friction across a surface. Keep claims aligned with the implemented noise experiment.

These items should be assigned or documented by the team; they are not all Alejandro's personal implementation obligations.

## Verification performed on this upload

- Confirmed the baseline and both contribution commits in Git history.
- All three offline v3 unittest cases passed, including failure handling and shared-noise checks.
- Loaded and validated both ten-seed pilot batches: 360 v1 rows and 360 v2 rows.
- Loaded and validated all three v3 matrices: 144 rows each, 432 total; each status records 84 unique scored rollouts.
- All source/input hashes recorded by the v3 folders match this checkpoint, including after the organization changes.
- The all-pair, ten-seed v3 configuration/cache check plans 3,348 scored rows without API calls. This does not establish physical feasibility for new materials.
- No fresh Drake simulation was run here: Drake is absent. The pytest command also could not run because pytest is absent; v3 tests ran through Python's unittest runner. Saved run validation is not a substitute for a fresh physics rerun.

The saved results are useful development evidence, not proof of general LLM superiority or calibration across materials.

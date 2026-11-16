# Repository map

All commands assume the repository root. This organization groups saved arm runs and provides a current entry point while preserving the runtime module layout and experimental hashes.

## Follow the active pipeline

| Step | Files | Role |
|---|---|---|
| Material descriptions | `demo_published_comparison.py`: `OBJECT`, `SURFACES` | Current Y01–Y03 descriptions; not a general final-material manifest |
| LLM estimate | `llm_prior.py`, `llm_cache_published_pilot.json` | Fixed request settings, validated JSON, settings-aware cached responses |
| Initial distributions | `demo_published_comparison.py`: `raw_llm_logprior`; `run_arm_comparison_v2.py`: prior constants | Convert raw LLM output and define the proposal-width generic prior |
| Practice observations | `run_arm_evaluation.py`: `make_observation`; `observe.py` | Noisy simulator frames, rest position and FK release point; ideal release timing |
| Bayesian update | `belief.py`: `update_with_slides` | Combine each initial prior with the same practice evidence |
| Plan and execute | `arm.py`, `push.py`, `physics.py`, `scene.py`, `drake_config.py`, `sim_config.yaml` | Expected landing-error planning and controlled Drake arm push |
| Orchestrate comparison | `run_arm_evaluation.py` | Methods × practice counts × targets × noise × seeds |
| Validate and summarize | `summarize_arm_evaluation.py` | Reject incomplete/invalid matrices; compute metrics without duplicating beliefs across targets |

V3 imports functions from the v2 runner and from `demo_published_comparison.py`. Despite their names, those files are active dependencies; do not delete them as obsolete demos.

## Data and caches

| File | Meaning |
|---|---|
| `ground_truth_published.csv` | Simulator truth for the three published pairs |
| `ground_truth_placeholder.csv` | Synthetic placeholders; not final measurements |
| `measured_friction.csv` | Expected team dataset; absent at the audited checkpoint |
| `friction_lookup.csv` | Independent lookup baseline; source caveats still require review |
| `llm_cache_published_pilot.json` | Responses used by the active published-pair runners |
| `llm_cache_pilot.json` | Earlier pilot responses |
| `measured_friction_interface.md`, `run_log_template.csv` | Measurement contract and run-log template |

## Other code and reference material

- `friction_analysis.py`, `aggregate_pairs.py`, `make_mock_measured.py`: physical measurement processing and mock-data generation.
- `lookup_table.py`, `friction_lookup_table.tex`, `rows_*.tex`: lookup compilation and report tables.
- `e0_*.py`, `check_drake.py`: earlier physics/contact/release/observation validation.
- `e1_arm_validation.py`: known-friction arm validation.
- `perception.py`, `demo_video.py`: rendered perception and visualization; not the v3 observation path.
- `simready_audit.py`, `simready_audit.md`, `simready_friction.png`: material-library investigation.
- `model_specification.tex`: historical detailed model, including a calibrated-LLM formulation that differs from active raw v3 priors.
- `demo_bayesian_integration.py`, `demo_compare_priors.py`: early integration demonstrations.
- `run_arm_comparison.py`, `summarize_arm_pilot.py`: v1 pilot.
- `run_arm_comparison_v2.py`, `summarize_arm_v2.py`: v2 pilot; v2 also supplies execution to v3.
- `tests/`: existing unit/physics/controller checks plus the new offline v3 contract tests.
- Root-level E0/E1 CSV/PNG/TXT artifacts: earlier team validation, kept at existing paths.

See [the results index](../results/README.md) for the moved arm runs and [the contribution audit](ALEJANDRO_CONTRIBUTION.md) for ownership and remaining work.

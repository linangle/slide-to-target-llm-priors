# Repository map

Paths are from the repository root. On 16 Nov the files were grouped into folders (README, *Repository layout*): this map is Alejandro's, with each file shown at its new path. The runs saved under `results/` record source hashes by bare file name, which the move does not change; the contents of `arm.py`, `push.py`, `physics.py`, `belief.py`, `scene.py`, `llm_prior.py` and the runners have changed since those runs.

## Follow the active pipeline

| Step | Files | Role |
|---|---|---|
| Material descriptions | `evaluation/demo_published_comparison.py`: `OBJECT`, `SURFACES` | Current Y01–Y03 descriptions; not a general final-material manifest |
| LLM estimate | `src/llm_prior.py`, `data/llm_cache_published_pilot.json` | Fixed request settings, validated JSON, settings-aware cached responses |
| Initial distributions | `evaluation/demo_published_comparison.py`: `raw_llm_logprior`; `evaluation/run_arm_comparison_v2.py`: prior constants | Convert raw LLM output and define the proposal-width generic prior |
| Practice observations | `evaluation/run_arm_evaluation.py`: `make_observation`; `src/observe.py` | Noisy simulator frames, rest position and FK release point; ideal release timing |
| Bayesian update | `src/belief.py`: `update_with_slides` | Combine each initial prior with the same practice evidence |
| Plan and execute | `src/arm.py`, `src/push.py`, `src/physics.py`, `src/scene.py`, `src/drake_config.py`, `src/sim_config.yaml` | Expected landing-error planning and controlled Drake arm push |
| Orchestrate comparison | `evaluation/run_arm_evaluation.py` | Methods × practice counts × targets × noise × seeds |
| Validate and summarize | `evaluation/summarize_arm_evaluation.py` | Reject incomplete/invalid matrices; compute metrics without duplicating beliefs across targets |

V3 imports functions from the v2 runner and from `evaluation/demo_published_comparison.py`. Despite their names, those files are active dependencies; do not delete them as obsolete demos.

## Data and caches

| File | Meaning |
|---|---|
| `data/ground_truth_published.csv` | Simulator truth for the three published pairs |
| `data/ground_truth_placeholder.csv` | Synthetic placeholders; not final measurements |
| `measured_friction.csv` | Expected team dataset; absent at the audited checkpoint |
| `data/friction_lookup.csv` | Independent lookup baseline; source caveats still require review |
| `data/llm_cache_published_pilot.json` | Responses used by the active published-pair runners |
| `data/llm_cache_pilot.json` | Earlier pilot responses |
| `docs/measured_friction_interface.md`, `data/run_log_template.csv` | Measurement contract and run-log template |

## Other code and reference material

- `friction/friction_analysis.py`, `friction/aggregate_pairs.py`, `friction/make_mock_measured.py`: physical measurement processing and mock-data generation.
- `friction/lookup_table.py`, `paper/friction_lookup_table.tex`, `paper/rows_*.tex`: lookup compilation and report tables.
- `experiments/e0_*.py`, `experiments/check_drake.py`: earlier physics/contact/release/observation validation.
- `experiments/e1_arm_validation.py`: known-friction arm validation.
- `src/perception.py`, `figures/demo_video.py`: rendered perception and visualization; not the v3 observation path.
- `friction/simready_audit.py`, `docs/simready_audit.md`, `figures/simready_friction.png`: material-library investigation.
- `paper/model_specification.tex`: historical detailed model, including a calibrated-LLM formulation that differs from active raw v3 priors.
- `evaluation/demo_bayesian_integration.py`, `evaluation/demo_compare_priors.py`: early integration demonstrations.
- `evaluation/run_arm_comparison.py`, `evaluation/summarize_arm_pilot.py`: v1 pilot.
- `evaluation/run_arm_comparison_v2.py`, `evaluation/summarize_arm_v2.py`: v2 pilot; v2 also supplies execution to v3.
- `tests/`: existing unit/physics/controller checks plus the new offline v3 contract tests.
- `results/e0/`, `results/e1/`, `results/arm_rerun/`: the E0/E1 CSV/PNG/TXT artifacts of the earlier team validation (moved from the root on 16 Nov).

See [the results index](../results/README.md) for the moved arm runs and [the contribution audit](ALEJANDRO_CONTRIBUTION.md) for ownership and remaining work.

# Slide-to-Target

MIT 6.4210 Robotic Manipulation · Team 3.4

A KUKA iiwa pushes an object toward a target. We compare generic, lookup, and text-only LLM friction priors, update each with the same 0/1/2 practice slides, and plan a separate scored push. A true-friction oracle provides a reference.

## Start here

| Contribution and completion audit(docs/ALEJANDRO_CONTRIBUTION.md) |
| Where does each part live? | [Repository map](docs/REPOSITORY_MAP.md) |
| Latest multi-target/noise settings | [V3 evaluation](EVALUATION_V3.md) |
| Saved results and which version to use | [Results index](results/README.md) |
| Arm implementation and limits | [Arm controller](ARM_CONTROLLER.md) |
| Fixed three-pair v2 pilot | [V2 protocol](PILOT_V2_PROTOCOL.md) |
| Measurement data contract | [Measured-friction interface](measured_friction_interface.md) |
| Earlier measurement instructions and team decisions | [Historical technical reference](docs/TECHNICAL_REFERENCE.md) |

## Current status

Integration and development evaluation are implemented. The final project protocol is **not frozen**. The runner currently accepts only Y01–Y03, and `measured_friction.csv` is absent. The proposal calls for at least 24 measured pairs plus published data; this requires data delivery, manifest/description integration, and a documented final configuration (or an explicitly revised scope).

Saved evidence: v1 and v2 each include 360 scored rows across ten noise seeds; v3 includes 432 scored rows across three pairs, three targets, and four noise conditions at one noisy seed. These are separate experiment versions, not one pooled dataset. The same three material pairs recur.

The current primary method uses raw LLM priors followed by Bayesian practice updates. Empirical calibration is not a requirement of the submitted proposal. Historical calibrated-model code remains available but is not the v3 method.

## Setup and checks

Run commands from this repository root in your existing virtual environment. For a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Offline checks (no API request or Drake simulation):

```bash
python -m unittest discover -s tests -p test_evaluation_v3.py -v
python run_arm_evaluation.py --pair all --seeds 0 1 2 3 4 5 6 7 8 9 --check-only
```

Full test suite with the project dependencies installed:

```bash
python -m pytest tests -q
```

Read existing summaries directly in [the results index](results/README.md). To regenerate a summary, choose a NEW output folder; existing summary directories are intentionally not overwritten:

```bash
python summarize_arm_evaluation.py results/evaluation_v3/arm_eval_v3_Y01_check --out results/evaluation_v3/Y01_recheck
python summarize_arm_v2.py --out results/pilot_v2/recheck_summary
python summarize_arm_pilot.py --out results/pilot_v1/recheck_summary
```

Example new Drake development run, using cached responses:

```bash
python run_arm_evaluation.py --pair Y01 --seeds 8 --out results/evaluation_v3/Y01_seed8
python summarize_arm_evaluation.py results/evaluation_v3/Y01_seed8
```

This is an example, not a request to start the final experiment. Evaluators never query the API on a cache miss. Live querying is explicit in `llm_prior.py --query`.

## Layout and provenance

Runtime Python modules, input CSV/JSON/YAML files, and older E0/E1 outputs remain at their established paths. Saved arm comparisons are grouped into `results/pilot_v1/`, `results/pilot_v2/`, and `results/evaluation_v3/`. Documentation lives in `docs/` and the linked protocol files.

Organization preserves saved experiment bytes and all source/input files hashed by the v3 runs. Only the v1/v2 summary scripts' default result locations change. Historic records retain their original settings. The ignored `EVALUATION_SPEC_DRAFT.md`, if present locally, is an earlier planning note, not the current protocol.

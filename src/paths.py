"""
paths.py -- where things live in the repository, so no script depends on the directory it is run from.

    src/           the library: physics, scene, push, arm, decide, belief, observe, perception, llm_prior, drake_config
                   (+ sim_config.yaml); importable from anywhere after `pip install -r requirements.txt`
    experiments/   E0 and E1 validation scripts and the arm re-run
    evaluation/    the method comparison: runners, summarisers, demonstrations
    friction/      rig data processing, the lookup table, the SimReady audit
    figures/       scripts that draw figures and render the video, and their outputs
    data/          inputs: ground truth, lookup table, LLM response caches, run-log template
    results/       saved outputs of experiments/ and evaluation/
    paper/         LaTeX: the model specification, the lookup-table document and their generated rows
    docs/          guides and protocols
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC, EXPERIMENTS, EVALUATION, FRICTION, FIGURES = (REPO/d for d in ("src", "experiments", "evaluation", "friction", "figures"))
DATA, RESULTS, PAPER, DOCS = (REPO/d for d in ("data", "results", "paper", "docs"))

def locate(name):
    """The file called `name` among the source and input folders (src, data, then the script folders).  The evaluation
    runners use it to hash the sources and inputs a run depended on, which they record by bare file name."""
    for d in (SRC, DATA, EVALUATION, EXPERIMENTS, FRICTION, FIGURES, PAPER):
        if (d/name).is_file():
            return d/name
    raise FileNotFoundError(f"{name} is not in src/, data/ or a script folder of {REPO}")

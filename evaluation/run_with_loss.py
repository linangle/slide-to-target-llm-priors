#!/usr/bin/env python3
"""
run_with_loss.py -- run_arm_evaluation.py with another loss in the planner, without changing the runner.

The evaluators call arm.plan_peak_speed with its default loss (decide.loss_sq_rel, the specification's).  This wrapper
swaps in another loss from decide.LOSSES for one run, so the two can be compared on the same setup; everything after the
loss name is passed to run_arm_evaluation.py as it is.  It writes loss_override.json in the run folder, because
settings.json has no field for the loss.

    python3 evaluation/run_with_loss.py sq_abs --pair all --seeds 7 --out results/side_base/evaluation/loss_sq_abs
    python3 evaluation/run_with_loss.py sq_abs --v3 --pair all --seeds 7 --out results/evaluation_v3/merged_code_seed7_sq_abs
"""
import functools, json, sys
from pathlib import Path
import arm, decide
import run_arm_evaluation as E

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in decide.LOSSES:
        sys.exit(f"usage: run_with_loss.py {{{','.join(decide.LOSSES)}}} <run_arm_evaluation.py options>")
    name = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    arm.plan_peak_speed = functools.partial(arm.plan_peak_speed, loss=decide.LOSSES[name])   # run() imports it from arm when it starts
    try:
        E.main()
    finally:
        if "--out" in sys.argv and Path(sys.argv[sys.argv.index("--out") + 1]).is_dir():
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            (out/"loss_override.json").write_text(json.dumps({"loss": name, "note": "planner loss set by evaluation/run_with_loss.py; "
                                                                             "the runner's default is sq_rel"}, indent=2) + "\n")

if __name__ == "__main__":
    main()

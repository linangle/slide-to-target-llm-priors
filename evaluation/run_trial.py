#!/usr/bin/env python3
"""
run_trial.py -- run_arm_evaluation.py with the planner's loss, the braking jerk, the board and the object's start gap
set from outside, and every push's release point recorded.  The 20 Nov trials for D3 (loss), D4 (what a miss is measured
from), D10 (braking jerk) and D15 (one-piece board).  No file the evaluation hashes is edited: this wrapper replaces
names in the loaded modules for the duration of one run.

    python3 evaluation/run_trial.py [--loss sq_rel|sq_abs|...] [--jerk 100] [--one-piece] [--gap-error SEED] -- <run_arm_evaluation.py options>

--loss       the loss arm.plan_peak_speed plans a belief's push with (decide.LOSSES; the default is the runner's, sq_rel)
--jerk       the braking jerk J in m/s^3, for the planner, the cap / practice / target rules and the push itself (default 100)
--one-piece  a uniform board in one piece instead of 24 equal strips (no seams: D15)
--gap-error  each push starts with the paddle 0.5-3 mm behind the object instead of the 1 mm the planner assumes
             (E1 part C's range), drawn from this seed.  Without it the gap is 1 mm, as in every saved run.

With none of the four options the run is the plain evaluator's, row for row.  Besides the evaluator's files the run folder gets
  trial_override.json   what was set here (settings.json has no field for the loss, the board or the gap)
  release_log.csv       one row per Drake push: the FK release point, the object's true position there, the start gap, the rest
  both_measures.csv     results.csv with the miss measured two ways (D4):
        miss_rest_mm   = x_rest - target                               the object's rest position on the board (what is scored)
        miss_slide_mm  = (x_rest - x_fk) - (target - x_release_model)  the slide from the release point forward kinematics
                         reports, against the slide the plan wanted from the release point the model expects for that push
    The two differ by x_fk - x_release_model: the start-gap error plus the release model's error.
"""
import argparse, csv, functools, json, sys
from pathlib import Path

GAP_NOMINAL, GAP_RANGE = 0.001, (0.0005, 0.003)
LOG_FIELDS = ["mu", "v_peak", "J", "gap_mm", "x_fk", "x_sep", "v_sep", "x_rest", "y_rest", "separated"]

def release_model_x(u, mu, J, x0=0.55):
    """Where the plan expects the object's centre at release, with the nominal start gap (arm.rest_x's first three terms)."""
    from physics import release_push_distance
    return x0 - GAP_NOMINAL + float(release_push_distance(u, mu, J))

def miss_slide(x_rest, x_fk, target, u, mu, J, x0=0.55):
    """D4's second measure, in m: the slide from the FK release point minus the slide the plan wanted."""
    return (x_rest - x_fk) - (target - release_model_x(u, mu, J, x0))

def both_measures(out, J):
    """Write both_measures.csv in a finished (or failed) run folder from results.csv and release_log.csv."""
    out = Path(out)
    # a push is identified by its friction, speed and rest position: two pairs can share a friction and a speed (Y01, Y02)
    log = {(float(r["mu"]), float(r["v_peak"]), float(r["x_rest"])): r for r in csv.DictReader((out/"release_log.csv").open())}
    rows = list(csv.DictReader((out/"results.csv").open()))
    for r in rows:
        p = log[(float(r["mu_true"]), float(r["peak_speed"]), float(r["x_rest"]))]
        target = float(r["target_x"])
        r["gap_mm"], r["x_fk"] = p["gap_mm"], p["x_fk"]
        r["x_release_model"] = release_model_x(float(r["peak_speed"]), float(r["mu_true"]), J)
        r["miss_rest_mm"] = 1e3*(float(r["x_rest"]) - target)
        r["miss_slide_mm"] = 1e3*miss_slide(float(r["x_rest"]), float(p["x_fk"]), target, float(r["peak_speed"]), float(r["mu_true"]), J)
    if rows:
        with (out/"both_measures.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    return rows

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        sys.exit(__doc__)
    cut = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    import decide
    p.add_argument("--loss", choices=list(decide.LOSSES), default=None)
    p.add_argument("--jerk", type=float, default=100.0)
    p.add_argument("--one-piece", action="store_true")
    p.add_argument("--gap-error", type=int, default=None, metavar="SEED")
    a = p.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    if "--out" not in rest:
        sys.exit("run_trial.py: give the evaluator --out <new folder>")
    out = Path(rest[rest.index("--out") + 1])

    import numpy as np
    import arm, push, scene
    import run_arm_evaluation as E
    if a.loss is not None:
        arm.plan_peak_speed = functools.partial(arm.plan_peak_speed, loss=decide.LOSSES[a.loss])   # run() imports it from arm when it starts
    E.JERK = float(a.jerk)                                    # the planner, the cap / practice / target rules, settings.json
    if a.one_piece:
        push.Table = functools.partial(scene.Table, strip=10)   # one strip covers the board
    rng = np.random.default_rng(a.gap_error) if a.gap_error is not None else None
    log, original = [], push.push_slide

    def push_slide(mu, v_peak, J, *args, pusher=None, **kw):   # run_arm_comparison_v2.execute imports this at each call, with J = 100
        gap = GAP_NOMINAL
        if rng is not None:
            gap = float(rng.uniform(*GAP_RANGE))
            pusher.gap = gap
        r = original(mu, v_peak, float(a.jerk), *args, pusher=pusher, **kw)
        log.append(dict(mu=repr(float(mu)), v_peak=repr(float(v_peak)), J=a.jerk, gap_mm=gap*1e3, x_fk=repr(r["x_fk"]), x_sep=repr(r["x_sep"]),
                        v_sep=r["v_sep"], x_rest=repr(r["x_rest"]), y_rest=r["y_rest"], separated=r["separated"]))
        if out.is_dir():
            with (out/"release_log.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_FIELDS, lineterminator="\n")
                w.writeheader(); w.writerows(log)
        return r

    push.push_slide = push_slide
    sys.argv = [sys.argv[0]] + rest
    try:
        E.main()
    finally:
        push.push_slide = original
        if out.is_dir():
            (out/"trial_override.json").write_text(json.dumps(dict(
                loss=a.loss or "sq_rel (the runner's default)", jerk=a.jerk, board="one piece" if a.one_piece else "24 equal strips (as saved runs)",
                start_gap=f"uniform {GAP_RANGE[0]*1e3:g}-{GAP_RANGE[1]*1e3:g} mm per push, seed {a.gap_error}; planned for 1 mm" if rng is not None else "1 mm",
                note="set by evaluation/run_trial.py from outside the runner; a row of release_log.csv per Drake push, practice pushes first per pair "
                     "after the oracle's preflight pushes"), indent=2) + "\n")
            if (out/"results.csv").exists() and (out/"release_log.csv").exists():
                both_measures(out, float(a.jerk))

if __name__ == "__main__":
    main()

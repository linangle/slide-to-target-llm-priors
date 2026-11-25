#!/usr/bin/env python3
"""
run_pair_set_trial.py -- run_pair_set.py (any set of pairs) with the arm setup and the trial switches of run_trial.py
set from outside: the brake cap, the acceleration check and the posture rule (D9), the planner's loss (D3), the braking
jerk (D10), a one-piece board (D15), a start-gap error, and both miss measures (D4).  No file an evaluation hashes is
edited.  run_pair_set.py runs its pairs in worker processes, so the switches are applied in each worker when it starts.

    python3 evaluation/run_pair_set_trial.py [switches] -- <run_pair_set.py options, with --out>
    python3 evaluation/run_pair_set_trial.py --summarize RUN [RUN ...] [--out summary.txt]

--a-brake A      brake cap in m/s^2 (default: the arm's 12)
--check-accel    planned joint accelerations must be inside the SDF limits (in the cap and in each push's validity)
--check-release  a push is valid only if the object left the paddle before the paddle stopped (run_arm_evaluation.released)
--posture NAME   speed (the arm's default) or limits (arm.POSTURES)
--loss NAME      the loss arm.plan_peak_speed plans a belief's push with (decide.LOSSES; default: the runner's, sq_rel)
--jerk J         braking jerk for the planner and the push (default 100).  The speed cap, the practice speed and the
                 targets are still computed at J = 100, so runs at different J share them.
--one-piece      a uniform board in one piece instead of 24 equal strips
--gap-error SEED each push starts 0.5-3 mm behind the object instead of the 1 mm the planner assumes; the gap of a push
                 is drawn from (SEED, friction, speed), so it does not depend on which worker runs the pair

With no switch the run is run_pair_set.py's.  Besides that runner's files the run folder gets trial_override.json,
release_log.csv (one row per Drake push) and both_measures.csv (results.csv with miss_rest_mm and miss_slide_mm, as in
run_trial.py).
"""
import argparse, csv, functools, json, math, os, sys, zlib
from pathlib import Path
from types import SimpleNamespace

from run_trial import GAP_NOMINAL, GAP_RANGE, LOG_FIELDS, miss_slide, release_model_x

SETUP_JERK = 100.0

def gap_of(seed, mu, v_peak):
    """The start gap of one push in m: a draw that depends only on the seed, the friction and the speed."""
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(f"{float(mu)!r}:{float(v_peak)!r}".encode())]))
    return float(rng.uniform(*GAP_RANGE))

def worker_setup(o):
    """Apply the switches in this process (a pool worker, or the parent for a check)."""
    import arm, decide, push, scene
    import run_arm_evaluation as E
    if o["loss"] is not None:
        arm.plan_peak_speed = functools.partial(arm.plan_peak_speed, loss=decide.LOSSES[o["loss"]])
    E.JERK = float(o["jerk"])
    if o["one_piece"]:
        push.Table = functools.partial(scene.Table, strip=10)
    original, parts = push.push_slide, o["parts"]

    def push_slide(mu, v_peak, J, *args, pusher=None, **kw):
        gap = GAP_NOMINAL
        if o["gap_error"] is not None:
            gap = gap_of(o["gap_error"], mu, v_peak)
            pusher.gap = gap
        r = original(mu, v_peak, float(o["jerk"]), *args, pusher=pusher, **kw)
        if parts is not None:
            Path(parts).mkdir(exist_ok=True)
            path = Path(parts)/f"{os.getpid()}.csv"
            new = not path.exists()
            with path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_FIELDS, lineterminator="\n")
                if new:
                    w.writeheader()
                w.writerow(dict(mu=repr(float(mu)), v_peak=repr(float(v_peak)), J=o["jerk"], gap_mm=gap*1e3, x_fk=repr(r["x_fk"]), x_sep=repr(r["x_sep"]),
                                v_sep=r["v_sep"], x_rest=repr(r["x_rest"]), y_rest=r["y_rest"], separated=r["separated"]))
        return r

    push.push_slide = push_slide

def both_measures(out, J):
    """both_measures.csv from results.csv and release_log.csv.  results.csv rounds speed and rest position to 1e-6."""
    out = Path(out)
    log = {(float(r["mu"]), round(float(r["v_peak"]), 6), round(float(r["x_rest"]), 6)): r for r in csv.DictReader((out/"release_log.csv").open())}
    rows = list(csv.DictReader((out/"results.csv").open()))
    for r in rows:
        p = log.get((float(r["mu_true"]), float(r["peak_speed"]), float(r["x_rest"]))) if r["valid"] == "True" else None
        target = float(r["target_x"])
        r["miss_rest_mm"] = 1e3*(float(r["x_rest"]) - target)
        if p is None:
            r["gap_mm"] = r["x_fk"] = r["x_release_model"] = r["miss_slide_mm"] = ""
            continue
        u = float(p["v_peak"])
        r["gap_mm"], r["x_fk"] = p["gap_mm"], p["x_fk"]
        r["x_release_model"] = release_model_x(u, float(r["mu_true"]), J)
        r["miss_slide_mm"] = 1e3*miss_slide(float(p["x_rest"]), float(p["x_fk"]), target, u, float(r["mu_true"]), J)
    if rows:
        with (out/"both_measures.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    return rows

# ----------------------------------------------------------------------------- summary of several runs

def _rmse(v):
    v = [x for x in v if x == x]
    return math.sqrt(sum(x*x for x in v)/len(v)) if v else float("nan")

def _mae(v):
    v = [x for x in v if x == x]
    return sum(abs(x) for x in v)/len(v) if v else float("nan")

def _load(folder):
    folder = Path(folder)
    rows = list(csv.DictReader((folder/"both_measures.csv").open()))
    for r in rows:
        r["slides"] = int(r["slides"])
        for k in ("target_x", "peak_speed", "miss_rest_mm", "miss_slide_mm"):
            r[k] = float(r[k]) if r[k] != "" else float("nan")
    return dict(name=folder.name, rows=rows, settings=json.loads((folder/"settings.json").read_text()),
                override=json.loads((folder/"trial_override.json").read_text()), status=json.loads((folder/"status.json").read_text()))

def summarize(folders, groups=None):
    runs = [_load(f) for f in folders if (Path(f)/"both_measures.csv").exists()]
    L = ["Pair-set trial runs (evaluation/run_pair_set_trial.py); landing error along x in mm, nominal noise", ""]
    for run in runs:
        s, o, rows = run["settings"], run["override"], run["rows"]
        bad = [r for r in rows if r["valid"] != "True"]
        L.append(f"{run['name']}: {len(s['pairs'])} pairs, friction {min(i['mu'] for i in s['pair_info'].values())}-{max(i['mu'] for i in s['pair_info'].values())}; "
                 f"seeds {s['seeds']}; loss {o['loss']}; J {o['jerk']:g}; board {o['board']}; start gap {o['start_gap']}; brake {s['arm']['brake_cap']:g} m/s^2, "
                 f"posture {s['arm'].get('posture', 'speed')}, limits checked {'/'.join(s['arm']['checked_limits'])}; cap {s['peak_cap']}, practice {s['practice_peak']}, "
                 f"targets {[round(t, 3) for t in s['targets']]}; {len(rows)} rows, {len(bad)} invalid"
                 + (": " + "; ".join(sorted({f"{r['pair']} at {r['peak_speed']:.3f} m/s ({r['separated_or_limits']})" for r in bad})[:6]) if bad else ""))
    methods = [m for m in dict.fromkeys(r["method"] for r in runs[0]["rows"]) if m != "oracle"] if runs else []
    groups = groups or {"all pairs": None}

    def table(key, stat, title):
        out = [title, f"{'run':24} {'group':16} " + " ".join(f"{'n=' + str(n) + ' ' + m:>15}" for n in (0, 1, 2) for m in methods) + f" {'oracle':>8}"]
        for run in runs:
            for gname, ps in groups.items():
                rows = [r for r in run["rows"] if r["valid"] == "True" and (ps is None or r["pair"] in ps)]
                if not rows:
                    continue
                cells = [stat(r[key] for r in rows if r["slides"] == n and r["method"] == m) for n in (0, 1, 2) for m in methods]
                out.append(f"{run['name']:24} {gname:16} " + " ".join(f"{c:15.2f}" for c in cells) + f" {stat(r[key] for r in rows if r['method'] == 'oracle'):8.2f}")
        return out

    L += [""] + table("miss_rest_mm", _rmse, "RMSE, miss measured from the rest position (the runner's score)")
    L += [""] + table("miss_rest_mm", _mae, "Mean |miss|, rest position")
    L += [""] + table("miss_slide_mm", _rmse, "RMSE, miss measured as the slide from the FK release point")
    L += ["", "Signed mean miss with no practice, rest position (mm): " + " / ".join(methods)]
    for run in runs:
        rows = [r for r in run["rows"] if r["valid"] == "True" and r["slides"] == 0]
        L.append(f"{run['name']:24} " + " / ".join(f"{sum(r['miss_rest_mm'] for r in rows if r['method'] == m)/max(1, sum(1 for r in rows if r['method'] == m)):+.1f}" for m in methods))
    L += ["", "Difference between the measures, miss_slide - miss_rest, valid rows (mm): mean, SD, largest"]
    for run in runs:
        d = [r["miss_slide_mm"] - r["miss_rest_mm"] for r in run["rows"] if r["miss_slide_mm"] == r["miss_slide_mm"]]
        m = sum(d)/len(d)
        L.append(f"{run['name']:24} {m:+.3f}  {math.sqrt(sum((x - m)**2 for x in d)/len(d)):.3f}  {max(abs(x) for x in d):.3f}")
    L += ["", "Which method is closest, per pair, with no practice (mean |miss| over the targets): pairs won by " + " / ".join(methods)]
    for run in runs:
        rows = [r for r in run["rows"] if r["valid"] == "True" and r["slides"] == 0]
        wins = dict.fromkeys(methods, 0)
        for p in run["settings"]["pairs"]:
            score = {m: _mae(r["miss_rest_mm"] for r in rows if r["pair"] == p and r["method"] == m) for m in methods}
            score = {m: v for m, v in score.items() if v == v}
            if score:
                wins[min(score, key=score.get)] += 1
        L.append(f"{run['name']:24} " + " / ".join(str(wins[m]) for m in methods))
    return "\n".join(L) + "\n"

# ----------------------------------------------------------------------------- the run

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--summarize" in argv:
        p = argparse.ArgumentParser()
        p.add_argument("--summarize", nargs="+", type=Path)
        p.add_argument("--out", type=Path, default=None)
        a = p.parse_args(argv)
        groups = {"all pairs": None, "Y01-Y03": {"Y01", "Y02", "Y03"}, "handbook": {f"L{i:02d}" for i in range(1, 21)}}
        text = summarize(a.summarize, groups)
        print(text, end="")
        if a.out:
            a.out.write_text(text)
        return
    if "--" not in argv:
        sys.exit(__doc__)
    cut = argv.index("--")
    import decide
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-brake", type=float, default=None)
    p.add_argument("--check-accel", action="store_true")
    p.add_argument("--check-release", action="store_true")
    p.add_argument("--posture", choices=["speed", "limits"], default=None)
    p.add_argument("--loss", choices=list(decide.LOSSES), default=None)
    p.add_argument("--jerk", type=float, default=SETUP_JERK)
    p.add_argument("--one-piece", action="store_true")
    p.add_argument("--gap-error", type=int, default=None, metavar="SEED")
    a = p.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    check_only = "--check-only" in rest
    if "--out" not in rest and not check_only:
        sys.exit("run_pair_set_trial.py: give the runner --out <new folder>")
    out = Path(rest[rest.index("--out") + 1]) if "--out" in rest else None

    import multiprocessing
    import arm
    import run_arm_evaluation as E
    import run_pair_set as P
    opts = dict(loss=a.loss, jerk=a.jerk, one_piece=a.one_piece, gap_error=a.gap_error, parts=None if out is None else str(out/"release_parts"))
    setup, grid = E.arm_setup, arm.target_grid

    def arm_setup(args, truth, post):                  # the cap and the practice speed, at J = 100 whatever the trial's J
        ns = SimpleNamespace(**vars(args))
        if a.a_brake is not None: ns.a_brake = a.a_brake
        if a.check_accel: ns.check_accel = True
        if a.check_release: ns.check_release = True
        if a.posture is not None: ns.posture = a.posture
        keep, E.JERK = E.JERK, SETUP_JERK
        try:
            return setup(ns, truth, post)
        finally:
            E.JERK = keep

    E.arm_setup = arm_setup
    arm.target_grid = lambda cap, mu, n, J, x0: grid(cap, mu, n, SETUP_JERK, x0)      # run() imports it from arm when it starts
    E.JERK = float(a.jerk)                             # settings.json records the trial's J
    P.Pool = functools.partial(multiprocessing.Pool, initializer=worker_setup, initargs=(opts,))
    sys.argv = [sys.argv[0]] + rest
    try:
        P.main()
    finally:
        if out is not None and out.is_dir():
            parts = sorted((out/"release_parts").glob("*.csv")) if (out/"release_parts").is_dir() else []
            if parts:
                rows = [r for part in parts for r in csv.DictReader(part.open())]
                with (out/"release_log.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=LOG_FIELDS, lineterminator="\n")
                    w.writeheader(); w.writerows(rows)
                for part in parts:
                    part.unlink()
                (out/"release_parts").rmdir()
            (out/"trial_override.json").write_text(json.dumps(dict(
                loss=a.loss or "sq_rel (the runner's default)", jerk=a.jerk, board="one piece" if a.one_piece else "24 equal strips (as saved runs)",
                start_gap=f"uniform {GAP_RANGE[0]*1e3:g}-{GAP_RANGE[1]*1e3:g} mm per push, seed {a.gap_error}; planned for 1 mm" if a.gap_error is not None else "1 mm",
                a_brake=a.a_brake, check_accel=a.check_accel, check_release=a.check_release, posture=a.posture,
                note="set by evaluation/run_pair_set_trial.py from outside the runner, in each worker process; the cap, the practice speed and the "
                     "targets are computed at J = 100; release_log.csv has a row per Drake push, in no fixed order"), indent=2) + "\n")
            if (out/"results.csv").exists() and (out/"release_log.csv").exists():
                both_measures(out, float(a.jerk))

if __name__ == "__main__":
    main()

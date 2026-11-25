#!/usr/bin/env python3
"""
base_trial.py -- the 19 Nov trial: is there a base position and brake cap that keeps EVERY push the comparison needs inside
the iiwa's joint speed, torque and acceleration limits?  (Decision D9.)

Two things are tried, neither changes arm.py:

  posture   arm.ArmPusher.make_plan picks, among the IK postures that cover the stroke, the one with the lowest joint speed
            per m/s of paddle speed.  With the acceleration limit in the picture that is the wrong criterion near the edge
            of the arm's reach: slow pushes brake close to the start pose and go over the acceleration limit although faster
            ones are inside it.  limit_aware() swaps the choice for the posture whose WORST of the three ratios (speed,
            torque, acceleration) is lowest for the push at hand.  It is a patch applied from outside (as
            patchy_table_study.patchy_board is); without it every script behaves as before.
  base      --scan plans every speed from 0.15 to 1.10 m/s over a grid of base positions and brake caps, with either
            posture choice, and prints which speeds are inside all three limits.  Planning only (no contact).

--validate then repeats the two Drake checks of the arm (e1_arm_validation.py: 85 pushes; e1_release_error.py: 35 pushes)
at one base and brake cap with the limit-aware posture; --camera repeats E0-5's camera replay with those postures in view.

    python3 experiments/base_trial.py --scan                                     # ~10 min on 8 workers -> results/side_base/base_trial/scan_limit_aware.txt
    python3 experiments/base_trial.py --scan --posture speed                     # the same with arm.py's own posture choice
    python3 experiments/base_trial.py --validate --base-x 0.6 --a-brake 6        # ~3 min -> results/side_base/base_trial/base060_brake6_limit_aware_*
    python3 experiments/base_trial.py --camera --base-x 0.6 --a-brake 6          # the camera replay with those postures in view
"""
import argparse, contextlib, csv, os
from multiprocessing import Pool
import numpy as np
from paths import RESULTS

OUT = RESULTS/"side_base"/"base_trial"
SPEEDS = [round(0.15 + 0.05*i, 2) for i in range(20)]        # 0.15 .. 1.10 m/s
KEYS = ("qd_ratio", "tau_ratio", "qdd_ratio")

def _limit_aware_make_plan(self, prof, _original=None):
    """ArmPusher.make_plan with the posture chosen for the lowest worst limit ratio of THIS push (speed, torque, acceleration)."""
    m = self.model
    cands = m.candidates(self.X0, seed=self.posture_seed)
    if not cands:
        return _original(self, prof)                       # raises arm.py's own error
    best, reasons, real = None, [], m.candidates
    try:
        for q in cands:
            m.candidates = lambda *a, _q=q, **k: [_q]      # one posture at a time through arm.py's own planning code
            try:
                p = _original(self, prof)
            except RuntimeError as ex:
                reasons.append(str(ex)); continue
            worst = max(p[k] for k in KEYS)
            if best is None or worst < best[0]:
                best = (worst, q)
        if best is None:
            raise RuntimeError("no posture covers the push: " + "; ".join(sorted(set(reasons))))
        m.candidates = lambda *a, _q=best[1], **k: [_q]
        plan = _original(self, prof)
        plan["n_candidates"] = len(cands)
        return plan
    finally:
        m.candidates = real

@contextlib.contextmanager
def limit_aware(on=True):
    """While open, arm.ArmPusher plans with the limit-aware posture choice (on=False: arm.py's own, a no-op)."""
    import arm
    original = arm.ArmPusher.make_plan
    if on:
        arm.ArmPusher.make_plan = lambda self, prof: _limit_aware_make_plan(self, prof, original)
    try:
        yield
    finally:
        arm.ArmPusher.make_plan = original

# ---------------------------------------------------------------------- the planning scan

def _scan_one(job):
    import arm
    x, y, cap, u, aware = job
    with limit_aware(aware):
        try:
            r = arm.feasibility(u, 100.0, (x, y, 0.0), 0.55, cap)
        except Exception:
            return job, None
    return job, tuple(r[k] for k in KEYS) if r["ok"] else None

def scan(xs, ys, caps, aware, workers):
    jobs = [(x, y, c, u, aware) for y in ys for c in caps for x in xs for u in SPEEDS]
    with Pool(workers) as pool:
        res = dict(pool.imap_unordered(_scan_one, jobs, chunksize=2))
    L = [f"posture chosen for the {'lowest worst limit ratio (limit-aware)' if aware else 'lowest joint speed (arm.py)'}; J 100 m/s^3, object start x0 = 0.55 m",
         "per base and brake cap, by commanded peak speed: '.' inside the speed, torque and acceleration limits; v, t, a = over that limit; x = no plan",
         f"{'base (x, y) m':>14} {'cap':>4} | " + " ".join(f"{u:>4.2f}" for u in SPEEDS) + " | inside all three, m/s        accel ratio"]
    for y in ys:
        for c in caps:
            for x in xs:
                marks, acc = [], []
                for u in SPEEDS:
                    v = res[(x, y, c, u, aware)]
                    marks.append("x" if v is None else ("".join(k for k, r in zip("vta", v) if r > 1.0) or "."))
                    if v is not None:
                        acc.append(v[2])
                inside = [u for u, mk in zip(SPEEDS, marks) if mk == "."]
                run, cur = [], []                           # the longest unbroken run of speeds inside the limits
                for u, mk in zip(SPEEDS, marks):
                    cur = cur + [u] if mk == "." else []
                    run = cur if len(cur) > len(run) else run
                L.append(f"{f'({x:.2f}, {y:.2f})':>14} {c:>4g} | " + " ".join(f"{mk:>4}" for mk in marks) + " | " +
                         (f"{run[0]:.2f}..{run[-1]:.2f} unbroken ({len(inside)} of {len(SPEEDS)})" if run else "none").ljust(28) +
                         (f" {min(acc):.2f}..{max(acc):.2f}" if acc else ""))
            L.append("")
    return "\n".join(L)

# ---------------------------------------------------------------------- the Drake checks, with the patch inside each worker

def _e1_run(job):
    import e1_arm_validation as E1
    with limit_aware():
        return E1.run(job)

def _rel_run(job):
    import e1_release_error as ER
    with limit_aware():
        return ER.run(job)

def validate(base, a_brake, stem, workers):
    import e1_arm_validation as E1, e1_release_error as ER
    js = E1.jobs(False, base, a_brake)
    print(f"{len(js)} arm-validation pushes on {workers} workers ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(_e1_run, js, chunksize=1)
        rel = pool.map(_rel_run, [(mu, v, base, a_brake) for mu in ER.MUS for v in ER.SPEEDS], chunksize=1)
    rows.sort(key=lambda r: (r["part"], r["case"], r["pusher"]))
    with open(f"{stem}_e1_arm_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, E1.FIELDS); w.writeheader(); w.writerows(rows)
    text = E1.summarize(E1.load(f"{stem}_e1_arm_results.csv"))
    open(f"{stem}_e1_arm_summary.txt", "w").write(text + "\n")
    for r in rel:
        for k in ER.FIELDS:
            r.setdefault(k, float("nan") if k != "error" else "")
    with open(f"{stem}_release_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ER.FIELDS); w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rel)
    rtext = ER.summarize(rel)
    open(f"{stem}_release_error_summary.txt", "w").write(rtext + "\n")
    print(text); print(); print(rtext)

def _e5_init(*args):
    limit_aware().__enter__()              # for the life of the worker: the replay poses the arm along its planned path
    import e0_5_observation as E5
    E5._init(*args)

def _e5_replicate(task):
    import e0_5_observation as E5
    return E5.replicate(task)

def camera_check(base, a_brake, stem, workers, n=20):
    """E0-5 (e0_5_observation.py's main, the full-coverage camera and the height gate) with the limit-aware postures in view."""
    import e0_5_observation as E5
    arm = {"base": tuple(base), "a_brake": a_brake}
    with limit_aware():
        cases = E5.push_cases(E5.CASES, arm)
    todo = [(ci, fps, rep) for ci in range(len(cases)) for fps in E5.RATES for rep in range(n)]
    print(f"{len(todo)} replayed slides on {workers} workers", flush=True)
    with Pool(workers, initializer=_e5_init, initargs=(cases, arm, "full", E5.FACE_MARGIN, E5.FACE_GATE)) as pool:
        reps = list(pool.imap_unordered(_e5_replicate, todo))
    reps.sort(key=lambda r: (r["case"], r["fps"], r["rep"]))
    results, noise_rows, _ = E5.analyse(cases, reps)
    fr = E5.frames_table(cases, reps)
    E5.write_csv(f"{stem}_e0_5_results.csv", results); E5.write_csv(f"{stem}_e0_5_frames.csv", fr); E5.write_csv(f"{stem}_e0_5_noise.csv", noise_rows)
    E5.summarize(results, noise_rows, f"{stem}_e0_5_summary.txt")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--camera", action="store_true", help="E0-5's camera replay with the limit-aware postures in view (~10 min)")
    ap.add_argument("--posture", choices=["limit", "speed"], default="limit", help="--scan: limit-aware posture choice, or arm.py's own (lowest joint speed)")
    ap.add_argument("--xs", type=float, nargs="+", default=[0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70])
    ap.add_argument("--ys", type=float, nargs="+", default=[0.65])
    ap.add_argument("--caps", type=float, nargs="+", default=[5.5, 6.0])
    ap.add_argument("--base-x", type=float, default=0.6); ap.add_argument("--base-y", type=float, default=0.65)
    ap.add_argument("--a-brake", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", default=None, help="--scan: the text file; --validate: the path prefix of the output files")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.scan:
        text = scan(a.xs, a.ys, a.caps, a.posture == "limit", a.workers)
        out = a.out or str(OUT/f"scan_{'limit_aware' if a.posture == 'limit' else 'speed_posture'}.txt")
        open(out, "w").write(text + "\n"); print(text)
    if a.validate:
        stem = a.out or str(OUT/f"base{round(a.base_x*100):03d}_brake{a.a_brake:g}_limit_aware")
        validate((a.base_x, a.base_y, 0.0), a.a_brake, stem, a.workers)
    if a.camera:
        stem = a.out or str(OUT/f"base{round(a.base_x*100):03d}_brake{a.a_brake:g}_limit_aware")
        camera_check((a.base_x, a.base_y, 0.0), a.a_brake, stem, a.workers)

if __name__ == "__main__":
    main()

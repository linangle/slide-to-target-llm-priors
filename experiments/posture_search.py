#!/usr/bin/env python3
"""
posture_search.py -- is there ONE start posture that keeps every push of the comparison inside the iiwa's joint speed,
torque and acceleration limits?  (Follow-up to decision D9, 21 Nov.)

With posture="limits" the arm picks, push by push, the IK posture whose worst limit ratio is lowest, and ends up using
three different start postures over the speed range.  This script looks for a single start posture that is inside all
three limits at every speed from --u-min to --u-max, with the brake cap of --a-brake.  Planning only (no contact), and
arm.py is not edited: the paths and ratios come from arm.ArmModel.path and arm.ArmPusher._trajectory.

A start posture fixes the whole push: the path follows the minimum-norm joint tangent from it (arm.ArmModel.path), so a
posture's path is the same at every speed and only the stroke and the timing change.  The start pose leaves the arm one
free direction (seven joints, a six-dimensional pose), so the start postures form curves in joint space.  The search
covers them in two ways:
  ik      the arm's own multi-start IK with many more random seeds (arm.py uses 40 with seed 0)
  sweep   from every IK solution, steps along the free direction (the Jacobian's null vector) projected back onto the
          pose, both ways, until a joint limit: the whole curve through that solution
Each posture is scored by its worst ratio to a limit over a grid of speeds; the best are rescored on a fine grid.

    python3 experiments/posture_search.py                       # ~10 min on 8 workers -> results/side_base/posture_search/
    python3 experiments/posture_search.py --a-brake 5           # another brake cap
    python3 experiments/posture_search.py --validate            # Drake: E1 and the release-error grid with the best single posture
"""
import argparse, contextlib, json, math
from multiprocessing import Pool
import numpy as np
from paths import RESULTS

OUT = RESULTS/"side_base"/"posture_search"
KEYS = ("qd_ratio", "tau_ratio", "qdd_ratio")
NAMES = dict(qd_ratio="speed", tau_ratio="torque", qdd_ratio="acceleration")

def setup(base, a_brake, x0=0.55):
    """An ArmPusher posed as arm.feasibility poses it (no scene)."""
    import arm
    pu = arm.ArmPusher(base=tuple(base), a_brake_max=a_brake)
    pu.x_touch0 = x0 - arm.OBJECT_SIZE[0]/2 - pu.gap
    pu.model = arm.ArmModel(base)
    pu.X0 = arm.link7_pose(pu.x_touch0, 0.0, 0.0, pu.clearance)
    return pu

def profiles(speeds, J, a_brake):
    import arm
    return {u: arm.profile(u, J, arm.A_PUSH, a_max=a_brake) for u in speeds}

def score(pu, q0, profs):
    """One start posture over all the speeds: {u: (speed, torque, acceleration ratios) or None where its path stops short}."""
    length = max(p["x_stop"] for p in profs.values()) + 0.002
    path = pu.model.path(q0, pu.X0, length)
    out = {}
    for u, prof in profs.items():
        if prof["x_stop"] + 0.002 > path["s_ok"] + 1e-9:
            out[u] = None
            continue
        tr = pu._trajectory(path, prof)
        out[u] = tuple(float(tr[k]) for k in KEYS) + tuple(int(tr[k + "_joint"]) for k in KEYS)
    return out, path

def worst(sc):
    """(worst ratio over the speeds, the speed, which limit, which joint); inf if any speed is not covered."""
    best = (0.0, None, None, None)
    for u, v in sc.items():
        if v is None:
            return (math.inf, u, "reach", 0)
        i = int(np.argmax(v[:3]))
        if v[i] > best[0]:
            best = (v[i], u, NAMES[KEYS[i]], v[3 + i])
    return best

def sweep(model, X0, q0, step=0.03, n_max=400):
    """The curve of start postures through q0: steps along the Jacobian's null vector, projected back onto the pose."""
    out = []
    for sign in (1.0, -1.0):
        q, prev = np.array(q0, float), None
        for _ in range(n_max):
            Jm = model.jacobian(q)
            n = np.linalg.svd(Jm)[2][-1]
            if prev is not None and n @ prev < 0:
                n = -n
            elif prev is None:
                n = sign*n
            prev = n
            q, err = model.project(q + step*n, X0, iters=6)
            if err > 1e-8 or np.any(q <= model.q_lo) or np.any(q >= model.q_hi):
                break
            if any(np.max(np.abs(q - o)) < step/3 for o in out[-3:]):
                break
            out.append(q.copy())
    return out

def _candidates(job):
    """IK solutions from one seed batch, and the curves through them."""
    base, a_brake, seed, n_random = job
    pu = setup(base, a_brake)
    cands = pu.model.candidates(pu.X0, n_random=n_random, seed=seed)
    curve = [q for c in cands for q in sweep(pu.model, pu.X0, c)]
    return [("ik", q.tolist()) for q in cands] + [("sweep", q.tolist()) for q in curve]

def _score(job):
    base, a_brake, J, speeds, kind, q = job
    pu = setup(base, a_brake)
    sc, path = score(pu, np.array(q), profiles(speeds, J, a_brake))
    return kind, q, {f"{u:g}": v for u, v in sc.items()}, bool(path["ok"]), float(path["s_ok"])

def distinct(items, tol=0.02):
    out = []
    for kind, q in items:
        if all(np.max(np.abs(np.array(q) - np.array(o[1]))) > tol for o in out):
            out.append((kind, q))
    return out

def run(a):
    import arm
    base = arm.layout_base(a.layout)
    coarse = [round(a.u_min + i*(a.u_max - a.u_min)/a.n_coarse, 4) for i in range(a.n_coarse + 1)]
    fine = sorted({round(u, 3) for u in np.arange(a.u_min, a.u_max + 1e-9, 0.01)} | {a.u_max})
    with Pool(a.procs) as pool:
        found = [c for part in pool.map(_candidates, [(base, a.a_brake, s, a.n_random) for s in range(a.seeds)], chunksize=1) for c in part]
        cands = distinct(sorted(found, key=lambda c: c[0]))          # the IK solutions first, then the curve points
        scored = pool.map(_score, [(base, a.a_brake, a.J, coarse, k, q) for k, q in cands], chunksize=4)
        ranked = sorted(scored, key=lambda r: worst({float(u): v for u, v in r[2].items()})[0])
        top = pool.map(_score, [(base, a.a_brake, a.J, fine, r[0], r[1]) for r in ranked[:a.top]], chunksize=1)
    top = sorted(top, key=lambda r: worst({float(u): v for u, v in r[2].items()})[0])
    # the arm's own choice, push by push, for reference
    own = {}
    for posture in ("limits", "speed"):
        rows = [arm.feasibility(u, a.J, base, 0.55, a.a_brake, posture=posture) for u in coarse]
        own[posture] = [(u, max(r[k] for k in KEYS) if r["ok"] else math.inf) for u, r in zip(coarse, rows)]
    covered = sum(1 for r in scored if math.isfinite(worst({float(u): v for u, v in r[2].items()})[0]))
    L = [f"One start posture for every push?  base {tuple(round(b, 3) for b in base)}, brake cap {a.a_brake:g} m/s^2 (releases friction below {a.a_brake/arm.G:.2f}), "
         f"J {a.J:g} m/s^3, speeds {a.u_min:g}-{a.u_max:g} m/s",
         f"{len(cands)} distinct start postures ({sum(1 for k, _ in cands if k == 'ik')} from IK over {a.seeds} x {a.n_random} random seeds, "
         f"{sum(1 for k, _ in cands if k == 'sweep')} more along the curves through them); {covered} cover the longest push",
         "", f"the best {len(top)}, rescored at every 0.01 m/s: worst ratio to a limit over all the speeds (<= 1 is inside)",
         f"{'rank':>4} {'worst':>6} {'at m/s':>7} {'limit':>13} {'joint':>5}   {'speed':>6} {'torque':>6} {'accel':>6}   {'found by':>8}   start posture, rad"]
    for i, r in enumerate(top, 1):
        sc = {float(u): v for u, v in r[2].items()}
        w = worst(sc)
        col = [max(v[j] for v in sc.values() if v) for j in range(3)] if math.isfinite(w[0]) else [math.nan]*3
        L.append(f"{i:>4} {w[0]:>6.3f} {w[1]:>7g} {w[2]:>13} {w[3]:>5}   {col[0]:>6.3f} {col[1]:>6.3f} {col[2]:>6.3f}   {r[0]:>8}   {np.round(r[1], 3).tolist()}")
    best = top[0]
    bs = {float(u): v for u, v in best[2].items()}
    bw = worst(bs)
    L += ["", f"best single posture: worst ratio {bw[0]:.3f} ({bw[2]}, joint {bw[3]}, at {bw[1]:g} m/s): "
          + ("inside all three limits at every speed" if bw[0] <= 1 else "NOT inside the limits at every speed")]
    inside = [u for u, v in sorted(bs.items()) if v and max(v[:3]) <= 1]
    if bw[0] > 1 and inside:
        L.append(f"  it is inside at {len(inside)} of {len(bs)} speeds; over at " + ", ".join(f"{u:g}" for u, v in sorted(bs.items()) if not v or max(v[:3]) > 1))
    L += ["", "by speed: the best single posture's three ratios, and the arm's own choice push by push (worst ratio)",
          f"{'m/s':>6} {'speed':>6} {'torque':>6} {'accel':>6} | {'posture=limits':>14} {'posture=speed':>14}"]
    for (u, wl), (_, ws) in zip(own["limits"], own["speed"]):
        key = min(bs, key=lambda x: abs(x - u))
        v = bs[key]
        L.append(f"{u:>6g} " + (" ".join(f"{x:>6.3f}" for x in v[:3]) if v else f"{'not covered':>20}") + f" | {wl:>14.3f} {ws:>14.3f}")
    text = "\n".join(L) + "\n"
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"brake{a.a_brake:g}" + ("" if a.layout == "side" else f"_{a.layout}") + ("" if a.J == 100 else f"_J{a.J:g}")
    (a.out/f"{stem}.txt").write_text(text)
    (a.out/f"{stem}.json").write_text(json.dumps(dict(
        base=list(base), a_brake=a.a_brake, J=a.J, u_min=a.u_min, u_max=a.u_max, n_candidates=len(cands), covered=covered,
        best=dict(q_start=best[1], worst=bw[0], at=bw[1], limit=bw[2], joint=bw[3], by_speed=best[2]),
        top=[dict(q_start=r[1], found_by=r[0], worst=worst({float(u): v for u, v in r[2].items()})[0]) for r in top],
        all_worst=sorted(worst({float(u): v for u, v in r[2].items()})[0] for r in scored if r[3])), indent=1) + "\n")
    print(text, end="")

# ---------------------------------------------------------------------- one fixed posture through arm.py's own planner

@contextlib.contextmanager
def fixed_posture(q):
    """While open, every ArmPusher plans from start posture q alone (its IK candidates are replaced by q)."""
    import arm
    original = arm.ArmModel.candidates
    arm.ArmModel.candidates = lambda self, *args, **kw: [np.array(q, float)]
    try:
        yield
    finally:
        arm.ArmModel.candidates = original

def _e1_fixed(job):
    import e1_arm_validation as E1
    q, job = job
    with fixed_posture(q):
        return E1.run(job)

def _rel_fixed(job):
    import e1_release_error as ER
    q, job = job
    with fixed_posture(q):
        return ER.run(job)

def validate(a):
    """The two Drake checks of the arm (e1_arm_validation.py, e1_release_error.py) with the best single posture of a saved search."""
    import csv
    import arm, e1_arm_validation as E1, e1_release_error as ER
    base = arm.layout_base(a.layout)
    stem = f"brake{a.a_brake:g}" + ("" if a.layout == "side" else f"_{a.layout}") + ("" if a.J == 100 else f"_J{a.J:g}")
    q = json.loads((a.out/f"{stem}.json").read_text())["best"]["q_start"]
    js = E1.jobs(False, base, a.a_brake)
    print(f"{len(js)} arm-validation pushes and {len(ER.MUS)*len(ER.SPEEDS)} release-error pushes from start posture {np.round(q, 3).tolist()} ...", flush=True)
    with Pool(a.procs) as pool:
        rows = pool.map(_e1_fixed, [(q, j) for j in js], chunksize=1)
        rel = pool.map(_rel_fixed, [(q, (mu, v, base, a.a_brake)) for mu in ER.MUS for v in ER.SPEEDS], chunksize=1)
    rows.sort(key=lambda r: (r["part"], r["case"], r["pusher"]))
    with open(a.out/f"{stem}_single_e1_arm_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, E1.FIELDS); w.writeheader(); w.writerows(rows)
    text = E1.summarize(E1.load(str(a.out/f"{stem}_single_e1_arm_results.csv")))
    (a.out/f"{stem}_single_e1_arm_summary.txt").write_text(text + "\n")
    for r in rel:
        for k in ER.FIELDS:
            r.setdefault(k, float("nan") if k != "error" else "")
    with open(a.out/f"{stem}_single_release_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ER.FIELDS); w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rel)
    rtext = ER.summarize(rel)
    (a.out/f"{stem}_single_release_error_summary.txt").write_text(rtext + "\n")
    print(text); print(); print(rtext)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layout", default="side")
    ap.add_argument("--a-brake", type=float, default=6.0)
    ap.add_argument("--J", type=float, default=100.0)
    ap.add_argument("--u-min", type=float, default=0.15)
    ap.add_argument("--u-max", type=float, default=1.158)
    ap.add_argument("--n-coarse", type=int, default=24, help="speeds in the first scoring pass")
    ap.add_argument("--seeds", type=int, default=16, help="batches of random IK seeds")
    ap.add_argument("--n-random", type=int, default=60, help="random IK seeds per batch")
    ap.add_argument("--top", type=int, default=12, help="postures rescored on the fine grid")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=OUT, type=lambda s: __import__("pathlib").Path(s))
    ap.add_argument("--validate", action="store_true", help="Drake: E1 and the release-error grid with the saved search's best single posture")
    a = ap.parse_args()
    if a.validate:
        return validate(a)
    run(a)

if __name__ == "__main__":
    main()

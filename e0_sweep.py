#!/usr/bin/env python3
"""
e0_sweep.py -- E0: does the simulated slide obey d = v0^2 / (2 mu g)?

Passive slides on a uniform table (scene.passive_slide): the object settles, is given speed v0
along +x, and slides to rest.  Each slide passes if it stops within 2% of Eq. (1) and turns less
than 2 deg (proposal Sec. III-A).  A contact setting passes for a shape if every (mu, v0) passes.

Stages:
    grid        shapes x approximation x time step x hydroelastic modulus x damping
                (Hunt-Crossley dissipation for lagged/similar; relaxation time for SAP, which ignores
                Hunt-Crossley)
    resolution  mesh resolution hint, at the default contact settings
    robust      footed objects with one long foot or 3 feet, and centre-of-mass offsets, at the
                default contact settings: the deviation a material prior cannot know

Writes one row per slide to --out and prints a summary per contact setting.

Usage:
    python3 e0_sweep.py                    # all stages, all cores
    python3 e0_sweep.py --quick            # a few slides, to check it runs
    python3 e0_sweep.py --stages robust
    python3 e0_sweep.py --summarize e0_results.csv   # re-print the summary from a saved run
"""
import argparse, csv, math, os, time
from collections import defaultdict
from multiprocessing import Pool
from scene import G, Contact, Obj, Table, passive_slide

MUS, V0S = (0.15, 0.30, 0.50), (0.6, 1.0)   # proposal Sec. III-A first check
SHAPES = ("box", "disc", "ring", "footed")
APPROX = ("lagged", "similar", "sap")
DTS = (1e-3, 2e-3, 5e-3)                    # s
MODULI = (1e5, 1e6, 1e7)                    # Pa, proposal's E0 range
DISSIPATION = (0.0, 1.0, 10.0, 100.0)       # s/m, Hunt-Crossley, proposal's 0-100 range
RELAXATION = (0.0, 0.01, 0.1, 1.0)          # s, SAP relaxation time
RESOLUTIONS = (0.02, 0.01, 0.005)           # m
ROBUST = ([("footed", dict(foot_error=e)) for e in (0.0005, 0.001, 0.002)]
          + [("footed", dict(n_feet=3)), ("footed", dict(n_feet=3, foot_error=0.001))]
          + [("box", dict(com_offset=c)) for c in ((0.01, 0, 0), (0.02, 0, 0), (0, 0.01, 0), (0, 0.02, 0), (0.02, 0.02, 0))])

def damping(approx):
    return [("relaxation", r) for r in RELAXATION] if approx == "sap" else [("dissipation", d) for d in DISSIPATION]

def tasks(stages, quick):
    mus, v0s = ((0.30,), (0.8,)) if quick else (MUS, V0S)
    out = []
    def add(stage, shape, objkw, contactkw):
        out.extend((stage, shape, objkw, contactkw, mu, v0) for mu in mus for v0 in v0s)
    if "grid" in stages:
        for shape in (("box", "ring") if quick else SHAPES):
            for approx in APPROX:
                for dt in ((1e-3,) if quick else DTS):
                    for E in ((1e7,) if quick else MODULI):
                        for key, val in (damping(approx)[1:2] if quick else damping(approx)):
                            add("grid", shape, {}, {"approx": approx, "time_step": dt, "modulus": E, key: val})
    if "resolution" in stages:
        for shape in (("box",) if quick else SHAPES):
            for res in RESOLUTIONS:
                add("resolution", shape, {}, {"resolution": res})
    if "robust" in stages:
        for shape, objkw in (ROBUST[:1] if quick else ROBUST):
            add("robust", shape, objkw, {})
    return out

def run(task):
    stage, shape, objkw, contactkw, mu, v0 = task
    obj, contact = Obj(shape=shape, **objkw), Contact(**contactkw)
    t0, msg = time.time(), ""
    try:
        d, yaw, _ = passive_slide({"mu_k_nominal": mu, "s_tab_log": 0.0}, v0, obj, Table(patchiness=0.0), contact)
    except Exception as e:   # record and carry on: a failing setting is a result too
        d = yaw = float("nan"); msg = str(e).splitlines()[0][:200]
    d_ideal = v0**2/(2*mu*G)
    err = d/d_ideal - 1
    return {"stage": stage, "shape": shape, "n_feet": obj.n_feet if shape == "footed" else "",
            "foot_error_mm": obj.foot_error*1e3 if shape == "footed" else "",
            "com_offset_cm": " ".join(f"{c*100:g}" for c in obj.com_offset), "approx": contact.approx,
            "dt_ms": contact.time_step*1e3, "modulus_Pa": f"{contact.modulus:g}",
            "dissipation": contact.dissipation if contact.approx != "sap" else "",
            "relaxation_s": "" if contact.approx != "sap" else ("default" if contact.relaxation is None else contact.relaxation),
            "resolution_m": contact.resolution, "mu": mu, "v0": v0, "d_cm": f"{d*100:.3f}",
            "d_ideal_cm": f"{d_ideal*100:.3f}", "err_pct": f"{err*100:.2f}", "yaw_deg": f"{yaw:.2f}",
            "pass": int(abs(err) < 0.02 and abs(yaw) < 2.0), "wall_s": f"{time.time() - t0:.2f}", "error": msg}

SETTING = ("stage", "approx", "dt_ms", "modulus_Pa", "dissipation", "relaxation_s", "resolution_m",
           "n_feet", "foot_error_mm", "com_offset_cm")
OBJECT_FIELDS = ("n_feet", "foot_error_mm", "com_offset_cm")

def summarize(rows):
    """One line per contact setting (or robustness case): worst error and yaw for each shape."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:   # object fields separate settings only in the robust stage
        groups[tuple("" if r["stage"] != "robust" and k in OBJECT_FIELDS else r[k] for k in SETTING)][r["shape"]].append(r)
    for stage in ("grid", "resolution", "robust"):
        keys = sorted((k for k in groups if k[0] == stage), key=lambda k: tuple(str(v) for v in k))
        if not keys:
            continue
        print(f"\n== {stage} ==  worst |error| and |yaw| over (mu, v0) per shape; * = all slides pass")
        n_all = 0
        for k in keys:
            label = " ".join(f"{n}={v}" for n, v in zip(SETTING[1:], k[1:]) if v not in ("", "0 0 0"))
            cells, all_ok = [], True
            for shape, rs in groups[k].items():
                errs = [abs(float(r["err_pct"])) for r in rs]
                yaws = [abs(float(r["yaw_deg"])) for r in rs]
                ok = all(int(r["pass"]) for r in rs)
                all_ok &= ok
                cells.append(f"{shape} {max(errs):5.1f}%/{max(yaws):4.1f}deg{'*' if ok else ' '}")
            n_all += all_ok
            print(f"{'PASS' if all_ok else '    '} {label:<70} " + "  ".join(cells))
        print(f"{n_all} of {len(keys)} {stage} settings pass for every shape")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="grid,resolution,robust")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="e0_results.csv")
    ap.add_argument("--summarize", metavar="CSV", help="summarize a saved run instead of running")
    a = ap.parse_args()
    if a.summarize:
        with open(a.summarize, newline="") as f:
            summarize(list(csv.DictReader(f)))
        return
    todo = tasks(a.stages.split(","), a.quick)
    print(f"{len(todo)} slides on {a.workers} workers")
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, todo, chunksize=4), 1):
            rows.append(r)
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: tuple(str(r[k]) for k in SETTING + ("shape", "mu", "v0")))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")
    summarize(rows)

if __name__ == "__main__":
    main()

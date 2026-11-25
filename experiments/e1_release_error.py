#!/usr/bin/env python3
"""
e1_release_error.py -- the arm's execution error over the speed and friction range: the Tier B execution model.

The decision rule plans with the release model: release speed v_r = u - (mu g)^2/(2J), release point
x_r = x0 - gap + release_push_distance(u, mu, J), rest point x_r + v_r^2/(2 mu g).  A harness that does not simulate the
arm needs to know how far the arm's actual push is from that: the actual release speed as the model's times (1 + eps_v),
the release point off by eps_x, and the landing off by the two together.  This runs the arm (default layout) over a grid
of friction and commanded peak speed and writes those errors to a small CSV, as e0_5_observation.py does for the camera
noise (e0_5_noise.csv).

The simulator has no actuator noise, so these are biases that repeat exactly, not a spread: use them as the systematic
part and add the stated sensor/actuation spread on top (5 Nov decision on Tier B noise).

    python3 experiments/e1_release_error.py              # 35 pushes, ~1 min -> results/side_base/e1_release_error.csv, _summary.txt
    python3 experiments/e1_release_error.py --layout origin --out results/e1/e1_release_error
    python3 experiments/e1_release_error.py --a-brake 4  # the soft brake (only friction below about 0.4 releases)
    python3 experiments/e1_release_error.py --base-x 0.68 --base-y 0.65 --a-brake 6 --out results/side_base/base_trial/release_error_base068_brake6   # another base (18 Nov trial)
"""
import argparse, csv, math, os
from multiprocessing import Pool
import numpy as np
from paths import RESULTS

MUS = (0.1, 0.14, 0.2, 0.26, 0.3, 0.4, 0.5)
SPEEDS = (0.3, 0.45, 0.6, 0.8, 1.0)
J, X0 = 100.0, 0.55
FIELDS = ["mu", "v_peak", "J", "base_x", "base_y", "a_brake", "posture", "v_r_model", "v_sep", "eps_v_pct", "x_r_model", "x_sep", "eps_x_release_mm",
          "x_rest_model", "x_rest", "landing_error_mm", "slide_err_pct", "fk_release_bias_mm", "lateral_miss_mm", "heading_deg",
          "track_rms_mm", "v_lim_frac", "tau_lim_frac", "a_lim_frac", "separated", "error"]

def run(job):
    from arm import ArmPusher, rest_x
    from physics import release_push_distance, release_speed
    from push import push_slide
    mu, v, base, a_brake, posture = (*job, None)[:5]
    pusher = ArmPusher(base=tuple(base), **({} if a_brake is None else {"a_brake_max": float(a_brake)}), **({} if posture is None else {"posture": posture}))
    row = {"mu": mu, "v_peak": v, "J": J, "base_x": base[0], "base_y": base[1], "a_brake": pusher.a_brake_max, "posture": pusher.posture, "error": ""}
    try:
        r = push_slide(mu, v, J, pusher=pusher, x0=X0, keep_traj=False)
    except Exception as ex:
        row["error"] = str(ex).splitlines()[0][:160]
        return row
    v_r, x_r = float(release_speed(v, mu, J)), X0 - pusher.gap + release_push_distance(v, mu, J)
    row.update(v_r_model=v_r, v_sep=r["v_sep"], eps_v_pct=r["v_r_cmd_err_pct"], x_r_model=x_r, x_sep=r["x_sep"], eps_x_release_mm=(r["x_sep"] - x_r)*1e3,
               x_rest_model=rest_x(v, mu, J, X0, pusher.gap), x_rest=r["x_rest"], landing_error_mm=(r["x_rest"] - rest_x(v, mu, J, X0, pusher.gap))*1e3,
               slide_err_pct=r["d_err_pct"], fk_release_bias_mm=(r["x_fk"] - r["x_sep"])*1e3, lateral_miss_mm=r["y_rest"]*1e3, heading_deg=r["psi_r_deg"],
               track_rms_mm=r["track_q_rms"]*1e3, v_lim_frac=r["arm_plan_qd_ratio"], tau_lim_frac=r["arm_plan_tau_ratio"], a_lim_frac=r["arm_plan_qdd_ratio"],
               separated=int(r["separated"]))
    return row

def summarize(rows):
    ok = [r for r in rows if not r["error"] and r["separated"] and r["v_r_model"] > 0.05]
    L = [f"arm execution error against the release model, base ({rows[0]['base_x']:g}, {rows[0]['base_y']:g}) m, J {J:g} m/s^3, brake cap {rows[0]['a_brake']:g} m/s^2"
         + (f", start posture chosen by {rows[0]['posture']}" if rows[0].get("posture") not in (None, "", "speed") else ""),
         "eps_v: release speed, % of the model's.  Rows mu, columns commanded peak speed (m/s).  * = over a joint speed or torque limit",
         f"{'mu':>5} | " + " ".join(f"{v:>8g}" for v in SPEEDS)]
    def cell(mu, v, key, fmt):
        r = next((r for r in rows if r["mu"] == mu and r["v_peak"] == v), None)
        if r is None or r["error"] or not r["separated"] or r["v_r_model"] <= 0.05:
            return f"{'--':>8}"
        return f"{format(r[key], fmt):>7}" + ("*" if max(r["v_lim_frac"], r["tau_lim_frac"]) > 1 else " ")
    for mu in MUS:
        L.append(f"{mu:>5} | " + " ".join(cell(mu, v, "eps_v_pct", "+.2f") for v in SPEEDS))
    L += ["", "landing error: where the object stopped minus the model's rest point, mm", f"{'mu':>5} | " + " ".join(f"{v:>8g}" for v in SPEEDS)]
    for mu in MUS:
        L.append(f"{mu:>5} | " + " ".join(cell(mu, v, "landing_error_mm", "+.2f") for v in SPEEDS))
    inlim = [r for r in ok if max(r["v_lim_frac"], r["tau_lim_frac"]) <= 1]
    for label, rs in (("all released pushes", ok), ("within joint speed and torque limits", inlim)):
        if rs:
            e, x, l = (np.array([r[k] for r in rs]) for k in ("eps_v_pct", "eps_x_release_mm", "landing_error_mm"))
            L.append(f"{label} ({len(rs)}): eps_v mean {e.mean():+.2f}%, SD {e.std(ddof=1):.2f}%, range {e.min():+.2f}..{e.max():+.2f}%; "
                     f"release point {x.mean():+.2f} mm (range {x.min():+.2f}..{x.max():+.2f}); landing {l.mean():+.2f} mm, RMS {np.sqrt(np.mean(l**2)):.2f}, "
                     f"worst {np.abs(l).max():.2f} mm")
    skipped = [r for r in rows if r not in ok]
    if skipped:
        L.append(f"left out ({len(skipped)}): " + ", ".join(f"mu {r['mu']} v {r['v_peak']} (" + (r["error"][:40] if r["error"] else "no release: the brake never out-brakes friction"
                                                                                               if not r.get("separated") else "release speed under 5 cm/s") + ")" for r in skipped))
    L.append("for a harness: v_r_actual = v_r_model * (1 + eps_v/100); these are deterministic biases, not noise.")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layout", choices=["origin", "side"], default="side")
    ap.add_argument("--a-brake", type=float, default=None)
    ap.add_argument("--posture", choices=["speed", "limits"], default=None, help="how the arm chooses its start posture (arm.POSTURES; default: speed)")
    ap.add_argument("--base-x", type=float, default=None, help="iiwa base position along the push, m (with --base-y: overrides --layout; needs --out)")
    ap.add_argument("--base-y", type=float, default=None, help="iiwa base offset from the push line along y, m")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default=None, help="path prefix (default results/side_base/e1_release_error, or results/e1/... for the origin)")
    a = ap.parse_args()
    from arm import layout_base
    custom = a.base_x is not None or a.base_y is not None
    if custom and a.out is None:
        ap.error("--base-x/--base-y need an explicit --out")
    base = layout_base((a.base_x or 0.0, a.base_y or 0.0, 0.0) if custom else a.layout)
    out = a.out or str((RESULTS/"side_base" if a.layout == "side" else RESULTS/"e1")/"e1_release_error") + ("" if a.a_brake is None else f"_brake{a.a_brake:g}") + ("" if a.posture is None else f"_posture_{a.posture}")
    jobs = [(mu, v, base, a.a_brake, a.posture) for mu in MUS for v in SPEEDS]
    with Pool(a.workers) as pool:
        rows = pool.map(run, jobs)
    for r in rows:
        for k in FIELDS:
            r.setdefault(k, float("nan") if k != "error" else "")
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    text = summarize(rows)
    print(text)
    open(out + "_summary.txt", "w").write(text + "\n")
    print(f"wrote {out}.csv, {out}_summary.txt")

if __name__ == "__main__":
    main()

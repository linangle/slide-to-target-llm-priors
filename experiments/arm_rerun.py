#!/usr/bin/env python3
"""
arm_rerun.py -- the 5 Nov "arm re-run": 12 pushes with the iiwa (arm.ArmPusher) in place of the prescribed plate (push.Pusher),
each beside the plate's result for the same case.

The cases are picked from the E0-3 and E0-4 grids to span mu, v_peak, J and the box/disc, COM, plate-yaw and paddle-friction
knobs; the grids themselves are not rerun (5 Nov decision).  Both pushers run the same push.profile command; the plate brakes
uncapped as in E0-3/E0-4, the arm with its 12 m/s^2 brake cap (the release is the same: a_max > mu g).

Per case and pusher: the object's speed at t* against the release model (E0-4's bar: within 1%), the slide against
v_sep^2/(2 mu g), release heading, lateral miss, FK release-point bias and paddle tracking.  Arm vs plate: the difference of
v_sep and of the rest position.  The arm rows also carry what the push asked of the iiwa: peak joint speed and torque against
the SDF limits (speed is not enforced by the simulator, torque is).

Writes arm_rerun_results.csv, arm_rerun_summary.txt, arm_rerun_summary.png.
    python3 experiments/arm_rerun.py                     # 24 pushes, ~1 min on 8 workers
    python3 experiments/arm_rerun.py --summarize results/arm_rerun/arm_rerun_results.csv
"""
import argparse, csv, math, os, time
from multiprocessing import Pool
import numpy as np
from scene import Obj
from drake_config import contact_from_config
from push import Pusher, push_slide, scalars
from arm import ArmPusher
from paths import RESULTS

W = Obj().size[1]
#        label,                 mu,  v,   J,     shape, com_y,  lateral, yaw, mu_p
CASES = (("E0-4 grid",          0.1, 0.8, 100.0, "box", 0.0,    0.0, 0.0, 0.3),
         ("E0-4 grid, acceptance", 0.3, 0.8, 100.0, "box", 0.0, 0.0, 0.0, 0.3),
         ("E0-4 grid",          0.5, 0.8, 100.0, "box", 0.0,    0.0, 0.0, 0.3),
         ("E0-4 grid",          0.3, 0.6, 50.0,  "box", 0.0,    0.0, 0.0, 0.3),
         ("E0-4 grid",          0.3, 1.0, 200.0, "box", 0.0,    0.0, 0.0, 0.3),
         ("E0-4 grid, worst",   0.5, 0.6, 30.0,  "box", 0.0,    0.0, 0.0, 0.3),
         ("E0-4 grid",          0.1, 1.0, 2000.0, "box", 0.0,   0.0, 0.0, 0.3),
         ("E0-4 grid",          0.3, 0.8, 500.0, "box", 0.0,    0.0, 0.0, 0.3),
         ("E0-3 disc",          0.3, 0.8, 100.0, "disc", 0.0,   0.0, 0.0, 0.3),
         ("E0-3 COM e = w/4",   0.3, 0.8, 100.0, "box", 0.25*W, 0.0, 0.0, 0.3),
         ("E0-3 plate yaw 2",   0.3, 0.8, 100.0, "box", 0.0,    0.0, 2.0, 0.3),
         ("E0-3 mu_p 0.6",      0.3, 0.8, 100.0, "box", 0.0,    0.0, 0.0, 0.6))
KEEP = ("v_sep", "v_r_cmd", "v_r_cmd_err_pct", "v_r_act_err_pct", "sep_lag_ms", "d_slide", "d_err_pct", "psi_r_deg", "omega_r",
        "lateral_miss", "yaw_rest_deg", "tilt_max_deg", "x_sep", "x_fk", "x_rest", "track_q_rms", "track_q_max", "track_v_rms",
        "f_max", "n_point_contacts", "separated", "steps", "wall_s", "a_max", "t_sep", "t_rest")
ARM = ("arm_tau_ratio", "arm_tau_joint", "arm_qd_ratio", "arm_qd_joint", "arm_plan_qd_ratio", "arm_plan_tau_ratio", "arm_m_eff",
       "arm_sigma_min", "arm_postures", "arm_z_drift_mm", "arm_y_drift_mm", "arm_tilt_max_deg")

def tasks():
    return [(i, who) for i in range(len(CASES)) for who in ("plate", "arm")]

def run(task):
    i, who = task
    label, mu, v, J, shape, com, lat, yaw, mu_p = CASES[i]
    row = {"case": i + 1, "label": label, "pusher": who, "mu": mu, "v_peak": v, "J": J, "shape": shape, "com_y_mm": com*1e3,
           "lateral_mm": lat*1e3, "yaw_cmd_deg": yaw, "mu_p": mu_p}
    pusher = Pusher(mu_p=mu_p, lateral=lat, yaw_deg=yaw) if who == "plate" else ArmPusher(mu_p=mu_p, lateral=lat, yaw_deg=yaw)
    try:
        r = scalars(push_slide(mu, v, J, contact_from_config(), Obj(shape=shape, com_offset=(0.0, com, 0.0)), pusher, keep_traj=False))
        row.update({k: r[k] for k in KEEP})
        row.update({k: r.get(k, "") for k in ARM})
        row["x_fk_bias_mm"] = (r["x_fk"] - r["x_sep"])*1e3
        row["ok"] = int(abs(r["v_r_cmd_err_pct"]) < 1.0 and r["track_q_rms"] < 1e-4 and r["n_point_contacts"] == 0)
        row["error"] = ""
    except Exception as ex:
        row.update(ok=0, error=str(ex).splitlines()[0][:200])
    return row

def f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return float("nan")

def summarize(rows, out=None):
    ok = [r for r in rows if not r.get("error")]
    pairs = []
    for i in range(len(CASES)):
        p = next((r for r in ok if int(r["case"]) == i + 1 and r["pusher"] == "plate"), None)
        a = next((r for r in ok if int(r["case"]) == i + 1 and r["pusher"] == "arm"), None)
        pairs.append((p, a))
    lines = [f"== arm re-run: 12 pushes with the iiwa beside the prescribed plate ({len(ok)} of {len(rows)} pushes ran; config contact) ==",
             "v_sep err: the object's speed at t* against v_peak - (mu g)^2/(2J) on the command, % (bar: 1%); d err: the free slide against v_sep^2/(2 mu g), %;",
             "track: paddle rms tracking error, mm; FK bias: FK release point minus the object's true position at t*, mm; arm - plate: v_sep and rest position",
             f"{'#':>2} {'case':<22}{'mu':>4}{'v':>4}{'J':>5} {'shape':<5}| {'v_sep err plate':>15} {'arm':>7} | {'d err plate':>11} {'arm':>7} | {'psi_r arm':>9} | "
             f"{'miss arm mm':>11} | {'track plate':>11} {'arm':>7} | {'FK bias arm':>11} | {'arm-plate v_sep %':>17} {'x_rest mm':>9}"]
    for i, (p, a) in enumerate(pairs):
        label, mu, v, J, shape, *_ = CASES[i]
        if p is None or a is None:
            lines.append(f"{i + 1:>2} {label:<22}{mu:>4}{v:>4}{J:>5g} {shape:<5}| missing: {'plate ' if p is None else ''}{'arm' if a is None else ''}")
            continue
        dv = (f(a, "v_sep")/f(p, "v_sep") - 1)*100
        lines.append(f"{i + 1:>2} {label:<22}{mu:>4}{v:>4}{J:>5g} {shape:<5}| {f(p, 'v_r_cmd_err_pct'):>+15.2f} {f(a, 'v_r_cmd_err_pct'):>+7.2f} | "
                     f"{f(p, 'd_err_pct'):>+11.2f} {f(a, 'd_err_pct'):>+7.2f} | {f(a, 'psi_r_deg'):>+9.3f} | {f(a, 'lateral_miss')*1e3:>+11.2f} | "
                     f"{f(p, 'track_q_rms')*1e3:>11.3f} {f(a, 'track_q_rms')*1e3:>7.3f} | {f(a, 'x_fk_bias_mm'):>+11.2f} | {dv:>+17.2f} {(f(a, 'x_rest') - f(p, 'x_rest'))*1e3:>+9.2f}")
    arms = [a for _, a in pairs if a is not None]
    plates = [p for p, _ in pairs if p is not None]
    if arms and plates:
        lines.append("")
        lines.append(f"arm: {sum(int(str(a['ok'])) for a in arms)} of {len(arms)} within E0-4's bar (v_sep within 1% of the model, tracking under 0.1 mm rms, no point contacts); "
                     f"largest |v_sep err| {max(abs(f(a, 'v_r_cmd_err_pct')) for a in arms):.2f}% (plate {max(abs(f(p, 'v_r_cmd_err_pct')) for p in plates):.2f}%); "
                     f"tracking rms {min(f(a, 'track_q_rms') for a in arms)*1e3:.3f}..{max(f(a, 'track_q_rms') for a in arms)*1e3:.3f} mm "
                     f"(plate {min(f(p, 'track_q_rms') for p in plates)*1e3:.3f}..{max(f(p, 'track_q_rms') for p in plates)*1e3:.3f})")
        both = [(p, a) for p, a in pairs if p is not None and a is not None]
        dv = [(f(a, "v_sep")/f(p, "v_sep") - 1)*100 for p, a in both]
        dx = [(f(a, "x_rest") - f(p, "x_rest"))*1e3 for p, a in both]
        lines.append(f"arm vs plate, same case: v_sep differs by {min(dv):+.2f}..{max(dv):+.2f}%, the rest position by {min(dx):+.1f}..{max(dx):+.1f} mm; "
                     f"FK release-point bias {min(f(a, 'x_fk_bias_mm') for a in arms):+.2f}..{max(f(a, 'x_fk_bias_mm') for a in arms):+.2f} mm; "
                     f"paddle z drift <= {max(f(a, 'arm_z_drift_mm') for a in arms):.2f} mm, tilt <= {max(f(a, 'arm_tilt_max_deg') for a in arms):.3f} deg while in contact")
        lines.append("")
        lines.append("what the pushes asked of the iiwa (base at the origin; speed limits are NOT enforced by the simulator, torque limits are):")
        lines.append(f"{'#':>2} {'mu':>4}{'v':>4}{'J':>5} | {'joint speed / limit':>19} {'(joint)':>7} | {'torque / limit':>14} {'(joint)':>7} | {'m_eff kg':>8} | {'sigma_min':>9} | {'postures':>8}")
        for i, (_, a) in enumerate(pairs):
            if a is None:
                continue
            _, mu, v, J, *_ = CASES[i]
            lines.append(f"{i + 1:>2} {mu:>4}{v:>4}{J:>5g} | {f(a, 'arm_qd_ratio'):>19.2f} {int(f(a, 'arm_qd_joint')):>7} | {f(a, 'arm_tau_ratio'):>14.2f} {int(f(a, 'arm_tau_joint')):>7} | "
                         f"{f(a, 'arm_m_eff'):>8.1f} | {f(a, 'arm_sigma_min'):>9.3f} | {int(f(a, 'arm_postures')):>8}")
        qd = [f(a, "arm_qd_ratio") for a in arms]
        lines.append(f"reading: every push runs joint 4 (75 deg/s) at {min(qd):.1f}..{max(qd):.1f} x its speed limit, so with the base on the push line the real iiwa delivers "
                     f"about {min(f(a, 'v_peak')/f(a, 'arm_qd_ratio') for a in arms):.2f}..{max(f(a, 'v_peak')/f(a, 'arm_qd_ratio') for a in arms):.2f} m/s, not the grid's 0.6-1.0; "
                     f"torque stays within the limits up to 0.8 m/s with the 12 m/s^2 brake cap (ratio <= {max(f(a, 'arm_tau_ratio') for a in arms if f(a, 'v_peak') <= 0.8):.2f}) "
                     f"and exceeds them at 1.0 m/s (clamped in the simulator: ratio {max(f(a, 'arm_tau_ratio') for a in arms if f(a, 'v_peak') > 0.8):.2f}).  See arm.py --scan for other base positions.")
    bad = [r for r in rows if r.get("error")]
    if bad:
        lines.append(f"{len(bad)} pushes failed: " + "; ".join(f"case {r['case']} {r['pusher']}: {r['error']}" for r in bad))
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w").write(text + "\n")

def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if not r.get("error")]
    n = len(CASES)
    x = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    col = {"plate": "#2a78d6", "arm": "#eb6834"}
    for who in ("plate", "arm"):
        rs = {int(r["case"]): r for r in ok if r["pusher"] == who}
        get = lambda k, s=1.0: np.array([f(rs[i], k)*s if i in rs else np.nan for i in x])
        axes[0].scatter(x - 0.1 + 0.2*(who == "arm"), get("v_r_cmd_err_pct"), s=30, color=col[who], label=who, zorder=3)
        axes[1].scatter(x - 0.1 + 0.2*(who == "arm"), get("track_q_rms", 1e3), s=30, color=col[who], label=who, zorder=3)
    arm = {int(r["case"]): r for r in ok if r["pusher"] == "arm"}
    axes[2].bar(x - 0.18, [f(arm[i], "arm_qd_ratio") if i in arm else np.nan for i in x], 0.36, color="#eb6834", label="joint speed / limit (not enforced)")
    axes[2].bar(x + 0.18, [f(arm[i], "arm_tau_ratio") if i in arm else np.nan for i in x], 0.36, color="#1baf7a", label="torque / effort limit (enforced)")
    axes[2].axhline(1.0, color="#52514e", linewidth=0.8, linestyle="--")
    for ax, label in zip(axes, ("object speed at t* vs release model, %", "paddle tracking rms, mm", "peak ratio to the iiwa's limits")):
        ax.set_ylabel(label); ax.set_xticks(x); ax.set_xlabel("case (arm_rerun_summary.txt)")
        ax.grid(color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].axhspan(-1, 1, color="#eeeeee", zorder=0); axes[0].axhline(0, color="#52514e", linewidth=0.6)
    axes[1].axhline(0.1, color="#52514e", linewidth=0.8, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8); axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("arm re-run: the iiwa (arm.py) beside the prescribed plate (push.py) on 12 pushes from the E0-3/E0-4 grids", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    ap.add_argument("--out", default=str(RESULTS/"arm_rerun"/"arm_rerun_results.csv"))
    ap.add_argument("--summary", default=str(RESULTS/"arm_rerun"/"arm_rerun_summary.txt"))
    ap.add_argument("--fig", default=str(RESULTS/"arm_rerun"/"arm_rerun_summary.png"))
    ap.add_argument("--summarize", metavar="CSV")
    a = ap.parse_args()
    if a.summarize:
        with open(a.summarize, newline="") as fh:
            rows = list(csv.DictReader(fh))
        summarize(rows, a.summary); plot(rows, a.fig); return
    todo = tasks()
    print(f"{len(todo)} pushes on {a.workers} workers")
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, todo), 1):
            rows.append(r)
            print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (int(r["case"]), r["pusher"]))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"wrote {a.out}")
    summarize(rows, a.summary)
    plot(rows, a.fig)
    print(f"wrote {a.summary}, {a.fig}")

if __name__ == "__main__":
    main()

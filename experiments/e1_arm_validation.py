#!/usr/bin/env python3
"""
e1_arm_validation.py -- the arm milestone (proposal, Nov 6): repeatable robot pushes, achievable release speeds and
targeting accuracy with known friction.  Three parts, all with the iiwa (arm.ArmPusher) under the frozen contact config:

  A  arm vs plate   12 pushes picked from the E0-3/E0-4 grids (mu, v_peak, J, box/disc; the 5 Nov "arm re-run" decision),
                    each run with the arm and with push.py's prescribed plate.  Bars (README, "The bar"): speed at t*
                    within 1% of v_peak - (mu g)^2/(2J), slide within 2% of v^2/(2 mu g), and the arm's landing within
                    2 mm of the plate's.  The arm's tracking error against the profile is the new number.
  B  targeting      target x in {0.65 ... 0.90} m x mu in {0.1, 0.3, 0.5}, planned with the TRUE mu (the reference
                    method): where the object stops vs the target.  Success = within TOL (1 cm).
  C  repeatability  3 target cases x N trials with small disturbances the planner does not know about: the object's
                    start (paddle gap 0.5-3 mm instead of 1 mm), a lateral offset of the paddle (+-3 mm) and the object's
                    mass (+-10%).  Spread of the landing (std, worst case).

Every arm row also reports the peak joint velocity / limit (v_lim_frac): above 1 the real iiwa could not make that push
(the simulator does not enforce joint velocity limits; see ARM_CONTROLLER.md).  The summary gives results for all rows
and for the rows within the limits.

Since the 16 Nov merge the pushes run on arm.py's merged controller (multi-start IK posture, brake capped at 12 m/s^2);
the planner, the cases and the bars are unchanged.  v_lim_frac and tau_lim_frac are the plan's peak joint-speed and
torque ratios (push_slide's arm_plan_qd_ratio, arm_plan_tau_ratio).  --base-x / --base-y move the iiwa's base, to repeat
the run once the team has decided where it stands (the default is the scene's origin).

    python3 experiments/e1_arm_validation.py                   # 85 pushes, ~1 min on 4 workers -> results/e1/e1_arm_results.csv, results/e1/e1_arm_summary.txt, results/e1/e1_arm_summary.png
    python3 experiments/e1_arm_validation.py --quick           # one case per part, to check it runs (~10 s)
    python3 experiments/e1_arm_validation.py --layout side --out results/side_base/e1_arm              # the same run with the base beside the board
    python3 experiments/e1_arm_validation.py --layout side --a-brake 4 --out results/side_base/e1_arm_brake4   # ... and a brake inside the acceleration limits
    python3 experiments/e1_arm_validation.py --summarize results/e1/e1_arm_results.csv
"""
import argparse, csv, math, os, time
from multiprocessing import Pool
import numpy as np
from paths import RESULTS

TOL = 0.01                     # m, target success window (decide.loss_miss's 1 cm floor)
TARGETS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
MUS = (0.1, 0.3, 0.5)
# part A: (shape, mu, v_peak, J) spanning the E0-3/E0-4 grids; v_peak <= the arm's reach at that J (arm.max_peak_speed)
CASES_A = [("box", 0.1, 0.6, 100.0), ("box", 0.1, 0.9, 30.0), ("box", 0.3, 0.6, 30.0), ("box", 0.3, 0.8, 100.0),
           ("box", 0.3, 1.0, 300.0), ("box", 0.5, 0.6, 300.0), ("box", 0.5, 0.8, 100.0), ("box", 0.5, 0.9, 30.0),
           ("disc", 0.1, 0.8, 100.0), ("disc", 0.3, 0.8, 30.0), ("disc", 0.3, 0.6, 300.0), ("disc", 0.5, 1.0, 100.0)]
CASES_C = [(0.3, 0.75), (0.1, 0.80), (0.5, 0.70), (0.3, 0.64), (0.1, 0.70)]   # (mu, target x); the last two within the joint limits
N_C = 8
FIELDS = ["part", "case", "pusher", "shape", "mu", "v_peak", "J", "x_target", "gap_mm", "lateral_mm", "mass",
          "v_r_model", "v_sep", "v_err_pct", "d_err_pct", "psi_deg", "yaw_rest_deg", "x_rest", "y_rest", "miss_mm",
          "track_rms_mm", "track_max_mm", "v_lim_frac", "tau_lim_frac", "joint_v_worst", "separated", "point_contacts", "wall_s",
          "a_lim_frac", "base_x", "base_y", "a_brake"]      # added 16 Nov: planned joint acceleration / SDF limit, and the arm setup

# ----------------------------------------------------------------------------- the jobs

def jobs(quick=False, base=(0.0, 0.0, 0.0), a_brake=None):
    out = []
    for i, (shape, mu, v, J) in enumerate(CASES_A[:1] if quick else CASES_A):
        for pusher in ("arm", "plate"):
            out.append(dict(part="A", case=f"A{i+1:02d}", pusher=pusher, shape=shape, mu=mu, v_peak=v, J=J))
    for mu in (MUS[1:2] if quick else MUS):
        for x in (TARGETS[2:3] if quick else TARGETS):
            out.append(dict(part="B", case=f"B_mu{mu}_x{x}", pusher="arm", shape="box", mu=mu, J=100.0, x_target=x))
    rng = np.random.default_rng(0)
    for k, (mu, x) in enumerate(CASES_C[:1] if quick else CASES_C):
        for n in range(2 if quick else N_C):
            out.append(dict(part="C", case=f"C{k+1}_{n}", pusher="arm", shape="box", mu=mu, J=100.0, x_target=x,
                            gap=float(rng.uniform(0.0005, 0.003)), lateral=float(rng.uniform(-0.003, 0.003)),
                            mass=float(0.5*rng.uniform(0.9, 1.1))))
    for j in out:
        j["base"], j["a_brake"] = tuple(base), a_brake
    return out

def run(job):
    from arm import ArmPusher, plan_peak_speed
    from push import Pusher, push_slide
    from scene import Obj
    from physics import release_speed
    t0 = time.time()
    mu, J = job["mu"], job["J"]
    gap, lateral, mass = job.get("gap", 0.001), job.get("lateral", 0.0), job.get("mass", 0.5)
    obj = Obj(shape=job["shape"], mass=mass)
    x_t = job.get("x_target", float("nan"))
    base = tuple(job.get("base", (0.0, 0.0, 0.0)))
    v = job["v_peak"] if "v_peak" in job else plan_peak_speed(x_t, mu, J, base=base)     # planned with the nominal 1 mm gap
    arm_options = {} if job.get("a_brake") is None else {"a_brake_max": float(job["a_brake"])}      # default: the arm's 12 m/s^2
    pusher = ArmPusher(gap=gap, lateral=lateral, base=base, **arm_options) if job["pusher"] == "arm" else Pusher(lateral=lateral)
    r = push_slide(mu, v, J, obj=obj, pusher=pusher, keep_traj=False)
    row = {k: job.get(k, "") for k in ("part", "case", "pusher", "shape", "mu", "J")}
    row.update(v_peak=v, x_target=x_t, gap_mm=gap*1e3, lateral_mm=lateral*1e3, mass=mass,
               v_r_model=float(release_speed(v, mu, J)), v_sep=r["v_sep"], v_err_pct=r["v_r_cmd_err_pct"],
               d_err_pct=r["d_err_pct"], psi_deg=r["psi_r_deg"], yaw_rest_deg=r["yaw_rest_deg"], x_rest=r["x_rest"],
               y_rest=r["y_rest"], miss_mm=(r["x_rest"] - x_t)*1e3 if not math.isnan(x_t) else float("nan"),
               track_rms_mm=r["track_q_rms"]*1e3, track_max_mm=r["track_q_max"]*1e3, separated=int(r["separated"]),
               point_contacts=r["n_point_contacts"], wall_s=time.time() - t0)
    if job["pusher"] == "arm":      # the plan's peak joint-speed and torque ratios to the iiwa's limits (arm.ArmPusher.summary)
        row.update(v_lim_frac=r["arm_plan_qd_ratio"], tau_lim_frac=r["arm_plan_tau_ratio"], joint_v_worst=r["arm_plan_qd_joint"],
                   a_lim_frac=r["arm_plan_qdd_ratio"], base_x=base[0], base_y=base[1], a_brake=pusher.a_brake_max)
    else:
        row.update(v_lim_frac=float("nan"), tau_lim_frac=float("nan"), joint_v_worst="", a_lim_frac=float("nan"), base_x="", base_y="", a_brake="")
    return row

# ----------------------------------------------------------------------------- summary

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
            rows.append(r)
    return rows

def summarize(rows):
    L = []
    ok = lambda b: "PASS" if b else "FAIL"
    A = [r for r in rows if r["part"] == "A"]
    if A:
        L.append("A  arm vs plate: the same push with each pusher (bars: speed at t* within 1% of the release model, slide within 2%,")
        L.append("   arm landing within 2 mm of the plate's).  v_lim = peak joint velocity / limit (>1: the real iiwa cannot do it)")
        L.append(f"   {'case':5} {'shape':5} {'mu':>4} {'v_pk':>5} {'J':>5} | {'v err% arm/plate':>17} | {'slide err% arm/plate':>20} | "
                 f"{'x_rest arm-plate':>16} | {'track rms/max mm':>16} | {'v_lim':>5}")
        all_ok = True
        for case in sorted({r["case"] for r in A}):
            a = next((r for r in A if r["case"] == case and r["pusher"] == "arm"), None)
            p = next((r for r in A if r["case"] == case and r["pusher"] == "plate"), None)
            if a is None or p is None:
                continue
            dx = (a["x_rest"] - p["x_rest"])*1e3
            good = abs(a["v_err_pct"]) < 1 and abs(a["d_err_pct"]) < 2 and abs(dx) < 2 and a["separated"] == 1
            all_ok &= good
            L.append(f"   {case:5} {a['shape']:5} {a['mu']:4.1f} {a['v_peak']:5.2f} {a['J']:5.0f} | {a['v_err_pct']:+7.2f} / {p['v_err_pct']:+7.2f} | "
                     f"{a['d_err_pct']:+9.2f} / {p['d_err_pct']:+8.2f} | {dx:+13.2f} mm | {a['track_rms_mm']:7.3f} / {a['track_max_mm']:6.3f} | "
                     f"{a['v_lim_frac']:5.2f}  {ok(good)}")
        L.append(f"   -> {ok(all_ok)}: {sum(1 for r in A if r['pusher'] == 'arm')} arm pushes; worst arm tracking rms "
                 f"{max(r['track_rms_mm'] for r in A if r['pusher'] == 'arm'):.3f} mm")
        L.append("")
    B = [r for r in rows if r["part"] == "B"]
    if B:
        L.append(f"B  targeting with known friction (planned with the true mu, J 100): miss = x_rest - x_target, success within {TOL*100:.0f} cm")
        L.append(f"   {'mu':>4} | " + " ".join(f"{x:>14.2f}" for x in TARGETS) + "   (target x, m: miss mm [v_lim])")
        for mu in MUS:
            cells = []
            for x in TARGETS:
                r = next((r for r in B if r["mu"] == mu and abs(r["x_target"] - x) < 1e-9), None)
                cells.append(f"{r['miss_mm']:+7.2f} [{r['v_lim_frac']:4.2f}]" if r else f"{'--':>14}")
            L.append(f"   {mu:4.1f} | " + " ".join(cells))
        L.extend(_miss_stats(B, "   "))
        L.append("")
    C = [r for r in rows if r["part"] == "C"]
    if C:
        L.append("C  repeatability: unknown start gap 0.5-3 mm (planned for 1), paddle lateral offset +-3 mm, mass +-10%")
        for k, (mu, x) in enumerate(CASES_C):
            rs = [r for r in C if r["case"].startswith(f"C{k+1}_")]
            if not rs:
                continue
            m, y = np.array([r["miss_mm"] for r in rs]), np.array([r["y_rest"]*1e3 for r in rs])
            gap = np.array([r["gap_mm"] for r in rs])
            L.append(f"   mu {mu} target {x:.2f} m, {len(rs)} trials: miss mean {m.mean():+.2f} mm, std {m.std(ddof=1) if len(m) > 1 else 0:.2f} mm, "
                     f"worst {np.abs(m).max():.2f} mm; lateral std {y.std(ddof=1) if len(y) > 1 else 0:.2f} mm, worst {np.abs(y).max():.2f} mm; "
                     f"miss with the gap error removed (mean) {np.mean(m + (gap - 1.0)):+.2f} mm; v_lim {rs[0]['v_lim_frac']:.2f}")
        L.extend(_miss_stats(C, "   "))
        L.append("   (the start-gap error moves the release point one-for-one, so most of the miss is the planner not knowing")
        L.append("    where the object is, not the arm; perception of the start pose would remove it)")
        L.append("")
    arm_rows = [r for r in rows if r["pusher"] == "arm"]
    if arm_rows:
        n_in = sum(1 for r in arm_rows if r["v_lim_frac"] <= 1.0)
        L.append(f"Joint limits: {n_in} of {len(arm_rows)} arm pushes are within the iiwa's joint velocity limits; the binding joint is "
                 f"{', '.join(sorted({f'joint {int(r['joint_v_worst'])}' for r in arm_rows if r['v_lim_frac'] > 1.0})) or 'none'}. "
                 f"Fastest within limits: arm.limited_peak_speed() (0.49 m/s at J 100 with the base at the origin, 1.02 m/s beside the board).")
        acc = [r for r in arm_rows if isinstance(r.get("a_lim_frac"), float) and not math.isnan(r["a_lim_frac"])]      # runs since 16 Nov
        if acc:
            a = np.array([r["a_lim_frac"] for r in acc])
            n_all = sum(1 for r in acc if max(r["v_lim_frac"], r["tau_lim_frac"], r["a_lim_frac"]) <= 1.0)
            L.append(f"Acceleration (planned joint acceleration / the SDF's drake:acceleration limit; not enforced by the simulator): "
                     f"{a.min():.2f}..{a.max():.2f}, within the limit in {int((a <= 1.0).sum())} of {len(acc)} arm pushes; "
                     f"{n_all} of {len(acc)} are within speed, torque and acceleration limits together.  "
                     f"Setup: base ({acc[0]['base_x']:g}, {acc[0]['base_y']:g}) m, brake cap {acc[0]['a_brake']:g} m/s^2.")
    return "\n".join(L)

def _miss_stats(rows, ind):
    out = []
    for label, rs in (("all", rows), ("within joint limits", [r for r in rows if r["v_lim_frac"] <= 1.0])):
        if not rs:
            out.append(f"{ind}{label}: no rows")
            continue
        m = np.array([r["miss_mm"] for r in rs])
        out.append(f"{ind}{label}: {len(rs)} slides, RMS miss {np.sqrt(np.mean(m**2)):.2f} mm, worst {np.abs(m).max():.2f} mm, "
                   f"within {TOL*100:.0f} cm {np.mean(np.abs(m) <= TOL*1e3)*100:.0f}%")
    return out

def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    A = [r for r in rows if r["part"] == "A"]
    cases = sorted({r["case"] for r in A})
    for j, (pz, col) in enumerate((("arm", "#c0392b"), ("plate", "#7f8c8d"))):
        vals = [next((r["v_err_pct"] for r in A if r["case"] == c and r["pusher"] == pz), np.nan) for c in cases]
        ax[0].bar(np.arange(len(cases)) + (j - 0.5)*0.38, vals, 0.38, color=col, label=pz)
    ax[0].axhspan(-1, 1, color="#2ecc71", alpha=0.12, label="1% bar")
    ax[0].set_xticks(range(len(cases))); ax[0].set_xticklabels(cases, rotation=60, fontsize=8)
    ax[0].set_ylabel("object speed at t* vs release model (%)"); ax[0].set_title("A  arm vs plate"); ax[0].legend(fontsize=8)
    B = [r for r in rows if r["part"] == "B"]
    for mu, col in zip(MUS, ("#2980b9", "#c0392b", "#27ae60")):
        rs = sorted([r for r in B if r["mu"] == mu], key=lambda r: r["x_target"])
        if rs:
            ax[1].plot([r["x_target"] for r in rs], [r["miss_mm"] for r in rs], "o-", color=col, label=f"mu {mu}")
            bad = [r for r in rs if r["v_lim_frac"] > 1]
            ax[1].plot([r["x_target"] for r in bad], [r["miss_mm"] for r in bad], "x", color="k", ms=9)
    ax[1].axhspan(-TOL*1e3, TOL*1e3, color="#2ecc71", alpha=0.12)
    ax[1].set_xlabel("target x (m)"); ax[1].set_ylabel("miss (mm)"); ax[1].set_title("B  targeting, true mu  (x: over joint limits)")
    ax[1].legend(fontsize=8)
    C = [r for r in rows if r["part"] == "C"]
    for k, (mu, x) in enumerate(CASES_C):
        rs = [r for r in C if r["case"].startswith(f"C{k+1}_")]
        if rs:
            ax[2].scatter([r["miss_mm"] for r in rs], [r["y_rest"]*1e3 for r in rs], label=f"mu {mu}, x* {x}", s=24)
    ax[2].add_patch(plt.Circle((0, 0), TOL*1e3, fill=False, ls="--", color="#27ae60"))
    ax[2].set_aspect("equal", "datalim"); ax[2].axhline(0, color="#ccc", lw=0.8); ax[2].axvline(0, color="#ccc", lw=0.8)
    ax[2].set_xlabel("miss along the push (mm)"); ax[2].set_ylabel("lateral (mm)"); ax[2].set_title("C  repeatability under disturbances")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default=str(RESULTS/"e1"/"e1_arm"), help="path prefix of the three output files")
    ap.add_argument("--base-x", type=float, default=0.0, help="iiwa base position along the push, m (the scene's is 0)")
    ap.add_argument("--base-y", type=float, default=0.0, help="iiwa base offset from the push line along y, m")
    ap.add_argument("--layout", choices=["origin", "side"], default=None, help="a named base position (arm.LAYOUTS); overrides --base-x/--base-y")
    ap.add_argument("--a-brake", type=float, default=None, help="cap on the arm's braking deceleration, m/s^2 (default: the arm's 12)")
    ap.add_argument("--summarize", default=None, help="reprint the summary (and plot) from a saved results CSV")
    a = ap.parse_args()
    if a.summarize:
        rows = load(a.summarize)
    else:
        from arm import layout_base
        js = jobs(a.quick, layout_base(a.layout) if a.layout else (a.base_x, a.base_y, 0.0), a.a_brake)
        print(f"{len(js)} pushes on {a.workers} workers ...", flush=True)
        t0 = time.time()
        with Pool(a.workers) as pool:
            rows = []
            for i, r in enumerate(pool.imap_unordered(run, js), 1):
                rows.append(r)
                if i % 10 == 0 or i == len(js):
                    print(f"  {i}/{len(js)}  ({time.time() - t0:.0f} s)", flush=True)
        rows.sort(key=lambda r: (r["part"], r["case"], r["pusher"]))
        with open(f"{a.out}_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
        rows = load(f"{a.out}_results.csv")
    text = summarize(rows)
    print(text)
    if not a.summarize:
        with open(f"{a.out}_summary.txt", "w") as f:
            f.write(text + "\n")
    plot(rows, f"{a.out}_summary.png")
    print(f"wrote {a.out}_summary.png" + ("" if a.summarize else f", {a.out}_results.csv, {a.out}_summary.txt"))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
e0_1_free_slide.py -- E0-1: does a free slide obey d = v^2 / (2 mu g)?  Freezes assumption A6.

A box is launched directly (no arm) on a uniform table built from sim_config.yaml, varying what the config
freezes: the discrete contact approximation with ITS OWN damping parameter (Hunt-Crossley dissipation for
lagged and similar, which ignore relaxation_time; relaxation_time for SAP, which ignores dissipation), the
time step {1, 2, 5} ms and the hydroelastic margin {0, 1e-4} m, at mu in {0.1, 0.3, 0.5} and v in
{0.6, 1.0} m/s.  Pass: stopping distance within 2% of d0 = v^2/(2 mu g) and yaw drift within 2 deg.
Two further launches at the config settings:
  * spin: a disc (uniform-pressure base) launched with R omega / v in {0.5, 1}; d/d0 is compared with the
    integrated disk model physics.disk_spin_slide (pass within 10%) and with 1 + 0.26 (R omega / v)^2;
  * heading: the box launched psi in {5, 10} deg off the +x axis; along-axis distance d0 cos psi, lateral
    d0 sin psi (within 2%), yaw within 2 deg.
Writes e0_1_results.csv, e0_1_summary.txt, e0_1_summary.png and prints the settings that pass.
    python3 e0_1_free_slide.py              # ~350 slides, about a minute on 10 cores
    python3 e0_1_free_slide.py --quick
    python3 e0_1_free_slide.py --summarize e0_1_results.csv
"""
import argparse, csv, math, os, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from pydrake.all import DiagramBuilder, RollPitchYaw, Simulator, SpatialVelocity
from scene import G, Obj, Table, build_scene
from drake_config import active_dissipation, contact_from_config, load_config
from physics import disk_spin_ratio, spin_factor

MUS, V0S = (0.1, 0.3, 0.5), (0.6, 1.0)
DTS = (1e-3, 2e-3, 5e-3)
MARGINS = (None, 1e-4)
DAMPING = {"lagged": [("dissipation", d) for d in (0.0, 1.0, 10.0, 50.0)],
           "similar": [("dissipation", d) for d in (0.0, 1.0, 10.0, 50.0)],
           "sap": [("relaxation", r) for r in (0.0, 0.01, 0.1)]}
SPIN_EPS, HEADINGS_DEG = (0.5, 1.0), (5.0, 10.0)
R_DISC = Obj(shape="disc").size[0]/2

def launch(mu, v0, contact, shape="box", omega=0.0, psi=0.0, x0=0.55):
    """Settle, launch with speed v0 at heading psi and spin omega, run to rest.  Returns dict(dx, dy, yaw_deg, t)."""
    builder = DiagramBuilder()
    sc = build_scene(builder, {"mu_k_nominal": mu, "s_tab_log": 0.0}, Obj(shape=shape), Table(patchiness=0.0), contact, arm=False, x0=x0)
    sim = Simulator(builder.Build())
    plant, body = sc.plant, sc.body
    ctx = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
    t = 0.3
    sim.AdvanceTo(t)
    X0 = plant.EvalBodyPoseInWorld(ctx, body)
    plant.SetFreeBodySpatialVelocity(ctx, body, SpatialVelocity([0, 0, omega], [v0*math.cos(psi), v0*math.sin(psi), 0]))
    t_max = t + 3*v0/(mu*G)*(1.6 if omega else 1.0)
    while t < t_max:
        t += 0.01
        sim.AdvanceTo(t)
        V = plant.EvalBodySpatialVelocityInWorld(ctx, body)
        if t > 0.35 and np.linalg.norm(V.translational()[:2]) < 1e-4 and abs(V.rotational()[2]) < 1e-3:
            break
    X1 = plant.EvalBodyPoseInWorld(ctx, body)
    d = X1.translation() - X0.translation()
    yaw = math.degrees(RollPitchYaw(X1.rotation()).yaw_angle() - RollPitchYaw(X0.rotation()).yaw_angle())
    return {"dx": float(d[0]), "dy": float(d[1]), "yaw_deg": yaw, "t": t - 0.3}

def tasks(quick):
    mus, v0s = ((0.3,), (1.0,)) if quick else (MUS, V0S)
    out = []
    for approx, damps in DAMPING.items():
        for key, val in (damps[:1] if quick else damps):
            for dt in ((1e-3,) if quick else DTS):
                for margin in MARGINS:
                    for mu in mus:
                        for v0 in v0s:
                            out.append(("grid", {"approx": approx, key: val, "time_step": dt, "margin": margin}, mu, v0, "box", 0.0, 0.0))
    for eps in SPIN_EPS:
        for mu, v0 in (((0.3, 1.0),) if quick else ((0.3, 0.6), (0.3, 1.0), (0.1, 1.0))):
            out.append(("spin", {}, mu, v0, "disc", eps, 0.0))
    for psi in HEADINGS_DEG:
        for mu, v0 in (((0.3, 1.0),) if quick else ((0.3, 1.0), (0.1, 0.6))):
            out.append(("heading", {}, mu, v0, "box", 0.0, psi))
    return out

def run(task):
    stage, over, mu, v0, shape, eps, psi_deg = task
    cfg = load_config()
    contact = contact_from_config(cfg, **over)
    if "relaxation" in over:
        contact.dissipation = cfg["dissipation"]     # SAP ignores it; kept at the config value for the record
    t0, msg = time.time(), ""
    psi, omega = math.radians(psi_deg), eps*v0/R_DISC
    try:
        r = launch(mu, v0, contact, shape, omega, psi)
    except Exception as e:
        r, msg = {"dx": float("nan"), "dy": float("nan"), "yaw_deg": float("nan"), "t": float("nan")}, str(e).splitlines()[0][:200]
    d0 = v0**2/(2*mu*G)
    row = {"stage": stage, "approx": contact.approx, "dt_ms": contact.time_step*1e3,
           "damping": active_dissipation(contact.approx),
           "damping_value": contact.relaxation if contact.approx == "sap" else contact.dissipation,
           "margin_m": 0.0 if contact.margin is None else contact.margin, "shape": shape, "mu": mu, "v0": v0,
           "spin_ratio": eps, "psi_deg": psi_deg, "dx_cm": f"{r['dx']*100:.3f}", "dy_cm": f"{r['dy']*100:.3f}",
           "d0_cm": f"{d0*100:.3f}", "yaw_deg": f"{r['yaw_deg']:.3f}", "t_slide_s": f"{r['t']:.2f}"}
    if stage == "grid":
        err = r["dx"]/d0 - 1
        row.update(predicted_cm=f"{d0*100:.3f}", err_pct=f"{err*100:.2f}", ok=int(abs(err) < 0.02 and abs(r["yaw_deg"]) < 2.0))
    elif stage == "spin":
        pred = d0*disk_spin_ratio(eps)
        err = r["dx"]/pred - 1
        row.update(predicted_cm=f"{pred*100:.3f}", err_pct=f"{err*100:.2f}", ok=int(abs(err) < 0.10),
                   quadratic_cm=f"{d0*spin_factor(1, eps, 1)*100:.3f}", ratio_sim=f"{r['dx']/d0:.4f}", ratio_disk=f"{pred/d0:.4f}")
    else:
        along, lat = d0*math.cos(psi), d0*math.sin(psi)
        err, err_lat = r["dx"]/along - 1, r["dy"]/lat - 1
        row.update(predicted_cm=f"{along*100:.3f}", err_pct=f"{err*100:.2f}", lateral_predicted_cm=f"{lat*100:.3f}",
                   lateral_err_pct=f"{err_lat*100:.2f}", ok=int(abs(err) < 0.02 and abs(err_lat) < 0.02 and abs(r["yaw_deg"]) < 2.0))
    row.update(wall_s=f"{time.time() - t0:.1f}", error=msg)
    return row

SETTING = ("approx", "damping", "damping_value", "dt_ms", "margin_m")

def summarize(rows, out=None):
    lines = []
    say = lines.append
    grid = [r for r in rows if r["stage"] == "grid"]
    groups = defaultdict(list)
    for r in grid:
        groups[tuple(str(r[k]) for k in SETTING)].append(r)
    say("== E0-1 grid: free slide of a box vs d = v^2/(2 mu g); worst |error| and |yaw| over mu in {0.1, 0.3, 0.5}, v in {0.6, 1.0} ==")
    passing = []
    for key in sorted(groups, key=lambda k: (k[0], float(k[3]), float(k[2]), float(k[4]))):
        rs = groups[key]
        errs = [abs(float(r["err_pct"])) for r in rs if r["err_pct"] != "nan"]
        yaws = [abs(float(r["yaw_deg"])) for r in rs if r["yaw_deg"] != "nan"]
        ok = all(int(r["ok"]) for r in rs) and not any(r["error"] for r in rs)
        if ok:
            passing.append(key)
        label = f"{key[0]:<8}{key[1]}={float(key[2]):<6g} dt={float(key[3]):g} ms margin={float(key[4]):g}"
        say(f"{'PASS' if ok else '    '} {label:<58} max err {max(errs) if errs else float('nan'):5.2f}%  max yaw {max(yaws) if yaws else float('nan'):5.2f} deg"
            + (f"  errors: {rs[0]['error']}" if any(r["error"] for r in rs) else ""))
    say(f"{len(passing)} of {len(groups)} settings pass")
    for stage, title in (("spin", "== spin launches (disc): d/d0 in the simulator vs the uniform-pressure disk integrator (pass within 10%) =="),
                         ("heading", "== heading launches (box): along-axis d0 cos psi and lateral d0 sin psi within 2%, yaw within 2 deg ==")):
        rs = [r for r in rows if r["stage"] == stage]
        if rs:
            say(title)
            for r in rs:
                if stage == "spin":
                    say(f"{'PASS' if int(r['ok']) else '    '} mu={r['mu']} v={r['v0']} R omega/v={r['spin_ratio']}: d/d0 sim {r['ratio_sim']}, "
                        f"disk model {r['ratio_disk']} ({float(r['err_pct']):+.1f}%), quadratic {float(r['quadratic_cm'])/float(r['d0_cm']):.3f}")
                else:
                    say(f"{'PASS' if int(r['ok']) else '    '} mu={r['mu']} v={r['v0']} psi={r['psi_deg']} deg: along {float(r['err_pct']):+.2f}%, "
                        f"lateral {float(r['lateral_err_pct']):+.2f}%, yaw {float(r['yaw_deg']):+.2f} deg")
    cfg = load_config()
    frozen = (cfg["discrete_contact_approximation"], active_dissipation(cfg["discrete_contact_approximation"]),
              str(cfg["relaxation_time"] if cfg["discrete_contact_approximation"] == "sap" else float(cfg["dissipation"])),
              str(float(cfg["time_step"])*1e3), str(float(cfg["margin"] or 0.0)))
    tested = [k for k in groups if k[0] == frozen[0] and float(k[3]) == float(frozen[3]) and float(k[4]) == float(frozen[4])]
    say("== A6 ==")
    say(f"sim_config.yaml: {frozen[0]}, dt {frozen[3]} ms, margin {frozen[4]} m, {frozen[1]}={frozen[2]}.  "
        + ("Its damping value is not in the sweep; " if frozen not in groups else "")
        + f"settings tested at that approximation, step and margin: " + ", ".join(f"{frozen[1]}={float(k[2]):g} {'PASS' if k in passing else 'fail'}" for k in tested))
    text = "\n".join(lines)
    print(text)
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")

def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hue = {"lagged": "#2a78d6", "similar": "#eb6834", "sap": "#1baf7a"}
    grid = [r for r in rows if r["stage"] == "grid" and r["err_pct"] != "nan"]
    fig, axes = plt.subplots(1, len(DTS) + 1, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    for ax, dt in zip(axes, DTS):
        rs = [r for r in grid if float(r["dt_ms"]) == dt*1e3]
        labels = sorted({(r["approx"], float(r["damping_value"])) for r in rs}, key=lambda k: (list(hue).index(k[0]), k[1]))
        for r in rs:
            x = labels.index((r["approx"], float(r["damping_value"])))
            hollow = float(r["margin_m"]) == 0.0
            ax.scatter(x + (-0.12 if hollow else 0.12), float(r["err_pct"]), s=22, facecolors="none" if hollow else hue[r["approx"]],
                       edgecolors=hue[r["approx"]], linewidths=1.2, zorder=3)
        ax.axhspan(-2, 2, color="#eeeeee", zorder=0)
        ax.set_xticks(range(len(labels)), [f"{ {'lagged': 'lag', 'similar': 'sim', 'sap': 'sap'}[a]}\n{v:g}" for a, v in labels], fontsize=7)
        ax.set_title(f"step {dt*1e3:g} ms", fontsize=10)
        ax.set_ylim(-12, 12) if dt > 1e-3 else ax.set_ylim(-4, 4)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("stopping-distance error, % (band: pass within 2%)")
    axes[0].text(0.02, 0.03, "points beyond the axis are clipped; see e0_1_summary.txt", transform=axes[0].transAxes, fontsize=7, color="#52514e")
    ax = axes[-1]
    eps = np.linspace(0, 1.05, 9)
    ax.plot(eps, [disk_spin_ratio(e, n=60) if e > 0 else 1.0 for e in eps], color="#0b0b0b", linewidth=2, label="disk integrator")
    ax.plot(eps, spin_factor(1, eps, 1), color="#52514e", linewidth=1, linestyle="--", label="1 + 0.26 (R omega/v)^2")
    for r in rows:
        if r["stage"] == "spin" and r["dx_cm"] != "nan":
            ax.scatter(float(r["spin_ratio"]), float(r["ratio_sim"]), color="#2a78d6", s=30, zorder=3)
    ax.scatter([], [], color="#2a78d6", s=30, label="Drake disc")
    ax.set_xlabel("R omega / v at launch"); ax.set_ylabel("d / d0"); ax.set_title("spin lengthens the slide", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle("E0-1: free slide against d = v^2 / (2 mu g)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.text(0.01, 0.905, "hollow: margin 0; filled: margin 1e-4 m.  x: approximation and its damping value (Hunt-Crossley dissipation in s/m, "
             "or SAP relaxation time in s)", fontsize=8, color="#52514e")
    fig.savefig(path, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="e0_1_results.csv")
    ap.add_argument("--summary", default="e0_1_summary.txt")
    ap.add_argument("--fig", default="e0_1_summary.png")
    ap.add_argument("--summarize", metavar="CSV")
    a = ap.parse_args()
    if a.summarize:
        with open(a.summarize, newline="") as f:
            rows = list(csv.DictReader(f))
        summarize(rows, a.summary); plot(rows, a.fig); return
    todo = tasks(a.quick)
    print(f"{len(todo)} slides on {a.workers} workers")
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, todo, chunksize=2), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    rows.sort(key=lambda r: (r["stage"], r["approx"], float(r["dt_ms"]), float(r["damping_value"]), float(r["margin_m"]), r["mu"], r["v0"], r["spin_ratio"], r["psi_deg"]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")
    summarize(rows, a.summary)
    plot(rows, a.fig)
    print(f"wrote {a.summary}, {a.fig}")

if __name__ == "__main__":
    main()

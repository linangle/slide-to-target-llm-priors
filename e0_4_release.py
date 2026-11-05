#!/usr/bin/env python3
"""
e0_4_release.py -- E0-4: does the release model v_r = v_peak - (mu g)^2 / (2J) hold?

Pushes a box with the prescribed paddle (push.py) at the frozen contact settings (sim_config.yaml):
  * grid: braking jerk J in {30, 50, 100, 200, 500, 2000} m/s^3, mu in {0.1, 0.3, 0.5}, v_peak in {0.6, 0.8, 1.0} m/s;
  * contact variants at mu = 0.3, v_peak = 0.8, J in {50, 200}: each discrete approximation with its own damping
    (lagged 0 s/m; similar 0 and 1 s/m; sap relaxation 0, 0.01 s and Drake's default), beside the config's lagged 1 s/m.
Metric: the object's speed at separation (simulator state at the last step with paddle-object contact force above
0.01 N) against the model evaluated on the commanded profile (what the decision rule knows) and on the paddle's actual
motion (physics.release_from_profile on the recorded joint speed).  Pass: within 1%.
Decision 7 (braking strategy): the same rows give the distance lost to the ramp, 1 - (v_r / v_peak)^2, per J and mu.

Writes e0_4_results.csv, e0_4_summary.txt, e0_4_summary.png.
    python3 e0_4_release.py                 # 66 pushes, ~1 min on 8 workers
    python3 e0_4_release.py --quick
    python3 e0_4_release.py --summarize e0_4_results.csv
"""
import argparse, csv, math, os, time
from multiprocessing import Pool
import numpy as np
from scene import G, Obj
from drake_config import active_dissipation, contact_from_config, load_config
from physics import release_speed
from push import Pusher, push_slide, scalars

JS = (30.0, 50.0, 100.0, 200.0, 500.0, 2000.0)
MUS = (0.1, 0.3, 0.5)
VS = (0.6, 0.8, 1.0)
VARIANT_MU, VARIANT_V, VARIANT_JS = 0.3, 0.8, (50.0, 200.0)
VARIANTS = ({"approx": "lagged", "dissipation": 0.0}, {"approx": "similar", "dissipation": 0.0}, {"approx": "similar", "dissipation": 1.0},
            {"approx": "sap", "relaxation": 0.0}, {"approx": "sap", "relaxation": 0.01}, {"approx": "sap", "relaxation": None})

def tasks(quick):
    js, mus, vs = ((50.0, 200.0, 2000.0), (0.1, 0.5), (0.8,)) if quick else (JS, MUS, VS)
    todo = [("grid", {}, mu, v, J) for J in js for mu in mus for v in vs]
    todo += [("variant", over, VARIANT_MU, VARIANT_V, J) for J in VARIANT_JS for over in (VARIANTS[:1] + VARIANTS[-1:] if quick else VARIANTS)]
    return todo

def run(task):
    stage, over, mu, v, J = task
    cfg = load_config()
    contact = contact_from_config(cfg, **over)
    if "relaxation" in over:
        contact.dissipation = float(cfg["dissipation"])    # sap ignores it; kept at the config value for the record
    row = {"stage": stage, "approx": contact.approx, "damping": active_dissipation(contact.approx),
           "damping_value": "default" if (contact.approx == "sap" and contact.relaxation is None) else
                            (contact.relaxation if contact.approx == "sap" else contact.dissipation),
           "mu": mu, "v_peak": v, "J": J, "ramp_loss_pct": (1 - (release_speed(v, mu, J)/v)**2)*100}
    try:
        r = scalars(push_slide(mu, v, J, contact, Obj(), Pusher(), keep_traj=False))
        for k in ("mu", "v_peak", "J", "approx", "shape", "mu_p", "lateral", "yaw_cmd_deg", "com_y"):
            r.pop(k, None)
        row.update(r)
        row["sim_loss_pct"] = (1 - (row["v_sep"]/v)**2)*100
        row["ok"] = int(abs(row["v_r_cmd_err_pct"]) < 1.0)
        row["error"] = ""
    except Exception as ex:
        row.update(ok=0, error=str(ex).splitlines()[0][:200])
    return row

def summarize(rows, out=None):
    ok = [r for r in rows if not r.get("error")]
    grid = [r for r in ok if r["stage"] == "grid"]
    lines = [f"== E0-4: object speed at separation vs v_peak - (mu g)^2/(2J), box, config contact ({len(grid)} pushes) ==",
             "cells: error of the simulator's v_sep against the model on the commanded profile, % (pass within 1%); in brackets, against the model on the paddle's actual motion",
             f"{'mu':>4}{'v_peak':>7} | " + " | ".join(f"J={J:g}".rjust(16) for J in JS)]
    n_pass = 0
    for mu in MUS:
        for v in VS:
            cells = []
            for J in JS:
                rs = [r for r in grid if float(r["mu"]) == mu and float(r["v_peak"]) == v and float(r["J"]) == J]
                if not rs:
                    cells.append("".rjust(16)); continue
                r = rs[0]
                e, ea = float(r["v_r_cmd_err_pct"]), float(r["v_r_act_err_pct"])
                n_pass += int(r["ok"]) if str(r["ok"]) in ("0", "1") else int(bool(r["ok"]))
                cells.append(f"{e:+.2f} [{ea:+.2f}]".rjust(16) + ("" if abs(e) < 1 else "!"))
            lines.append(f"{mu:>4}{v:>7} | " + " | ".join(c[:17] for c in cells))
    lines.append(f"pass: {n_pass} of {len(grid)} grid pushes within 1% on the commanded profile; "
                 f"largest |error| {max(abs(float(r['v_r_cmd_err_pct'])) for r in grid):.2f}% (commanded), "
                 f"{max(abs(float(r['v_r_act_err_pct'])) for r in grid if r['v_r_act_err_pct'] != 'nan'):.2f}% (actual paddle)")
    lines.append(f"the paddle-object force vanished {min(float(r['sep_lag_ms']) for r in grid):+.1f} .. {max(float(r['sep_lag_ms']) for r in grid):+.1f} ms after t* (the compliant tail); "
                 f"the object's speed when it did was {min(float(r['v_sep_force'])/float(r['v_r_cmd']) - 1 for r in grid)*100:+.2f}% .. "
                 f"{max(float(r['v_sep_force'])/float(r['v_r_cmd']) - 1 for r in grid)*100:+.2f}% from the model")
    lines.append(f"free slide after separation vs v_sep^2/(2 mu g): {min(float(r['d_err_pct']) for r in grid):+.2f}% .. {max(float(r['d_err_pct']) for r in grid):+.2f}%; "
                 f"release heading |psi_r| <= {max(abs(float(r['psi_r_deg'])) for r in grid):.3f} deg; tilt <= {max(float(r['tilt_max_deg']) for r in grid):.3f} deg; "
                 f"paddle tracking rms <= {max(float(r['track_q_rms']) for r in grid)*1e3:.2f} mm, {max(float(r['track_v_rms']) for r in grid)*1e3:.1f} mm/s")
    lines.append("")
    lines.append("== contact variants at mu=0.3, v_peak=0.8 ==")
    lines.append(f"{'approx':<8}{'damping':<22}{'J':>6} | {'v_sep err % cmd':>16} | {'act':>8} | {'d err %':>8} | {'psi_r deg':>10} | {'tilt deg':>9} | {'spin rad/s':>11}")
    for r in sorted((r for r in ok if r["stage"] == "variant" or (r["stage"] == "grid" and float(r["mu"]) == VARIANT_MU and float(r["v_peak"]) == VARIANT_V and float(r["J"]) in VARIANT_JS)),
                    key=lambda r: (["lagged", "similar", "sap"].index(r["approx"]), str(r["damping_value"]), float(r["J"]))):
        lines.append(f"{r['approx']:<8}{r['damping'] + ' ' + str(r['damping_value']):<22}{float(r['J']):>6g} | {float(r['v_r_cmd_err_pct']):>+16.2f} | "
                     f"{float(r['v_r_act_err_pct']):>+8.2f} | {float(r['d_err_pct']):>+8.2f} | {float(r['psi_r_deg']):>+10.3f} | {float(r['tilt_max_deg']):>9.3f} | {float(r['omega_r']):>+11.4f}")
    lines.append("")
    lines.append("== decision 7: distance lost to the braking ramp, 1 - (v_r/v_peak)^2, % (model; simulator in brackets) ==")
    lines.append(f"{'mu':>4}{'v_peak':>7} | " + " | ".join(f"J={J:g}".rjust(14) for J in JS))
    for mu in MUS:
        for v in VS:
            cells = []
            for J in JS:
                rs = [r for r in grid if float(r["mu"]) == mu and float(r["v_peak"]) == v and float(r["J"]) == J]
                model = (1 - (release_speed(v, mu, J)/v)**2)*100
                cells.append((f"{model:.1f} [{float(rs[0]['sim_loss_pct']):.1f}]" if rs else f"{model:.1f}").rjust(14))
            lines.append(f"{mu:>4}{v:>7} | " + " | ".join(cells))
    lines.append("reading: the ramp costs more than 2% of the distance at mu = 0.3 for J below about 500 m/s^3 and at mu = 0.5 for J below about 2000; "
                 "where the model holds within 1%, the decision rule can keep v_r(mu) and brake as hard as the joints allow, both.")
    bad = [r for r in rows if r.get("error")]
    if bad:
        lines.append(f"{len(bad)} pushes failed: " + "; ".join(f"{r['approx']} {r['damping_value']} mu={r['mu']} v={r['v_peak']} J={r['J']}: {r['error']}" for r in bad[:5]))
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w").write(text + "\n")

def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hue_mu = dict(zip(MUS, ("#2a78d6", "#eb6834", "#1baf7a")))
    hue_ap = {"lagged": "#2a78d6", "similar": "#eb6834", "sap": "#1baf7a"}
    mark = dict(zip(VS, ("o", "s", "^")))
    ok = [r for r in rows if not r.get("error")]
    grid = [r for r in ok if r["stage"] == "grid"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 0.9]})
    ax = axes[0]
    for r in grid:
        ax.scatter(float(r["J"]), float(r["v_r_cmd_err_pct"]), s=28, marker=mark[float(r["v_peak"])], color=hue_mu[float(r["mu"])], zorder=3)
    ax.axhspan(-1, 1, color="#eeeeee", zorder=0); ax.axhline(0, color="#52514e", linewidth=0.6)
    ax.set_xscale("log"); ax.set_xlabel("braking jerk J, m/s^3"); ax.set_ylabel("object speed at t* vs the model, %")
    for mu in MUS:
        ax.scatter([], [], color=hue_mu[mu], s=28, label=f"mu {mu}")
    for v in VS:
        ax.scatter([], [], marker=mark[v], facecolors="none", edgecolors="#0b0b0b", s=28, label=f"v_peak {v} m/s")
    ax.legend(fontsize=7, frameon=False, ncol=2); ax.set_title("release model at the config contact (band: pass within 1%)", fontsize=10)
    ax = axes[1]
    Jf = np.logspace(math.log10(JS[0]), math.log10(JS[-1]), 100)
    for mu in MUS:
        ax.plot(Jf, (1 - (release_speed(0.8, mu, Jf)/0.8)**2)*100, color=hue_mu[mu], linewidth=1.5, label=f"model, mu {mu}, v 0.8")
    for r in grid:
        if float(r["v_peak"]) == 0.8:
            ax.scatter(float(r["J"]), float(r["sim_loss_pct"]), s=28, color=hue_mu[float(r["mu"])], zorder=3)
    ax.axhline(2, color="#52514e", linewidth=0.6, linestyle="--"); ax.text(JS[0]*1.05, 2.4, "2% of the distance", fontsize=7, color="#52514e")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("braking jerk J, m/s^3"); ax.set_ylabel("distance lost to the ramp, 1 - (v_r/v_peak)^2, %")
    ax.legend(fontsize=7, frameon=False); ax.set_title("decision 7: what braking costs", fontsize=10)
    ax = axes[2]
    var = [r for r in ok if r["stage"] == "variant" or (r["stage"] == "grid" and float(r["mu"]) == VARIANT_MU and float(r["v_peak"]) == VARIANT_V and float(r["J"]) in VARIANT_JS)]
    labels = sorted({(r["approx"], str(r["damping_value"])) for r in var}, key=lambda k: (list(hue_ap).index(k[0]), k[1]))
    for r in var:
        x = labels.index((r["approx"], str(r["damping_value"])))
        hollow = float(r["J"]) == VARIANT_JS[0]
        ax.scatter(x + (-0.12 if hollow else 0.12), float(r["v_r_cmd_err_pct"]), s=28, facecolors="none" if hollow else hue_ap[r["approx"]],
                   edgecolors=hue_ap[r["approx"]], linewidths=1.2, zorder=3)
    ax.axhspan(-1, 1, color="#eeeeee", zorder=0); ax.axhline(0, color="#52514e", linewidth=0.6)
    ax.set_xticks(range(len(labels)), [f"{ {'lagged': 'lag', 'similar': 'sim', 'sap': 'sap'}[a]}\n{v}" for a, v in labels], fontsize=7)
    ax.set_ylabel("object speed at t* vs the model, %"); ax.set_title(f"approximation and damping, mu {VARIANT_MU}, v {VARIANT_V}", fontsize=10)
    ax.text(0.02, 0.96, f"hollow: J {VARIANT_JS[0]:g}; filled: J {VARIANT_JS[1]:g} m/s^3", transform=ax.transAxes, fontsize=7, color="#52514e", va="top")
    ax.set_ylim(-5, 13)
    for ax in axes:
        ax.grid(color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("E0-4: the release model against the simulator", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.text(0.01, 0.935, "right panel: 'default' is Drake's sap relaxation time; points beyond the axis are clipped (see e0_4_summary.txt)", fontsize=8, color="#52514e")
    fig.savefig(path, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    ap.add_argument("--out", default="e0_4_results.csv")
    ap.add_argument("--summary", default="e0_4_summary.txt")
    ap.add_argument("--fig", default="e0_4_summary.png")
    ap.add_argument("--summarize", metavar="CSV")
    a = ap.parse_args()
    if a.summarize:
        with open(a.summarize, newline="") as f:
            rows = list(csv.DictReader(f))
        summarize(rows, a.summary); plot(rows, a.fig); return
    todo = tasks(a.quick)
    print(f"{len(todo)} pushes on {a.workers} workers")
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, todo), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["stage"], r["approx"], str(r["damping_value"]), float(r["mu"]), float(r["v_peak"]), float(r["J"])))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"wrote {a.out}")
    summarize(rows, a.summary)
    plot(rows, a.fig)
    print(f"wrote {a.summary}, {a.fig}")

if __name__ == "__main__":
    main()

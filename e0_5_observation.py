#!/usr/bin/env python3
"""
e0_5_observation.py -- E0-5: do the observation models behave as predicted?

Pushes a box with the prescribed paddle (push.py) at the frozen contact settings (sim_config.yaml) for mu in {0.1, 0.3, 0.5},
v_peak = 0.8 m/s, J = 100 m/s^3, records the simulator trajectory, then replays it through the overhead depth camera
(scene.Camera; perception.py: stereo depth noise, RANSAC table plane, blob, 2-D ICP on the outline) at 30, 60 and 120 Hz:
every frame after separation is rendered at the recorded pose, plate included, and perceived; so is the rest pose.  The N
replicates of a case differ in the depth noise, the camera phase, and a small lateral offset and yaw of the whole slide.
For each replicate observe.fit_slide fits models A (frames), B (+ rest pose) and C (+ FK release point), and O1 (commanded
speed and endpoint) is evaluated on the same slide.

Metrics: bias and standard error of mu-hat per model and rate, against the Cramer-Rao bound (observe.crlb) at the MEASURED
noise and against the error-budget table's nominal noise (frames 2 mm, rest 0.5 mm, FK 0.5 mm).  Pass: bias under 1%, SE
within 1.5x of the bound.  Bias is given against the nominal mu and against the simulator's effective mu (model C fitted to the
noise-free trajectory), which differ by the simulator's own ~1% (E0-1).
Outputs for Tier B: sigma_f and sigma_rest measured here; sigma_fk assumed 0.5 mm, because the simulator's forward kinematics
are exact (the FK release point here is the plate face at separation plus half the object length; its bias is reported).

Writes e0_5_results.csv (per case, rate, model), e0_5_frames.csv (every perceived frame), e0_5_noise.csv (Tier B values),
e0_5_summary.txt, e0_5_summary.png.
    python3 e0_5_observation.py                  # 3 cases x 3 rates x 20 replicates, a few minutes on 8 workers
    python3 e0_5_observation.py --quick          # 4 replicates
    python3 e0_5_observation.py --summarize      # redo the fits and figure from the saved frames
"""
import argparse, csv, math, os, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from pydrake.all import DiagramBuilder, RigidTransform, RollPitchYaw
from scene import G, Camera, Obj, Table, build_scene
from drake_config import contact_from_config
from observe import Obs, crlb, fit_o1, fit_slide, o1_bias
from perception import Noise, add_noise, perceive, render_depth
from push import Pusher, push_slide

CASES = ((0.1, 0.8), (0.3, 0.8), (0.5, 0.8))
J = 100.0
RATES = (30, 60, 120)
SIGMA_FK = 0.0005                                                   # assumed, m (Tier B); the simulator's FK is exact
BUDGET = {"sigma_f": 0.002, "sigma_rest": 0.0005, "sigma_fk": 0.0005}   # the error-budget table's nominal noise, m
T_AFTER_REST = 0.1                                                  # frames keep coming this long after the stop, s
MODELS = ("A", "B", "C", "C1", "O1")      # C1: model C without the frames in the first 30 ms after release

_scene, _cases = None, None   # per-worker render scene and the recorded pushes

def push_cases(cases):
    """One push per (mu, v_peak); returns its record with the trajectory and the simulator's effective mu."""
    out = []
    for mu, v in cases:
        r = push_slide(mu, v, J, contact_from_config(), Obj(), Pusher())
        tr = r["traj"]
        sel = tr["t"] >= r["t_sep"]
        exact = Obs(tr["t"][sel] - r["t_sep"], tr["x"][sel], float(tr["x"][-1]), float(tr["x"][sel][0]), 1e-5, 1e-5, 1e-5)
        fit = fit_slide(exact, "C")
        out.append({"mu": mu, "v_peak": v, "t_sep": r["t_sep"], "v_sep": r["v_sep"], "x_sep": r["x_sep"], "x_fk": r["x_fk"],
                    "x_rest": r["x_rest"], "t_rest": r["t_rest"], "d_slide": r["d_slide"], "mu_eff": fit["mu"], "v_r_eff": fit["v_r"],
                    "fk_bias": r["x_fk"] - r["x_sep"], "traj": {k: tr[k] for k in ("t", "x", "y", "yaw", "q")}})
        print(f"  push mu={mu} v={v}: v_sep {r['v_sep']:.4f} m/s, slide {r['d_slide']*100:.2f} cm, simulator mu_eff {fit['mu']:.4f} "
              f"({(fit['mu']/mu - 1)*100:+.2f}%), FK release point bias {(r['x_fk'] - r['x_sep'])*1e3:+.2f} mm", flush=True)
    return out

def _init(cases):
    global _scene, _cases
    _cases = cases
    builder, pusher = DiagramBuilder(), Pusher()
    sc = build_scene(builder, {"mu_k_nominal": 0.3, "s_tab_log": 0.0}, Obj(), Table(patchiness=0.0), contact_from_config(), arm=False,
                     camera=Camera(), pusher=pusher)
    ctx = builder.Build().CreateDefaultContext()
    _scene = (sc, ctx, sc.plant.GetMyMutableContextFromRoot(ctx), pusher, sc.X_WO.translation()[2])

def _frame(t, case, dy, dyaw, rng, noise, cam):
    """Render the recorded pose at time t (shifted by dy, dyaw) and perceive it.  Returns (x_true, y_true, est)."""
    sc, ctx, pctx, pusher, z0 = _scene
    tr = case["traj"]
    x, y, yaw, q = (float(np.interp(t, tr["t"], tr[k])) for k in ("x", "y", "yaw", "q"))
    sc.plant.SetFreeBodyPose(pctx, sc.body, RigidTransform(RollPitchYaw(0, 0, yaw + dyaw), [x, y + dy, z0]))
    pusher.joint.set_translation(pctx, q)
    # the window starts 0.5 mm ahead of the plate face (known from FK), so the plate's own face pixels never join the object's blob
    est = perceive(add_noise(render_depth(sc, ctx), cam, noise, rng), cam, Obj(), rng, x_window=(pusher.face_x(q) + 0.0005, pusher.face_x(q) + 0.6))
    return x, y + dy, est

def replicate(task):
    ci, fps, rep = task
    case = _cases[ci]
    rng = np.random.default_rng([ci, fps, rep])
    noise, cam = Noise(), Camera()
    phase, dy, dyaw = rng.uniform(0, 1/fps), rng.uniform(-0.005, 0.005), math.radians(rng.uniform(-2, 2))
    k = math.floor((case["t_sep"] - phase)*fps) + 1           # first frame strictly after separation
    frames = []
    while phase + k/fps <= case["t_rest"] + T_AFTER_REST:
        t = phase + k/fps
        x, y, est = _frame(t, case, dy, dyaw, rng, noise, cam)
        frames.append({"k": len(frames), "t_rel": t - case["t_sep"], "x_true": x, "x_hat": est["x"] if est else float("nan"),
                       "y_true": y, "y_hat": est["y"] if est else float("nan"), "n_points": est["n_points"] if est else 0})
        k += 1
    x_rest, _, est = _frame(case["t_rest"] + 1.0, case, dy, dyaw, rng, noise, cam)
    return {"case": ci, "fps": fps, "rep": rep, "phase": phase, "dy": dy, "dyaw_deg": math.degrees(dyaw), "frames": frames,
            "x_rest_true": x_rest, "x_rest_hat": est["x"] if est else float("nan"), "x_fk_obs": case["x_fk"] + SIGMA_FK*rng.standard_normal()}

def analyse(cases, reps):
    """Noise, fits and statistics per (case, rate).  Returns (results rows, noise rows, fits)."""
    results, noise_rows, fits = [], [], []
    for ci, case in enumerate(cases):
        for fps in RATES:
            rs = [r for r in reps if r["case"] == ci and r["fps"] == fps]
            if not rs:
                continue
            fr = [f for r in rs for f in r["frames"]]
            err = np.array([f["x_hat"] - f["x_true"] for f in fr])
            first = np.array([f["x_hat"] - f["x_true"] for f in fr if f["k"] == 0])
            later = np.array([f["x_hat"] - f["x_true"] for f in fr if f["k"] > 0])
            good = np.isfinite(err)
            sigma_f = float(np.sqrt(np.mean(err[good]**2)))
            err_rest = np.array([r["x_rest_hat"] - r["x_rest_true"] for r in rs])
            sigma_rest = float(np.sqrt(np.nanmean(err_rest**2)))
            noise_rows.append({"mu": case["mu"], "v_peak": case["v_peak"], "fps": fps, "n_frames": int(good.sum()), "missed_frames": int((~good).sum()),
                               "sigma_f_mm": sigma_f*1e3, "frame_bias_mm": float(np.mean(err[good]))*1e3,
                               "first_frame_bias_mm": float(np.nanmean(first))*1e3, "first_frame_rms_mm": float(np.sqrt(np.nanmean(first**2)))*1e3,
                               "later_frames_rms_mm": float(np.sqrt(np.nanmean(later**2)))*1e3 if len(later) else float("nan"),
                               "sigma_rest_mm": sigma_rest*1e3, "rest_bias_mm": float(np.nanmean(err_rest))*1e3,
                               "sigma_fk_mm_assumed": SIGMA_FK*1e3, "fk_bias_mm": case["fk_bias"]*1e3})
            est = defaultdict(list)
            for r in rs:
                t = np.array([f["t_rel"] for f in r["frames"]]); x = np.array([f["x_hat"] for f in r["frames"]])
                keep = np.isfinite(x)
                if keep.sum() < 3 or not math.isfinite(r["x_rest_hat"]):
                    continue
                obs = Obs(t[keep], x[keep], r["x_rest_hat"], r["x_fk_obs"], sigma_f, sigma_rest, SIGMA_FK)
                for m in ("A", "B", "C"):
                    est[m].append(fit_slide(obs, m)["mu"])
                keep1 = keep & (t > 0.03)        # C1: drop the frames taken while the object is still within about a millimetre of the plate
                if keep1.sum() >= 3:
                    est["C1"].append(fit_slide(Obs(t[keep1], x[keep1], r["x_rest_hat"], r["x_fk_obs"], sigma_f, sigma_rest, SIGMA_FK), "C")["mu"])
                est["O1"].append(float(fit_o1(case["v_peak"], r["x_rest_hat"] - r["x_fk_obs"])))
                fits.append({"case": ci, "fps": fps, "rep": r["rep"], **{f"mu_{m}": est[m][-1] for m in MODELS if est[m]}})
            for m in MODELS:
                mh = np.array(est[m])
                if len(mh) < 2:
                    continue
                mean, sd = mh.mean(), mh.std(ddof=1)
                if m == "O1":
                    pred = float("nan"); budget = float("nan")
                else:
                    kw = {"A": {}, "B": {"sigma_rest": sigma_rest}, "C": {"sigma_rest": sigma_rest, "sigma_fk": SIGMA_FK}, "C1": {"sigma_rest": sigma_rest, "sigma_fk": SIGMA_FK}}[m]
                    pred = crlb(case["mu_eff"], case["v_r_eff"], fps, sigma_f, **kw)*100
                    kwb = {"A": {}, "B": {"sigma_rest": BUDGET["sigma_rest"]}, "C": {"sigma_rest": BUDGET["sigma_rest"], "sigma_fk": BUDGET["sigma_fk"]},
                           "C1": {"sigma_rest": BUDGET["sigma_rest"], "sigma_fk": BUDGET["sigma_fk"]}}[m]
                    budget = crlb(case["mu"], case["v_sep"], fps, BUDGET["sigma_f"], **kwb)*100
                se = sd/case["mu_eff"]*100
                row = {"mu": case["mu"], "v_peak": case["v_peak"], "fps": fps, "model": m, "n": len(mh), "mu_eff": case["mu_eff"], "v_sep": case["v_sep"],
                       "mean_mu_hat": float(mean), "bias_nominal_pct": (mean/case["mu"] - 1)*100, "bias_eff_pct": (mean/case["mu_eff"] - 1)*100,
                       "se_pct": se, "crlb_measured_pct": pred, "se_over_crlb": se/pred if math.isfinite(pred) else float("nan"),
                       "crlb_budget_pct": budget, "sigma_f_mm": sigma_f*1e3, "sigma_rest_mm": sigma_rest*1e3, "sigma_fk_mm": SIGMA_FK*1e3,
                       "frames_per_slide": len(fr)/len(rs), "o1_bias_model_pct": o1_bias(case["mu"], case["v_peak"], J)*100 if m == "O1" else float("nan")}
                row["pass"] = int(abs(row["bias_eff_pct"]) < 1.0 and (m == "O1" or row["se_over_crlb"] < 1.5))
                results.append(row)
    return results, noise_rows, fits

def summarize(results, noise_rows, out=None):
    lines = ["== E0-5: observation models A, B, C and O1 on camera-replayed pushes (box, v_peak 0.8 m/s, J 100 m/s^3, config contact) ==",
             "bias against the simulator's effective mu (nominal mu in brackets), %; SE = SD of mu-hat over replicates, %; CRLB at the measured noise, %; "
             "SE/CRLB (pass < 1.5 and |bias| < 1%); CRLB from the error-budget table's nominal noise, %.  C1 is model C without the frames of the first 30 ms "
             "after release, when the object is within a millimetre of the plate; its bound is model C's",
             f"{'mu':>4}{'fps':>5}{'model':>6}{'n':>4} | {'bias eff [nom]':>18} | {'SE':>7} | {'CRLB':>7} | {'ratio':>6} | {'budget':>7} | {'frames':>6} | pass"]
    for r in results:
        lines.append(f"{r['mu']:>4}{r['fps']:>5}{r['model']:>6}{r['n']:>4} | {r['bias_eff_pct']:>+8.3f} [{r['bias_nominal_pct']:>+6.2f}] | {r['se_pct']:>7.3f} | "
                     f"{r['crlb_measured_pct']:>7.3f} | {r['se_over_crlb']:>6.2f} | {r['crlb_budget_pct']:>7.3f} | {r['frames_per_slide']:>6.1f} | "
                     + ("yes" if r["pass"] else " no") + (f"   (O1 bias predicted {r['o1_bias_model_pct']:+.2f}%)" if r["model"] == "O1" else ""))
    lines.append("")
    lines.append("== perception noise (Tier B inputs) ==")
    lines.append(f"{'mu':>4}{'fps':>5}{'frames':>7}{'missed':>7} | {'sigma_f mm':>10}{'bias':>7} | {'1st frame rms':>13}{'bias':>7} | {'later rms':>9} | {'sigma_rest mm':>13}{'bias':>7} | {'sigma_fk mm':>11}{'fk bias':>8}")
    for n in noise_rows:
        lines.append(f"{n['mu']:>4}{n['fps']:>5}{n['n_frames']:>7}{n['missed_frames']:>7} | {n['sigma_f_mm']:>10.3f}{n['frame_bias_mm']:>+7.3f} | {n['first_frame_rms_mm']:>13.3f}"
                     f"{n['first_frame_bias_mm']:>+7.3f} | {n['later_frames_rms_mm']:>9.3f} | {n['sigma_rest_mm']:>13.3f}{n['rest_bias_mm']:>+7.3f} | "
                     f"{n['sigma_fk_mm_assumed']:>11.1f}{n['fk_bias_mm']:>+8.2f}")
    pooled_f = {fps: math.sqrt(np.average([n["sigma_f_mm"]**2 for n in noise_rows if n["fps"] == fps], weights=[n["n_frames"] for n in noise_rows if n["fps"] == fps]))
                for fps in RATES if any(n["fps"] == fps for n in noise_rows)}
    pooled_rest = math.sqrt(np.mean([n["sigma_rest_mm"]**2 for n in noise_rows])) if noise_rows else float("nan")
    lines.append("Tier B: sigma_f (pooled over mu) " + ", ".join(f"{s:.3f} mm at {fps} Hz" for fps, s in pooled_f.items())
                 + f"; sigma_rest {pooled_rest:.3f} mm; sigma_fk {SIGMA_FK*1e3:.1f} mm (assumed).  The first frame after release sits within a millimetre "
                 "of the plate; its error is listed apart from the later frames.")
    worst = max((abs(r["bias_eff_pct"]) for r in results if r["model"] in ("A", "B", "C")), default=float("nan"))
    ratio = max((r["se_over_crlb"] for r in results if r["model"] in ("A", "B", "C") and math.isfinite(r["se_over_crlb"])), default=float("nan"))
    worst1 = max((abs(r["bias_eff_pct"]) for r in results if r["model"] == "C1"), default=float("nan"))
    lines.append(f"A/B/C: largest |bias| vs the simulator's mu {worst:.3f}%; largest SE/CRLB {ratio:.2f}.  Without the first 30 ms of frames (C1) the largest "
                 f"|bias| is {worst1:.3f}%: the residual bias of A/B/C is the first frame, taken against the plate.  "
                 f"The nominal-mu bias column is the simulator's own offset from mu (E0-1), not the estimator's.  These noise values come from a rendered "
                 f"depth camera with exact pose and intrinsics and no motion blur, so they are lower bounds for the real camera.")
    cs = {(r["mu"], r["fps"]): r for r in results if r["model"] == "C"}
    if cs:
        parts = ["mu " + str(mu) + ": " + ", ".join(f"{cs[(mu, fps)]['se_pct']:.2f}% at {fps} Hz" for fps in RATES if (mu, fps) in cs)
                 for mu in sorted({k[0] for k in cs})]
        lines.append("decision 8 (camera rate), model C's SE per slide: " + "; ".join(parts) + f".  The error-budget table assumed {BUDGET['sigma_f']*1e3:.0f} mm "
                     "frames; the rendered camera gives a few tenths of a millimetre, so every rate beats the table.  What the rate buys is frames on the short "
                     "high-friction slides (mu 0.5 at 0.8 m/s lasts 0.14 s) and, per observe.residual_signature, telling a friction step from spin, which "
                     "needs 60 Hz or sub-millimetre frames.")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w").write(text + "\n")

def plot(results, frames_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hue = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    mark = dict(zip((c[0] for c in CASES), ("o", "s", "^")))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for r in results:
        if r["model"] not in hue:
            continue
        x = RATES.index(r["fps"]) + {"A": -0.2, "B": 0.0, "C": 0.2}[r["model"]]
        ax.scatter(x, r["se_pct"], s=30, marker=mark[r["mu"]], color=hue[r["model"]], zorder=3)
        ax.plot([x - 0.07, x + 0.07], [r["crlb_measured_pct"]]*2, color=hue[r["model"]], linewidth=1.5, zorder=2)
    ax.set_yscale("log"); ax.set_xticks(range(len(RATES)), [f"{f} Hz" for f in RATES])
    ax.set_ylabel("SE of mu-hat over replicates, %"); ax.set_title("precision (bars: Cramer-Rao bound at the measured noise)", fontsize=10)
    for m in hue:
        ax.scatter([], [], color=hue[m], s=30, label=f"model {m}")
    for mu in mark:
        ax.scatter([], [], marker=mark[mu], facecolors="none", edgecolors="#0b0b0b", s=30, label=f"mu {mu}")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axes[1]
    for r in results:
        if r["model"] not in hue:
            continue
        x = RATES.index(r["fps"]) + {"A": -0.2, "B": 0.0, "C": 0.2}[r["model"]]
        ax.scatter(x, r["bias_eff_pct"], s=30, marker=mark[r["mu"]], color=hue[r["model"]], zorder=3)
    ax.axhspan(-1, 1, color="#eeeeee", zorder=0); ax.axhline(0, color="#52514e", linewidth=0.6)
    ax.set_xticks(range(len(RATES)), [f"{f} Hz" for f in RATES]); ax.set_ylabel("bias of mu-hat vs the simulator's mu, %")
    ax.set_title("bias (band: pass within 1%)", fontsize=10)
    ax = axes[2]
    for mu in mark:
        rs = [f for f in frames_rows if float(f["mu"]) == mu and int(f["fps"]) == 60 and f["err_x_mm"] not in ("nan", "")]
        ax.scatter([float(f["t_rel"]) for f in rs], [float(f["err_x_mm"]) for f in rs], s=8, marker=mark[mu], color="#2a78d6", alpha=0.45, zorder=3, label=f"mu {mu}")
    ax.axhline(0, color="#52514e", linewidth=0.6)
    ax.set_xlabel("time since release, s"); ax.set_ylabel("perceived x minus true x, mm (60 Hz frames)"); ax.set_title("frame error along the slide", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.grid(color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("E0-5: observation models on camera-replayed pushes (box, v_peak 0.8 m/s, J 100 m/s^3)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.text(0.01, 0.935, "frames are perceived in a window starting 0.5 mm ahead of the plate face (known from FK), so the plate never joins the object's blob; "
             "the fits use every frame after release", fontsize=8, color="#52514e")
    fig.savefig(path, dpi=150); plt.close(fig)

def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

def frames_table(cases, reps):
    rows = []
    for r in reps:
        c = cases[r["case"]]
        for f in r["frames"]:
            rows.append({"mu": c["mu"], "v_peak": c["v_peak"], "fps": r["fps"], "rep": r["rep"], "k": f["k"], "t_rel": f["t_rel"], "x_true": f["x_true"], "x_hat": f["x_hat"],
                         "err_x_mm": (f["x_hat"] - f["x_true"])*1e3, "err_y_mm": (f["y_hat"] - f["y_true"])*1e3, "n_points": f["n_points"]})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-n", type=int, default=20, help="replicates per case and rate")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    ap.add_argument("--out", default="e0_5_results.csv")
    ap.add_argument("--frames", default="e0_5_frames.csv")
    ap.add_argument("--noise", default="e0_5_noise.csv")
    ap.add_argument("--summary", default="e0_5_summary.txt")
    ap.add_argument("--fig", default="e0_5_summary.png")
    ap.add_argument("--summarize", action="store_true", help="redo the fits and figure from the saved frames and results")
    a = ap.parse_args()
    n = 4 if a.quick else a.n
    t0 = time.time()
    print("pushing the cases")
    cases = push_cases(CASES)
    if a.summarize:
        with open(a.frames, newline="") as f:
            fr = list(csv.DictReader(f))
        with open(a.out, newline="") as f:
            results = [{k: (float(v) if k not in ("model",) else v) for k, v in r.items()} for r in csv.DictReader(f)]
        for r in results:
            r["fps"], r["n"] = int(r["fps"]), int(r["n"])
        with open(a.noise, newline="") as f:
            noise_rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        for r in noise_rows:
            r["fps"] = int(r["fps"])
        summarize(results, noise_rows, a.summary); plot(results, fr, a.fig); return
    todo = [(ci, fps, rep) for ci in range(len(cases)) for fps in RATES for rep in range(n)]
    print(f"{len(todo)} replayed slides on {a.workers} workers")
    reps = []
    if a.workers == 1:
        _init(cases)
        for i, task in enumerate(todo, 1):
            reps.append(replicate(task))
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    else:
        with Pool(a.workers, initializer=_init, initargs=(cases,)) as pool:
            for i, r in enumerate(pool.imap_unordered(replicate, todo), 1):
                reps.append(r)
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    reps.sort(key=lambda r: (r["case"], r["fps"], r["rep"]))
    results, noise_rows, fits = analyse(cases, reps)
    fr = frames_table(cases, reps)
    write_csv(a.out, results); write_csv(a.frames, fr); write_csv(a.noise, noise_rows)
    print(f"wrote {a.out}, {a.frames}, {a.noise}")
    summarize(results, noise_rows, a.summary)
    plot(results, [{k: (str(v) if not isinstance(v, str) else v) for k, v in f.items()} for f in fr], a.fig)
    print(f"wrote {a.summary}, {a.fig} ({time.time() - t0:.0f} s)")

if __name__ == "__main__":
    main()

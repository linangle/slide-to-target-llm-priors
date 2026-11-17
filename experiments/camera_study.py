#!/usr/bin/env python3
"""
camera_study.py -- where should the overhead camera be?  Coverage, frame noise and the friction estimate against the
camera's position and the perception window's margin, with the iiwa pushing in view.

Why.  Moving the camera for full coverage of the board (16 Nov) raised E0-5's worst friction bias from 0.9% to 6.9%.  The
cause is the viewing direction: a camera AHEAD of the paddle sees the paddle's front face, and with depth noise some of
those pixels fall inside the perception window (which starts FACE_MARGIN ahead of the face) and attach to the object's
blob while the gap between them is still a few pixels wide.  The position error then swings by about a millimetre for
30-40 ms after release, which a short slide cannot average out.  E0-5's original camera sat right above the paddle at
the release of its 0.8 m/s pushes, which hid the face; that was a property of those cases, not of the camera.

What is varied.
  camera   position (x along the push, height z; looking straight down, y = 0): scene.CAMERAS and others
  window   how the perception window treats pixels just ahead of the paddle face: the 0.5 mm margin E0-5 has always used,
           a 3 mm margin, or the 0.5 mm margin plus a height gate (within 4 mm of the face, keep only pixels at the
           object's top height; the paddle face's leaked pixels are spread over all heights)
  pushes   (mu, v_peak) that release at different places and slide different lengths, pushed by the arm at the default
           layout: a short high-friction slide, the E0-5 comparison case, a slow push, and the longest practice slide

Per camera, margin and push, at 60 Hz (decision 8): frames lost, whether the rest pose is seen, frame noise, the largest
mean position error over the frames (the transient), and the bias and standard error of observation model C's friction.

    python3 experiments/camera_study.py                 # 11 cameras x 3 windows x 4 pushes x 12 replays, ~35 min on 8 workers
    python3 experiments/camera_study.py --cameras "x0.90 z1.15" --append     # add or redo one position
    python3 experiments/camera_study.py --quick         # 3 cameras, 4 replays
Writes results/camera_study/camera_study.csv and camera_study_summary.txt.
"""
import argparse, csv, math, os, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import e0_5_observation as E5
from arm import BASE
from paths import RESULTS
from physics import BOARD_X_MAX, OBJECT_SIZE
from scene import CAMERAS, Camera

#           name                 (x,    y,   z)
POSITIONS = {"original":         CAMERAS["original"],          # (0.75, 0, 0.80): to 16 Nov
             "full":             CAMERAS["full"],              # (1.00, 0, 1.00): the first full-coverage setting
             "x0.80 z0.85":      (0.80, 0.0, 0.85),
             "x0.85 z0.90":      (0.85, 0.0, 0.90),
             "x0.90 z0.95":      (0.90, 0.0, 0.95),
             "x0.75 z1.00":      (0.75, 0.0, 1.00),
             "x0.70 z1.10":      (0.70, 0.0, 1.10),
             "x0.60 z1.20":      (0.60, 0.0, 1.20),           # behind nearly every release
             "x0.75 z1.40":      (0.75, 0.0, 1.40),           # behind every release, high enough for the whole board
             "x0.95 z1.05":      (0.95, 0.0, 1.05),           # the whole board from lower viewing angles than "full"
             "x0.90 z1.15":      (0.90, 0.0, 1.15)}
#          label                        window margin, m   face gate (depth, tol), m
WINDOWS = (("0.5 mm margin",            0.0005,            None),            # as E0-5 has always run
           ("3 mm margin",              0.003,             None),
           ("0.5 mm + height gate",     0.0005,            (0.004, 0.006)))  # near the face, keep only pixels at the object's top
#        (mu,   v_peak): what it tests
PUSHES = ((0.5,  0.8),     # short slide (7 cm, 0.14 s): the case that went from 0.9% to 6.9%
          (0.3,  0.8),     # E0-5's middle case
          (0.26, 0.5),     # a slow push: releases at x = 0.61 m, behind most cameras
          (0.14, 1.0))     # the longest practice slide: rests near x = 1.2 m
FPS = 60

def view_x(position, cam=None):
    """x range of the board plane the camera sees (it looks straight down; the image's long side is along x)."""
    cam = cam or Camera()
    half = position[2]*math.tan(cam.fov_y/2)*cam.width/cam.height
    return position[0] - half, position[0] + half

def run_config(cases, arm, position, margin, gate, n, workers):
    todo = [(ci, FPS, rep) for ci in range(len(cases)) for rep in range(n)]
    with Pool(workers, initializer=E5._init, initargs=(cases, arm, tuple(position), margin, gate)) as pool:
        reps = list(pool.imap_unordered(E5.replicate, todo, chunksize=2))
    results, noise, _ = E5.analyse(cases, reps)
    rows = []
    for ci, case in enumerate(cases):
        rs = [r for r in reps if r["case"] == ci]
        by_k = defaultdict(list)
        for r in rs:
            for f in r["frames"]:
                if math.isfinite(f["x_hat"]):
                    by_k[f["k"]].append(f["x_hat"] - f["x_true"])
        transient = max((abs(float(np.mean(v))) for k, v in by_k.items() if k > 0 and len(v) >= max(2, len(rs)//2)), default=float("nan"))
        nz = next(r for r in noise if r["mu"] == case["mu"] and r["v_peak"] == case["v_peak"])
        fit = {r["model"]: r for r in results if r["mu"] == case["mu"] and r["v_peak"] == case["v_peak"]}
        rest_seen = float(np.mean([math.isfinite(r["x_rest_hat"]) for r in rs]))
        rows.append({"mu": case["mu"], "v_peak": case["v_peak"], "x_release": case["x_sep"], "x_rest": case["x_rest"], "slide_cm": case["d_slide"]*100,
                     "frames": nz["n_frames"], "missed_frames": nz["missed_frames"], "rest_seen": rest_seen,
                     "sigma_f_mm": nz["sigma_f_mm"], "first_frame_bias_mm": nz["first_frame_bias_mm"], "transient_mm": transient*1e3,
                     "rest_bias_mm": nz["rest_bias_mm"], "sigma_rest_mm": nz["sigma_rest_mm"],
                     **{f"bias_{m}_pct": fit[m]["bias_eff_pct"] if m in fit else float("nan") for m in ("C", "C1")},
                     **{f"se_{m}_pct": fit[m]["se_pct"] if m in fit else float("nan") for m in ("C", "C1")}})
    return rows

def summarize(rows):
    L = ["camera study: the iiwa at the default layout pushes; frames at 60 Hz; model C = frames + rest pose + FK release point",
         "view x: what the camera sees of the board along the push.  transient: largest mean position error over the frames after the first.",
         "bias: observation model C's friction against the simulator's effective mu; C1 drops the first 30 ms of frames.", ""]
    names = list(dict.fromkeys(r["camera"] for r in rows))
    for margin in dict.fromkeys(r["window"] for r in rows):
        L.append(f"== perception window: {margin} ==")
        L.append(f"{'camera':<13}{'view x, m':>13} | " + " | ".join(f"mu {r['mu']:g} v {r['v_peak']:g}".center(31) for r in rows if r["camera"] == names[0] and r["window"] == margin)
                 + " | worst |bias C|")
        L.append(f"{'':<13}{'':>13} | " + " | ".join(f"{'lost':>4} {'sig_f':>5} {'trans':>5} {'bias C':>6} {'C1':>6}" for r in rows if r["camera"] == names[0] and r["window"] == margin) + " |")
        for name in names:
            rs = [r for r in rows if r["camera"] == name and r["window"] == margin]
            cells = []
            for r in rs:
                lost = "REST" if r["rest_seen"] < 1.0 else f"{r['missed_frames']:d}"
                cells.append(f"{lost:>4} {r['sigma_f_mm']:5.2f} {r['transient_mm']:5.2f} {r['bias_C_pct']:+6.2f} {r['bias_C1_pct']:+6.2f}")
            worst = max((abs(r["bias_C_pct"]) for r in rs if math.isfinite(r["bias_C_pct"])), default=float("nan"))
            L.append(f"{name:<13}{rs[0]['view_lo']:6.2f}..{rs[0]['view_hi']:.2f} | " + " | ".join(cells) + f" | {worst:6.2f}%")
        L.append("")
    L.append("lost: frames with no pose found (REST: the rest pose itself was not seen in some replays, so the camera does not cover that slide);")
    L.append("sig_f and trans in mm; biases in %.  Releases at x = " + ", ".join(f"{r['x_release']:.2f}" for r in rows if r["camera"] == names[0] and r["window"] == rows[0]["window"])
             + " m; rest at " + ", ".join(f"{r['x_rest']:.2f}" for r in rows if r["camera"] == names[0] and r["window"] == rows[0]["window"]) + " m.")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-n", type=int, default=12, help="replays per push")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    ap.add_argument("--out", default=str(RESULTS/"camera_study"/"camera_study"), help="path prefix of the CSV and the summary")
    ap.add_argument("--cameras", nargs="+", default=None, help="only these positions (names in POSITIONS)")
    ap.add_argument("--append", action="store_true", help="add to the rows already in the CSV (replacing those cameras) instead of starting over")
    a = ap.parse_args()
    positions = {k: POSITIONS[k] for k in (a.cameras or (("original", "full", "x0.75 z1.40") if a.quick else POSITIONS))}
    n = 4 if a.quick else a.n
    arm = {"base": BASE, "a_brake": None}
    t0 = time.time()
    print("pushing the cases with the arm")
    cases = E5.push_cases(PUSHES, arm)
    rows = []
    for name, pos in positions.items():
        lo, hi = view_x(pos)
        for label, margin, gate in WINDOWS:
            rs = run_config(cases, arm, pos, margin, gate, n, a.workers)
            for r in rs:
                r.update(camera=name, cam_x=pos[0], cam_z=pos[2], view_lo=lo, view_hi=hi, window=label)
            rows += rs
            print(f"  {name:<13} {label:<22}: worst |bias C| {max(abs(r['bias_C_pct']) for r in rs):.2f}%, "
                  f"worst transient {max(r['transient_mm'] for r in rs):.2f} mm ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.append and os.path.exists(a.out + ".csv"):
        with open(a.out + ".csv", newline="") as f:
            old = [{k: (float(v) if k not in ("camera", "window") else v) for k, v in r.items()} for r in csv.DictReader(f)]
        for r in old:
            r["missed_frames"], r["frames"] = int(r["missed_frames"]), int(r["frames"])
        rows = [r for r in old if r["camera"] not in positions] + rows
        order = {name: i for i, name in enumerate(POSITIONS)}
        rows.sort(key=lambda r: order.get(r["camera"], 99))
    head = ["camera", "cam_x", "cam_z", "view_lo", "view_hi", "window"]
    keys = head + [k for k in rows[0] if k not in head]
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    text = summarize(rows)
    print(text)
    open(a.out + "_summary.txt", "w").write(text + "\n")
    print(f"wrote {a.out}.csv, {a.out}_summary.txt ({time.time() - t0:.0f} s)")

if __name__ == "__main__":
    main()

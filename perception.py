#!/usr/bin/env python3
"""
perception.py -- landing pose of the object from the overhead depth camera.

Pipeline (proposal Sec. III-C; course notes Ch. 4):
  1. render a depth image from the scene's camera (Drake renders it noise-free);
  2. add stereo-camera depth noise, sigma = z^2 * subpixel / (f * baseline), and round to the
     sensor's 1 mm depth unit;
  3. back-project to a point cloud in the world frame;
  4. fit the table plane with RANSAC and keep points between 5 mm and 15 cm above it;
  5. keep the largest connected blob in the image (optionally inside an x window);
  6. fit the object's top-view outline to the blob's boundary pixels with 2-D ICP.  The object rests
     flat with vertical sides, so every boundary point seen from above lies on its footprint outline,
     and the pose reduces to (x, y, yaw).  Points inside the top face carry no information about
     position along the face, and in 3-D point-to-point ICP they held yaw at its starting value.
     Boundary pixels on a top edge sit on average half a pixel inside the edge, while those on a
     visible side face sit on it, so top-edge points are fitted to an outline inset by half a pixel.
The landing distance is the perceived x minus the release x.  Its error against the simulator's pose
is the perception error that Tier B replays.

Usage:
    python3 perception.py --study 50                   # 50 random resting poses per shape
    python3 perception.py --study 50 --no-noise        # noise-free depth: the algorithm's own error
    python3 perception.py --pair M07 --slide 0.8       # passive slide, then perceive where it stopped
"""
import argparse, csv, math, sys, time
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from pydrake.all import CameraInfo, DiagramBuilder, RigidTransform, RollPitchYaw, Simulator
from scene import DEFAULT_PAIRS, Camera, Obj, Table, build_scene, camera_pose, load_pairs, run_slide

@dataclass
class Noise:
    on: bool = True
    baseline: float = 0.055   # stereo baseline, m (RealSense D415)
    subpixel: float = 0.08    # disparity noise, px
    quantum: float = 0.001    # depth unit, m

def intrinsics(cam):
    return CameraInfo(cam.width, cam.height, cam.fov_y)

def render_depth(sc, root_context):
    port = sc.sensor.depth_image_32F_output_port()
    return port.Eval(sc.sensor.GetMyContextFromRoot(root_context)).data[:, :, 0].copy()

def add_noise(depth, cam, noise, rng):
    if not noise.on:
        return depth
    ok = np.isfinite(depth)
    sigma = depth[ok]**2*noise.subpixel/(intrinsics(cam).focal_y()*noise.baseline)
    out = depth.copy()
    out[ok] = np.round((depth[ok] + sigma*rng.standard_normal(ok.sum()))/noise.quantum)*noise.quantum
    return out

def to_points(depth, cam):
    """World point for every pixel, shape (h, w, 3), NaN where the depth is invalid.  Drake's renderer
    samples pixel u at u + 0.5 from CameraInfo's principal point; without the half pixel, positions
    come out 0.65 mm off at 0.75 m."""
    info = intrinsics(cam)
    v, u = np.mgrid[0:cam.height, 0:cam.width]
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    p_C = np.stack([(u + 0.5 - info.center_x())*z/info.focal_x(), (v + 0.5 - info.center_y())*z/info.focal_y(), z], -1)
    X = camera_pose(cam)
    return p_C @ X.rotation().matrix().T + X.translation()

def fit_plane(P, rng, tol=0.005, iters=100):
    """Table plane n.p + c = 0 (n pointing up): RANSAC on a subsample, refit by least squares on its inliers."""
    Q = P[rng.choice(len(P), min(len(P), 4000), replace=False)]
    best = (-1, None, None)
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n/np.linalg.norm(n)*(1 if n[2] >= 0 else -1)
        count = np.count_nonzero(np.abs((Q - a) @ n) < tol)
        if count > best[0]:
            best = (count, n, -n @ a)
    _, n, c = best
    inl = P[np.abs(P @ n + c) < tol]
    centroid = inl.mean(axis=0)
    n = np.linalg.svd(inl - centroid, full_matrices=False)[2][2]
    n = n if n[2] >= 0 else -n
    return n, -n @ centroid

def segment(P_img, plane, band=(0.005, 0.15), x_window=None):
    """(points, boundary points) of the largest connected image blob between band[0] and band[1]
    above the plane; the boundary includes the edges of holes (the ring's inside)."""
    n, c = plane
    with np.errstate(invalid="ignore"):
        h = P_img @ n + c
        mask = (h > band[0]) & (h < band[1])
        if x_window:
            mask &= (P_img[..., 0] > x_window[0]) & (P_img[..., 0] < x_window[1])
    labels, k = ndimage.label(mask)
    if k == 0:
        return None
    sizes = ndimage.sum(mask, labels, range(1, k + 1))
    blob = labels == 1 + int(np.argmax(sizes))
    return P_img[blob], P_img[blob & ~ndimage.binary_erosion(blob)]

def top_height(obj):
    return obj.size[2] + (obj.foot[1] + max(obj.foot_error, 0.0) if obj.shape == "footed" else 0.0)

def outline(obj, inset=0.0, spacing=0.00025):
    """Top-view outline of the object in its body frame (xy), about `spacing` apart, moved `inset`
    into the material."""
    sx, sy = obj.size[0] - 2*inset, obj.size[1] - 2*inset
    if obj.shape in ("disc", "ring"):
        rings = [sx/2] + ([obj.inner/2 + inset] if obj.shape == "ring" else [])
        return np.vstack([np.c_[r*np.cos(t), r*np.sin(t)]
                          for r in rings for t in [np.linspace(0, 2*math.pi, round(2*math.pi*r/spacing), endpoint=False)]])
    ux, uy = np.arange(-sx/2, sx/2, spacing), np.arange(-sy/2, sy/2, spacing)   # box, or a footed block's top
    return np.vstack([np.c_[ux, np.full_like(ux, -sy/2)], np.c_[ux + spacing, np.full_like(ux, sy/2)],
                      np.c_[np.full_like(uy, sx/2), uy], np.c_[np.full_like(uy, -sx/2), uy + spacing]])

def icp_outline(groups, init, max_iter=100, max_dist=0.01, tol=1e-7):
    """(x, y, yaw) placing outlines (body frame) on boundary points (world xy) by point-to-point ICP.
    groups = [(boundary points, outline), ...], each set matched to its own outline, solved together."""
    groups = [(b, m, cKDTree(m)) for b, m in groups if len(b)]
    x, y, th = init
    for it in range(1, max_iter + 1):
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        ms, ps, ds = [], [], []
        for b, model, tree in groups:
            dist, idx = tree.query((b - [x, y]) @ R, distance_upper_bound=max_dist)   # in the body frame
            keep = np.isfinite(dist)
            ms.append(model[idx[keep]]); ps.append(b[keep]); ds.append(dist[keep])
        m, p, dist = np.vstack(ms), np.vstack(ps), np.concatenate(ds)
        mc, pc = m.mean(axis=0), p.mean(axis=0)
        H = (m - mc).T @ (p - pc)                                                          # 2-D Kabsch
        th_new = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        R = np.array([[math.cos(th_new), -math.sin(th_new)], [math.sin(th_new), math.cos(th_new)]])
        x_new, y_new = pc - R @ mc
        step = abs(x_new - x) + abs(y_new - y) + 0.05*abs(th_new - th)
        x, y, th = x_new, y_new, th_new
        if step < tol:
            break
    return {"x": x, "y": y, "yaw": th, "rmse": float(np.sqrt(np.mean(dist**2))), "iters": it,
            "matched": len(dist)/sum(len(b) for b, _, _ in groups)}

def perceive(depth, cam, obj, rng, x_window=None):
    """Object pose (x, y, yaw) from a depth image, or None if no object blob is found."""
    P = to_points(depth, cam)
    n, c = plane = fit_plane(P[np.isfinite(P[..., 2])], rng)
    found = segment(P, plane, x_window=x_window)
    if found is None or len(found[0]) < 50:
        return None
    blob, edge = found
    top = edge @ n + c > top_height(obj) - 0.003                       # top-edge points vs side-face points
    half_px = 0.5*(cam.position[2] + c/n[2] - top_height(obj))/intrinsics(cam).focal_y()
    est = icp_outline([(edge[top, :2], outline(obj, inset=half_px)), (edge[~top, :2], outline(obj))],
                      (*blob[:, :2].mean(axis=0), 0.0))
    est["n_points"], est["n_edge"] = len(blob), len(edge)
    return est

def yaw_error(est, true, shape):
    """Yaw error in degrees, modulo the shape's symmetry; NaN for round shapes (yaw is unobservable)."""
    if shape in ("disc", "ring"):
        return float("nan")
    e = math.degrees(est - true)
    return (e + 45) % 90 - 45

def study(n, shapes, cam, noise, seed):
    """n random resting poses per shape; returns one row per pose."""
    rng, rows = np.random.default_rng(seed), []
    for shape in shapes:
        obj = Obj(shape=shape)
        builder = DiagramBuilder()
        sc = build_scene(builder, {"mu_k_nominal": 0.3, "s_tab_log": 0.0}, obj, Table(patchiness=0.0), arm=False,
                         camera=cam)
        ctx = builder.Build().CreateDefaultContext()
        pctx, z0 = sc.plant.GetMyMutableContextFromRoot(ctx), sc.X_WO.translation()[2]
        for _ in range(n):
            # landing zone; lateral offset and yaw about as large as passive slides and E0 produce
            x, y, yaw = rng.uniform(0.6, 1.0), rng.uniform(-0.005, 0.005), math.radians(rng.uniform(-2, 2))
            sc.plant.SetFreeBodyPose(pctx, sc.body, RigidTransform(RollPitchYaw(0, 0, yaw), [x, y, z0]))
            t0 = time.time()
            est = perceive(add_noise(render_depth(sc, ctx), cam, noise, rng), cam, obj, rng)
            row = {"shape": shape, "noise": int(noise.on), "x_true": x, "y_true": y, "yaw_true_deg": math.degrees(yaw)}
            if est is None:
                row["error"] = "no object found"
            else:
                row.update({"err_x_mm": (est["x"] - x)*1e3, "err_y_mm": (est["y"] - y)*1e3,
                            "err_yaw_deg": yaw_error(est["yaw"], yaw, shape), "rmse_mm": est["rmse"]*1e3,
                            "n_points": est["n_points"], "n_edge": est["n_edge"], "iters": est["iters"],
                            "matched": est["matched"]})
            row["wall_s"] = time.time() - t0
            rows.append(row)
    return rows

def summarize(rows):
    print(f"{'shape':<8}{'n':>4}  {'x error mm (bias / SD / max)':>30}  {'y error mm':>22}  {'yaw error deg':>22}  wall s")
    for shape in dict.fromkeys(r["shape"] for r in rows):
        rs = [r for r in rows if r["shape"] == shape and "err_x_mm" in r]
        cells = []
        for k in ("err_x_mm", "err_y_mm", "err_yaw_deg"):
            e = np.array([r[k] for r in rs], float)
            e = e[np.isfinite(e)]
            cells.append(f"{e.mean():+.2f} / {e.std(ddof=1):.2f} / {np.abs(e).max():.2f}" if len(e) > 1 else "n/a")
        missed = sum(1 for r in rows if r["shape"] == shape) - len(rs)
        print(f"{shape:<8}{len(rs):>4}  {cells[0]:>30}  {cells[1]:>22}  {cells[2]:>22}  "
              f"{np.mean([r['wall_s'] for r in rs]):.2f}" + (f"  ({missed} missed)" if missed else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, metavar="N", help="N random resting poses per shape")
    ap.add_argument("--shapes", default="box,disc,ring,footed")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="perception_results.csv")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--pair", default="M01")
    ap.add_argument("--shape", default="box", choices=["box", "disc", "ring", "footed"])
    ap.add_argument("--slide", type=float, metavar="V0", help="passive slide at this speed, then perceive")
    a = ap.parse_args()
    cam, noise = Camera(), Noise(on=not a.no_noise)
    if a.study:
        rows = study(a.study, a.shapes.split(","), cam, noise, a.seed)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
        print(f"wrote {a.out} ({'noisy' if noise.on else 'noise-free'} depth)")
        summarize(rows)
    elif a.slide is not None:
        pair, obj = load_pairs(a.pairs)[a.pair], Obj(shape=a.shape)
        builder = DiagramBuilder()
        sc = build_scene(builder, pair, obj, Table(), arm=False, camera=cam)
        sim = Simulator(builder.Build())
        X0, X1 = run_slide(sim, sc, a.slide)
        rng = np.random.default_rng(a.seed)
        est = perceive(add_noise(render_depth(sc, sim.get_context()), cam, noise, rng), cam, obj, rng)
        if est is None:
            sys.exit("no object found in the depth image")
        d, d_hat = X1.translation()[0] - X0.translation()[0], est["x"] - X0.translation()[0]
        print(f"{a.pair} {a.shape} v0={a.slide} m/s: true d = {d*100:.2f} cm, perceived d = {d_hat*100:.2f} cm "
              f"(error {(d_hat - d)*1e3:+.2f} mm), lateral error {(est['y'] - X1.translation()[1])*1e3:+.2f} mm, "
              f"{est['n_points']} points, ICP rmse {est['rmse']*1e3:.2f} mm")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()

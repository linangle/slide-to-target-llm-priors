#!/usr/bin/env python3
"""
arm_clearance.py -- does the iiwa stay clear of the board, the object and the camera during a push?

The scene loads the iiwa without collision geometry (iiwa14_no_collision.sdf: only the paddle touches anything), so the
simulation cannot show a link passing through the table.  This check plans the push as arm.ArmPusher does, poses the arm
along the planned joint path, and takes the convex hull of each link's visual mesh (the link's real shape, slightly
enlarged where it is concave, so the clearances are on the safe side).  Per link, over the whole stroke:

  plane     lowest point of the link above z = 0, the plane of the board's top (scene.Table: x 0.30..1.52 m, 1.2 m wide,
            centred on scene.BOARD_Y for the layout).
            Link 0 is the base and stands on that plane; for it the question is whether its footprint overlaps the
            board, reported apart.  The paddle is meant to run 3 mm above the board.
  object    least distance to the object riding ahead of the paddle (a lower bound; the paddle itself touches it)
  camera    least distance to the camera's position (scene.Camera)

    python3 experiments/arm_clearance.py                        # the default layout at its speed cap
    python3 experiments/arm_clearance.py --layout origin        # the frontal layout
    python3 experiments/arm_clearance.py --base-x 0.6 --base-y 0.75 --v 1.0
"""
import argparse, math
import numpy as np
from scipy.spatial import ConvexHull
from pydrake.all import AddMultibodyPlantSceneGraph, Box, DiagramBuilder, Mesh, RigidTransform, Role
from physics import BOARD_X_MAX, BOARD_X_MIN, OBJECT_SIZE
from scene import BOARD_Y, Camera, Table, add_arm, board_y, layout_of
from push import profile
from arm import A_BRAKE, POSTURE, ArmModel, ArmPusher, layout_base, limited_peak_speed, link7_pose

def plan_path(base, v_peak, J=100.0, a_brake=A_BRAKE, x0=0.55, posture=POSTURE):
    """Joint path of the push ArmPusher would make: (s, q(s)) over the stroke, and the pusher with its plan."""
    pu = ArmPusher(base=tuple(base), a_brake_max=a_brake, posture=posture)
    pu.x_touch0, pu.model = x0 - OBJECT_SIZE[0]/2 - pu.gap, ArmModel(tuple(base))
    pu.X0 = link7_pose(pu.x_touch0, 0.0, 0.0, pu.clearance)
    pu.make_plan(profile(v_peak, J, a_max=a_brake))
    return pu.plan["s"], pu.plan["q"], pu

def link_hulls(base):
    """(plant, context, iiwa, [(link name, body, X_BG, hull vertices in G, hull face planes [n | d] in G)]) from the visual meshes."""
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, 0.0)
    iiwa, _ = add_arm(plant, None, RigidTransform(np.asarray(base, float)), collision=False)
    plant.Finalize()
    insp, parts = sg.model_inspector(), []
    for gid in insp.GetAllGeometryIds(Role.kIllustration):
        shape, body = insp.GetShape(gid), plant.GetBodyFromFrameId(insp.GetFrameId(gid))
        if isinstance(shape, Mesh):
            hull = shape.GetConvexHull()                       # in the geometry frame, scale applied
            V = np.array([hull.vertex(i) for i in range(hull.num_vertices())])
        elif isinstance(shape, Box):
            V = 0.5*np.array([[sx*shape.width(), sy*shape.depth(), sz*shape.height()] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        else:
            continue
        name = "paddle" if body.name() == "paddle" else body.name().replace("iiwa_link_", "link ")
        parts.append((name, body, insp.GetPoseInFrame(gid), V, ConvexHull(V).equations))
    return plant, plant.CreateDefaultContext(), iiwa, parts

def _box_points(centre, size, n=5):
    """Points on the surface of an axis-aligned box (a grid on each face)."""
    g = np.linspace(-0.5, 0.5, n)
    pts = [(a, b, c) for a in g for b in g for c in g if max(abs(a), abs(b), abs(c)) == 0.5]
    return np.asarray(centre, float) + np.array(pts)*np.asarray(size, float)

def clearance(base, v_peak, J=100.0, a_brake=A_BRAKE, x0=0.55, step=5, camera=None, board_centre=None, posture=POSTURE):
    """Least clearances of each link along the planned push: dict(links -> {plane, object, camera}, base footprint, plan)."""
    s, q, pu = plan_path(base, v_peak, J, a_brake, x0, posture)
    cam = np.asarray((camera or Camera()).position, float)
    plant, ctx, iiwa, parts = link_hulls(base)
    names = list(dict.fromkeys(name for name, *_ in parts))
    out = {name: {"plane": math.inf, "object": math.inf, "camera": math.inf} for name in names}
    foot = None
    for k in sorted(set(list(range(0, len(s), step)) + [len(s) - 1])):
        plant.SetPositions(ctx, iiwa, q[k])
        # the object rides in contact with the paddle face: its centre is half a length ahead of the face
        obj = _box_points([pu.x_touch0 + s[k] + pu.gap + OBJECT_SIZE[0]/2, 0.0, OBJECT_SIZE[2]/2], OBJECT_SIZE)
        for name, body, X_BG, V, planes in parts:
            X = plant.EvalBodyPoseInWorld(ctx, body) @ X_BG
            R, t = X.rotation().matrix(), X.translation()
            W = V @ R.T + t
            d = out[name]
            d["plane"] = min(d["plane"], float(W[:, 2].min()))
            for key, P in (("object", obj), ("camera", cam[None, :])):
                local = (P - t) @ R                                   # the points in the geometry frame
                sd = (local @ planes[:, :3].T + planes[:, 3]).max(axis=1)   # > 0 outside the hull; a lower bound on the distance
                d[key] = min(d[key], float(sd.min()))
            if name == "link 0" and foot is None:
                foot = W[W[:, 2] < t[2] + 0.03]                     # the vertices near the mounting face
    centre = board_y(base) if board_centre is None else board_centre
    table = Table()
    edge = centre + math.copysign(table.width/2, base[1] or 1.0)      # the board's side edge nearest the base
    fx, fy = (float(foot[:, 0].min()), float(foot[:, 0].max())), (float(foot[:, 1].min()), float(foot[:, 1].max()))
    beside = BOARD_X_MIN < base[0] < BOARD_X_MAX              # level with the board along x: the side edge is the one that matters
    overlap = (abs(edge) - min(abs(fy[0]), abs(fy[1]))) if beside else (fx[1] - BOARD_X_MIN)
    return {"links": out, "stroke": float(s[-1]), "plan": pu.plan, "base": tuple(base), "foot_x": fx, "foot_y": fy,
            "beside": beside, "foot_overlap": float(overlap), "edge": float(edge), "board_centre": float(centre), "camera": tuple(round(float(v), 3) for v in cam)}

def report(c, v_peak, a_brake):
    b = c["base"]
    L = [f"base at ({b[0]:g}, {b[1]:g}) m; push at {v_peak:.3f} m/s, brake cap {a_brake:g} m/s^2, stroke {c['stroke']*1e3:.0f} mm "
         f"(joint speed {c['plan']['qd_ratio']:.2f} x limit, joint {c['plan']['qd_ratio_joint']})"
         + (f"; start posture chosen by {c['plan']['posture']}" if c["plan"].get("posture", "speed") != "speed" else ""),
         f"{'':8}{'above z=0, mm':>15}{'to object, mm':>15}{'to camera, mm':>15}"]
    for name, d in c["links"].items():
        L.append(f"{name:8}{d['plane']*1e3:>15.0f}{d['object']*1e3:>15.0f}{d['camera']*1e3:>15.0f}")
    moving = {k: v for k, v in c["links"].items() if k not in ("link 0", "paddle")}
    lo, near = min(moving, key=lambda k: moving[k]["plane"]), min(moving, key=lambda k: moving[k]["object"])
    L.append(f"lowest moving link: {lo}, {moving[lo]['plane']*1e3:.0f} mm above the board's plane; nearest to the object: {near}, "
             f"{moving[near]['object']*1e3:.0f} mm; nearest any link comes to the camera at {c['camera']}: {min(v['camera'] for v in c['links'].values())*1e3:.0f} mm")
    edge = f"side edge (y = {c['edge']:+.2f} m, board centred on y = {c['board_centre']:+.2f} m)" if c["beside"] else "near edge (x = 0.30 m)"
    L.append(f"base footprint: x {c['foot_x'][0]:.3f}..{c['foot_x'][1]:.3f} m, y {c['foot_y'][0]:.3f}..{c['foot_y'][1]:.3f} m: "
             + (f"it OVERLAPS the board's {edge} by {c['foot_overlap']*1e3:.0f} mm" if c["foot_overlap"] > 0
                else f"it clears the board's {edge} by {-c['foot_overlap']*1e3:.0f} mm"))
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layout", default=None, help="origin or side (arm.LAYOUTS); default: the default layout")
    ap.add_argument("--base-x", type=float, default=None); ap.add_argument("--base-y", type=float, default=None)
    ap.add_argument("--v", type=float, default=None, help="peak speed, m/s (default: the layout's joint-speed bound)")
    ap.add_argument("--J", type=float, default=100.0)
    ap.add_argument("--a-brake", type=float, default=A_BRAKE)
    ap.add_argument("--posture", choices=["speed", "limits"], default=POSTURE, help="how the arm chooses its start posture (arm.POSTURES)")
    ap.add_argument("--board-y", type=float, default=None, help="y of the board's centre line (default: the layout's, scene.BOARD_Y)")
    ap.add_argument("--camera", choices=["original", "full"], default="full", help="camera position to measure against (scene.CAMERAS)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    a = ap.parse_args()
    from arm import BASE
    base = (a.base_x, a.base_y, 0.0) if a.base_x is not None else (layout_base(a.layout) if a.layout else BASE)
    v = a.v if a.v is not None else limited_peak_speed(a.J, base=tuple(base), a_max=a.a_brake, posture=a.posture)
    from scene import CAMERAS
    text = report(clearance(base, v, a.J, a.a_brake, camera=Camera(position=CAMERAS[a.camera]), board_centre=a.board_y, posture=a.posture), v, a.a_brake)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")

if __name__ == "__main__":
    main()

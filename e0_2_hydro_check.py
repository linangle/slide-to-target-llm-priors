#!/usr/bin/env python3
"""
e0_2_hydro_check.py -- E0-2: is every contact a hydroelastic patch, not a point-contact fallback?

With two rigid hydroelastic geometries, Drake's default `hydroelastic_with_fallback` silently switches that
pair to point contact; if mesh elements are much larger than the contact surface, contact also degenerates
to a point.  This check builds the scenes from sim_config.yaml (strict `hydroelastic`: a rigid-rigid pair
throws) and queries the contact results at EVERY time step:
  * passive: the object slides on the strips at 0.8 m/s (object-table contact);
  * push: the iiwa is posed by IK with the paddle upright behind the object and its joints locked; the object
    is driven into the paddle face (paddle-object contact);
for each object shape, in strict and in fallback mode.  Pass: zero point-pair contacts involving the paddle,
the object or the table, and at least one hydroelastic contact of each kind.  Two negative controls show what
the check guards against: (1) the paddle lowered 1 mm into the table (rigid on rigid) makes strict mode throw;
(2) a free RIGID box on the rigid table in fallback mode runs silently on point contact.  (Control 1 cannot show
the point pair in fallback mode: a contact between a locked arm and the world has no participating degree of
freedom, so Drake's discrete solver drops it from the contact results even though the geometries overlap.)

Writes e0_2_results.csv and e0_2_contacts.png.
    python3 e0_2_hydro_check.py            # all scenes, ~1 min
    python3 e0_2_hydro_check.py --shapes box --steps 100
"""
import argparse, csv, math, sys, time
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, InverseKinematics, RotationMatrix, Simulator,
                         Solve, SpatialVelocity)
import scene
from scene import PADDLE, Obj, Table, build_scene
from drake_config import contact_from_config, contact_pairs, describe, load_config

SHAPES = ("box", "disc", "ring", "footed")

def arm_pose_behind(x0, obj, face_gap=0.0, bottom_clearance=0.003):
    """iiwa joint angles with the paddle upright, face along +x, the face `face_gap` behind the object's back
    face (negative = overlap) and the plate's lower edge `bottom_clearance` above the table."""
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
    _, paddle = scene.add_arm(plant, scene.Contact())
    plant.Finalize()
    p = np.array([x0 - obj.size[0]/2 - face_gap - PADDLE[2]/2, 0, bottom_clearance + PADDLE[1]/2])
    R = RotationMatrix(np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]]))   # paddle x, y, z along world y, z, x
    ik = InverseKinematics(plant, plant.CreateDefaultContext())
    ik.AddPositionConstraint(paddle.body_frame(), np.zeros(3), plant.world_frame(), p - 1e-4, p + 1e-4)
    ik.AddOrientationConstraint(plant.world_frame(), R, paddle.body_frame(), RotationMatrix(), 1e-3)
    q_nom = np.array([0, 0.8, 0, -1.8, 0, 0.8, math.pi/2])
    ik.prog().AddQuadraticErrorCost(np.diag([10, 0.1, 10, 0.1, 10, 0.1, 1]), q_nom, ik.q())
    ik.prog().SetInitialGuess(ik.q(), q_nom)
    res = Solve(ik.prog())
    if not res.is_success():
        raise RuntimeError(f"no arm pose puts the paddle behind an object at x0 = {x0}")
    return res.GetSolution(ik.q())

def classify(kind, a, b):
    """'paddle-object', 'object-table', 'paddle-table' or 'other' from two geometry or body names."""
    tag = lambda n: "paddle" if n.endswith("paddle") else "object" if n.startswith("object") else "table" if n.startswith("strip") or n == "world" else n
    return "-".join(sorted((tag(a), tag(b)), key=("paddle", "object", "table").index if all(t in ("paddle", "object", "table") for t in (tag(a), tag(b))) else str))

def run_scene(cfg, shape, arm, contact_model, steps=300, x0=0.55, v_approach=-0.1, bottom_clearance=0.003):
    contact = contact_from_config(cfg, contact_model=contact_model)
    obj = Obj(shape=shape)
    row = {"scene": "push" if arm else "passive", "shape": shape, "contact_model": contact_model, "bottom_clearance_mm": bottom_clearance*1e3,
           "steps": steps, "steps_hydro_object_table": 0, "steps_hydro_paddle_object": 0, "point_pair_contacts": 0,
           "point_pairs": "", "hydro_pairs": "", "error": ""}
    t0 = time.time()
    try:
        builder = DiagramBuilder()
        sc = build_scene(builder, {"mu_k_nominal": 0.3, "s_tab_log": 0.0}, obj, Table(patchiness=0.0), contact, arm=arm, x0=x0)
        sim = Simulator(builder.Build())
        plant = sc.plant
        ctx = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
        inspector = sc.scene_graph.model_inspector()
        if arm:
            plant.SetPositions(ctx, sc.iiwa, arm_pose_behind(x0, obj, 0.0, bottom_clearance))
            for i in range(1, 8):
                plant.GetJointByName(f"iiwa_joint_{i}", sc.iiwa).Lock(ctx)
            plant.SetFreeBodySpatialVelocity(ctx, sc.body, SpatialVelocity(np.zeros(3), [v_approach, 0, 0]))
        else:
            plant.SetFreeBodySpatialVelocity(ctx, sc.body, SpatialVelocity(np.zeros(3), [0.8, 0, 0]))
        port = plant.get_contact_results_output_port()
        points, hydros = set(), set()
        for k in range(steps):
            sim.AdvanceTo((k + 1)*contact.time_step)
            kinds = {"paddle-object": False, "object-table": False}
            for kind, a, b in contact_pairs(port.Eval(ctx), plant, inspector):
                c = classify(kind, a, b)
                if kind == "point":
                    row["point_pair_contacts"] += 1; points.add(c)
                else:
                    hydros.add(c); kinds[c] = True if c in kinds else kinds.get(c, False)
            row["steps_hydro_object_table"] += kinds["object-table"]
            row["steps_hydro_paddle_object"] += kinds["paddle-object"]
        row["point_pairs"], row["hydro_pairs"] = ";".join(sorted(points)), ";".join(sorted(hydros))
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:200]
    row["wall_s"] = f"{time.time() - t0:.1f}"
    needed = row["steps_hydro_object_table"] > 0 and (not arm or row["steps_hydro_paddle_object"] > 0)
    row["pass"] = int(row["point_pair_contacts"] == 0 and needed and not row["error"])
    return row

def rigid_box_control(cfg, contact_model, steps=50):
    """A free rigid-hydroelastic box dropped onto a rigid table, outside scene.py: strict mode throws, fallback
    mode silently uses point contact.  Returns a result row."""
    from pydrake.all import (AddContactMaterial, AddMultibodyPlant, AddRigidHydroelasticProperties, Box, CoulombFriction,
                             MultibodyPlantConfig, ProximityProperties, RigidTransform, SpatialInertia)
    contact = contact_from_config(cfg, contact_model=contact_model)
    row = {"scene": "negative control: rigid box", "shape": "box", "contact_model": contact_model, "bottom_clearance_mm": "",
           "steps": steps, "steps_hydro_object_table": 0, "steps_hydro_paddle_object": 0, "point_pair_contacts": 0,
           "point_pairs": "", "hydro_pairs": "", "error": ""}
    t0 = time.time()
    try:
        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlant(MultibodyPlantConfig(time_step=contact.time_step, contact_model=contact_model,
                                                          discrete_contact_approximation=contact.approx), builder)
        def rigid(name):
            p = ProximityProperties(); AddRigidHydroelasticProperties(p)
            AddContactMaterial(contact.dissipation, None, CoulombFriction(0.3, 0.3), p); return p
        plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform([0, 0, -0.05]), Box(1, 1, 0.1), "table", rigid("table"))
        body = plant.AddRigidBody("object", SpatialInertia.SolidBoxWithMass(0.5, 0.1, 0.1, 0.05))
        plant.RegisterCollisionGeometry(body, RigidTransform(), Box(0.1, 0.1, 0.05), "object_0", rigid("object"))
        plant.Finalize()
        sim = Simulator(builder.Build())
        ctx = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
        plant.SetFreeBodyPose(ctx, body, RigidTransform([0, 0, 0.0249]))
        port = plant.get_contact_results_output_port()
        for k in range(steps):
            sim.AdvanceTo((k + 1)*contact.time_step)
            res = port.Eval(ctx)
            row["point_pair_contacts"] += res.num_point_pair_contacts()
            row["steps_hydro_object_table"] += res.num_hydroelastic_contacts() > 0
        row["point_pairs"] = "object-table" if row["point_pair_contacts"] else ""
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:200]
    row["wall_s"] = f"{time.time() - t0:.1f}"
    row["pass"] = int(bool(row["error"])) if contact_model == "hydroelastic" else int(row["point_pair_contacts"] > 0)
    return row

def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if not r["scene"].startswith("negative control")]
    labels = [f"{r['scene']} {r['shape']} ({'strict' if r['contact_model'] == 'hydroelastic' else 'fallback'})" for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.32*len(rows) + 1.4))
    ax.barh(y - 0.2, [r["steps_hydro_object_table"] for r in rows], 0.38, color="#2a78d6", label="steps with object-table hydroelastic contact")
    ax.barh(y + 0.2, [r["steps_hydro_paddle_object"] for r in rows], 0.38, color="#1baf7a", label="steps with paddle-object hydroelastic contact")
    pts = [r["point_pair_contacts"] for r in rows]
    ax.scatter(pts, y, color="#eb6834", zorder=3, s=28, label="point-pair contacts (must be 0)")
    ax.set_yticks(y, labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel(f"time steps (of {rows[0]['steps']})"); ax.set_title("E0-2: contact representation at every step")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="e0_2_results.csv")
    ap.add_argument("--fig", default="e0_2_contacts.png")
    ap.add_argument("--no-negative-control", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    print(describe(contact_from_config(cfg)))
    rows = []
    for model in ("hydroelastic", "hydroelastic_with_fallback"):
        for shape in a.shapes.split(","):
            for arm in (False, True):
                r = run_scene(cfg, shape, arm, model, a.steps)
                rows.append(r)
                print(f"{r['scene']:<8}{shape:<7}{model:<27} hydro object-table {r['steps_hydro_object_table']:>4} steps, "
                      f"paddle-object {r['steps_hydro_paddle_object']:>4} steps, point pairs {r['point_pair_contacts']:>3} "
                      f"{r['point_pairs']} {'PASS' if r['pass'] else 'FAIL ' + r['error']}")
    if not a.no_negative_control:
        r = run_scene(cfg, "box", True, "hydroelastic", min(a.steps, 50), bottom_clearance=-0.001)
        r["scene"], r["pass"] = "negative control: paddle into table", int(bool(r["error"]))   # strict must throw
        rows.append(r)
        print(f"negative control, paddle 1 mm into the table, strict: " + (f"threw: {r['error'][:110]}" if r["error"] else "no exception")
              + f" -> {'as expected' if r['pass'] else 'UNEXPECTED'}")
        for model in ("hydroelastic", "hydroelastic_with_fallback"):
            r = rigid_box_control(cfg, model, min(a.steps, 50))
            rows.append(r)
            print(f"negative control, rigid box on rigid table, {model}: "
                  + (f"threw: {r['error'][:110]}" if r["error"] else f"point-pair contacts {r['point_pair_contacts']} over {r['steps']} steps, hydro steps {r['steps_hydro_object_table']}")
                  + f" -> {'as expected' if r['pass'] else 'UNEXPECTED'}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot(rows, a.fig)
    ok = all(r["pass"] for r in rows)
    print(f"wrote {a.out}, {a.fig}; E0-2 {'PASS' if ok else 'FAIL'}: "
          f"{sum(r['point_pair_contacts'] for r in rows if not r['scene'].startswith('negative control'))} point-pair contacts in the real scenes")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

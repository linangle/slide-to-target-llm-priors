#!/usr/bin/env python3
"""
check_drake.py -- smoke test for the Drake environment.

1. Loads the iiwa 14 model (checks pydrake and the drake_models package download).
2. Slides a compliant-hydroelastic box on a rigid table: the box settles, is given a speed
   v0 along +x, and slides to rest.  The stopping distance should match the release model
       d = v0^2 / (2 mu g)
   within the proposal's E0 bar (2% in distance, 2 deg of yaw).  Exits 1 if it does not.

Usage:
    python3 check_drake.py                             # mu=0.30, v0=0.8 m/s, 1 ms step, lagged
    python3 check_drake.py --mu 0.15 --v0 1.0 --approx sap
    python3 check_drake.py --meshcat                   # replay the slide in the browser
"""
import argparse, math, sys
import numpy as np
from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddDefaultVisualization,
                         AddMultibodyPlant, AddRigidHydroelasticProperties, Box, CoulombFriction,
                         DiagramBuilder, MultibodyPlant, MultibodyPlantConfig, Parser,
                         ProximityProperties, RigidTransform, RollPitchYaw, Simulator, SpatialInertia,
                         SpatialVelocity, StartMeshcat)

G = 9.81
IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf"
BOX = (0.10, 0.10, 0.05)   # m; squat, flat base, as in the proposal
MASS = 0.5                 # kg; cancels in d, kept realistic for the contact stiffness

def check_iiwa():
    plant = MultibodyPlant(0.0)
    Parser(plant).AddModelsFromUrl(IIWA_URL)
    plant.Finalize()
    return plant.num_positions()

def slide(mu, v0, time_step, approx, modulus, dissipation, meshcat=None):
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlant(MultibodyPlantConfig(time_step=time_step, contact_model="hydroelastic",
                                                      discrete_contact_approximation=approx), builder)
    friction = CoulombFriction(mu, mu)   # same mu on both bodies, so the pair's mu is mu
    table = ProximityProperties()
    AddRigidHydroelasticProperties(table)
    AddContactMaterial(dissipation, None, friction, table)
    X_table, table_box = RigidTransform([1.0, 0, -0.05]), Box(4.0, 1.0, 0.1)   # top face at z = 0
    plant.RegisterCollisionGeometry(plant.world_body(), X_table, table_box, "table", table)
    plant.RegisterVisualGeometry(plant.world_body(), X_table, table_box, "table", [0.8, 0.7, 0.6, 1.0])
    body = plant.AddRigidBody("box", SpatialInertia.SolidBoxWithMass(MASS, *BOX))
    props = ProximityProperties()
    AddCompliantHydroelasticProperties(0.01, modulus, props)
    AddContactMaterial(dissipation, None, friction, props)
    plant.RegisterCollisionGeometry(body, RigidTransform(), Box(*BOX), "box", props)
    plant.RegisterVisualGeometry(body, RigidTransform(), Box(*BOX), "box", [0.3, 0.5, 0.9, 1.0])
    plant.Finalize()
    if meshcat:
        AddDefaultVisualization(builder, meshcat)
    sim = Simulator(builder.Build())
    ctx = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
    plant.SetFreeBodyPose(ctx, body, RigidTransform([0, 0, BOX[2]/2]))
    if meshcat:
        meshcat.StartRecording()
    t = 0.3
    sim.AdvanceTo(t)                     # settle onto the table
    X0 = plant.EvalBodyPoseInWorld(ctx, body)
    plant.SetFreeBodySpatialVelocity(ctx, body, SpatialVelocity(np.zeros(3), [v0, 0, 0]))
    t_max = t + 3*v0/(mu*G)              # 3x the ideal slide time
    while t < t_max:
        t += 0.01
        sim.AdvanceTo(t)
        if t > 0.35 and abs(plant.EvalBodySpatialVelocityInWorld(ctx, body).translational()[0]) < 1e-4:
            break
    X1 = plant.EvalBodyPoseInWorld(ctx, body)
    if meshcat:
        meshcat.StopRecording(); meshcat.PublishRecording()
    d = X1.translation()[0] - X0.translation()[0]
    yaw = math.degrees(RollPitchYaw(X1.rotation()).yaw_angle() - RollPitchYaw(X0.rotation()).yaw_angle())
    return d, yaw, t - 0.3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.30)
    ap.add_argument("--v0", type=float, default=0.8, help="speed given to the box, m/s")
    ap.add_argument("--dt", type=float, default=1e-3, help="plant time step, s")
    ap.add_argument("--approx", default="lagged", choices=["lagged", "similar", "sap"])
    ap.add_argument("--modulus", type=float, default=1e7, help="box hydroelastic modulus, Pa")
    ap.add_argument("--dissipation", type=float, default=1.0, help="Hunt-Crossley dissipation, s/m")
    ap.add_argument("--meshcat", action="store_true")
    a = ap.parse_args()
    print(f"iiwa 14 loaded: {check_iiwa()} joint positions")
    meshcat = StartMeshcat() if a.meshcat else None
    d, yaw, dur = slide(a.mu, a.v0, a.dt, a.approx, a.modulus, a.dissipation, meshcat)
    d_ideal = a.v0**2/(2*a.mu*G)
    err = d/d_ideal - 1
    ok = abs(err) < 0.02 and abs(yaw) < 2.0
    print(f"slide mu={a.mu} v0={a.v0} m/s dt={a.dt*1e3:g} ms {a.approx}: d={d*100:.2f} cm, "
          f"ideal {d_ideal*100:.2f} cm ({err:+.1%}), yaw {yaw:+.2f} deg, {dur:.2f} s -> {'PASS' if ok else 'FAIL'}")
    if meshcat:
        input(f"Replay at {meshcat.web_url()} ; press Enter to quit. ")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scene.py -- the Drake scene for Slide-to-Target.

One MultibodyPlant holds:
  * a table whose top (z = 0) is tiled with strips across the travel direction (+x).  Each
    strip's friction is the pair's mu perturbed in log space (the "patchy table"), sampled per seed;
  * the object, a compliant-hydroelastic body: box, disc, or block on 3-4 feet, optionally with a
    centre-of-mass offset or one long foot (the E0 robustness cases);
  * optionally the iiwa 14, base at the origin on the table, with a flat paddle welded to its flange
    (rigid hydroelastic).  No controller is added: connect one to the plant's actuation input.

Friction comes from the ground-truth file, one row per object-surface pair, in the columns of
measured_friction.csv.  Until that file exists, ground_truth_placeholder.csv holds synthetic values
in the same columns; they are not friction estimates.

Drake gives a contact pair the coefficient 2*mu_a*mu_b/(mu_a + mu_b).  The object's geometry carries
mu = OBJECT_MU, and every geometry it can touch (strips, paddle) gets the value that makes the *pair*
coefficient equal to its target (partner_mu below).

Usage:
    python3 scene.py --pair M07 --meshcat                     # show the scene with the arm
    python3 scene.py --pair M07 --slide 0.8                   # passive slide vs prediction
    python3 scene.py --pair M07 --slide 0.8 --shape disc --patchiness 2 --seed 3
"""
import argparse, csv, math, sys
from dataclasses import dataclass
import numpy as np
from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddDefaultVisualization,
                         AddMultibodyPlant, AddRigidHydroelasticProperties, Box, CoulombFriction, Cylinder,
                         DiagramBuilder, MultibodyPlantConfig, Parser, ProximityProperties, RigidTransform,
                         RollPitchYaw, Simulator, SpatialInertia, SpatialVelocity, StartMeshcat)

G = 9.81
DEFAULT_PAIRS = "ground_truth_placeholder.csv"   # switch to measured_friction.csv once it exists
IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf"
OBJECT_MU = 1.0               # object geometry's own mu; partners are set relative to it
FLANGE_Z = 0.045              # flange face along iiwa_link_7 z, m
PADDLE = (0.12, 0.18, 0.01)   # paddle plate along link-7 x, y, z (thickness), m
PADDLE_Y = -0.05              # plate centre offset along link-7 y: with +y up it hangs below the flange
PADDLE_MU = 0.3               # paddle-object pair coefficient

@dataclass
class Contact:
    time_step: float = 1e-3
    approx: str = "lagged"     # discrete contact approximation: lagged | similar | sap
    modulus: float = 1e7       # object hydroelastic modulus, Pa
    dissipation: float = 1.0   # Hunt-Crossley dissipation, s/m
    resolution: float = 0.01   # object mesh resolution hint, m

@dataclass
class Table:
    x_min: float = -0.3        # strips cover x_min..x_max along travel, m
    x_max: float = 1.5
    width: float = 1.2         # across travel (y), m
    strip: float = 0.05        # strip length along travel, m
    patchiness: float = 1.0    # multiplier on the pair's s_tab_log; 0 = uniform table
    bimodal: bool = False      # strips at mu*exp(+-s) instead of mu*exp(s*z), z ~ N(0, 1)

@dataclass
class Obj:
    shape: str = "box"                    # box | disc | footed
    size: tuple = (0.10, 0.10, 0.05)      # x, y, z; disc uses size[0] as diameter
    mass: float = 0.5                     # kg
    com_offset: tuple = (0.0, 0.0, 0.0)   # CoM from the body's geometric centre, m
    n_feet: int = 4                       # footed: 3 or 4
    foot: tuple = (0.02, 0.005)           # footed: foot diameter, foot height, m
    foot_error: float = 0.0               # footed: first foot this much longer, m

@dataclass
class Scene:
    plant: object
    body: object               # the object
    iiwa: object               # model instance, or None
    paddle: object             # body, or None
    mu_strips: np.ndarray      # target pair coefficient of each strip
    x_edges: np.ndarray        # strip boundaries along x, len = len(mu_strips) + 1
    X_WO: RigidTransform       # object's initial pose

def load_pairs(path=DEFAULT_PAIRS):
    with open(path, newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}

def partner_mu(target):
    """mu for a geometry touching the object so that Drake's pair value 2ab/(a+b) equals target."""
    if not 0 < target < 2*OBJECT_MU:
        raise ValueError(f"pair mu {target} outside (0, {2*OBJECT_MU})")
    return OBJECT_MU*target/(2*OBJECT_MU - target)

def strip_targets(mu, s, table, seed):
    n = int(round((table.x_max - table.x_min)/table.strip))
    rng = np.random.default_rng(seed)
    z = rng.choice([-1.0, 1.0], n) if table.bimodal else rng.standard_normal(n)
    return mu*np.exp(table.patchiness*s*z)

def proximity(compliant, mu, contact, resolution=None):
    props = ProximityProperties()
    if compliant:
        AddCompliantHydroelasticProperties(resolution or contact.resolution, contact.modulus, props)
    else:
        AddRigidHydroelasticProperties(props)
    AddContactMaterial(contact.dissipation, None, CoulombFriction(mu, mu), props)
    return props

def foot_xy(obj):
    sx, sy, _ = obj.size
    ax, ay = sx/2 - obj.foot[0], sy/2 - obj.foot[0]
    if obj.n_feet == 3:
        return [(ax, 0.0), (-ax, ay), (-ax, -ay)]
    return [(ax, ay), (ax, -ay), (-ax, ay), (-ax, -ay)]

def add_object(plant, obj, contact):
    """Adds the object; returns (body, height of the body origin above the table when resting)."""
    sx, sy, sz = obj.size
    if obj.shape == "disc":
        M = SpatialInertia.SolidCylinderWithMass(obj.mass, sx/2, sz, [0, 0, 1])
    else:
        M = SpatialInertia.SolidBoxWithMass(obj.mass, sx, sy, sz)
    body = plant.AddRigidBody("object", M.Shift(-np.asarray(obj.com_offset, float)))
    shapes = [(RigidTransform(), Cylinder(sx/2, sz) if obj.shape == "disc" else Box(sx, sy, sz), None)]
    z0 = sz/2
    if obj.shape == "footed":
        d, h = obj.foot
        for i, (x, y) in enumerate(foot_xy(obj)):
            hi = h + (obj.foot_error if i == 0 else 0.0)
            shapes.append((RigidTransform([x, y, -sz/2 - hi/2]), Cylinder(d/2, hi), d/4))
        z0 += h + max(obj.foot_error, 0.0)
    props = {}
    for i, (X, shape, res) in enumerate(shapes):
        props = proximity(True, OBJECT_MU, contact, res)
        plant.RegisterCollisionGeometry(body, X, shape, f"object_{i}", props)
        plant.RegisterVisualGeometry(body, X, shape, f"object_{i}", [0.3, 0.5, 0.9, 1.0])
    return body, z0

def add_arm(plant, contact):
    iiwa = Parser(plant).AddModelsFromUrl(IIWA_URL)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa))
    paddle = plant.AddRigidBody("paddle", iiwa, SpatialInertia.SolidBoxWithMass(0.3, *PADDLE))
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa), paddle.body_frame(),
                     RigidTransform([0, PADDLE_Y, FLANGE_Z + PADDLE[2]/2]))
    plant.RegisterCollisionGeometry(paddle, RigidTransform(), Box(*PADDLE), "paddle",
                                    proximity(False, partner_mu(PADDLE_MU), contact))
    plant.RegisterVisualGeometry(paddle, RigidTransform(), Box(*PADDLE), "paddle", [0.6, 0.6, 0.6, 1.0])
    return iiwa, paddle

def build_scene(builder, pair, obj=Obj(), table=Table(), contact=Contact(), arm=True, seed=0, x0=0.55):
    """Adds the plant to builder and returns a Scene; the plant is finalized."""
    plant, _ = AddMultibodyPlant(MultibodyPlantConfig(time_step=contact.time_step,
                                                      contact_model="hydroelastic_with_fallback",
                                                      discrete_contact_approximation=contact.approx), builder)
    mu_strips = strip_targets(float(pair["mu_k_nominal"]), float(pair["s_tab_log"]), table, seed)
    x_edges = table.x_min + table.strip*np.arange(len(mu_strips) + 1)
    lo, hi = mu_strips.min(), mu_strips.max()
    for i, mu in enumerate(mu_strips):
        X = RigidTransform([(x_edges[i] + x_edges[i+1])/2, 0, -0.025])   # 5 cm thick, top at z = 0
        strip = Box(table.strip, table.width, 0.05)
        plant.RegisterCollisionGeometry(plant.world_body(), X, strip, f"strip_{i:02d}",
                                        proximity(False, partner_mu(mu), contact))
        shade = 0.85 - 0.35*(mu - lo)/(hi - lo) if hi > lo else 0.7   # darker = more friction
        plant.RegisterVisualGeometry(plant.world_body(), X, strip, f"strip_{i:02d}", [shade, shade*0.9, shade*0.8, 1.0])
    body, z0 = add_object(plant, obj, contact)
    iiwa, paddle = add_arm(plant, contact) if arm else (None, None)
    X_WO = RigidTransform([x0, 0, z0])
    plant.SetDefaultFloatingBaseBodyPose(body, X_WO)
    plant.Finalize()
    return Scene(plant, body, iiwa, paddle, mu_strips, x_edges, X_WO)

def footprint_x(obj):
    """x offsets of the contact area under the body origin and their weights (uniform pressure)."""
    sx = obj.size[0]
    if obj.shape == "footed":
        r = obj.foot[0]/2
        u = np.linspace(-r, r, 21)
        px = np.concatenate([x + u for x, _ in foot_xy(obj)])
        return px, np.tile(np.sqrt(np.maximum(r*r - u*u, 0)), len(foot_xy(obj)))
    px = np.linspace(-sx/2, sx/2, 101)
    return px, (np.sqrt(np.maximum(sx*sx/4 - px*px, 0)) if obj.shape == "disc" else np.ones_like(px))

def predicted_distance(sc, obj, v0):
    """Stopping distance if friction is the strip mu averaged over the footprint: g*int(mu) = v0^2/2."""
    px, w = footprint_x(obj)
    x0 = sc.X_WO.translation()[0]
    x = np.arange(0, sc.x_edges[-1] - x0 - obj.size[0], 1e-4)
    i = np.clip(np.searchsorted(sc.x_edges, x0 + x[:, None] + px[None, :], side="right") - 1, 0, len(sc.mu_strips) - 1)
    mu_bar = (sc.mu_strips[i]*w).sum(axis=1)/w.sum()
    work = G*np.concatenate([[0], np.cumsum(0.5*(mu_bar[1:] + mu_bar[:-1])*np.diff(x))])
    return float(np.interp(v0**2/2, work, x))

def passive_slide(pair, v0, obj=Obj(), table=Table(), contact=Contact(), seed=0, meshcat=None):
    """Object settles, gets speed v0 along +x, slides to rest.  Returns (d, yaw_deg, d_predicted)."""
    builder = DiagramBuilder()
    sc = build_scene(builder, pair, obj, table, contact, arm=False, seed=seed)
    if meshcat:
        AddDefaultVisualization(builder, meshcat)
    sim = Simulator(builder.Build())
    plant, body = sc.plant, sc.body
    ctx = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
    if meshcat:
        meshcat.StartRecording()
    t = 0.3
    sim.AdvanceTo(t)                     # settle onto the table
    X0 = plant.EvalBodyPoseInWorld(ctx, body)
    plant.SetFreeBodySpatialVelocity(ctx, body, SpatialVelocity(np.zeros(3), [v0, 0, 0]))
    t_max = t + 3*v0/(sc.mu_strips.min()*G)
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
    return d, yaw, predicted_distance(sc, obj, v0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--pair", default="M01")
    ap.add_argument("--shape", default="box", choices=["box", "disc", "footed"])
    ap.add_argument("--n-feet", type=int, default=4, choices=[3, 4])
    ap.add_argument("--foot-error", type=float, default=0.0, help="first foot this much longer, m")
    ap.add_argument("--com-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    ap.add_argument("--patchiness", type=float, default=1.0, help="multiplier on s_tab_log; 0 = uniform")
    ap.add_argument("--bimodal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--approx", default="lagged", choices=["lagged", "similar", "sap"])
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--slide", type=float, metavar="V0", help="passive slide at this speed (no arm)")
    ap.add_argument("--meshcat", action="store_true")
    a = ap.parse_args()
    pair = load_pairs(a.pairs)[a.pair]
    if "PLACEHOLDER" in pair["sources"]:
        print(f"note: {a.pairs} holds placeholder friction values")
    obj = Obj(shape=a.shape, n_feet=a.n_feet, foot_error=a.foot_error, com_offset=tuple(a.com_offset))
    table = Table(patchiness=a.patchiness, bimodal=a.bimodal)
    contact = Contact(time_step=a.dt, approx=a.approx)
    meshcat = StartMeshcat() if a.meshcat else None
    print(f"{a.pair}: {pair['object_material']} on {pair['surface_material']}, mu={pair['mu_k_nominal']}")
    if a.slide is not None:
        d, yaw, d_pred = passive_slide(pair, a.slide, obj, table, contact, a.seed, meshcat)
        err = d/d_pred - 1
        ok = abs(err) < 0.02 and abs(yaw) < 2.0
        print(f"slide v0={a.slide} m/s {a.shape} patchiness={a.patchiness}{' bimodal' if a.bimodal else ''} "
              f"seed={a.seed}: d={d*100:.2f} cm, "
              f"predicted {d_pred*100:.2f} cm ({err:+.1%}), yaw {yaw:+.2f} deg -> {'PASS' if ok else 'FAIL'}")
    else:
        builder = DiagramBuilder()
        sc = build_scene(builder, pair, obj, table, contact, arm=True, seed=a.seed)
        print(f"{sc.plant.num_positions()} positions, {len(sc.mu_strips)} strips, "
              f"strip mu {sc.mu_strips.min():.3f}-{sc.mu_strips.max():.3f}")
        if meshcat:
            AddDefaultVisualization(builder, meshcat)
            diagram = builder.Build()
            ctx = diagram.CreateDefaultContext()
            sc.plant.SetPositions(sc.plant.GetMyMutableContextFromRoot(ctx), sc.iiwa,
                                  [0, 0.6, 0, -1.4, 0, 1.1, 0])   # display pose only
            diagram.ForcedPublish(ctx)
        ok = True
    if meshcat:
        input(f"Open {meshcat.web_url()} ; press Enter to quit. ")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

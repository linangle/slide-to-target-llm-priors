#!/usr/bin/env python3
"""
push.py -- prescribed-motion paddle: the push harness behind E0-3, E0-4 and E0-5.

The arm has no controller yet, so these experiments drive the paddle plate directly.  A rigid "paddle" body
(the plate of scene.PADDLE, face normal along +x) sits on a prismatic joint along world x.  Drake's implicit
PD actuator (JointActuator.set_controller_gains; it needs the SAP-family solver, so lagged, similar or sap)
tracks a constant-jerk speed profile (physics.constant_jerk_profile) with a feed-forward force m a_d.  The
implicit scheme applies the desired state of step k over [t_k, t_k+1], so the plate lags the command by one
time step (0.4 mm at 0.4 m/s and 1 ms) and otherwise tracks it to the solver tolerance; the recorded paddle
state, not the command, is what the release model is checked against.

Scene: uniform table at the pair's mu (Table(patchiness=0)), object from scene.Obj, plate `gap` behind the
object's back face and `clearance` above the table.  The paddle-object pair coefficient is mu_p
(scene.partner_mu makes Drake's 2ab/(a+b) equal to it).  Knobs for E0-3: lateral offset of the plate, yaw
misalignment of its face, and the object's COM offset (Obj.com_offset).

push_slide() runs settle -> push -> brake -> free slide to rest, stepping at the plant time step and reading
ContactResults every step.  The release is read three ways: (1) the model on the command, t* = t_peak + mu g / J,
with the simulator's object state interpolated there (v_sep, heading, spin, x_sep, the FK release point); (2) the
model on the paddle's recorded motion (physics.release_from_profile) and the object's speed at that instant;
(3) the instant the paddle-object force actually fell to zero (the compliant tail), extrapolated between steps,
with its lag after t*.  Near separation both bodies decelerate at about mu g, so which instant is used moves the
speed by mu g per time step (1.4% of a 0.36 m/s release at 1 ms); (1) is the quantity the decision rule predicts.
It returns those scalars plus the rest pose, slide and lateral error, contact-patch centroid, tracking error, and
the per-step trajectory for E0-5's camera replay.

    python3 push.py                                   # one push at the config contact: mu 0.3, v_peak 0.8, J 100
    python3 push.py --mu 0.1 --v 1.0 --J 50 --shape disc --com 0.02 --yaw 2 --mu-p 0.6
"""
import argparse, math, time
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from pydrake.all import (AddDefaultVisualization, Box, DiagramBuilder, FixedOffsetFrame, PdControllerGains,
                         PiecewisePolynomial, PrismaticJoint, RigidTransform, RollPitchYaw, RotationMatrix, Simulator,
                         SpatialInertia, StartMeshcat, TrajectorySource, UnitInertia)
from scene import G, PADDLE, Obj, Table, build_scene, partner_mu, proximity
from drake_config import contact_from_config, describe
from physics import A_PUSH, release_from_profile, release_speed, stop_distance

PLATE = (PADDLE[2], PADDLE[0], PADDLE[1])   # thickness along x (push), width along y, height along z: scene.PADDLE as the arm holds it

@dataclass
class Pusher:
    mass: float = 5.0          # kg: the plate plus the reflected arm inertia the PD acts on
    kp: float = 1e5            # N/m
    kd: float = 2e3            # N s/m
    mu_p: float = 0.3          # paddle-object pair coefficient
    lateral: float = 0.0       # plate centre offset along y from the object's centre, m
    yaw_deg: float = 0.0       # face misalignment about z, deg; 0 = face normal along the push
    gap: float = 0.001         # initial gap between the face's nearest point and the object's back face, m
    clearance: float = 0.003   # plate's lower edge above the table, m
    plate: tuple = PLATE
    # filled by add()
    body: object = field(default=None, repr=False, compare=False)
    joint: object = field(default=None, repr=False, compare=False)
    instance: object = field(default=None, repr=False, compare=False)
    x_touch0: float = field(default=None, repr=False, compare=False)   # world x of the face's nearest point at q = 0, plus the gap

    def add(self, plant, contact, obj, x0):
        """Adds the plate on a prismatic joint along world x, `gap` behind the object's back face; returns the body.
        The joint translation q is the plate's displacement from this start pose."""
        th, wd, ht = self.plate
        yaw = math.radians(self.yaw_deg)
        self.x_touch0 = x0 - obj.size[0]/2 - self.gap        # where the face's nearest point would touch the object
        x_centre = self.x_touch0 - wd/2*abs(math.sin(yaw)) - th/2*math.cos(yaw)   # a yawed plate's corner reaches forward
        self.instance = plant.AddModelInstance("pusher")
        self.body = plant.AddRigidBody("paddle", self.instance, SpatialInertia(self.mass, np.zeros(3), UnitInertia.SolidBox(th, wd, ht)))
        base = plant.AddFrame(FixedOffsetFrame("pusher_base", plant.world_frame(),
                                               RigidTransform([x_centre, self.lateral, self.clearance + ht/2])))
        self.joint = plant.AddJoint(PrismaticJoint("push_x", base, self.body.body_frame(), [1, 0, 0]))
        X_BG = RigidTransform(RotationMatrix.MakeZRotation(yaw), np.zeros(3))
        plant.RegisterCollisionGeometry(self.body, X_BG, Box(th, wd, ht), "paddle", proximity(False, partner_mu(self.mu_p), contact))
        plant.RegisterVisualGeometry(self.body, X_BG, Box(th, wd, ht), "paddle", [0.6, 0.6, 0.6, 1.0])
        plant.AddJointActuator("push_x", self.joint).set_controller_gains(PdControllerGains(p=self.kp, d=self.kd))
        return self.body

    def face_x(self, q):
        """World x of the face's nearest point to the object at joint translation q."""
        return self.x_touch0 + q

def profile(v_peak, J, a_push=A_PUSH, settle=0.3, hold=2.0, dt=1e-4):
    """Commanded paddle motion: rest until `settle`; accelerate at a_push; then ONE constant-jerk segment (-J) that
    takes the acceleration through zero exactly at v_peak and on into braking until the plate stops; then hold.
    So the brake starts from zero deceleration at the speed peak, as the release model assumes, and the command has
    no acceleration jump at the peak (a jump makes the PD plate dip and the object leave early at low mu).
    Returns dict(t, x, v, a, t_peak, t_stop, x_stop, a_push); x is the displacement from the start (the joint q)."""
    a_push = min(a_push, math.sqrt(2*J*v_peak))          # the ramp from a_push down to 0 at jerk J gains a_push^2/(2J)
    v1 = v_peak - a_push**2/(2*J)
    t1 = v1/a_push
    n1 = max(int(math.ceil(t1/dt)), 1)
    ta = np.linspace(0, t1, n1 + 1)
    tau_stop = (a_push + math.sqrt(a_push**2 + 2*J*v1))/J   # v1 + a_push tau - J tau^2/2 = 0
    tau = np.linspace(0, tau_stop, int(math.ceil(tau_stop/dt)) + 1)[1:]
    vb = np.maximum(v1 + a_push*tau - 0.5*J*tau**2, 0.0)
    ab = np.where(vb > 0, a_push - J*tau, 0.0)
    t = np.concatenate([[0.0], settle + ta, settle + t1 + tau, [settle + t1 + tau_stop + hold]])
    v = np.concatenate([[0.0], a_push*ta, vb, [0.0]])
    a = np.concatenate([[0.0], np.full(n1 + 1, a_push), ab, [0.0]])
    x = np.concatenate([[0.0], np.cumsum(0.5*(v[1:] + v[:-1])*np.diff(t))])
    return {"t": t, "x": x, "v": v, "a": a, "t_peak": settle + t1 + a_push/J, "t_stop": settle + t1 + tau_stop,
            "x_stop": float(x[-1]), "a_push": a_push}

def connect(builder, plant, pusher, prof, dt=0.0):
    """Desired state [x_d, v_d] and feed-forward force m a_d into the pusher's actuator.  The implicit PD reads the desired
    state at the start of a step and reaches it at the end, so the command is fed one time step (dt) ahead."""
    t = prof["t"] - dt
    pp_x = PiecewisePolynomial.CubicHermite(t, prof["x"][None, :], prof["v"][None, :])
    desired = builder.AddSystem(TrajectorySource(pp_x, output_derivative_order=1))
    builder.Connect(desired.get_output_port(), plant.get_desired_state_input_port(pusher.instance))
    pp_f = PiecewisePolynomial.FirstOrderHold(t, (pusher.mass*prof["a"])[None, :])
    builder.Connect(builder.AddSystem(TrajectorySource(pp_f)).get_output_port(), plant.get_actuation_input_port(pusher.instance))

def _paddle_object(res, inspector):
    """(force magnitude, area-weighted centroid, area) of the paddle-object hydroelastic contact in a ContactResults."""
    f, c, area = 0.0, np.zeros(3), 0.0
    for i in range(res.num_hydroelastic_contacts()):
        info = res.hydroelastic_contact_info(i)
        s = info.contact_surface()
        names = (inspector.GetName(s.id_M()), inspector.GetName(s.id_N()))
        if any("paddle" in n for n in names) and any("object" in n for n in names):
            f += float(np.linalg.norm(info.F_Ac_W().translational()))
            a = s.total_area()
            c, area = c + a*s.centroid(), area + a
    return f, (c/area if area > 0 else np.full(3, np.nan)), area

def _points_involving(res, plant, names=("paddle", "object")):
    n = 0
    for i in range(res.num_point_pair_contacts()):
        p = res.point_pair_contact_info(i)
        if plant.get_body(p.bodyA_index()).name() in names or plant.get_body(p.bodyB_index()).name() in names:
            n += 1
    return n

def push_slide(mu, v_peak, J, contact=None, obj=None, pusher=None, a_push=A_PUSH, settle=0.3, x0=0.55, f_min=0.01,
               camera=None, keep_traj=True, meshcat=None):
    """One push-and-slide.  Returns a dict of scalars plus 'traj' (per-step arrays) when keep_traj."""
    contact = contact if contact is not None else contact_from_config()
    obj = obj if obj is not None else Obj()
    pusher = pusher if pusher is not None else Pusher()
    prof = profile(v_peak, J, a_push, settle)
    builder = DiagramBuilder()
    sc = build_scene(builder, {"mu_k_nominal": mu, "s_tab_log": 0.0}, obj, Table(patchiness=0.0), contact, arm=False, x0=x0,
                     camera=camera, pusher=pusher)
    connect(builder, sc.plant, pusher, prof, contact.time_step)
    if meshcat:
        AddDefaultVisualization(builder, meshcat)
    sim = Simulator(builder.Build())
    plant, body, joint = sc.plant, sc.body, pusher.joint
    ctx = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
    inspector = sc.scene_graph.model_inspector()
    port = plant.get_contact_results_output_port()
    dt, rec = contact.time_step, defaultdict(list)
    if meshcat:
        meshcat.StartRecording()
    t0, t, k = time.time(), 0.0, 0
    t_end = prof["t_stop"] + 3*v_peak/(mu*G) + 0.5
    while t < t_end - 1e-9:
        k += 1
        t = k*dt
        sim.AdvanceTo(t)
        X = plant.EvalBodyPoseInWorld(ctx, body)
        V = plant.EvalBodySpatialVelocityInWorld(ctx, body)
        rpy = RollPitchYaw(X.rotation())
        res = port.Eval(ctx)
        f, c, area = _paddle_object(res, inspector)
        p, w = X.translation(), V.translational()
        rec["t"].append(t); rec["x"].append(p[0]); rec["y"].append(p[1]); rec["z"].append(p[2])
        rec["roll"].append(rpy.roll_angle()); rec["pitch"].append(rpy.pitch_angle()); rec["yaw"].append(rpy.yaw_angle())
        rec["vx"].append(w[0]); rec["vy"].append(w[1]); rec["wz"].append(V.rotational()[2])
        rec["q"].append(joint.get_translation(ctx)); rec["qd"].append(joint.get_translation_rate(ctx))
        rec["f"].append(f); rec["cy"].append(c[1] - pusher.lateral); rec["cz"].append(c[2]); rec["area"].append(area)
        rec["n_point"].append(_points_involving(res, plant))
        if t > prof["t_stop"] + 0.05 and math.hypot(w[0], w[1]) < 1e-4 and abs(V.rotational()[2]) < 1e-3:
            break
    if meshcat:
        meshcat.StopRecording(); meshcat.PublishRecording()
    tr = {key: np.array(val) for key, val in rec.items()}
    sx = obj.size[0]
    out = {"mu": mu, "v_peak": v_peak, "J": J, "a_push": prof["a_push"], "shape": obj.shape, "mu_p": pusher.mu_p,
           "lateral": pusher.lateral, "yaw_cmd_deg": pusher.yaw_deg, "com_y": obj.com_offset[1],
           "approx": contact.approx, "dt_ms": contact.time_step*1e3, "steps": len(tr["t"]), "wall_s": time.time() - t0,
           "n_point_contacts": int(tr["n_point"].sum()), "separated": bool(tr["f"][-1] <= f_min), "t_peak": prof["t_peak"], "t_stop": prof["t_stop"]}
    at = lambda k, tq: float(np.interp(tq, tr["t"], tr[k]))
    # (1) the release model on the command: separation when the commanded deceleration reaches mu g; the simulator's state there
    v_r_cmd = float(release_speed(v_peak, mu, J))
    t_star = prof["t_peak"] + mu*G/J if v_r_cmd > 0 else prof["t_stop"]      # v_r = 0: the object stops with the plate
    vx, vy = at("vx", t_star), at("vy", t_star)
    out.update(t_sep=t_star, v_sep=float(math.hypot(vx, vy)), psi_r_deg=math.degrees(math.atan2(vy, vx)), omega_r=at("wz", t_star),
               x_sep=at("x", t_star), y_sep=at("y", t_star), q_sep=at("q", t_star), v_r_cmd=v_r_cmd)
    out["x_fk"] = float(pusher.face_x(out["q_sep"]) + sx/2)
    out["v_r_cmd_err_pct"] = (out["v_sep"]/v_r_cmd - 1)*100 if v_r_cmd > 0 else float("nan")
    # (2) the release model on the paddle's actual motion: where its measured deceleration crosses mu g, and the object's speed there
    i0 = int(np.searchsorted(tr["t"], settle))
    rel = release_from_profile(tr["t"][i0:], tr["qd"][i0:], np.gradient(tr["qd"], tr["t"])[i0:], mu)
    if rel and rel[1] > 0:
        out.update(t_sep_act=rel[0], v_r_act=rel[1], v_obj_at_act=float(math.hypot(at("vx", rel[0]), at("vy", rel[0]))))
        out["v_r_act_err_pct"] = (out["v_obj_at_act"]/rel[1] - 1)*100
    else:
        out.update(t_sep_act=float("nan"), v_r_act=float("nan"), v_obj_at_act=float("nan"), v_r_act_err_pct=float("nan"))
    # (3) when the paddle-object force actually vanished (the compliant tail): last step above f_min after the peak, extrapolated to zero
    ip = int(np.searchsorted(tr["t"], prof["t_peak"]))
    in_contact = np.nonzero(tr["f"][ip:] > f_min)[0]
    i_sep = ip if len(in_contact) == 0 else ip + int(in_contact[-1])
    i1 = min(i_sep + 1, len(tr["t"]) - 1)
    f0, f1 = tr["f"][i_sep], tr["f"][i1]
    frac = min(max(f0/(f0 - f1), 0.0), 1.0) if f0 > f1 else 0.0
    t_zero = float(tr["t"][i_sep] + frac*(tr["t"][i1] - tr["t"][i_sep]))
    out.update(t_force_zero=t_zero, sep_lag_ms=(t_zero - t_star)*1e3, v_sep_force=float(math.hypot(at("vx", t_zero), at("vy", t_zero))))
    t_sep = t_zero
    # rest pose and the slide after separation
    x_rest, y_rest = tr["x"][-1], tr["y"][-1]
    d, d_pred = x_rest - out["x_sep"], float(stop_distance(out["v_sep"], mu))
    out.update(x_rest=float(x_rest), y_rest=float(y_rest), yaw_rest_deg=math.degrees(tr["yaw"][-1]), d_slide=float(d), d_pred=d_pred,
               d_err_pct=(d/d_pred - 1)*100 if d_pred > 1e-4 else float("nan"), lateral_miss=float(y_rest),
               lateral_after_release=float(y_rest - out["y_sep"]), t_rest=float(tr["t"][-1]),
               tilt_max_deg=math.degrees(np.max(np.abs(np.r_[tr["roll"], tr["pitch"]]))))
    # contact patch on the plate during the push, and command tracking
    push = (tr["t"] >= settle) & (tr["t"] <= t_sep) & (tr["f"] > f_min)
    cy = tr["cy"][push]
    out.update(patch_cy_mean=float(np.mean(cy)) if len(cy) else float("nan"), patch_cy_min=float(np.min(cy)) if len(cy) else float("nan"),
               patch_cy_max=float(np.max(cy)) if len(cy) else float("nan"),
               patch_cz_mean=float(np.mean(tr["cz"][push])) if len(cy) else float("nan"), f_max=float(tr["f"].max()))
    move = (tr["t"] >= settle) & (tr["t"] <= prof["t_stop"])
    eq = tr["q"][move] - np.interp(tr["t"][move], prof["t"], prof["x"])
    ev = tr["qd"][move] - np.interp(tr["t"][move], prof["t"], prof["v"])
    out.update(track_q_rms=float(np.sqrt(np.mean(eq**2))), track_q_max=float(np.abs(eq).max()),
               track_v_rms=float(np.sqrt(np.mean(ev**2))), track_v_max=float(np.abs(ev).max()))
    if keep_traj:
        out["traj"], out["profile"] = tr, prof
    return out

def scalars(r):
    """The result without its per-step arrays, for CSV rows."""
    return {k: v for k, v in r.items() if k not in ("traj", "profile")}

def summary(r):
    lines = [f"{r['shape']} mu={r['mu']} v_peak={r['v_peak']} J={r['J']} mu_p={r['mu_p']} lateral={r['lateral']*1e3:.0f} mm "
             f"yaw={r['yaw_cmd_deg']}deg com_y={r['com_y']*1e3:.0f} mm  ({r['approx']}, {r['dt_ms']:g} ms, {r['steps']} steps, {r['wall_s']:.1f} s)",
             f"  model on the command: separation at t*={r['t_sep']:.4f} s with v_r={r['v_r_cmd']:.4f} m/s; the object there: {r['v_sep']:.4f} m/s ({r['v_r_cmd_err_pct']:+.2f}%)",
             f"  model on the actual paddle: t*={r['t_sep_act']:.4f} s, v_r={r['v_r_act']:.4f}; the object there: {r['v_obj_at_act']:.4f} ({r['v_r_act_err_pct']:+.2f}%); "
             f"contact force vanished at {r['t_force_zero']:.4f} s ({r['sep_lag_ms']:+.1f} ms after t*), object then at {r['v_sep_force']:.4f} m/s",
             f"  release heading {r['psi_r_deg']:+.3f} deg, spin {r['omega_r']:+.4f} rad/s; patch centroid on plate y = {r['patch_cy_mean']*1e3:+.1f} mm "
             f"[{r['patch_cy_min']*1e3:+.1f}, {r['patch_cy_max']*1e3:+.1f}], z = {r['patch_cz_mean']*1e3:.1f} mm; f_max {r['f_max']:.2f} N",
             f"  slide {r['d_slide']*100:.2f} cm vs v_sep^2/(2 mu g) = {r['d_pred']*100:.2f} cm ({r['d_err_pct']:+.2f}%); lateral miss {r['lateral_miss']*1e3:+.2f} mm; "
             f"rest yaw {r['yaw_rest_deg']:+.3f} deg; max tilt {r['tilt_max_deg']:.3f} deg; FK release point minus true {(r['x_fk'] - r['x_sep'])*1e3:+.2f} mm",
             f"  paddle tracking: q rms {r['track_q_rms']*1e3:.2f} mm (max {r['track_q_max']*1e3:.2f}), v rms {r['track_v_rms']*1e3:.1f} mm/s "
             f"(max {r['track_v_max']*1e3:.1f}); point-pair contacts {r['n_point_contacts']}; separated {r['separated']}"]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--v", type=float, default=0.8, help="commanded peak speed, m/s")
    ap.add_argument("--J", type=float, default=100.0, help="braking jerk, m/s^3")
    ap.add_argument("--a-push", type=float, default=2.0)
    ap.add_argument("--shape", default="box", choices=["box", "disc", "ring", "footed"])
    ap.add_argument("--com", type=float, default=0.0, help="object COM offset along y, m")
    ap.add_argument("--lateral", type=float, default=0.0, help="plate centre offset along y, m")
    ap.add_argument("--yaw", type=float, default=0.0, help="plate face misalignment, deg")
    ap.add_argument("--mu-p", type=float, default=0.3)
    ap.add_argument("--approx", default=None); ap.add_argument("--dissipation", type=float, default=None)
    ap.add_argument("--meshcat", action="store_true")
    a = ap.parse_args()
    over = {k: v for k, v in (("approx", a.approx), ("dissipation", a.dissipation)) if v is not None}
    contact = contact_from_config(**over)
    print(describe(contact))
    meshcat = StartMeshcat() if a.meshcat else None
    r = push_slide(a.mu, a.v, a.J, contact, Obj(shape=a.shape, com_offset=(0.0, a.com, 0.0)),
                   Pusher(mu_p=a.mu_p, lateral=a.lateral, yaw_deg=a.yaw), a.a_push, meshcat=meshcat)
    print(summary(r))
    if meshcat:
        input("meshcat is showing the recording; press enter to quit")

if __name__ == "__main__":
    main()

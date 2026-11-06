#!/usr/bin/env python3
r"""
arm.py -- the iiwa arm controller: slide the object to a target point.

Three layers, each usable on its own:

1. plan_peak_speed(x_target, mu, J)   target point -> commanded peak paddle speed u
   The object's centre comes to rest at
        x_rest(u, mu) = x0 - gap + release_push_distance(u, mu, J)  +  v_r(u, mu, J)^2 / (2 mu g)
                        \_______ release point (rides on the paddle) _/   \___ free slide d = v_r^2/(2 mu g) ___/
   with v_r = u - (mu g)^2 / (2 J), the release model (physics.release_speed).  For a point estimate of mu we
   solve x_rest(u) = x_target for u; for a belief over mu (belief.posterior) we minimise E[(x_rest - x_target)^2].

2. ArmPusher.plan(prof)               paddle speed profile -> joint trajectory q(t), qdot(t), tau_ff(t)
   The paddle face must move along +x following push.profile(u, J), face square to the push, lower edge
   `clearance` above the table, centred on the object.  Inverse kinematics gives the start pose; closed-loop
   differential kinematics (qdot = J^+ [V_des + K e]) integrated at 0.1 ms carries the face along the line;
   inverse dynamics on the arm gives the feed-forward torque (inertia + Coriolis + gravity).

3. ArmPusher in the scene             joint trajectory -> torques
   Each iiwa joint gets Drake's implicit PD (JointActuator.set_controller_gains, needs the lagged/SAP solver that
   sim_config.yaml already uses) tracking [q(t), qdot(t)], plus tau_ff(t) on the actuation port.
   ArmPusher has the same hook as push.Pusher (add / connect / face_x / joint), so push.push_slide and the E0
   scripts run with pusher=ArmPusher() unchanged, and the release is read the same three ways.

    python3 arm.py --target 1.0 --mu 0.3                 # plan, push with the arm, report where it landed
    python3 arm.py --target 1.0 --mu 0.3 --mu-est 0.25   # plan with a wrong friction estimate
    python3 arm.py --target 1.0 --mu 0.3 --meshcat       # watch it (open the printed URL)
    python3 arm.py --target 1.0 --mu 0.3 --plate         # same plan, but the prescribed plate instead of the arm
    python3 arm.py --reach                               # the arm's reach along the push line and the fastest feasible push
"""
import argparse, math
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from pydrake.all import (InverseKinematics, JacobianWrtVariable, MultibodyForces, MultibodyPlant, PdControllerGains,
                         PiecewisePolynomial, RotationMatrix, SceneGraph, Solve, StartMeshcat, TrajectorySource)

from physics import A_PUSH, G, release_push_distance, release_speed
from scene import PADDLE, PADDLE_MU, Contact, Obj, add_arm
from push import Pusher, profile, push_slide, summary

# The paddle as the arm holds it (scene.add_arm): plate along link-7 x, y, z = 0.12 (world y), 0.18 (world z),
# 0.01 thick (world x).  So link 7 (and the paddle body) has x -> world +y, y -> world +z, z -> world +x (push).
R_PUSH = RotationMatrix(np.array([[0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]]))
P_FACE = np.array([0.0, 0.0, PADDLE[2]/2])      # centre of the plate's front face, in the paddle body frame
Q_SEED = np.array([-0.66, 1.35, 0.69, -2.05, -0.38, -1.47, 0.88])   # IK seed: paddle face at x ~ 0.50 m, low over the table
REACH_X = (0.50, 0.92)       # face x the arm reaches with this posture (python3 arm.py --reach); start x0 = 0.55 sits at the near end

# ----------------------------------------------------------------------------- 1. target point -> peak speed

def rest_x(u, mu, J, x0=0.55, gap=0.001, a_push=A_PUSH, g=G):
    """Where the object's centre comes to rest for commanded peak speed u (the release model plus the free slide)."""
    v_r = float(release_speed(u, mu, J, g))
    return x0 - gap + release_push_distance(u, mu, J, a_push, g) + v_r*v_r/(2*mu*g)

def plan_peak_speed(x_target, mu, J=100.0, x0=0.55, gap=0.001, a_push=A_PUSH, u_max=None, g=G):
    """Commanded peak paddle speed that lands the object's centre at x_target.

    mu: a friction value (point estimate, or the true value for the reference method), or a belief dict with
        'grid' (ln mu) and 'pmf' (belief.posterior's output), in which case u minimises E[(x_rest - x_target)^2].
    u_max: the fastest push the arm can make from x0 (default: max_peak_speed, from the arm's reach).
    Raises ValueError if the target is closer than the release point or farther than the arm can send it."""
    u_max = max_peak_speed(J, x0, gap, a_push) if u_max is None else u_max
    if isinstance(mu, dict):
        mus, w = np.exp(np.asarray(mu["grid"])), np.asarray(mu["pmf"])
        keep = w > 1e-12*w.max()
        mus, w = mus[keep], w[keep]/w[keep].sum()
        cost = lambda u: float(np.sum(w*(np.array([rest_x(u, m, J, x0, gap, a_push, g) for m in mus]) - x_target)**2))
        us = np.linspace(0.0, u_max, 241)
        c = [cost(u) for u in us]
        k = int(np.argmin(c))
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(cost, bounds=(us[max(k-1, 0)], us[min(k+1, len(us)-1)]), method="bounded")
        return float(r.x)
    f = lambda u: rest_x(u, mu, J, x0, gap, a_push, g) - x_target
    if f(0.0) >= 0:
        raise ValueError(f"target x = {x_target:.3f} m is behind the object's start ({x0:.3f} m)")
    if f(u_max) < 0:
        raise ValueError(f"target x = {x_target:.3f} m needs more than the arm's fastest push u = {u_max:.2f} m/s "
                         f"(reaches {rest_x(u_max, mu, J, x0, gap, a_push, g):.3f} m at mu = {mu}); move x0 or lower a_push/J limits")
    return float(brentq(f, 0.0, u_max, xtol=1e-9))

def limited_peak_speed(J=100.0, x0=0.55, gap=0.001, a_push=A_PUSH, obj_len=Obj().size[0], which=("v",), tol=0.005):
    """Fastest commanded peak speed whose joint trajectory stays within the iiwa's limits: which = ("v",) checks joint
    velocities (the binding one), add "tau" for the feed-forward torques.  Bisection on u, tol m/s; ~10 trajectory plans."""
    face0 = [x0 - obj_len/2 - gap, 0.0, 0.003 + PADDLE[1]/2]
    def worst(u):
        rep = joint_limit_report(joint_trajectory(face0, profile(u, J, a_push), dt=5e-4))
        return max(float(rep[k].max()) for k in which)
    lo, hi = 0.02, max_peak_speed(J, x0, gap, a_push, obj_len)
    if worst(hi) <= 1.0:
        return hi
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if worst(mid) <= 1.0 else (lo, mid)
    return lo

def max_peak_speed(J=100.0, x0=0.55, gap=0.001, a_push=A_PUSH, obj_len=Obj().size[0], x_reach=REACH_X[1]):
    """Fastest commanded peak speed whose whole paddle stroke (profile x_stop) stays inside the arm's reach."""
    room = x_reach - (x0 - obj_len/2 - gap)
    f = lambda u: profile(u, J, a_push)["x_stop"] - room
    return float(brentq(f, 1e-3, 5.0, xtol=1e-6)) if f(5.0) > 0 else 5.0

# ----------------------------------------------------------------------------- 2. paddle line -> joint trajectory

_KEEP = []

@lru_cache(maxsize=1)
def _arm_model():
    """A continuous-time plant with only the iiwa and its paddle, for IK, Jacobians and inverse dynamics."""
    plant, sg = MultibodyPlant(0.0), SceneGraph()
    plant.RegisterAsSourceForSceneGraph(sg)
    iiwa, paddle = add_arm(plant, Contact())
    plant.Finalize()
    _KEEP.append(sg)                 # the plant keeps a pointer to its SceneGraph: keep it alive
    return plant, iiwa, paddle

def solve_ik(p_face, q_seed=Q_SEED, R=R_PUSH, tol=1e-6):
    """Joint angles that put the paddle's face centre at p_face (world) with the face normal along +x."""
    plant, _, paddle = _arm_model()
    ik = InverseKinematics(plant, with_joint_limits=True)
    p = np.asarray(p_face, float)
    ik.AddPositionConstraint(paddle.body_frame(), P_FACE, plant.world_frame(), p - tol, p + tol)
    ik.AddOrientationConstraint(paddle.body_frame(), RotationMatrix(), plant.world_frame(), R, 1e-5)
    ik.prog().AddQuadraticErrorCost(np.eye(7), q_seed, ik.q())
    ik.prog().SetInitialGuess(ik.q(), q_seed)
    r = Solve(ik.prog())
    if not r.is_success():
        raise RuntimeError(f"IK failed for the paddle face at {np.round(p, 3)} (outside the reach {REACH_X}?)")
    return r.GetSolution(ik.q())

def joint_trajectory(face_xyz0, prof, dt=1e-4, K=200.0, q0=None):
    """Joint trajectory that moves the paddle face from face_xyz0 along +x by prof['x'](t), orientation fixed.
    Closed-loop differential kinematics: qdot = J^+ (V_des + K * [rotation error; position error]), integrated at dt.
    Returns dict(t, q, v, vdot, tau, face_err) on the grid, tau the feed-forward torque (inverse dynamics with gravity)."""
    plant, _, paddle = _arm_model()
    ctx = plant.CreateDefaultContext()
    W, F = plant.world_frame(), paddle.body_frame()
    t_move = prof["t_stop"] + 0.01
    t = np.arange(0.0, t_move + dt/2, dt)
    xd, vd = np.interp(t, prof["t"], prof["x"]), np.interp(t, prof["t"], prof["v"])
    p0 = np.asarray(face_xyz0, float)
    q = solve_ik(p0) if q0 is None else np.array(q0, float)
    Q, V, E = np.zeros((len(t), 7)), np.zeros((len(t), 7)), np.zeros(len(t))
    for k in range(len(t)):
        plant.SetPositions(ctx, q)
        X = plant.CalcRelativeTransform(ctx, W, F)
        p = X @ P_FACE
        p_des = p0 + np.array([xd[k], 0.0, 0.0])
        aa = (R_PUSH @ X.rotation().inverse()).ToAngleAxis()
        e_rot = aa.axis()*aa.angle()
        Vcmd = np.r_[K*e_rot, vd[k]*np.array([1.0, 0, 0]) + K*(p_des - p)]
        Jv = plant.CalcJacobianSpatialVelocity(ctx, JacobianWrtVariable.kV, F, P_FACE, W, W)
        qd = np.linalg.lstsq(Jv, Vcmd, rcond=None)[0]       # minimum-norm joint velocity
        Q[k], V[k], E[k] = q, qd, np.linalg.norm(p_des - p)
        q = q + qd*dt
    A = np.gradient(V, dt, axis=0)
    tau = np.zeros_like(Q)
    f_ext = MultibodyForces(plant)
    for k in range(len(t)):
        plant.SetPositions(ctx, Q[k]); plant.SetVelocities(ctx, V[k])
        tau[k] = plant.CalcInverseDynamics(ctx, A[k], f_ext) - plant.CalcGravityGeneralizedForces(ctx)
    return {"t": t, "q": Q, "v": V, "vdot": A, "tau": tau, "face_err": E}

def joint_limit_report(traj):
    """Peak |q|, |qdot| and |tau_ff| per joint as a fraction of the iiwa's limits (1.0 = at the limit)."""
    plant, iiwa, _ = _arm_model()
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    vmax = plant.GetVelocityUpperLimits()
    tmax = np.array([plant.get_joint_actuator(i).effort_limit() for i in plant.GetJointActuatorIndices(iiwa)])
    qfrac = np.max(np.maximum(traj["q"]/hi, traj["q"]/lo), axis=0)
    return {"q": qfrac, "v": np.max(np.abs(traj["v"]), axis=0)/vmax, "tau": np.max(np.abs(traj["tau"]), axis=0)/tmax}

# ----------------------------------------------------------------------------- 3. the arm in the scene

class _FaceCoordinate:
    """Stands in for Pusher.joint in push_slide's records: the paddle face's travel along x from its start, and its rate."""
    def __init__(self, arm):
        self.arm = arm
    def get_translation(self, ctx):
        X = self.arm.plant.EvalBodyPoseInWorld(ctx, self.arm.body)
        return float((X @ P_FACE)[0] - self.arm.x_touch0)
    def get_translation_rate(self, ctx):
        return float(self.arm.plant.EvalBodySpatialVelocityInWorld(ctx, self.arm.body).translational()[0])

@dataclass
class ArmPusher:
    """The iiwa with its paddle as a pusher: build_scene(..., pusher=ArmPusher()) and push.push_slide(..., pusher=ArmPusher())."""
    kp: tuple = (2e4, 2e4, 1e4, 1e4, 5e3, 2e3, 2e3)     # N m / rad, joints 1..7 (tracks the face to ~0.06 mm rms at 0.8 m/s)
    kd: tuple = (450., 450., 225., 225., 112., 45., 45.)  # N m s / rad
    gap: float = 0.001         # initial gap between the face and the object's back face, m
    clearance: float = 0.003   # plate's lower edge above the table, m
    lateral: float = 0.0       # plate centre offset along y from the object's centre, m
    feedforward: bool = True   # inverse-dynamics torque on top of the PD
    mu_p: float = PADDLE_MU    # paddle-object pair coefficient (fixed by scene.add_arm)
    yaw_deg: float = 0.0       # face misalignment: the arm keeps the face square (0)
    # filled by add() and connect()
    plant: object = field(default=None, repr=False, compare=False)
    instance: object = field(default=None, repr=False, compare=False)
    body: object = field(default=None, repr=False, compare=False)
    joint: object = field(default=None, repr=False, compare=False)
    x_touch0: float = field(default=None, repr=False, compare=False)
    face0: np.ndarray = field(default=None, repr=False, compare=False)
    traj: dict = field(default=None, repr=False, compare=False)

    def add(self, plant, contact, obj, x0):
        """Adds the iiwa and paddle (scene.add_arm) with an implicit PD on every joint; returns the paddle body."""
        self.plant = plant
        self.instance, self.body = add_arm(plant, contact)
        for i in range(7):      # (GetJointActuatorIndices is empty before Finalize, so look the actuators up by name)
            act = plant.get_mutable_joint_actuator(plant.GetJointActuatorByName(f"iiwa_joint_{i+1}", self.instance).index())
            act.set_controller_gains(PdControllerGains(p=self.kp[i], d=self.kd[i]))
        self.x_touch0 = x0 - obj.size[0]/2 - self.gap
        self.face0 = np.array([self.x_touch0, self.lateral, self.clearance + PADDLE[1]/2])
        self.joint = _FaceCoordinate(self)
        return self.body

    def face_x(self, q):
        """World x of the paddle face at travel q (as Pusher.face_x)."""
        return self.x_touch0 + q

    def plan(self, prof):
        self.traj = joint_trajectory(self.face0, prof)
        return self.traj

    def connect(self, builder, plant, prof, dt=0.0):
        """Plans the joint trajectory for prof, sets the arm's start pose, and wires the desired state [q, qdot] and
        the feed-forward torque into the plant.  Like push.connect, the command is fed one time step ahead."""
        tr = self.plan(prof)
        t_end = prof["t"][-1]
        t = np.r_[tr["t"], t_end] - dt
        q = np.vstack([tr["q"], tr["q"][-1]]).T
        v = np.vstack([tr["v"], np.zeros(7)]).T
        tau = np.vstack([tr["tau"], tr["tau"][-1]]) if self.feedforward else np.zeros((len(t), 7))
        if self.feedforward:      # holding still at the end: gravity only
            pa, _, _ = _arm_model()
            c = pa.CreateDefaultContext(); pa.SetPositions(c, tr["q"][-1])
            tau[-1] = -pa.CalcGravityGeneralizedForces(c)
        plant.SetDefaultPositions(self.instance, tr["q"][0])
        desired = builder.AddSystem(TrajectorySource(PiecewisePolynomial.CubicHermite(t, q, v), output_derivative_order=1))
        builder.Connect(desired.get_output_port(), plant.get_desired_state_input_port(self.instance))
        ff = builder.AddSystem(TrajectorySource(PiecewisePolynomial.FirstOrderHold(t, tau.T)))
        builder.Connect(ff.get_output_port(), plant.get_actuation_input_port(self.instance))

# ----------------------------------------------------------------------------- the whole thing

def slide_to_target(x_target, mu, mu_est=None, J=100.0, x0=0.55, a_push=A_PUSH, obj=None, pusher=None, meshcat=None,
                    keep_traj=False):
    """Plan with mu_est (default: the true mu, i.e. the reference method), push with the arm on a table of friction mu,
    and report where the object stopped.  mu_est may be a belief dict (see plan_peak_speed).
    Returns push_slide's dict plus x_target, mu_est, u (the planned peak speed), x_pred, miss (x_rest - x_target)
    and joint_limits (peak fraction of each joint's position, velocity and torque limit)."""
    obj = obj if obj is not None else Obj()
    pusher = pusher if pusher is not None else ArmPusher()
    est = mu if mu_est is None else mu_est
    u = plan_peak_speed(x_target, est, J, x0, pusher.gap, a_push)
    r = push_slide(mu, u, J, obj=obj, pusher=pusher, a_push=a_push, x0=x0, meshcat=meshcat, keep_traj=keep_traj)
    r.update(x_target=x_target, mu_est=est if not isinstance(est, dict) else float("nan"), u=u,
             x_pred=rest_x(u, mu, J, x0, pusher.gap, a_push), miss=r["x_rest"] - x_target, miss_y=r["y_rest"])
    if isinstance(pusher, ArmPusher):
        r["joint_limits"] = jl = joint_limit_report(pusher.traj)
        r["ik_face_err_max"] = float(pusher.traj["face_err"].max())
        r["over_limits"] = [f"joint {i+1} {k} x{jl[k][i]:.2f}" for k in ("q", "v", "tau") for i in range(7) if jl[k][i] > 1.0]
    return r

def report(r):
    lines = [f"target x = {r['x_target']:.3f} m, true mu = {r['mu']}, planned with mu = {r['mu_est']}, J = {r['J']:g} m/s^3 "
             f"-> peak paddle speed u = {r['u']:.4f} m/s, release speed (model) {r['v_r_cmd']:.4f} m/s",
             f"  landed at x = {r['x_rest']:.4f} m, y = {r['y_rest']*1e3:+.2f} mm: miss {r['miss']*1e3:+.2f} mm along the push "
             f"(model with the true mu predicted {r['x_pred']:.4f} m)",
             summary(r)]
    if "joint_limits" in r:
        jl = r["joint_limits"]
        lines.append("  joint use (peak / limit, joints 1-7): position " + " ".join(f"{x:.2f}" for x in jl["q"]) +
                     " | velocity " + " ".join(f"{x:.2f}" for x in jl["v"]) + " | ff torque " + " ".join(f"{x:.2f}" for x in jl["tau"]))
        lines.append(f"  differential-IK face error max {r['ik_face_err_max']*1e6:.1f} um")
        if r["over_limits"]:
            lines.append("  WARNING: this push exceeds the real iiwa's limits (the simulator does not enforce joint velocity limits): "
                         + ", ".join(r["over_limits"]))
    return "\n".join(lines)

def print_reach(J=100.0, x0=0.55):
    print(f"paddle face x reachable with the face square and 3 mm over the table (IK from the seed posture):")
    q = Q_SEED
    for xf in np.arange(0.46, 1.0, 0.02):
        try:
            q = solve_ik([xf, 0.0, 0.003 + PADDLE[1]/2], q)
            print(f"  x = {xf:.2f} m  ok   q = {np.round(q, 2)}")
        except RuntimeError:
            print(f"  x = {xf:.2f} m  --")
    print(f"fastest peak speed with the stroke inside x <= {REACH_X[1]} m (x0 = {x0}, J = {J:g}): {max_peak_speed(J, x0):.3f} m/s")
    print(f"fastest peak speed within the joint velocity limits: {limited_peak_speed(J, x0):.3f} m/s; "
          f"within velocity and torque limits: {limited_peak_speed(J, x0, which=('v', 'tau')):.3f} m/s")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", type=float, default=1.0, help="where the object's centre should stop, world x, m")
    ap.add_argument("--mu", type=float, default=0.3, help="true table friction in the simulator")
    ap.add_argument("--mu-est", type=float, default=None, help="friction the planner believes (default: the true mu)")
    ap.add_argument("--J", type=float, default=100.0, help="braking jerk, m/s^3")
    ap.add_argument("--a-push", type=float, default=A_PUSH)
    ap.add_argument("--x0", type=float, default=0.55, help="object start x, m")
    ap.add_argument("--shape", default="box", choices=["box", "disc", "ring", "footed"])
    ap.add_argument("--plate", action="store_true", help="use push.py's prescribed plate instead of the arm")
    ap.add_argument("--meshcat", action="store_true")
    ap.add_argument("--reach", action="store_true", help="print the arm's reach and the fastest push, then exit")
    a = ap.parse_args()
    if a.reach:
        return print_reach(a.J, a.x0)
    meshcat = StartMeshcat() if a.meshcat else None
    r = slide_to_target(a.target, a.mu, a.mu_est, a.J, a.x0, a.a_push, Obj(shape=a.shape),
                        Pusher() if a.plate else ArmPusher(), meshcat)
    print(report(r))
    if meshcat:
        input(f"Open {meshcat.web_url()} and press play on the recording; press Enter to quit. ")

if __name__ == "__main__":
    main()

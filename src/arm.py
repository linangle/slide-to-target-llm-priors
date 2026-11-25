#!/usr/bin/env python3
"""
arm.py -- the iiwa 14 as the pusher: a joint-space controller that drives the paddle along push.profile(), and the
target planner that picks the push's peak speed for a point on the board.

Two 6 Nov implementations are merged here (16 Nov).  The controller (ArmModel, ArmPusher, feasibility) is Angelina's; the
target planner (rest_x, plan_peak_speed, max_peak_speed, limited_peak_speed, slide_to_target, report, the --target and
--reach commands) is Diego's, now running on that controller.  Diego's differential-IK trajectory and his ArmPusher are
replaced by the controller below; joint_limit_report(pusher.traj) and the planner's signatures are kept, so callers of his
arm.py (e1_arm_validation.py, the evaluation scripts) run unchanged.

ArmPusher has the plate's hook (push.Pusher: add, connect, q, qd, face_x, set_displacement, record, summary), so
push.push_slide and the E0 scripts run with the arm by swapping `Pusher()` for `ArmPusher()`.

How it works.  The push is a straight line: the paddle face moves along world +x with its normal along the push, the
plate's lower edge `clearance` above the table and the plate centred on the object (lateral, yaw knobs as the plate's).
  1. The paddle pose fixes link 7's pose (scene.add_arm welds the plate 5 cm below the flange, face along link-7 +z), so
     the task is a 6-DoF frame path with one redundant joint.  The start posture is chosen among multi-start IK
     solutions (Drake InverseKinematics, random seeds) by integrating each along the whole path with the minimum-norm
     joint tangent dq/ds = J^+ e_x and keeping the one with the smallest peak joint-speed ratio (|dq_i/ds| / v_lim_i)
     that stays inside the joint limits and away from singularities (posture="speed", the default).  posture="limits"
     (19 Nov) plans the push with every such posture and keeps the one whose worst ratio of joint speed, torque and
     acceleration to its limit is lowest; the posture then depends on the speed of the push.
  2. The chosen path q(s) is a Hermite spline over the displacement s; the profile x(t), v(t), a(t) then gives
     q_d = q(x), qdot_d = q'(x) v, qddot_d = q''(x) v^2 + q'(x) a.
  3. Each joint runs Drake's implicit PD (JointActuator.set_controller_gains, SAP-family solver) on (q_d, qdot_d) fed one
     time step ahead, as the plate does, with the inverse dynamics of the arm along the desired trajectory (gravity,
     Coriolis, inertia incl. the reflected rotor inertia) as the feed-forward torque.  The plant clamps the total torque
     at the SDF effort limits (320, 320, 176, 176, 110, 40, 40 N m), so a brake the iiwa cannot deliver shows up as a
     tracking error, not as an unphysical push.  The braking deceleration is capped at `a_brake_max` (profile's a_max).
  4. q(ctx) and qd(ctx) report the paddle body's x displacement and speed from forward kinematics, the quantities the
     release model is checked against; record() adds the joint positions, speeds and net torques to the trajectory.

What the iiwa can deliver (decision 7).  The joint velocity limits are NOT enforced by the simulator, so plan() reports
the peak ratio of the planned joint speeds to the limits (85, 85, 100, 75, 130, 135, 135 deg/s) and feasibility()
tabulates it with the torque ratio over v_peak, J and the base position.  With the base on the push line the motion of
the wrist is radial from the shoulder, which only the elbow (joint 4, 75 deg/s) can produce: that caps the paddle speed
near 0.5 m/s.  A base beside the line lets joints 1 and 2 sweep the paddle instead.

Layouts (scene.LAYOUTS).  The default base stands beside the board at (0.6, 0.65) m ("side", since 16 Nov): 1.02 m/s within
the joint speed and torque limits.  "origin" is the layout before that, with the base on the push line: 0.49 m/s.  Every
function here takes base=, and --layout origin runs the old layout; experiments/arm_clearance.py checks that the links,
the base's footprint and the camera stay clear at a base.  Joint accelerations are set by the brake cap (a_brake_max):
over the SDF's limits at the default 12 m/s^2, inside them at 4 m/s^2, which only releases friction below about 0.4.
With posture="limits" and a 6 m/s^2 brake (releases friction below 0.61) every push from 0.15 to 1.14 m/s is inside the
speed, torque and acceleration limits at the side base (experiments/base_trial.py, results/side_base/base_trial/).

The target planner.  The object rides on the paddle to the release point and then slides d = v_r^2 / (2 mu g), so its centre
rests at
    x_rest(u, mu) = x0 - gap + release_push_distance(u, mu, J)  +  v_r(u, mu, J)^2 / (2 mu g),    v_r = u - (mu g)^2 / (2 J).
plan_peak_speed(x_target, mu) solves x_rest(u) = x_target for a friction value.  For a belief over mu it is decide.py's rule
with the target slide moving with the speed, d*(u, mu) = x_target - x_r(u, mu) (decide.point_target): one decision rule and one
loss for slide targets and point targets.  Speeds are bounded by u_max: by default the arm's reach along the line
(max_peak_speed); limited_peak_speed is the tighter bound from the joint speed (and torque) limits, which the simulator does
not enforce.

    python3 src/arm.py --mu 0.3 --v 0.8 --J 100            # one push with the arm, same report as push.py (+ arm metrics); --meshcat
    python3 src/arm.py --target 0.75 --mu 0.3              # plan the speed for a target point with the true mu, push, report the miss
    python3 src/arm.py --target 0.75 --mu 0.3 --mu-est 0.25   # plan with a wrong friction estimate
    python3 src/arm.py --target 0.75 --mu 0.3 --plate      # the same plan pushed by push.py's prescribed plate
    python3 src/arm.py --reach                             # the arm's stroke along the push line and the fastest pushes it allows
    python3 src/arm.py --reach --layout origin             # the same for the layout before 16 Nov (base on the push line)
    python3 src/arm.py --target 1.0 --mu 0.3 --a-max 4     # a push inside the speed, torque and acceleration limits
    python3 src/arm.py --plan --v 0.8 --J 100              # the plan alone: posture, joint-speed and torque ratios, no contact
    python3 src/arm.py --plan --v 1.0 --a-max 6 --posture limits   # the posture chosen for all three limits, 6 m/s^2 brake
    python3 src/arm.py --feasibility                       # ratios over v_peak x J at the default base
    python3 src/arm.py --scan                              # ratios over the base's lateral offset from the push line
"""
import argparse, math, time
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from pydrake.all import (AddDefaultVisualization, InverseKinematics, JacobianWrtVariable, JointActuatorIndex, MultibodyForces,
                         MultibodyPlant, PdControllerGains, PiecewisePolynomial, RigidTransform, RotationMatrix, Solve, StartMeshcat,
                         TrajectorySource)
from physics import A_PUSH, G, OBJECT_SIZE, release_push_distance, release_speed
from decide import Release, choose_speed, loss_sq_rel, point_target
from scene import DEFAULT_LAYOUT, FLANGE_Z, LAYOUTS, PADDLE, PADDLE_MU, PADDLE_Y, add_arm
from push import PLATE, Pusher, profile, push_slide, summary

# link-7 axes in the world during the push: x_L7 = +y_W (plate width), y_L7 = +z_W (the plate hangs below the flange),
# z_L7 = +x_W (the plate's far face pushes).  Columns are the link-7 axes.
R_WL7 = RotationMatrix(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
X_L7P = RigidTransform([0.0, PADDLE_Y, FLANGE_Z + PADDLE[2]/2])        # paddle body (plate centre) in link 7, as add_arm welds it
Q_DISPLAY = np.array([0.0, 0.6, 0.0, -1.4, 0.0, 1.1, 0.0])              # scene.py's display pose, one IK seed
V_LIM_DEG = (85, 85, 100, 75, 130, 135, 135)                            # KUKA LBR iiwa 14 R820 joint speed limits, deg/s (the SDF's)
A_BRAKE = 12.0                                                          # m/s^2, the arm's cap on the braking deceleration (ArmPusher.a_brake_max)
POSTURES = ("speed", "limits")                                          # how ArmPusher chooses its start posture among the IK solutions
POSTURE = "speed"                                                       # the default: the choice every result up to 19 Nov was made with
# Base positions by layout name (scene.LAYOUTS): "side" (beside the board, the default since 16 Nov) and "origin" (the layout
# before: the push runs radially out from the base, so the elbow makes nearly all of the paddle speed).
BASE = LAYOUTS[DEFAULT_LAYOUT]                                          # the default base (link 0) position in the world, m

def layout_results(layout):
    """Folder for a layout's saved outputs: results/side_base/ for "side"; the folders in use before 16 Nov (results/e1/,
    results/arm_rerun/, ...) for "origin", returned as results/ itself.  None for any other base: pass paths explicitly."""
    from paths import RESULTS
    return {"side": RESULTS/"side_base", "origin": RESULTS}.get(layout)

def start_posture(x_face, base=BASE, clearance=0.003, length=0.3):
    """Joint angles that put the paddle face at world x_face on the push line, in the posture ArmPusher would start a push
    of `length` m from (for scripts that pose the arm without simulating it)."""
    m = ArmModel(tuple(base))
    X0 = link7_pose(x_face, 0.0, 0.0, clearance)
    paths = [m.path(q, X0, length) for q in m.candidates(X0)]
    ok = [p for p in paths if p["ok"]] or paths
    if not ok:
        raise RuntimeError(f"no IK solution for the paddle face at x = {x_face:.3f} m (base {tuple(base)})")
    return min(ok, key=lambda p: (p["qd_per_mps"], p["tau_g_ratio"]))["q"][0]

def layout_base(layout):
    """Base position for a layout name ("origin", "side") or an (x, y[, z]) position."""
    if isinstance(layout, str):
        return LAYOUTS[layout]
    b = tuple(float(v) for v in layout)
    return b if len(b) == 3 else (*b, 0.0)

def link7_pose(x_face, lateral, yaw_deg, clearance, plate=PLATE):
    """Link-7 pose that puts the plate's face (nearest point to the object) at world x_face, the plate centre at y = lateral,
    its lower edge `clearance` above the table, face normal yawed by yaw_deg about z."""
    th, wd, ht = plate
    yaw = math.radians(yaw_deg)
    x_centre = x_face - wd/2*abs(math.sin(yaw)) - th/2*math.cos(yaw)   # a yawed plate's corner reaches forward (push.Pusher.add)
    X_WP = RigidTransform(RotationMatrix.MakeZRotation(yaw) @ R_WL7, [x_centre, lateral, clearance + ht/2])
    return X_WP @ X_L7P.inverse()

def pose_error(X_target, X):
    """6-vector [rotation (axis-angle), translation] taking X to X_target, in the world frame; matches Drake's spatial Jacobian."""
    aa = (X_target.rotation() @ X.rotation().inverse()).ToAngleAxis()
    return np.concatenate([aa.angle()*aa.axis(), X_target.translation() - X.translation()])

class ArmModel:
    """Kinematics-and-dynamics copy of the arm (no contact, continuous plant) for IK, Jacobians and inverse dynamics."""

    def __init__(self, base=BASE):
        self.plant = MultibodyPlant(0.0)
        self.iiwa, self.paddle = add_arm(self.plant, None, RigidTransform(np.asarray(base, float)), collision=False)
        self.plant.Finalize()
        self.ctx = self.plant.CreateDefaultContext()
        self.L7 = self.plant.GetFrameByName("iiwa_link_7", self.iiwa)
        self.W = self.plant.world_frame()
        self.q_lo, self.q_hi = self.plant.GetPositionLowerLimits(), self.plant.GetPositionUpperLimits()
        self.v_lim = self.plant.GetVelocityUpperLimits()
        self.a_lim = self.plant.GetAccelerationUpperLimits()
        acts = [self.plant.get_joint_actuator(i) for i in self.plant.GetJointActuatorIndices(self.iiwa)]
        self.tau_lim = np.array([a.effort_limit() for a in acts])
        self.rotor = np.array([a.default_rotor_inertia()*a.default_gear_ratio()**2 for a in acts])   # reflected inertia, kg m^2
        self.n = self.plant.num_positions()

    def fk(self, q):
        self.plant.SetPositions(self.ctx, q)
        return self.plant.CalcRelativeTransform(self.ctx, self.W, self.L7)

    def jacobian(self, q):
        """6 x n spatial Jacobian of link 7's origin in the world: rows [omega; v]."""
        self.plant.SetPositions(self.ctx, q)
        return self.plant.CalcJacobianSpatialVelocity(self.ctx, JacobianWrtVariable.kV, self.L7, np.zeros(3), self.W, self.W)

    def mass_matrix(self, q):
        self.plant.SetPositions(self.ctx, q)
        return self.plant.CalcMassMatrix(self.ctx)

    def inverse_dynamics(self, q, v, vdot):
        """Joint torques that produce vdot at (q, v) under gravity (rotor inertia included)."""
        self.plant.SetPositions(self.ctx, q); self.plant.SetVelocities(self.ctx, v)
        forces = MultibodyForces(self.plant)
        self.plant.CalcForceElementsContribution(self.ctx, forces)
        return self.plant.CalcInverseDynamics(self.ctx, vdot, forces)

    def project(self, q, X_target, iters=4, tol=1e-10):
        """Newton steps onto the pose constraint from a nearby q; returns (q, pose error norm)."""
        for _ in range(iters):
            e = pose_error(X_target, self.fk(q))
            if np.linalg.norm(e) < tol:
                break
            q = q + np.linalg.lstsq(self.jacobian(q), e, rcond=None)[0]
        return q, float(np.linalg.norm(pose_error(X_target, self.fk(q))))

    def ik(self, X_target, q_seed, tol=1e-6):
        """One IK solve near q_seed; None if it fails."""
        ik = InverseKinematics(self.plant, self.ctx)
        p = X_target.translation()
        ik.AddPositionConstraint(self.L7, np.zeros(3), self.W, p - tol, p + tol)
        ik.AddOrientationConstraint(self.W, X_target.rotation(), self.L7, RotationMatrix(), tol)
        prog, q = ik.prog(), ik.q()
        prog.AddQuadraticErrorCost(1e-3*np.eye(self.n), q_seed, q)
        prog.SetInitialGuess(q, q_seed)
        res = Solve(prog)
        if not res.is_success():
            return None
        qs, err = self.project(res.GetSolution(q), X_target)
        return qs if err < 1e-8 and np.all(qs > self.q_lo) and np.all(qs < self.q_hi) else None

    def candidates(self, X_target, n_random=40, seed=0):
        """Distinct IK solutions at X_target from the display pose, its mirror and random seeds inside the joint limits."""
        rng = np.random.default_rng(seed)
        seeds = [Q_DISPLAY, Q_DISPLAY*np.array([1, 1, 1, -1, 1, -1, 1])] + \
                [rng.uniform(self.q_lo, self.q_hi) for _ in range(n_random)]
        out = []
        for s in seeds:
            q = self.ik(X_target, s)
            if q is not None and all(np.max(np.abs(q - o)) > 0.05 for o in out):
                out.append(q)
        return out

    def path(self, q0, X0, length, ds=1e-3, margin=0.02, sigma_min=0.03):
        """Integrates the straight +x path of length `length` from posture q0 at pose X0 with the minimum-norm joint tangent.
        Returns dict(s, q, dqds, sigma, ok, reason, s_ok, qd_per_mps, qd_joint, tau_g_ratio, limit_margin) -- qd_per_mps is the
        peak over the path of max_i |dq_i/ds| / v_lim_i, the joint-speed ratio per 1 m/s of paddle speed; s_ok is how far along
        the line this posture stays inside the joint limits and away from singularities (= length when ok)."""
        s = np.arange(0.0, length + ds/2, ds)
        if s[-1] < length:
            s = np.append(s, length)
        n = len(s)
        q, dq, sig = np.empty((n, self.n)), np.empty((n, self.n)), np.empty(n)
        qk, ok, reason, s_ok = np.array(q0, float), True, "", float(s[-1])
        e_x = np.array([0, 0, 0, 1.0, 0, 0])
        for k, sk in enumerate(s):
            X = RigidTransform(X0.rotation(), X0.translation() + np.array([sk, 0, 0]))
            qk, err = self.project(qk, X)
            Jk = self.jacobian(qk)
            sv = np.linalg.svd(Jk, compute_uv=False)
            q[k], sig[k] = qk, sv[-1]
            dq[k] = np.linalg.lstsq(Jk, e_x, rcond=None)[0]
            if ok and (err > 1e-6 or sv[-1] < sigma_min):
                ok, reason, s_ok = False, f"singular at s = {sk:.3f} m (sigma_min {sv[-1]:.3f}, pose error {err:.1e})", float(s[max(k - 1, 0)])
            if ok and (np.any(qk < self.q_lo + margin) or np.any(qk > self.q_hi - margin)):
                j = int(np.argmin(np.minimum(qk - self.q_lo, self.q_hi - qk)))
                ok, reason, s_ok = False, f"joint {j + 1} at its limit at s = {sk:.3f} m", float(s[max(k - 1, 0)])
            if k + 1 < n:
                qk = qk + dq[k]*(s[k + 1] - sk)
        ratio = np.abs(dq)/self.v_lim
        kmax = np.unravel_index(np.argmax(ratio), ratio.shape)
        tau_g = np.array([self.inverse_dynamics(qq, np.zeros(self.n), np.zeros(self.n)) for qq in q[::max(1, n//20)]])
        return {"s": s, "q": q, "dqds": dq, "sigma": sig, "ok": ok, "reason": reason, "s_ok": s_ok, "qd_per_mps": float(ratio[kmax]),
                "qd_joint": int(kmax[1]) + 1, "tau_g_ratio": float(np.max(np.abs(tau_g)/self.tau_lim)),
                "limit_margin": float(np.min(np.minimum(q - self.q_lo, self.q_hi - q)))}

    def effective_mass(self, q):
        """Apparent mass of the paddle along +x at q: 1 / (J_x M^-1 J_x^T), kg (what the plate harness modelled as 5 kg)."""
        J = self.jacobian(q)[3]
        return float(1.0/(J @ np.linalg.solve(self.mass_matrix(q), J)))

@dataclass
class ArmPusher:
    """The iiwa pushing with its paddle; the same hook as push.Pusher (see the module docstring)."""
    mu_p: float = PADDLE_MU    # paddle-object pair coefficient
    lateral: float = 0.0       # plate centre offset along y from the object's centre, m
    yaw_deg: float = 0.0       # face misalignment about z, deg
    gap: float = 0.001         # initial gap between the face's nearest point and the object's back face, m
    clearance: float = 0.003   # plate's lower edge above the table, m
    plate: tuple = PLATE
    base: tuple = BASE         # iiwa base (link 0) position in the world, m; the default layout's (scene.LAYOUTS)
    a_brake_max: float = A_BRAKE   # cap on the braking deceleration, m/s^2 (> mu g for every pair, so the release is unchanged)
    kp: float = 4e4            # joint PD stiffness, N m/rad (about 1e5 N/m at the paddle, the plate's value)
    zeta: float = 1.0          # joint PD damping ratio against the diagonal mass (incl. rotor inertia) at the display pose
    posture_seed: int = 0      # RNG seed of the multi-start IK
    posture: str = POSTURE     # which IK posture starts the push: "speed" = lowest peak joint speed per m/s of paddle speed (the
                               # default); "limits" = lowest worst ratio of the planned joint speed, torque and acceleration to
                               # their limits for this push (19 Nov; with a 6 m/s^2 brake, inside all three up to 1.14 m/s)
    # filled by add() / connect()
    model: ArmModel = field(default=None, repr=False, compare=False)
    body: object = field(default=None, repr=False, compare=False)      # paddle body in the scene plant
    instance: object = field(default=None, repr=False, compare=False)  # iiwa model instance in the scene plant
    plant: object = field(default=None, repr=False, compare=False)
    x_touch0: float = field(default=None, repr=False, compare=False)
    X0: RigidTransform = field(default=None, repr=False, compare=False)   # link-7 start pose
    x_paddle0: float = field(default=None, repr=False, compare=False)     # paddle body x at the start
    plan: dict = field(default=None, repr=False, compare=False)
    spline: object = field(default=None, repr=False, compare=False)
    mass: float = field(default=None, repr=False, compare=False)         # effective mass along x at the start (report only)

    # ------------------------------------------------------------------ the hook
    def add(self, plant, contact, obj, x0):
        """Adds the iiwa and its paddle to the scene plant with the plate `gap` behind the object's back face; returns the paddle."""
        self.plant = plant
        self.x_touch0 = x0 - obj.size[0]/2 - self.gap
        self.model = ArmModel(self.base)
        self.X0 = link7_pose(self.x_touch0, self.lateral, self.yaw_deg, self.clearance, self.plate)
        self.x_paddle0 = (self.X0 @ X_L7P).translation()[0]
        self.instance, self.body = add_arm(plant, contact, RigidTransform(np.asarray(self.base, float)), self.mu_p)
        # implicit PD on every iiwa joint (GetJointActuatorIndices(instance) is empty before Finalize, so scan all actuators;
        # the iiwa's are added in joint order 1..7, the order of the mass-matrix diagonal)
        M = np.diag(self.model.mass_matrix(Q_DISPLAY))
        acts = [plant.get_mutable_joint_actuator(JointActuatorIndex(i)) for i in range(plant.num_actuators())]
        acts = [a for a in acts if a.model_instance() == self.instance]
        assert len(acts) == self.model.n, "expected one actuator per iiwa joint"
        for i, act in enumerate(acts):
            act.set_controller_gains(PdControllerGains(p=self.kp, d=2*self.zeta*math.sqrt(self.kp*M[i])))
        return self.body

    def __post_init__(self):
        if self.posture not in POSTURES:
            raise ValueError(f"posture is one of {POSTURES}, not {self.posture!r}")

    def _trajectory(self, path, prof):
        """The desired joint trajectory of one posture's path along the profile, the arm's inverse dynamics as feed-forward,
        and the peak ratios of joint speed, acceleration and torque to their limits while the paddle moves."""
        m = self.model
        spline = PiecewisePolynomial.CubicHermite(path["s"], path["q"].T, path["dqds"].T)
        t, x, v, a = prof["t"], prof["x"], prof["v"], prof["a"]
        xs = np.clip(x, 0.0, path["s"][-1])
        q_d = np.array([spline.value(s)[:, 0] for s in xs])
        dq = np.array([spline.EvalDerivative(s, 1)[:, 0] for s in xs])
        d2q = np.array([spline.EvalDerivative(s, 2)[:, 0] for s in xs])
        qd_d = dq*v[:, None]
        qdd_d = d2q*(v*v)[:, None] + dq*a[:, None]
        tau = np.array([m.inverse_dynamics(q_d[k], qd_d[k], qdd_d[k]) for k in range(len(t))])
        move = (t >= t[1]) & (t <= prof["t_stop"] + 1e-9)
        rv, ra, rt = np.abs(qd_d)/m.v_lim, np.abs(qdd_d)/m.a_lim, np.abs(tau)/m.tau_lim
        i_v, i_a, i_t = (np.unravel_index(np.argmax(r[move]), r[move].shape) for r in (rv, ra, rt))
        return dict(spline=spline, t=t, q_d=q_d, qd_d=qd_d, qdd_d=qdd_d, tau_ff=tau, move=move,
                    qd_ratio=float(rv[move][i_v]), qd_ratio_joint=int(i_v[1]) + 1, qdd_ratio=float(ra[move][i_a]), qdd_ratio_joint=int(i_a[1]) + 1,
                    tau_ratio=float(rt[move][i_t]), tau_ratio_joint=int(i_t[1]) + 1, tau_max=np.abs(tau[move]).max(axis=0),
                    qd_max=np.abs(qd_d[move]).max(axis=0))

    def make_plan(self, prof):
        """Chooses the posture (self.posture) and plans q(s) over the profile's travel; fills self.plan and self.spline."""
        m, length = self.model, prof["x_stop"] + 0.002
        cands = m.candidates(self.X0, seed=self.posture_seed)
        if not cands:
            raise RuntimeError(f"no IK solution for the paddle face at x = {self.x_touch0:.3f} m (base {self.base})")
        paths = [m.path(q, self.X0, length) for q in cands]
        ok = [p for p in paths if p["ok"]]
        if not ok:
            raise RuntimeError("no posture covers the push: " + "; ".join(sorted({p["reason"] for p in paths})))
        if self.posture == "limits":
            # plan this push with every posture that covers it and keep the one whose worst limit ratio is lowest.  The
            # joint accelerations are highest where the paddle brakes, which depends on the speed, so the posture does too
            plans = [self._trajectory(p, prof) for p in ok]
            i = min(range(len(ok)), key=lambda k: max(plans[k][r] for r in ("qd_ratio", "tau_ratio", "qdd_ratio")))
            best, tr = ok[i], plans[i]
        else:
            best = min(ok, key=lambda p: (p["qd_per_mps"], p["tau_g_ratio"]))
            tr = self._trajectory(best, prof)
        self.spline = tr.pop("spline")
        self.mass = m.effective_mass(best["q"][0])
        self.plan = dict(best, n_candidates=len(cands), n_ok=len(ok), q_start=best["q"][0], **tr, m_eff=self.mass, length=length,
                         sigma_min=float(best["sigma"].min()), posture=self.posture)
        return self.plan

    def connect(self, builder, plant, prof, dt=0.0):
        """Plans the joint trajectory for `prof`, poses the arm at its start and wires desired state + feed-forward torque into
        the plant's implicit PD, one time step ahead as for the plate."""
        plan = self.make_plan(prof)
        plant.SetDefaultPositions(self.instance, plan["q_start"])
        t = prof["t"] - dt
        pp_q = PiecewisePolynomial.CubicHermite(t, plan["q_d"].T, plan["qd_d"].T)
        desired = builder.AddSystem(TrajectorySource(pp_q, output_derivative_order=1))
        builder.Connect(desired.get_output_port(), plant.get_desired_state_input_port(self.instance))
        pp_tau = PiecewisePolynomial.FirstOrderHold(t, plan["tau_ff"].T)
        builder.Connect(builder.AddSystem(TrajectorySource(pp_tau)).get_output_port(), plant.get_actuation_input_port(self.instance))

    def face_x(self, q):
        """World x of the face's nearest point to the object at displacement q."""
        return self.x_touch0 + q

    @property
    def traj(self):
        """The planned joint trajectory while the paddle moves, in the layout joint_limit_report reads: dict(t, q, v, vdot, tau)
        of n x 7 arrays plus the iiwa's limits (q_lo, q_hi, v_lim, tau_lim, a_lim).  None before a push has been planned."""
        if self.plan is None:
            return None
        p, m, k = self.plan, self.model, self.plan["move"]
        return {"t": p["t"][k], "q": p["q_d"][k], "v": p["qd_d"][k], "vdot": p["qdd_d"][k], "tau": p["tau_ff"][k],
                "q_lo": m.q_lo, "q_hi": m.q_hi, "v_lim": m.v_lim, "tau_lim": m.tau_lim, "a_lim": m.a_lim}

    def q(self, ctx):
        """Paddle displacement along the push from its start pose, from forward kinematics, m."""
        return float(self.plant.EvalBodyPoseInWorld(ctx, self.body).translation()[0] - self.x_paddle0)

    def qd(self, ctx):
        return float(self.plant.EvalBodySpatialVelocityInWorld(ctx, self.body).translational()[0])

    def set_displacement(self, ctx, q):
        """Poses the arm along its planned path at displacement q (E0-5's camera replay)."""
        if self.spline is None:
            self.make_plan(profile(0.8, 100.0))
        self.plant.SetPositions(ctx, self.instance, self.spline.value(np.clip(q, 0.0, self.plan["s"][-1]))[:, 0])

    def record(self, ctx):
        """Per-step joint positions, speeds and net (clamped) actuation torques."""
        return {"q_arm": self.plant.GetPositions(ctx, self.instance), "v_arm": self.plant.GetVelocities(ctx, self.instance),
                "tau": self.plant.get_net_actuation_output_port(self.instance).Eval(ctx)}

    def summary(self, tr):
        """Arm scalars for push_slide's result: planned and actual joint-speed and torque ratios to the iiwa's limits."""
        m, p = self.model, self.plan
        rt, rv = np.abs(tr["tau"])/m.tau_lim, np.abs(tr["v_arm"])/m.v_lim
        it, iv = np.unravel_index(np.argmax(rt), rt.shape), np.unravel_index(np.argmax(rv), rv.shape)
        # the paddle's pose error against the straight line while it touches the object: z (clearance) and y drift, and its tilt
        X = [self.model.fk(q) for q in tr["q_arm"][tr["f"] > 0.01][::5]] or [self.model.fk(tr["q_arm"][0])]
        pz = np.array([Xk.translation()[2] for Xk in X]); py = np.array([Xk.translation()[1] for Xk in X])
        ang = np.array([np.linalg.norm(pose_error(RigidTransform(self.X0.rotation(), Xk.translation()), Xk)[:3]) for Xk in X])
        return {"arm_tau_ratio": float(rt[it]), "arm_tau_joint": int(it[1]) + 1, "arm_qd_ratio": float(rv[iv]), "arm_qd_joint": int(iv[1]) + 1,
                "arm_plan_qd_ratio": p["qd_ratio"], "arm_plan_qd_joint": p["qd_ratio_joint"], "arm_plan_tau_ratio": p["tau_ratio"],
                "arm_plan_tau_joint": p["tau_ratio_joint"], "arm_plan_qdd_ratio": p["qdd_ratio"], "arm_plan_qdd_joint": p["qdd_ratio_joint"],
                "arm_m_eff": p["m_eff"],
                "arm_sigma_min": p["sigma_min"], "arm_postures": p["n_ok"], "arm_z_drift_mm": float((pz.max() - pz.min())*1e3),
                "arm_y_drift_mm": float((py.max() - py.min())*1e3), "arm_tilt_max_deg": float(math.degrees(ang.max())), "arm_base_y": self.base[1]}

    def describe(self):
        p = self.plan
        return (f"arm plan: {p['n_ok']} of {p['n_candidates']} postures cover the {p['length']*1e3:.0f} mm push; start q = {np.round(p['q_start'], 3)}; "
                f"effective mass along x {p['m_eff']:.1f} kg; sigma_min {p['sigma_min']:.3f}\n"
                f"  joint speed: peak {p['qd_ratio']:.2f} x limit (joint {p['qd_ratio_joint']}); max |qd| deg/s = {np.round(np.degrees(p['qd_max']), 0)}\n"
                f"  torque:      peak {p['tau_ratio']:.2f} x effort limit (joint {p['tau_ratio_joint']}); max |tau| N m = {np.round(p['tau_max'], 0)} of {m_lim(self)}\n"
                f"  acceleration: peak {p['qdd_ratio']:.2f} x the SDF's drake:acceleration limit (joint {p['qdd_ratio_joint']}); set by the brake cap, "
                f"see --reach")

def m_lim(pusher):
    return np.round(pusher.model.tau_lim, 0)

# ---------------------------------------------------------------------- feasibility: what the iiwa can deliver

def feasibility(v_peak, J, base=BASE, x0=0.55, a_max=A_BRAKE, lateral=0.0, yaw_deg=0.0, obj_size_x=OBJECT_SIZE[0], gap=0.001, a_push=A_PUSH,
                posture=POSTURE):
    """Plans the push without contact and returns the plan's ratios (or the reason it is impossible)."""
    pu = ArmPusher(lateral=lateral, yaw_deg=yaw_deg, gap=gap, base=tuple(base), a_brake_max=a_max, posture=posture)
    pu.x_touch0 = x0 - obj_size_x/2 - pu.gap
    pu.model = ArmModel(base)
    pu.X0 = link7_pose(pu.x_touch0, lateral, yaw_deg, pu.clearance)
    prof = profile(v_peak, J, a_push, a_max=a_max)
    try:
        p = pu.make_plan(prof)
    except RuntimeError as ex:
        return {"v_peak": v_peak, "J": J, "base_y": base[1], "ok": False, "reason": str(ex), "x_face_end": pu.x_touch0 + prof["x_stop"]}
    return {"v_peak": v_peak, "J": J, "base_y": base[1], "ok": True, "qd_ratio": p["qd_ratio"], "qd_joint": p["qd_ratio_joint"], "tau_ratio": p["tau_ratio"],
            "tau_joint": p["tau_ratio_joint"], "qdd_ratio": p["qdd_ratio"], "qdd_joint": p["qdd_ratio_joint"], "m_eff": p["m_eff"], "sigma_min": p["sigma_min"], "x_face_end": pu.x_touch0 + prof["x_stop"],
            "v_max_by_speed": v_peak/p["qd_ratio"], "postures": p["n_ok"]}

def table(rows, key, title):
    vs, js = sorted({r["v_peak"] for r in rows}), sorted({r["J"] for r in rows})
    lines = [title, f"{'v_peak':>7} | " + " | ".join(f"J={J:g}".rjust(14) for J in js)]
    for v in vs:
        cells = []
        for J in js:
            r = next((r for r in rows if r["v_peak"] == v and r["J"] == J), None)
            cells.append("".rjust(14) if r is None else (f"{r[key]:.2f} (j{r[key.replace('ratio', 'joint')]})".rjust(14) if r["ok"] else "unreachable".rjust(14)))
        lines.append(f"{v:>7} | " + " | ".join(cells))
    return "\n".join(lines)

# ---------------------------------------------------------------------- the target planner: target point -> peak paddle speed

def rest_x(u, mu, J, x0=0.55, gap=0.001, a_push=A_PUSH, g=G):
    """Where the object's centre comes to rest for commanded peak speed u (the release model plus the free slide)."""
    v_r = float(release_speed(u, mu, J, g))
    return x0 - gap + release_push_distance(u, mu, J, a_push, g) + v_r*v_r/(2*mu*g)

def plan_peak_speed(x_target, mu, J=100.0, x0=0.55, gap=0.001, a_push=A_PUSH, u_max=None, g=G, loss=None, base=BASE, clip=False):
    """Commanded peak paddle speed that lands the object's centre at x_target.

    mu: a friction value (a point estimate, or the true value for the reference method): solves rest_x(u) = x_target.
        Or a belief dict with 'grid' (ln mu) and 'pmf' (belief.posterior's output): decide.choose_speed with the target slide
        d*(u, mu) = x_target - x_r(u, mu) (decide.point_target) under `loss` (default decide.loss_sq_rel, the specification's;
        decide.loss_sq_abs is the squared miss E[(x_rest - x_target)^2]).
    u_max: the fastest push allowed (default: max_peak_speed, the arm's reach from x0 with the base at `base`; pass
        limited_peak_speed(...) to stay inside the joint speed limits).  A belief's speed is clipped at u_max.
    Raises ValueError for a friction value whose target is behind the release point or beyond what u_max can reach.
    clip=True returns u_max for a target beyond reach instead of raising: the push is made at the cap and its miss is
    scored, as a belief's is (the proposal's rule for requests above the speed bound)."""
    u_max = max_peak_speed(J, x0, gap, a_push, base=base) if u_max is None else u_max
    if isinstance(mu, dict):
        return float(choose_speed(mu, Release(J=J, g=g), point_target(x_target, x0, gap, a_push), loss or loss_sq_rel, u_max=u_max)["u"])
    f = lambda u: rest_x(u, mu, J, x0, gap, a_push, g) - x_target
    if f(0.0) >= 0:
        raise ValueError(f"target x = {x_target:.3f} m is behind the object's start ({x0:.3f} m)")
    if f(u_max) < 0:
        if clip:
            return float(u_max)
        raise ValueError(f"target x = {x_target:.3f} m needs more than the fastest push allowed, u = {u_max:.2f} m/s "
                         f"(reaches {rest_x(u_max, mu, J, x0, gap, a_push, g):.3f} m at mu = {mu}); move x0 or the base, or raise u_max")
    return float(brentq(f, 0.0, u_max, xtol=1e-9))

@lru_cache(maxsize=64)
def reach(base=BASE, x0=0.55, gap=0.001, obj_len=OBJECT_SIZE[0], lateral=0.0, clearance=0.003):
    """Longest straight stroke of the paddle from its start pose behind an object at x0, m: the furthest any IK posture
    follows the push line inside the joint limits and away from singularities (ArmModel.path's s_ok)."""
    m = ArmModel(tuple(base))
    X0 = link7_pose(x0 - obj_len/2 - gap, lateral, 0.0, clearance)
    return max((m.path(q, X0, 1.2)["s_ok"] for q in m.candidates(X0)), default=0.0)

def max_peak_speed(J=100.0, x0=0.55, gap=0.001, a_push=A_PUSH, obj_len=OBJECT_SIZE[0], base=BASE, a_max=A_BRAKE):
    """Fastest commanded peak speed whose whole paddle stroke (profile x_stop, plus the 2 mm ArmPusher plans past it) stays
    inside the arm's reach.  Says nothing about the joint speed limits: see limited_peak_speed."""
    room = reach(tuple(base), x0, gap, obj_len) - 0.002
    f = lambda u: profile(u, J, a_push, a_max=a_max)["x_stop"] - room
    if f(1e-3) >= 0:
        return 0.0
    return float(brentq(f, 1e-3, 5.0, xtol=1e-6)) if f(5.0) > 0 else 5.0

@lru_cache(maxsize=64)
def limited_peak_speed(J=100.0, x0=0.55, gap=0.001, a_push=A_PUSH, obj_len=OBJECT_SIZE[0], which=("v",), tol=0.005, base=BASE, a_max=A_BRAKE,
                       posture=POSTURE):
    """Fastest commanded peak speed whose planned joint trajectory stays within the iiwa's limits: which = ("v",) checks the
    joint speeds (the binding ones, and not enforced by the simulator), add "tau" for the inverse-dynamics torques and "a"
    for the joint accelerations (the SDF's drake:acceleration values, 8.6-15.7 rad/s^2; KUKA publishes none).  The
    accelerations are set mostly by the brake cap a_max, not by the speed: at 12 m/s^2 they are over the limit at any useful
    speed, at 4 m/s^2 they are inside it up to the speed limit (but a cap must stay above mu g for the object to leave).
    Bisection on u to tol m/s over feasibility(), about 8 plans (a few seconds); cached.  0.0 if even a 2 cm/s push is over.
    The bisection takes every slower push to be inside the limits too.  With "a" checked that can fail: a slow push brakes
    near the start pose, and the posture can change with the stroke.  limit_scan() checks a whole range of speeds."""
    key = {"v": "qd_ratio", "tau": "tau_ratio", "a": "qdd_ratio"}
    def worst(u):
        r = feasibility(u, J, tuple(base), x0, a_max, obj_size_x=obj_len, gap=gap, a_push=a_push, posture=posture)
        return max(r[key[k]] for k in which) if r["ok"] else math.inf
    lo, hi = 0.02, max_peak_speed(J, x0, gap, a_push, obj_len, tuple(base), a_max)
    if hi <= lo or worst(lo) > 1.0:
        return 0.0
    if worst(hi) <= 1.0:
        return hi
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if worst(mid) <= 1.0 else (lo, mid)
    return lo

def joint_limit_report(traj):
    """Peak |q|, |qdot|, |tau| and |qddot| per joint as a fraction of the iiwa's limits (1.0 = at the limit), from a planned
    joint trajectory (ArmPusher.traj): keys "q", "v", "tau", "a".  These are the plan's numbers (desired motion and
    inverse-dynamics torque); the torques the simulation actually applied are in push_slide's result (arm_tau_ratio).
    The acceleration limits are the SDF's drake:acceleration values; the simulator enforces neither them nor the speeds."""
    if "a_lim" not in traj:
        m = ArmModel()
        traj = dict(traj, q_lo=m.q_lo, q_hi=m.q_hi, v_lim=m.v_lim, tau_lim=m.tau_lim, a_lim=m.a_lim)
    return {"q": np.max(np.maximum(traj["q"]/traj["q_hi"], traj["q"]/traj["q_lo"]), axis=0),
            "v": np.max(np.abs(traj["v"]), axis=0)/traj["v_lim"], "tau": np.max(np.abs(traj["tau"]), axis=0)/traj["tau_lim"],
            "a": np.max(np.abs(traj["vdot"]), axis=0)/traj["a_lim"]}

def affordable(u_max, mu, J=100.0, x0=0.55, gap=0.001, a_push=A_PUSH, g=G):
    """What a speed bound affords on friction mu: dict(slide, x_release, x_rest) for a push at u_max -- the longest free slide
    v_r^2 / (2 mu g), where the paddle lets go, and the farthest point the object's centre can be sent to."""
    v_r = float(release_speed(u_max, mu, J, g))
    x_r = x0 - gap + release_push_distance(u_max, mu, J, a_push, g)
    return {"slide": v_r*v_r/(2*mu*g), "x_release": x_r, "x_rest": x_r + v_r*v_r/(2*mu*g)}

def target_grid(u_max, mu_hi, n=3, J=100.0, x0=0.55, near=0.05, headroom=0.9, gap=0.001, a_push=A_PUSH):
    """Target points (world x of the object's centre) re-derived from a speed bound: n evenly spaced from x0 + near to the
    farthest point the HIGHEST-friction pair reaches at `headroom` of u_max, rounded down to the centimetre.  Every pair
    then reaches every target with a plan below the cap, so a push at the cap means a method asked for more than the arm
    has; the headroom leaves room for that to show without the true-friction plan sitting on the cap itself."""
    far = math.floor(affordable(headroom*u_max, mu_hi, J, x0, gap, a_push)["x_rest"]*100)/100
    if far <= x0 + near:
        raise ValueError(f"a {u_max:.2f} m/s cap sends mu = {mu_hi} only to x = {far:.2f} m: no room for targets beyond x0 + {near} m")
    return [round(float(x), 3) for x in np.linspace(x0 + near, far, n)]

def slide_to_target(x_target, mu, mu_est=None, J=100.0, x0=0.55, a_push=A_PUSH, obj=None, pusher=None, meshcat=None,
                    keep_traj=False, u_max=None, loss=None, contact=None):
    """Plan with mu_est (default: the true mu, i.e. the reference method), push on a table of friction mu with the arm
    (or any pusher, e.g. push.Pusher()), and report where the object stopped.  mu_est may be a belief dict (see plan_peak_speed).
    Returns push_slide's dict plus x_target, mu_est, u (the planned peak speed), x_pred, miss (x_rest - x_target), miss_y
    and, for the arm, joint_limits (peak fraction of each joint's position, speed and torque limit) and over_limits."""
    from scene import Obj
    obj = obj if obj is not None else Obj()
    pusher = pusher if pusher is not None else ArmPusher()
    is_arm = isinstance(pusher, ArmPusher)
    est = mu if mu_est is None else mu_est
    if u_max is None:      # the arm's reach from this start; the plate has no reach to respect
        u_max = max_peak_speed(J, x0, pusher.gap, a_push, obj.size[0], tuple(pusher.base), pusher.a_brake_max) if is_arm else 5.0
    u = plan_peak_speed(x_target, est, J, x0, pusher.gap, a_push, u_max, loss=loss)
    r = push_slide(mu, u, J, contact, obj, pusher, a_push, x0=x0, meshcat=meshcat, keep_traj=keep_traj)
    r.update(x_target=x_target, mu_est=est if not isinstance(est, dict) else float("nan"), u=u,
             x_pred=rest_x(u, mu, J, x0, pusher.gap, a_push), miss=r["x_rest"] - x_target, miss_y=r["y_rest"])
    if is_arm:
        r["joint_limits"] = jl = joint_limit_report(pusher.traj)
        r["over_limits"] = [f"joint {i+1} {k} x{jl[k][i]:.2f}" for k in ("q", "v", "tau", "a") for i in range(len(jl[k])) if jl[k][i] > 1.0]
    return r

def report(r):
    """slide_to_target's result as text: the plan, the landing, push.summary and the joint use."""
    lines = [f"target x = {r['x_target']:.3f} m, true mu = {r['mu']}, planned with mu = {r['mu_est']}, J = {r['J']:g} m/s^3 "
             f"-> peak paddle speed u = {r['u']:.4f} m/s, release speed (model) {r['v_r_cmd']:.4f} m/s",
             f"  landed at x = {r['x_rest']:.4f} m, y = {r['y_rest']*1e3:+.2f} mm: miss {r['miss']*1e3:+.2f} mm along the push "
             f"(model with the true mu predicted {r['x_pred']:.4f} m)",
             summary(r)]
    if "joint_limits" in r:
        jl = r["joint_limits"]
        lines.append("  joint use, plan (peak / limit, joints 1-7): position " + " ".join(f"{x:.2f}" for x in jl["q"]) +
                     " | speed " + " ".join(f"{x:.2f}" for x in jl["v"]) + " | torque " + " ".join(f"{x:.2f}" for x in jl["tau"]) +
                     " | acceleration " + " ".join(f"{x:.2f}" for x in jl["a"]))
        if r["over_limits"]:
            lines.append("  WARNING: this push exceeds the real iiwa's limits (the simulator enforces neither joint speed nor acceleration limits): "
                         + ", ".join(r["over_limits"]))
    return "\n".join(lines)

def limit_scan(speeds, J=100.0, base=BASE, a_max=A_BRAKE, posture=POSTURE, which=("v", "tau", "a"), x0=0.55):
    """The worst ratio of the planned joint speed / torque / acceleration (`which`) to its limit at each commanded peak speed:
    [(u, worst ratio)], inf where no posture covers the push.  A push is inside the limits where the ratio is <= 1."""
    key = {"v": "qd_ratio", "tau": "tau_ratio", "a": "qdd_ratio"}
    rows = [feasibility(u, J, tuple(base), x0, a_max, posture=posture) for u in speeds]
    return [(u, max(r[key[k]] for k in which) if r["ok"] else math.inf) for u, r in zip(speeds, rows)]

def print_reach(J=100.0, x0=0.55, base=BASE, a_max=A_BRAKE, posture=POSTURE):
    base = tuple(base)
    face0, stroke = x0 - OBJECT_SIZE[0]/2 - 0.001, reach(base, x0)
    print(f"base at ({base[0]:+.2f}, {base[1]:+.2f}) m, object at x0 = {x0} m: the paddle face starts at x = {face0:.3f} m and can follow the "
          f"push line for {stroke:.3f} m, to x = {face0 + stroke:.3f} m")
    print(f"fastest peak speed with the stroke inside that reach (J = {J:g}, brake cap {a_max:g} m/s^2): {max_peak_speed(J, x0, base=base, a_max=a_max):.3f} m/s")
    lim = {w: limited_peak_speed(J, x0, which=w, base=base, a_max=a_max, posture=posture) for w in (("v",), ("v", "tau"), ("v", "tau", "a"))}
    print(f"fastest peak speed within the joint speed limits: {lim[('v',)]:.3f} m/s; within speed and torque: {lim[('v', 'tau')]:.3f} m/s; "
          f"within speed, torque and acceleration: {lim[('v', 'tau', 'a')]:.3f} m/s")
    print("the acceleration bound against the brake cap (the cap must stay above mu g, or the object never leaves the paddle):")
    for cap in (4.0, 6.0, 8.0, 12.0):
        u = limited_peak_speed(J, x0, which=("v", "tau", "a"), base=base, a_max=cap, posture=posture)
        over = [v for v, w in limit_scan([v for v in (round(0.15 + 0.05*i, 2) for i in range(40)) if v <= u], J, base, cap, posture) if w > 1.0]
        print(f"  brake cap {cap:4.0f} m/s^2 (releases mu < {cap/G:.2f}): {u:.3f} m/s within speed, torque and acceleration"
              + (f"; slower pushes over a limit: {', '.join(f'{v:g}' for v in over)} m/s" if over else ""))
    u = lim[("v", "tau")]
    print(f"what {u:.2f} m/s affords (longest slide, farthest rest point of the object's centre):")
    for mu in (0.1, 0.3, 0.5):
        a = affordable(u, mu, J, x0)
        print(f"  mu {mu}: slide {a['slide']*100:.0f} cm, to x = {a['x_rest']:.2f} m")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.3, help="true table friction in the simulator")
    ap.add_argument("--v", type=float, default=0.8, help="commanded peak speed, m/s (ignored with --target)")
    ap.add_argument("--target", type=float, default=None, help="plan the speed for this target: where the object's centre should stop, world x, m")
    ap.add_argument("--mu-est", type=float, default=None, help="with --target: the friction the planner believes (default: the true mu)")
    ap.add_argument("--x0", type=float, default=0.55, help="object start x, m")
    ap.add_argument("--a-push", type=float, default=A_PUSH)
    ap.add_argument("--plate", action="store_true", help="with --target: push with push.py's prescribed plate instead of the arm")
    ap.add_argument("--reach", action="store_true", help="print the arm's stroke along the push line and the fastest pushes, then exit")
    ap.add_argument("--J", type=float, default=100.0, help="braking jerk, m/s^3")
    ap.add_argument("--a-max", type=float, default=12.0, help="cap on the braking deceleration, m/s^2")
    ap.add_argument("--shape", default="box", choices=["box", "disc", "ring", "footed"])
    ap.add_argument("--com", type=float, default=0.0); ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0); ap.add_argument("--mu-p", type=float, default=PADDLE_MU)
    ap.add_argument("--layout", choices=list(LAYOUTS), default=DEFAULT_LAYOUT, help="where the base stands: side (beside the board, the default) or origin")
    ap.add_argument("--base-x", type=float, default=None, help="iiwa base position along the push, m (overrides --layout, with --base-y)")
    ap.add_argument("--base-y", type=float, default=None, help="iiwa base offset from the push line along y, m")
    ap.add_argument("--posture", choices=list(POSTURES), default=POSTURE, help="start posture: lowest joint speed (speed, the default) or "
                    "lowest worst ratio of joint speed, torque and acceleration to their limits for this push (limits)")
    ap.add_argument("--kp", type=float, default=4e4)
    ap.add_argument("--plan", action="store_true", help="plan only (no contact): posture and ratios")
    ap.add_argument("--feasibility", action="store_true"); ap.add_argument("--scan", action="store_true")
    ap.add_argument("--meshcat", action="store_true")
    a = ap.parse_args()
    base = layout_base(a.layout) if a.base_x is None and a.base_y is None else (a.base_x or 0.0, a.base_y or 0.0, 0.0)
    if a.reach:
        return print_reach(a.J, a.x0, base, a.a_max, a.posture)
    if a.feasibility or a.scan:
        vs, js = (0.6, 0.8, 1.0), (30.0, 100.0, 500.0, 2000.0) if a.feasibility else (100.0, 2000.0)
        # --scan: the team's base at the origin, then bases beside the push line (y = 0.65 m is just off the 1.2 m wide board),
        # level with the start of the push and further along it, so the paddle sweeps across the arm's front instead of radially
        bases = ((0.0, 0.0), (0.0, 0.4), (0.3, 0.65), (0.45, 0.65), (0.6, 0.65), (0.75, 0.65)) if a.scan else (base[:2],)
        for bx, by in bases:
            t0 = time.time()
            rows = [feasibility(v, J, (bx, by, 0.0), a_max=a.a_max, posture=a.posture) for v in vs for J in js]
            print(f"\n== base at x = {bx:+.2f}, y = {by:+.2f} m (push line y = 0, object at x0 = 0.55), brake cap {a.a_max} m/s^2 ({time.time() - t0:.0f} s) ==")
            print(table(rows, "qd_ratio", "peak joint speed / limit (joint): the iiwa delivers v_peak only where this is <= 1"))
            print(table(rows, "tau_ratio", "peak joint torque / effort limit (joint), inverse dynamics along the plan, no contact force"))
            okr = [r for r in rows if r["ok"]]
            if okr:
                print(f"paddle speed the joint limits allow on this line: {min(r['v_max_by_speed'] for r in okr):.2f} .. {max(r['v_max_by_speed'] for r in okr):.2f} m/s; "
                      f"effective mass along x {min(r['m_eff'] for r in okr):.1f} .. {max(r['m_eff'] for r in okr):.1f} kg; face travels to x = {max(r['x_face_end'] for r in okr):.2f} m")
            for r in rows:
                if not r["ok"]:
                    reasons = r["reason"].split(": ", 1)[-1].split("; ")
                    print(f"  v={r['v_peak']} J={r['J']:g}: " + "; ".join(sorted(set(x.split(" (")[0] for x in reasons))[:4]))
        return
    if a.plan:
        r = feasibility(a.v, a.J, base, a_max=a.a_max, lateral=a.lateral, yaw_deg=a.yaw, posture=a.posture)
        print(r); return
    from scene import Obj
    from drake_config import contact_from_config, describe
    contact = contact_from_config()
    print(describe(contact))
    meshcat = StartMeshcat() if a.meshcat else None
    obj = Obj(shape=a.shape, com_offset=(0.0, a.com, 0.0))
    pu = ArmPusher(mu_p=a.mu_p, lateral=a.lateral, yaw_deg=a.yaw, base=base, a_brake_max=a.a_max, kp=a.kp, posture=a.posture)
    if a.target is not None:
        pu = Pusher(mu_p=a.mu_p, lateral=a.lateral, yaw_deg=a.yaw) if a.plate else pu
        r = slide_to_target(a.target, a.mu, a.mu_est, a.J, a.x0, a.a_push, obj, pu, meshcat, contact=contact)
        print(report(r))
        if a.plate:
            if meshcat:
                input("meshcat is showing the recording; press enter to quit")
            return
    else:
        r = push_slide(a.mu, a.v, a.J, contact, obj, pu, a.a_push, x0=a.x0, meshcat=meshcat)
        print(summary(r))
    print(pu.describe())
    print(f"  in the push: torque peak {r['arm_tau_ratio']:.2f} x limit (joint {r['arm_tau_joint']}), joint speed peak {r['arm_qd_ratio']:.2f} x limit (joint {r['arm_qd_joint']}); "
          f"paddle z drift {r['arm_z_drift_mm']:.2f} mm, y drift {r['arm_y_drift_mm']:.2f} mm, tilt {r['arm_tilt_max_deg']:.3f} deg")
    if meshcat:
        input("meshcat is showing the recording; press enter to quit")

if __name__ == "__main__":
    main()

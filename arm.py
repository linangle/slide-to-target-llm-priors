#!/usr/bin/env python3
"""
arm.py -- the iiwa 14 as the pusher: a joint-space controller that drives the paddle along push.profile().

ArmPusher has the plate's hook (push.Pusher: add, connect, q, qd, face_x, set_displacement, record, summary), so
push.push_slide and the E0 scripts run with the arm by swapping `Pusher()` for `ArmPusher()`.

How it works.  The push is a straight line: the paddle face moves along world +x with its normal along the push, the
plate's lower edge `clearance` above the table and the plate centred on the object (lateral, yaw knobs as the plate's).
  1. The paddle pose fixes link 7's pose (scene.add_arm welds the plate 5 cm below the flange, face along link-7 +z), so
     the task is a 6-DoF frame path with one redundant joint.  The start posture is chosen among multi-start IK
     solutions (Drake InverseKinematics, random seeds) by integrating each along the whole path with the minimum-norm
     joint tangent dq/ds = J^+ e_x and keeping the one with the smallest peak joint-speed ratio (|dq_i/ds| / v_lim_i)
     that stays inside the joint limits and away from singularities.
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

    python3 arm.py --mu 0.3 --v 0.8 --J 100            # one push with the arm, same report as push.py (+ arm metrics); --meshcat
    python3 arm.py --plan --v 0.8 --J 100              # the plan alone: posture, joint-speed and torque ratios, no contact
    python3 arm.py --feasibility                       # ratios over v_peak x J at the default base
    python3 arm.py --scan                              # ratios over the base's lateral offset from the push line
"""
import argparse, math, time
from dataclasses import dataclass, field
import numpy as np
from pydrake.all import (AddDefaultVisualization, InverseKinematics, JacobianWrtVariable, JointActuatorIndex, MultibodyForces,
                         MultibodyPlant, PdControllerGains, PiecewisePolynomial, RigidTransform, RotationMatrix, Solve, StartMeshcat,
                         TrajectorySource)
from scene import FLANGE_Z, PADDLE, PADDLE_MU, PADDLE_Y, add_arm
from push import PLATE, profile

# link-7 axes in the world during the push: x_L7 = +y_W (plate width), y_L7 = +z_W (the plate hangs below the flange),
# z_L7 = +x_W (the plate's far face pushes).  Columns are the link-7 axes.
R_WL7 = RotationMatrix(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
X_L7P = RigidTransform([0.0, PADDLE_Y, FLANGE_Z + PADDLE[2]/2])        # paddle body (plate centre) in link 7, as add_arm welds it
Q_DISPLAY = np.array([0.0, 0.6, 0.0, -1.4, 0.0, 1.1, 0.0])              # scene.py's display pose, one IK seed
V_LIM_DEG = (85, 85, 100, 75, 130, 135, 135)                            # KUKA LBR iiwa 14 R820 joint speed limits, deg/s (the SDF's)

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

    def __init__(self, base=(0.0, 0.0, 0.0)):
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
        Returns dict(s, q, dqds, sigma, ok, reason, qd_per_mps, qd_joint, tau_g_ratio, limit_margin) -- qd_per_mps is the peak
        over the path of max_i |dq_i/ds| / v_lim_i, the joint-speed ratio per 1 m/s of paddle speed."""
        s = np.arange(0.0, length + ds/2, ds)
        if s[-1] < length:
            s = np.append(s, length)
        n = len(s)
        q, dq, sig = np.empty((n, self.n)), np.empty((n, self.n)), np.empty(n)
        qk, ok, reason = np.array(q0, float), True, ""
        e_x = np.array([0, 0, 0, 1.0, 0, 0])
        for k, sk in enumerate(s):
            X = RigidTransform(X0.rotation(), X0.translation() + np.array([sk, 0, 0]))
            qk, err = self.project(qk, X)
            Jk = self.jacobian(qk)
            sv = np.linalg.svd(Jk, compute_uv=False)
            q[k], sig[k] = qk, sv[-1]
            dq[k] = np.linalg.lstsq(Jk, e_x, rcond=None)[0]
            if ok and (err > 1e-6 or sv[-1] < sigma_min):
                ok, reason = False, f"singular at s = {sk:.3f} m (sigma_min {sv[-1]:.3f}, pose error {err:.1e})"
            if ok and (np.any(qk < self.q_lo + margin) or np.any(qk > self.q_hi - margin)):
                j = int(np.argmin(np.minimum(qk - self.q_lo, self.q_hi - qk)))
                ok, reason = False, f"joint {j + 1} at its limit at s = {sk:.3f} m"
            if k + 1 < n:
                qk = qk + dq[k]*(s[k + 1] - sk)
        ratio = np.abs(dq)/self.v_lim
        kmax = np.unravel_index(np.argmax(ratio), ratio.shape)
        tau_g = np.array([self.inverse_dynamics(qq, np.zeros(self.n), np.zeros(self.n)) for qq in q[::max(1, n//20)]])
        return {"s": s, "q": q, "dqds": dq, "sigma": sig, "ok": ok, "reason": reason, "qd_per_mps": float(ratio[kmax]),
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
    base: tuple = (0.0, 0.0, 0.0)   # iiwa base (link 0) position in the world, m; the team's scene has it at the origin
    a_brake_max: float = 12.0  # cap on the braking deceleration, m/s^2 (> mu g for every pair, so the release is unchanged)
    kp: float = 4e4            # joint PD stiffness, N m/rad (about 1e5 N/m at the paddle, the plate's value)
    zeta: float = 1.0          # joint PD damping ratio against the diagonal mass (incl. rotor inertia) at the display pose
    posture_seed: int = 0      # RNG seed of the multi-start IK
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

    def make_plan(self, prof):
        """Chooses the posture and plans q(s) over the profile's travel; fills self.plan and self.spline."""
        m, length = self.model, prof["x_stop"] + 0.002
        cands = m.candidates(self.X0, seed=self.posture_seed)
        if not cands:
            raise RuntimeError(f"no IK solution for the paddle face at x = {self.x_touch0:.3f} m (base {self.base})")
        paths = [m.path(q, self.X0, length) for q in cands]
        ok = [p for p in paths if p["ok"]]
        if not ok:
            raise RuntimeError("no posture covers the push: " + "; ".join(sorted({p["reason"] for p in paths})))
        best = min(ok, key=lambda p: (p["qd_per_mps"], p["tau_g_ratio"]))
        self.spline = PiecewisePolynomial.CubicHermite(best["s"], best["q"].T, best["dqds"].T)
        # the desired joint trajectory along the profile and the arm's inverse dynamics as feed-forward
        t, x, v, a = prof["t"], prof["x"], prof["v"], prof["a"]
        xs = np.clip(x, 0.0, best["s"][-1])
        q_d = np.array([self.spline.value(s)[:, 0] for s in xs])
        dq = np.array([self.spline.EvalDerivative(s, 1)[:, 0] for s in xs])
        d2q = np.array([self.spline.EvalDerivative(s, 2)[:, 0] for s in xs])
        qd_d = dq*v[:, None]
        qdd_d = d2q*(v*v)[:, None] + dq*a[:, None]
        tau = np.array([m.inverse_dynamics(q_d[k], qd_d[k], qdd_d[k]) for k in range(len(t))])
        move = (t >= t[1]) & (t <= prof["t_stop"] + 1e-9)
        rv, ra, rt = np.abs(qd_d)/m.v_lim, np.abs(qdd_d)/m.a_lim, np.abs(tau)/m.tau_lim
        i_v, i_a, i_t = (np.unravel_index(np.argmax(r[move]), r[move].shape) for r in (rv, ra, rt))
        self.mass = m.effective_mass(best["q"][0])
        self.plan = dict(best, n_candidates=len(cands), n_ok=len(ok), q_start=best["q"][0], t=t, q_d=q_d, qd_d=qd_d, qdd_d=qdd_d, tau_ff=tau,
                         qd_ratio=float(rv[move][i_v]), qd_ratio_joint=int(i_v[1]) + 1, qdd_ratio=float(ra[move][i_a]), qdd_ratio_joint=int(i_a[1]) + 1,
                         tau_ratio=float(rt[move][i_t]), tau_ratio_joint=int(i_t[1]) + 1, tau_max=np.abs(tau[move]).max(axis=0),
                         qd_max=np.abs(qd_d[move]).max(axis=0), m_eff=self.mass, length=length, sigma_min=float(best["sigma"].min()))
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
                "arm_plan_tau_joint": p["tau_ratio_joint"], "arm_plan_qdd_ratio": p["qdd_ratio"], "arm_m_eff": p["m_eff"],
                "arm_sigma_min": p["sigma_min"], "arm_postures": p["n_ok"], "arm_z_drift_mm": float((pz.max() - pz.min())*1e3),
                "arm_y_drift_mm": float((py.max() - py.min())*1e3), "arm_tilt_max_deg": float(math.degrees(ang.max())), "arm_base_y": self.base[1]}

    def describe(self):
        p = self.plan
        return (f"arm plan: {p['n_ok']} of {p['n_candidates']} postures cover the {p['length']*1e3:.0f} mm push; start q = {np.round(p['q_start'], 3)}; "
                f"effective mass along x {p['m_eff']:.1f} kg; sigma_min {p['sigma_min']:.3f}\n"
                f"  joint speed: peak {p['qd_ratio']:.2f} x limit (joint {p['qd_ratio_joint']}); max |qd| deg/s = {np.round(np.degrees(p['qd_max']), 0)}\n"
                f"  torque:      peak {p['tau_ratio']:.2f} x effort limit (joint {p['tau_ratio_joint']}); max |tau| N m = {np.round(p['tau_max'], 0)} of {m_lim(self)}\n"
                f"  acceleration (SDF drake:acceleration, informational): peak {p['qdd_ratio']:.2f} x (joint {p['qdd_ratio_joint']})")

def m_lim(pusher):
    return np.round(pusher.model.tau_lim, 0)

# ---------------------------------------------------------------------- feasibility: what the iiwa can deliver

def feasibility(v_peak, J, base=(0.0, 0.0, 0.0), x0=0.55, a_max=12.0, lateral=0.0, yaw_deg=0.0, obj_size_x=0.10):
    """Plans the push without contact and returns the plan's ratios (or the reason it is impossible)."""
    from scene import Obj
    pu = ArmPusher(lateral=lateral, yaw_deg=yaw_deg, base=base, a_brake_max=a_max)
    pu.x_touch0 = x0 - obj_size_x/2 - pu.gap
    pu.model = ArmModel(base)
    pu.X0 = link7_pose(pu.x_touch0, lateral, yaw_deg, pu.clearance)
    prof = profile(v_peak, J, a_max=a_max)
    try:
        p = pu.make_plan(prof)
    except RuntimeError as ex:
        return {"v_peak": v_peak, "J": J, "base_y": base[1], "ok": False, "reason": str(ex), "x_face_end": pu.x_touch0 + prof["x_stop"]}
    return {"v_peak": v_peak, "J": J, "base_y": base[1], "ok": True, "qd_ratio": p["qd_ratio"], "qd_joint": p["qd_ratio_joint"], "tau_ratio": p["tau_ratio"],
            "tau_joint": p["tau_ratio_joint"], "qdd_ratio": p["qdd_ratio"], "m_eff": p["m_eff"], "sigma_min": p["sigma_min"], "x_face_end": pu.x_touch0 + prof["x_stop"],
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--v", type=float, default=0.8, help="commanded peak speed, m/s")
    ap.add_argument("--J", type=float, default=100.0, help="braking jerk, m/s^3")
    ap.add_argument("--a-max", type=float, default=12.0, help="cap on the braking deceleration, m/s^2")
    ap.add_argument("--shape", default="box", choices=["box", "disc", "ring", "footed"])
    ap.add_argument("--com", type=float, default=0.0); ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0); ap.add_argument("--mu-p", type=float, default=PADDLE_MU)
    ap.add_argument("--base-x", type=float, default=0.0, help="iiwa base position along the push, m (the scene's is 0)")
    ap.add_argument("--base-y", type=float, default=0.0, help="iiwa base offset from the push line along y, m")
    ap.add_argument("--kp", type=float, default=4e4)
    ap.add_argument("--plan", action="store_true", help="plan only (no contact): posture and ratios")
    ap.add_argument("--feasibility", action="store_true"); ap.add_argument("--scan", action="store_true")
    ap.add_argument("--meshcat", action="store_true")
    a = ap.parse_args()
    base = (a.base_x, a.base_y, 0.0)
    if a.feasibility or a.scan:
        vs, js = (0.6, 0.8, 1.0), (30.0, 100.0, 500.0, 2000.0) if a.feasibility else (100.0, 2000.0)
        # --scan: the team's base at the origin, then bases beside the push line (y = 0.65 m is just off the 1.2 m wide board),
        # level with the start of the push and further along it, so the paddle sweeps across the arm's front instead of radially
        bases = ((0.0, 0.0), (0.0, 0.4), (0.3, 0.65), (0.45, 0.65), (0.6, 0.65), (0.75, 0.65)) if a.scan else ((a.base_x, a.base_y),)
        for bx, by in bases:
            t0 = time.time()
            rows = [feasibility(v, J, (bx, by, 0.0), a_max=a.a_max) for v in vs for J in js]
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
        r = feasibility(a.v, a.J, base, a_max=a.a_max, lateral=a.lateral, yaw_deg=a.yaw)
        print(r); return
    from scene import Obj
    from drake_config import contact_from_config, describe
    from push import push_slide, summary
    contact = contact_from_config()
    print(describe(contact))
    meshcat = StartMeshcat() if a.meshcat else None
    pu = ArmPusher(mu_p=a.mu_p, lateral=a.lateral, yaw_deg=a.yaw, base=base, a_brake_max=a.a_max, kp=a.kp)
    r = push_slide(a.mu, a.v, a.J, contact, Obj(shape=a.shape, com_offset=(0.0, a.com, 0.0)), pu, meshcat=meshcat)
    print(summary(r))
    print(pu.describe())
    print(f"  in the push: torque peak {r['arm_tau_ratio']:.2f} x limit (joint {r['arm_tau_joint']}), joint speed peak {r['arm_qd_ratio']:.2f} x limit (joint {r['arm_qd_joint']}); "
          f"paddle z drift {r['arm_z_drift_mm']:.2f} mm, y drift {r['arm_y_drift_mm']:.2f} mm, tilt {r['arm_tilt_max_deg']:.3f} deg")
    if meshcat:
        input("meshcat is showing the recording; press enter to quit")

if __name__ == "__main__":
    main()

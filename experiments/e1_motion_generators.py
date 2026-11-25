#!/usr/bin/env python3
"""
e1_motion_generators.py -- the proposal's E1 comparison: differential inverse kinematics against kinematic trajectory
optimisation as the push's motion generator, with the generator arm.py uses beside them.

What arm.py does ("current").  It is neither of the two.  The push is a straight line, so the paddle pose leaves one
redundant joint.  arm.py takes the start postures a multi-start IK finds, integrates each along the line with the
minimum-norm joint tangent dq/ds = J^+ e_x (resolved-rate motion, computed off line, with no limits in the loop), and
keeps one posture: the lowest joint speed per m/s (posture "speed"), or the lowest worst ratio of joint speed, torque and
acceleration to their limits for this push (posture "limits", used here).  The path is then timed by push.profile.  So it
is a differential-IK path with a search over the one free choice, the start posture.

The two generators of the proposal, both given the same commanded paddle motion x(t) (push.profile) and the same start pose:

  diffik    Drake's DoDifferentialInverseKinematics in a 1 ms loop from the current planner's start posture, tracking the
            commanded pose with joint position and velocity limits as hard bounds.  When a limit binds, Drake scales the
            commanded spatial velocity down, the paddle falls behind its command, and nothing fails: the push is slower
            than asked.  That is the "clipping" the proposal wanted counted.
  diffik_acc  The same with the joint acceleration limits as hard bounds too.
  trajopt   Drake's KinematicTrajectoryOptimization: a cubic B-spline q(t) over the push, with the paddle pose pinned to
            the commanded line at 70 sample times (100 control points), joint position,
            velocity and acceleration bounds, zero joint velocity at both ends, and the path-energy cost.  The current
            plan is the initial guess.  If no trajectory satisfies the bounds the solver says so and the push is not made.
  trajopt_vel The same without the acceleration bounds, so that each differential IK has a trajectory optimisation under
            the same limits beside it (diffik with trajopt_vel, diffik_acc with trajopt).
            (The proposal's version minimised duration with a release-velocity constraint; here the timing is the
            commanded profile's, because the release model and the decision rule are built on that profile.)

Each plan is executed in Drake by the same joint controller (implicit PD with inverse-dynamics feed-forward, arm.py) with
the object in contact, and the release is read as in every other experiment.  arm.py is not edited: GenPusher subclasses
ArmPusher and replaces the plan.

    python3 experiments/e1_motion_generators.py              # 85 pushes, a few minutes -> results/side_base/e1_motion_generators/
    python3 experiments/e1_motion_generators.py --summarize results/side_base/e1_motion_generators/results.csv
    python3 experiments/e1_motion_generators.py --a-brake 12 --out results/side_base/e1_motion_generators_brake12
"""
import argparse, csv, time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from pydrake.all import KinematicTrajectoryOptimization, PiecewisePolynomial, RigidTransform, SnoptSolver, TrajectorySource
from pydrake.multibody.inverse_kinematics import (DifferentialInverseKinematicsParameters, DifferentialInverseKinematicsStatus,
                                                  DoDifferentialInverseKinematics, PositionConstraint)
from arm import ArmModel, ArmPusher, LAYOUTS, link7_pose, pose_error, rest_x
from paths import RESULTS
from physics import OBJECT_SIZE
from push import profile, push_slide

GENERATORS = ("current", "diffik", "trajopt_vel", "diffik_acc", "trajopt")
X0_OBJ, J_BRAKE, MU = 0.55, 100.0, 0.26          # object start, braking jerk, friction of the executed pushes (Y03's)
SPEEDS = {"side": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1), "origin": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
TAIL = 0.25                                       # s after the commanded stop that a plan may still move (a lagging diff IK)
FIELDS = ["layout", "generator", "v_peak", "mu", "a_brake", "delivered", "reason", "clipped", "alpha_min", "peak_speed_err_pct",
          "release_err_pct", "landing_err_mm", "separated", "plan_v_ratio", "plan_a_ratio", "plan_tau_ratio", "inside_limits",
          "path_x_err_mm", "path_yz_err_mm", "path_tilt_deg", "track_rms_mm", "stuck_steps", "plan_s"]

def desired_pose(X0, x):
    return RigidTransform(X0.rotation(), X0.translation() + np.array([x, 0.0, 0.0]))

def plan_times(prof, dt=1e-3):
    """The profile's time grid, with samples every dt for TAIL after the commanded stop (the profile has none there)."""
    t = prof["t"]
    return np.concatenate([t[:-1], np.arange(prof["t_stop"] + dt, prof["t_stop"] + TAIL + dt/2, dt), [max(t[-1], prof["t_stop"] + TAIL + dt)]])

def path_error(model, X0, prof, t, q):
    """How far the planned paddle is from the commanded line and timing while it moves: max |x error|, max y-z drift (mm), max tilt (deg)."""
    k = np.nonzero((t >= prof["t"][1]) & (t <= prof["t_stop"]))[0][::20]
    ex, eyz, tilt = 0.0, 0.0, 0.0
    for i in k:
        e = pose_error(desired_pose(X0, float(np.interp(t[i], prof["t"], prof["x"]))), model.fk(q[i]))
        ex, eyz, tilt = max(ex, abs(e[3])), max(eyz, float(np.hypot(e[4], e[5]))), max(tilt, float(np.linalg.norm(e[:3])))
    return dict(path_x_err_mm=1e3*ex, path_yz_err_mm=1e3*eyz, path_tilt_deg=float(np.degrees(tilt)))

def timed_plan(model, prof, t, q_d, qd_d, qdd_d):
    """What ArmPusher._trajectory returns, for a joint trajectory given in time: feed-forward torque and the peak limit ratios."""
    tau = np.array([model.inverse_dynamics(q_d[k], qd_d[k], qdd_d[k]) for k in range(len(t))])
    move = (t >= t[1]) & (t <= prof["t_stop"] + TAIL + 1e-9)
    out = dict(t=t, q_d=q_d, qd_d=qd_d, qdd_d=qdd_d, tau_ff=tau, move=move, tau_max=np.abs(tau[move]).max(axis=0),
               qd_max=np.abs(qd_d[move]).max(axis=0))
    for name, r in (("qd", np.abs(qd_d)/model.v_lim), ("qdd", np.abs(qdd_d)/model.a_lim), ("tau", np.abs(tau)/model.tau_lim)):
        i = np.unravel_index(np.argmax(r[move]), r[move].shape)
        out[f"{name}_ratio"], out[f"{name}_ratio_joint"] = float(r[move][i]), int(i[1]) + 1
    return out

# ----------------------------------------------------------------------------- generator: differential inverse kinematics

def diffik(model, X0, prof, q0, t_plan, dt=1e-3, accel_limits=False):
    """Roll Drake's differential IK along the commanded pose, 1 ms steps.  Returns (q, qd, qdd on t_plan, info)."""
    n = model.n
    par = DifferentialInverseKinematicsParameters(n, n)
    par.set_time_step(dt)
    par.set_joint_position_limits((model.q_lo, model.q_hi))
    par.set_joint_velocity_limits((-model.v_lim, model.v_lim))
    par.set_nominal_joint_position(np.asarray(q0, float))
    if accel_limits:
        par.set_joint_acceleration_limits((-model.a_lim, model.a_lim))
    tk = np.arange(prof["t"][1], prof["t_stop"] + TAIL + dt/2, dt)
    xd, vd = np.interp(tk, prof["t"], prof["x"]), np.interp(tk, prof["t"], prof["v"])
    q, v = np.array(q0, float), np.zeros(n)
    Q, V, alpha, got, stuck = [q.copy()], [], [], [], 0
    for k in range(len(tk) - 1):
        Jk = model.jacobian(q)
        Vd = pose_error(desired_pose(X0, xd[k + 1]), model.fk(q))/dt       # the pose step asked of this time step
        res = DoDifferentialInverseKinematics(q, v, Vd, Jk, par)
        if res.status == DifferentialInverseKinematicsStatus.kSolutionFound:
            v = np.asarray(res.joint_velocities, float)
        else:                                                              # stuck or no solution: the arm holds still this step
            v, stuck = np.zeros(n), stuck + 1
        vx = float((Jk @ v)[3])
        got.append(vx); alpha.append(vx/Vd[3] if vd[k + 1] > 0.05*prof["v"].max() else 1.0)
        q = q + v*dt
        Q.append(q.copy()); V.append(v.copy())
    Q, V = np.array(Q), np.array(V + [np.zeros(n)])
    node_v = np.vstack([np.zeros(n), 0.5*(V[:-2] + V[1:-1]), np.zeros(n)])   # C1 through the piecewise-linear rollout
    pp = PiecewisePolynomial.CubicHermite(tk, Q.T, node_v.T)
    ts = np.clip(t_plan, tk[0], tk[-1])
    q_d = np.array([pp.value(s)[:, 0] for s in ts])
    inside = ((t_plan >= tk[0]) & (t_plan <= tk[-1]))[:, None]
    qd_d = np.array([pp.EvalDerivative(s, 1)[:, 0] for s in ts])*inside
    # acceleration as this controller commands it: the change of the velocity command from one step to the next.  (The
    # interpolant's second derivative overshoots by a third for one step wherever the commanded acceleration jumps.)
    step_acc = np.vstack([np.diff(np.vstack([np.zeros(n), V[:-1]]), axis=0)/dt, np.zeros((1, n))])
    qdd_d = step_acc[np.clip(np.searchsorted(tk, ts, side="right") - 1, 0, len(tk) - 1)]*inside
    info = dict(alpha_min=float(min(alpha)), clipped=bool(min(alpha) < 0.98), stuck_steps=stuck, peak_paddle_speed=float(max(got)),
                lag_max_mm=float(1e3*max(xd[k + 1] - (model.fk(Q[k + 1]).translation()[0] - X0.translation()[0]) for k in range(0, len(tk) - 1, 10))))
    return q_d, qd_d, qdd_d, info

# ----------------------------------------------------------------------------- generator: kinematic trajectory optimisation

SNOPT = {"Major optimality tolerance": 1e-4, "Major iterations limit": 400, "Iterations limit": 100000}

def trajopt(model, X0, prof, guess, t_plan, n_ctrl=100, n_samples=70, tol=2e-5, margin=0.02, accel_limits=True, options=None):
    """B-spline q(t) over the push with the paddle pinned to the commanded pose at sample times and the joint bounds.
    guess: a trajectory q(t) for the initial control points.  Returns (q, qd, qdd on t_plan, info); raises RuntimeError
    when the solver finds no trajectory inside the bounds."""
    t0, t1 = float(prof["t"][1]), float(prof["t_stop"])
    D, n = t1 - t0, model.n
    opt = KinematicTrajectoryOptimization(n, n_ctrl, 4, D)
    opt.AddDurationConstraint(D, D)
    opt.AddPositionBounds(model.q_lo + margin, model.q_hi - margin)
    opt.AddVelocityBounds(-model.v_lim, model.v_lim)
    if accel_limits:
        opt.AddAccelerationBounds(-model.a_lim, model.a_lim)
    opt.AddPathVelocityConstraint(np.zeros(n), np.zeros(n), 0.0)
    opt.AddPathVelocityConstraint(np.zeros(n), np.zeros(n), 1.0)
    ts = np.linspace(t0, t1, n_samples)      # fewer pins than control points: a pin fixes six of a control point's seven numbers
    # The pose is pinned through three points of link 7, not an orientation constraint: the angle between two frames has zero
    # gradient where the frames agree, which is where the constraint has to hold, and the solver cannot use it there.
    R, arm_len = X0.rotation().matrix(), 0.2
    a1 = np.argsort(np.abs(R[:, 0]))[:2]                     # the two world axes across link 7's x
    a2 = np.array([int(np.argmax(np.abs(R[:, 2])))])         # the world axis a turn about link 7's x moves its y along
    pins = [(np.zeros(3), np.arange(3)), (np.array([arm_len, 0, 0]), a1), (np.array([0, arm_len, 0]), a2)]
    for ti in ts:
        X, s = desired_pose(X0, float(np.interp(ti, prof["t"], prof["x"]))), float(np.clip((ti - t0)/D, 0.0, 1.0))
        for p_L7, axes in pins:                              # six independent coordinates of three points fix the pose
            p = X @ p_L7
            lo, hi = np.full(3, -np.inf), np.full(3, np.inf)
            lo[axes], hi[axes] = p[axes] - tol, p[axes] + tol
            opt.AddPathPositionConstraint(PositionConstraint(model.plant, model.W, lo, hi, model.L7, p_L7, model.ctx), s)
    opt.AddPathEnergyCost(1.0)
    prog = opt.get_mutable_prog()
    knots, order = np.asarray(opt.basis().knots()), opt.basis().order()
    greville = np.array([knots[i + 1:i + order].mean() for i in range(n_ctrl)])     # where each control point acts, in [0, 1]
    prog.SetInitialGuess(opt.control_points(), np.column_stack([guess.value(t0 + s*D)[:, 0] for s in greville]))
    prog.SetInitialGuess(opt.duration(), D)
    for name, value in (SNOPT if options is None else options).items():
        prog.SetSolverOption(SnoptSolver.id(), name, value)
    tic = time.time()
    result = SnoptSolver().Solve(prog)
    if not result.is_success():
        raise RuntimeError(f"trajectory optimisation found no trajectory inside the bounds ({result.get_solver_id().name()} "
                           f"info {getattr(result.get_solver_details(), 'info', '?')}, {time.time() - tic:.0f} s)")
    bs = opt.ReconstructTrajectory(result)
    d1, d2 = bs.MakeDerivative(1), bs.MakeDerivative(2)
    ts_plan = np.clip(t_plan - t0, 0.0, D)
    inside = ((t_plan >= t0) & (t_plan <= t1))[:, None]
    q_d = np.array([bs.value(s)[:, 0] for s in ts_plan])
    qd_d = np.array([d1.value(s)[:, 0] for s in ts_plan])*inside
    qdd_d = np.array([d2.value(s)[:, 0] for s in ts_plan])*inside
    return q_d, qd_d, qdd_d, dict(solve_s=time.time() - tic, solver=result.get_solver_id().name(), samples=len(ts), control_points=n_ctrl)

# ----------------------------------------------------------------------------- the pusher

@dataclass
class GenPusher(ArmPusher):
    """ArmPusher with the plan made by another motion generator.  'current' is ArmPusher unchanged."""
    generator: str = "current"
    accel_limits: bool = None        # None: each generator's own default (diffik: velocity limits only; trajopt: both)

    def make_plan(self, prof):
        base = super().make_plan(prof)
        if self.generator == "current":
            base["gen"] = path_error(self.model, self.X0, prof, base["t"], base["q_d"])
            return base
        t_plan = plan_times(prof)
        if self.generator in ("diffik", "diffik_acc"):
            q, qd, qdd, info = diffik(self.model, self.X0, prof, base["q_start"], t_plan,
                                      accel_limits=self.generator == "diffik_acc" or bool(self.accel_limits))
        elif self.generator in ("trajopt", "trajopt_vel"):
            guess = PiecewisePolynomial.CubicHermite(prof["t"], base["q_d"].T, base["qd_d"].T)
            q, qd, qdd, info = trajopt(self.model, self.X0, prof, guess, t_plan,
                                       accel_limits=self.generator == "trajopt" if self.accel_limits is None else self.accel_limits)
        else:
            raise ValueError(f"generator is one of {GENERATORS}")
        info.update(path_error(self.model, self.X0, prof, t_plan, q))
        self.plan = dict(base, **timed_plan(self.model, prof, t_plan, q, qd, qdd), q_start=q[0], gen=info, posture=f"{self.generator} from {self.posture}")
        return self.plan

    def connect(self, builder, plant, prof, dt=0.0):
        if self.generator == "current":
            return super().connect(builder, plant, prof, dt)
        plan = self.make_plan(prof)                        # as ArmPusher.connect, on the plan's own time grid
        plant.SetDefaultPositions(self.instance, plan["q_start"])
        t = plan["t"] - dt
        desired = builder.AddSystem(TrajectorySource(PiecewisePolynomial.CubicHermite(t, plan["q_d"].T, plan["qd_d"].T), output_derivative_order=1))
        builder.Connect(desired.get_output_port(), plant.get_desired_state_input_port(self.instance))
        ff = builder.AddSystem(TrajectorySource(PiecewisePolynomial.FirstOrderHold(t, plan["tau_ff"].T)))
        builder.Connect(ff.get_output_port(), plant.get_actuation_input_port(self.instance))

def plan_only(layout, generator, v_peak, a_brake=6.0, J=J_BRAKE, x0=X0_OBJ, **kw):
    """The plan without a scene or contact (as arm.feasibility sets a pusher up).  Returns (plan, profile)."""
    pu = GenPusher(base=tuple(LAYOUTS[layout]), a_brake_max=a_brake, posture="limits", generator=generator, **kw)
    pu.x_touch0 = x0 - OBJECT_SIZE[0]/2 - pu.gap
    pu.model = ArmModel(pu.base)
    pu.X0 = link7_pose(pu.x_touch0, 0.0, 0.0, pu.clearance)
    prof = profile(v_peak, J, a_max=a_brake)
    return pu.make_plan(prof), prof

# ----------------------------------------------------------------------------- the comparison

def _job(job):
    layout, gen, u, a_brake = job
    row = dict(layout=layout, generator=gen, v_peak=u, mu=MU, a_brake=a_brake, delivered=0, reason="")
    pusher = GenPusher(base=tuple(LAYOUTS[layout]), a_brake_max=a_brake, posture="limits", generator=gen)
    tic = time.time()
    try:
        r = push_slide(MU, u, J_BRAKE, x0=X0_OBJ, pusher=pusher, keep_traj=True)
    except RuntimeError as exc:
        return dict(row, reason=str(exc).replace("\n", " ")[:200], plan_s=round(time.time() - tic, 1))
    p, g = pusher.plan, pusher.plan["gen"]
    ratios = (p["qd_ratio"], p["qdd_ratio"], p["tau_ratio"])
    row.update(delivered=1, clipped=int(bool(g.get("clipped", False))), alpha_min=round(g.get("alpha_min", 1.0), 4),
               peak_speed_err_pct=round(100*(float(np.max(r["traj"]["qd"]))/u - 1), 3), release_err_pct=round(r["v_r_cmd_err_pct"], 3),
               landing_err_mm=round(1e3*(r["x_rest"] - rest_x(u, MU, J_BRAKE, X0_OBJ)), 2), separated=int(bool(r["separated"])),
               plan_v_ratio=round(ratios[0], 3), plan_a_ratio=round(ratios[1], 3), plan_tau_ratio=round(ratios[2], 3),
               inside_limits=int(max(ratios) <= 1.001), path_x_err_mm=round(g["path_x_err_mm"], 3), path_yz_err_mm=round(g["path_yz_err_mm"], 3),
               path_tilt_deg=round(g["path_tilt_deg"], 4), track_rms_mm=round(1e3*r["track_q_rms"], 3), stuck_steps=g.get("stuck_steps", ""),
               plan_s=round(time.time() - tic, 1))
    return row

def summarize(rows, a_brake):
    f = lambda r, k, fmt: "    -" if r.get(k) in ("", None) else format(float(r[k]), fmt)
    lines = [f"E1 motion generators: {len(rows)} pushes in Drake, friction {MU}, J = {J_BRAKE:g} m/s^3, brake cap {a_brake:g} m/s^2.",
             "current = arm.py (posture 'limits'); diffik = Drake differential IK, joint position and velocity limits; diffik_acc = the same",
             "with acceleration limits too; trajopt = Drake kinematic trajectory optimisation, position, velocity and acceleration bounds;",
             "trajopt_vel = the same without the acceleration bounds.  Pairs under the same limits: diffik / trajopt_vel, diffik_acc / trajopt.",
             "Same commanded paddle motion, same joint controller.",
             "peak = paddle peak speed against the command; release = object speed at the model's release instant against the release",
             "model on the command; landing = rest position against the model; v / a / tau = planned peak over the limit; held = 1 ms",
             "steps in which differential IK returned no step and the arm was held still (as Drake's integrator does).", ""]
    for layout in dict.fromkeys(r["layout"] for r in rows):
        lines += [f"base: {layout} {tuple(LAYOUTS[layout])[:2]}",
                  "  generator    speed  made  clipped  peak %  release %  landing mm    v     a    tau   line x / yz mm   held"]
        for gen in GENERATORS:
            for r in (r for r in rows if r["layout"] == layout and r["generator"] == gen):
                if not int(r["delivered"]):
                    lines.append(f"  {gen:11s}  {float(r['v_peak']):4.2f}   no    {r['reason'][:88]}")
                    continue
                lines.append(f"  {gen:11s}  {float(r['v_peak']):4.2f}   yes   {'yes' if int(r['clipped']) else ' no'}   {f(r, 'peak_speed_err_pct', '+7.2f')}"
                             f"   {f(r, 'release_err_pct', '+7.2f')}    {f(r, 'landing_err_mm', '+8.1f')}  {f(r, 'plan_v_ratio', '5.2f')} {f(r, 'plan_a_ratio', '5.2f')}"
                             f" {f(r, 'plan_tau_ratio', '5.2f')}   {f(r, 'path_x_err_mm', '6.3f')} / {f(r, 'path_yz_err_mm', '5.3f')}   {r.get('stuck_steps') or '-':>4}")
        lines.append("")
        for gen in GENERATORS:
            rs = [r for r in rows if r["layout"] == layout and r["generator"] == gen]
            good = [float(r["v_peak"]) for r in rs if int(r["delivered"]) and int(r["inside_limits"]) and not int(r["clipped"])
                    and abs(float(r["release_err_pct"])) < 1.5]
            clipped = sum(int(r["delivered"]) and int(r["clipped"]) for r in rs)
            refused = sum(not int(r["delivered"]) for r in rs)
            lines.append(f"  {gen:11s}: {len(good)} of {len(rs)} pushes inside all three limits with the release within 1.5% of the model"
                         f"{' (up to ' + format(max(good), '.1f') + ' m/s)' if good else ''}; clipped {clipped}; refused {refused}")
        lines.append("")
    return "\n".join(lines)

# Three hues for the three kinds of generator (fixed order, checked for colour-vision separation on this surface); the
# limit set is the line style and the marker fill, so no series is told apart by colour alone.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"current":     dict(color="#2a78d6", ls="-",  marker="o", mfc="#2a78d6", label="arm.py (start-posture search)"),
         "diffik":      dict(color="#eb6834", ls="-",  marker="s", mfc="#eb6834", label="differential IK, velocity limits"),
         "diffik_acc":  dict(color="#eb6834", ls="--", marker="s", mfc=SURFACE,   label="differential IK, + acceleration limits"),
         "trajopt_vel": dict(color="#1baf7a", ls="-",  marker="^", mfc="#1baf7a", label="trajectory optimisation, velocity bounds"),
         "trajopt":     dict(color="#1baf7a", ls="--", marker="^", mfc=SURFACE,   label="trajectory optimisation, + acceleration bounds")}
TITLES = {"side": "base beside the board", "origin": "base on the push line"}

def figure(rows, path):
    """Release-speed error and the worst planned limit ratio against the commanded speed, one column per base."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    layouts = list(dict.fromkeys(r["layout"] for r in rows))
    fig, axes = plt.subplots(2, len(layouts), figsize=(4.6*len(layouts), 6.4), sharex="col", squeeze=False, facecolor=SURFACE)
    num = lambda r, k: float(r[k])
    for c, layout in enumerate(layouts):
        for gen in GENERATORS:
            rs = [r for r in rows if r["layout"] == layout and r["generator"] == gen and int(r["delivered"])]
            st = STYLE[gen]
            kw = dict(color=st["color"], ls=st["ls"], lw=2, marker=st["marker"], ms=6.5, mfc=st["mfc"], mec=st["color"], mew=1.5, label=st["label"])
            u = [num(r, "v_peak") for r in rs]
            axes[0][c].plot(u, [num(r, "release_err_pct") for r in rs], **kw)
            axes[1][c].plot(u, [max(num(r, k) for k in ("plan_v_ratio", "plan_a_ratio", "plan_tau_ratio")) for r in rs], **kw)
        refused = sorted({(r["generator"], float(r["v_peak"])) for r in rows if r["layout"] == layout and not int(r["delivered"])})
        if refused:
            names = sorted({g for g, _ in refused}, key=GENERATORS.index)
            axes[0][c].text(0.03, 0.05, "no point: the generator refused the push\n(" + ", ".join(names) + ")", transform=axes[0][c].transAxes,
                            fontsize=8, color=INK2, va="bottom")
        axes[1][c].axhline(1.0, color=INK2, lw=1, zorder=0)
        axes[1][c].set_yscale("log")
        axes[1][c].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axes[1][c].yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v in (0.5, 2, 3, 5, 20, 50) else ""))
        axes[1][c].text(0.015, 1.0, "limit", transform=axes[1][c].get_yaxis_transform(), ha="left", va="top", fontsize=8, color=INK2)
        axes[0][c].set_title(TITLES.get(layout, layout), fontsize=11, color=INK, loc="left")
        axes[1][c].set_xlabel("commanded peak paddle speed, m/s", color=INK2)
    axes[0][0].set_ylabel("release speed error, %", color=INK2)
    axes[1][0].set_ylabel("worst planned limit ratio", color=INK2)
    same = [g for g in ("diffik", "diffik_acc") if all(abs(float(a["release_err_pct"]) - float(b["release_err_pct"])) < 0.15
            for a in rows for b in rows if a["layout"] == b["layout"] == layouts[0] and a["generator"] == "current" and b["generator"] == g
            and a["v_peak"] == b["v_peak"] and int(a["delivered"]) and int(b["delivered"]))]
    if same:
        axes[0][0].text(0.97, 0.93, "arm.py and differential IK coincide here", transform=axes[0][0].transAxes, fontsize=8, color=INK2, ha="right", va="top")
    for ax in axes.flat:
        ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-brake", type=float, default=6.0, help="brake cap, m/s^2 (default 6, the rerun's)")
    ap.add_argument("--layouts", nargs="+", default=["side", "origin"], choices=list(SPEEDS))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RESULTS/"side_base"/"e1_motion_generators")
    ap.add_argument("--summarize", type=Path, default=None, metavar="CSV")
    a = ap.parse_args()
    if a.summarize:
        with a.summarize.open(newline="") as f:
            rows = list(csv.DictReader(f))
        print(summarize(rows, float(rows[0]["a_brake"])))
        figure(rows, a.summarize.with_name("summary.png"))
        return
    jobs = [(layout, gen, u, a.a_brake) for layout in a.layouts for gen in GENERATORS for u in SPEEDS[layout]]
    with Pool(a.procs) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/"results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n", restval="")
        w.writeheader(); w.writerows(rows)
    with (a.out/"results.csv").open(newline="") as f:
        saved = list(csv.DictReader(f))
    text = summarize(saved, a.a_brake)
    (a.out/"summary.txt").write_text(text + "\n")
    figure(saved, a.out/"summary.png")
    print(text)

if __name__ == "__main__":
    main()

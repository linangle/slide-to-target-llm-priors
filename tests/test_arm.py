"""arm.py: the iiwa as the pusher, and the target planner on top of it.  The controller tests come first (the first needs none
of Drake), then the planner's (Diego's 6 Nov tests, on the merged controller); the two marked slow each run one short push."""
import math
import numpy as np
import pytest
from physics import G, release_from_profile, release_speed, release_push_distance
from push import PLATE, profile

def test_brake_cap_leaves_the_release_alone():
    """With the deceleration capped at a_max > mu g the ramp still releases at v_peak - (mu g)^2/(2J); only the stop moves."""
    for v, J, mu in ((0.8, 100.0, 0.3), (1.0, 2000.0, 0.5), (0.6, 30.0, 0.5), (1.0, 200.0, 0.1)):
        p0, p1 = profile(v, J), profile(v, J, a_max=12.0)
        t_star, v_r = release_from_profile(p1["t"], p1["v"], p1["a"], mu)
        assert abs(v_r - release_speed(v, mu, J)) < 1e-6
        assert abs(float(np.interp(t_star, p1["t"], p1["x"])) - release_push_distance(v, mu, J)) < 1e-6
        assert p1["a"].min() >= -12.0 - 1e-9 and p1["a_max"] == 12.0 and p0["a_max"] == math.inf
        assert p1["x_stop"] >= p0["x_stop"] - 1e-9 and p1["t_stop"] >= p0["t_stop"] - 1e-9
        brake = p1["t"] > p1["t_peak"]
        assert np.all(np.diff(p1["v"][brake]) <= 1e-12) and p1["v"][-1] == 0.0
    assert profile(0.8, 100.0, a_max=math.inf)["x_stop"] == profile(0.8, 100.0)["x_stop"]

def test_link7_pose_puts_the_plate_where_the_prescribed_plate_is():
    from pydrake.all import RigidTransform
    from arm import X_L7P, link7_pose
    th, wd, ht = PLATE
    for x_face, lat, yaw in ((0.499, 0.0, 0.0), (0.6, 0.02, 2.0), (0.45, -0.07, 0.5)):
        X_WP = link7_pose(x_face, lat, yaw, 0.003) @ X_L7P                   # the paddle body = plate centre
        c = X_WP.translation()
        assert abs(c[1] - lat) < 1e-12 and abs(c[2] - (0.003 + ht/2)) < 1e-12
        R = X_WP.rotation().matrix()
        n = R[:, 2]                                                           # the face normal is link-7 z
        assert abs(math.degrees(math.atan2(n[1], n[0])) - yaw) < 1e-9 and abs(n[2]) < 1e-12
        assert np.allclose(R[:, 1], [0, 0, 1])                                # the plate's long side is vertical
        # the nearest point of the (yawed) face to the object is at x_face, as in push.Pusher.add
        corners = [c + R @ np.array([sx*wd/2, sy*ht/2, th/2]) for sx in (-1, 1) for sy in (-1, 1)]
        assert abs(max(k[0] for k in corners) - x_face) < 1e-12

def test_plan_follows_the_push_line_and_reports_the_iiwa_limits():
    """The planned joint path keeps the paddle on the straight line to 1e-7 inside the joint limits, and the plan reports what
    the push asks of the iiwa: with the base on the push line the elbow (joint 4, 75 deg/s) is over its speed limit at 0.8 m/s."""
    from pydrake.all import RigidTransform
    from arm import ArmModel, ArmPusher, pose_error
    pu = ArmPusher()
    pu.x_touch0, pu.model = 0.499, ArmModel(pu.base)
    from arm import link7_pose
    pu.X0 = link7_pose(pu.x_touch0, 0.0, 0.0, pu.clearance)
    prof = profile(0.8, 100.0, a_max=pu.a_brake_max)
    plan = pu.make_plan(prof)
    m = pu.model
    for s in np.linspace(0.0, plan["s"][-1], 7):
        q = pu.spline.value(s)[:, 0]
        X = RigidTransform(pu.X0.rotation(), pu.X0.translation() + np.array([s, 0, 0]))
        assert np.linalg.norm(pose_error(X, m.fk(q))) < 1e-7
        assert np.all(q > m.q_lo) and np.all(q < m.q_hi)
    assert plan["n_ok"] >= 1 and plan["sigma_min"] > 0.03 and plan["tau_ratio"] < 1.0
    assert plan["qd_ratio_joint"] == 4 and plan["qd_ratio"] > 1.0       # the 6 Nov finding; see arm.py and the README
    assert 10.0 < plan["m_eff"] < 50.0                                    # reflected rotor inertia: far more than the plate's 5 kg
    assert plan["q_d"].shape == (len(prof["t"]), 7) and plan["tau_ff"].shape == plan["q_d"].shape

@pytest.mark.slow
def test_one_push_with_the_arm_meets_the_plate_bar():
    """Box, config contact, mu 0.3, v_peak 0.8, J 100 with the iiwa: the same bar as tests/test_push.py's plate test."""
    from scene import Obj
    from drake_config import contact_from_config
    from push import push_slide
    from arm import ArmPusher
    pu = ArmPusher()
    r = push_slide(0.3, 0.8, 100.0, contact_from_config(), Obj(), pu, keep_traj=False)
    assert r["n_point_contacts"] == 0 and r["separated"]
    assert abs(r["v_r_cmd_err_pct"]) < 1.0 and abs(r["v_r_act_err_pct"]) < 1.0
    assert abs(r["sep_lag_ms"]) < 3.0
    assert abs(r["d_err_pct"]) < 2.0 and abs(r["psi_r_deg"]) < 0.05 and abs(r["yaw_rest_deg"]) < 0.1
    assert abs(r["x_fk"] - r["x_sep"]) < 2e-4
    assert r["track_q_rms"] < 2e-4
    assert r["arm_tau_ratio"] < 1.0 and r["arm_z_drift_mm"] < 0.5 and r["arm_tilt_max_deg"] < 0.1   # no torque clamping, plate on its line
    assert r["a_max"] == pu.a_brake_max

# ----------------------------------------------------------------------------- the target planner (plan_peak_speed and friends)

def _belief(m, S, n=2000):
    """ln mu ~ N(m, S^2) on a grid, in belief.posterior's layout (grid, pmf)."""
    grid = np.linspace(math.log(0.02), math.log(2.0), n)
    pmf = np.exp(-0.5*((grid - m)/S)**2)
    return {"grid": grid, "pmf": pmf/pmf.sum()}

def test_plan_lands_on_target_under_the_model():
    from arm import plan_peak_speed, rest_x
    for x_t, mu, J in ((0.75, 0.3, 100.0), (0.70, 0.5, 300.0), (0.9, 0.15, 100.0)):
        u = plan_peak_speed(x_t, mu, J)
        assert abs(rest_x(u, mu, J) - x_t) < 1e-6

def test_plan_rejects_unreachable_targets():
    from arm import plan_peak_speed
    with pytest.raises(ValueError):
        plan_peak_speed(0.50, 0.3)            # behind the start
    with pytest.raises(ValueError):
        plan_peak_speed(1.50, 0.5)            # farther than the arm's stroke can send it
    with pytest.raises(ValueError):
        plan_peak_speed(0.75, 0.3, u_max=0.45)   # needs 0.66 m/s: over a caller's speed cap

def test_plan_with_a_belief_matches_a_point_belief():
    """A belief concentrated on one mu gives the same speed as that mu."""
    from arm import plan_peak_speed
    u0 = plan_peak_speed(0.8, 0.3)
    assert abs(plan_peak_speed(0.8, _belief(math.log(0.3), 0.002)) - u0) < 2e-3

def test_belief_plan_is_decides_rule_with_the_target_slide_moving_with_the_speed():
    """plan_peak_speed on a belief is decide.choose_speed with d*(u, mu) = x* - x_r(u, mu).  With the squared miss as the loss it
    is the 6 Nov planner's objective, E[(x_rest - x*)^2]; the specification's squared relative loss lands within 3 mm/s of it.
    A FIXED d* taken from the nominal speed's release point is the convention error: it picks a slower push that releases
    earlier and so lands about 2 cm short of the point."""
    import decide as D
    from arm import plan_peak_speed, rest_x
    from scipy.optimize import minimize_scalar
    x_t, x0, J, post = 0.75, 0.55, 100.0, _belief(math.log(0.3), 0.3)
    rel, u_max = D.Release(J=J), 1.0
    u = plan_peak_speed(x_t, post, J, x0, u_max=u_max)
    assert u == D.choose_speed(post, rel, D.point_target(x_t, x0), D.loss_sq_rel, u_max=u_max)["u"]
    mus, w = np.exp(post["grid"]), post["pmf"]
    direct = minimize_scalar(lambda v: float(np.sum(w*(np.array([rest_x(v, m, J, x0) for m in mus]) - x_t)**2)), bounds=(0.3, 1.0),
                             method="bounded", options={"xatol": 1e-7}).x
    u_abs = plan_peak_speed(x_t, post, J, x0, u_max=u_max, loss=D.loss_sq_abs)
    assert abs(u_abs - direct) < 1e-4
    assert abs(u - u_abs) < 3e-3
    u_nom = plan_peak_speed(x_t, 0.3, J, x0, u_max=u_max)
    d_fixed = x_t - (x0 - 0.001 + release_push_distance(u_nom, 0.3, J))        # the slide the nominal speed needs
    u_fixed = D.choose_speed(post, rel, d_fixed, D.loss_sq_rel, u_max=u_max)["u"]
    assert u_fixed < u - 0.03 and 0.015 < rest_x(u, 0.3, J, x0) - rest_x(u_fixed, 0.3, J, x0) < 0.035

def test_reach_speed_bounds_and_the_joint_limit_report():
    """With the base at the origin the paddle follows the push line to x = 0.92 m, and joint 4's speed limit caps the push
    near 0.48 m/s (both as in Diego's 6 Nov controller).  joint_limit_report(pusher.traj) is the hook the evaluation scripts read."""
    from arm import ArmModel, ArmPusher, joint_limit_report, limited_peak_speed, link7_pose, max_peak_speed, reach
    assert 0.40 < reach() < 0.45
    assert 1.0 < max_peak_speed(100.0) < 1.2
    u_lim = limited_peak_speed(100.0)
    assert 0.42 < u_lim < 0.52
    pu = ArmPusher()
    assert pu.traj is None
    pu.x_touch0, pu.model = 0.499, ArmModel(pu.base)
    pu.X0 = link7_pose(pu.x_touch0, 0.0, 0.0, pu.clearance)
    pu.make_plan(profile(u_lim, 100.0, a_max=pu.a_brake_max))
    jl = joint_limit_report(pu.traj)
    assert all(jl[k].shape == (7,) for k in ("q", "v", "tau"))
    assert 0.95 < jl["v"].max() <= 1.0 and int(np.argmax(jl["v"])) == 3 and jl["q"].max() < 1.0 and jl["tau"].max() < 1.0
    assert abs(jl["v"].max() - pu.plan["qd_ratio"]) < 1e-12 and abs(jl["tau"].max() - pu.plan["tau_ratio"]) < 1e-12

@pytest.mark.slow
def test_slide_to_target_with_true_mu():
    from arm import slide_to_target
    r = slide_to_target(0.75, 0.3)
    assert abs(r["miss"]) < 0.005 and abs(r["miss_y"]) < 0.002
    assert r["over_limits"] and r["joint_limits"]["v"].max() > 1.0      # 0.66 m/s from the origin is over joint 4's speed limit

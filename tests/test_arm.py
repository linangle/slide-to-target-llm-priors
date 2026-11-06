"""arm.py: the iiwa as the pusher.  The first tests need no simulation (the first none of Drake); the last runs one short push."""
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

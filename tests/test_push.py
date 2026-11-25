"""push.py: the prescribed-paddle harness.  The first two tests need no Drake; the last runs one short push."""
import math
import numpy as np
import pytest
from physics import G, release_from_profile, release_push_distance, release_speed
from push import PLATE, Pusher, profile

def test_profile_peaks_at_v_peak_with_zero_acceleration():
    for v, J in ((0.8, 100.0), (1.0, 50.0), (0.6, 30.0), (1.0, 2000.0)):
        p = profile(v, J)
        i = int(np.argmax(p["v"]))
        assert abs(p["v"][i] - v) < 1e-6          # the sample grid straddles the exact peak by up to dt/2
        assert abs(p["t"][i] - p["t_peak"]) < 2e-4 and abs(p["a"][i]) < 0.01
        assert p["v"][-1] == 0.0 and p["a"][-1] == 0.0 and np.all(np.diff(p["t"]) > 0)

def test_profile_peak_deceleration_is_min_of_cap_and_sqrt_2Ju():
    """The brake is one jerk ramp that runs until the paddle stops (no ramp back down), so the commanded deceleration peaks
    at sqrt(2 J u), or at the cap once sqrt(2 J u) exceeds it: from u = a_max^2 / (2 J), i.e. 0.72 m/s for the arm's
    12 m/s^2 cap and 0.18 m/s for a 6 m/s^2 cap at J = 100.  release_speed > 0 is the same condition as mu g < sqrt(2 J u)."""
    J = 100.0
    for u in (0.2, 0.33, 0.6, 1.0):
        for cap in (math.inf, 12.0, 6.0):
            peak = float((-profile(u, J, a_max=cap)["a"]).max())
            assert abs(peak - min(cap, math.sqrt(2*J*u))) < 0.02          # the sample before the stop, one dt of jerk short
        mu_edge = math.sqrt(2*J*u)/G
        assert release_speed(u, 0.98*mu_edge, J) > 0 and release_speed(u, 1.02*mu_edge, J) == 0
    assert float((-profile(0.70, J, a_max=12.0)["a"]).max()) < 12.0 - 0.1 and float((-profile(0.17, J, a_max=6.0)["a"]).max()) < 6.0 - 0.1

def test_profile_brake_matches_release_model():
    """One constant-jerk segment through the peak: release_from_profile on the command reproduces release_speed."""
    for v, J, mu in ((0.8, 100.0, 0.3), (1.0, 50.0, 0.1), (0.6, 30.0, 0.5)):
        p = profile(v, J)
        t_star, v_r = release_from_profile(p["t"], p["v"], p["a"], mu)
        assert abs(v_r - release_speed(v, mu, J)) < 1e-6
        assert abs(t_star - (p["t_peak"] + mu*G/J)) < 2e-4

def test_release_push_distance_matches_profile():
    """physics.release_push_distance (decide.slide_room's release point) is the commanded travel up to t*, or up to
    the paddle's stop when the ramp never out-brakes friction (the last case: (mu g)^2 / 2J > v_peak)."""
    for v, J, mu in ((0.8, 100.0, 0.3), (1.2, 100.0, 0.1), (1.0, 50.0, 0.5), (0.6, 2000.0, 0.3), (0.05, 100.0, 0.3),
                     (0.2, 30.0, 0.5)):
        p = profile(v, J)
        t_star = min(p["t_peak"] + mu*G/J, p["t_stop"])
        assert abs(release_push_distance(v, mu, J) - float(np.interp(t_star, p["t"], p["x"]))) < 1e-6, (v, J, mu)
    assert release_push_distance(0.0, 0.3, 100.0) == 0.0
    assert abs(release_push_distance(1.0, 0.3, math.inf) - 0.25) < 1e-12        # v^2 / (2 a_push), a_push = 2

def test_pusher_geometry():
    assert PLATE == (0.01, 0.12, 0.18)
    pu = Pusher(gap=0.002)
    assert pu.body is None and pu.x_touch0 is None

@pytest.mark.slow
def test_one_push_matches_release_model_and_slide():
    """Box, config contact, mu 0.3, v_peak 0.8, J 100: the object leaves within 1% of the model and slides within 2% of v^2/(2 mu g)."""
    from scene import Obj
    from drake_config import contact_from_config
    from push import push_slide
    r = push_slide(0.3, 0.8, 100.0, contact_from_config(), Obj(), Pusher(), keep_traj=False)
    assert r["n_point_contacts"] == 0 and r["separated"]
    assert abs(r["v_r_cmd_err_pct"]) < 1.0 and abs(r["v_r_act_err_pct"]) < 1.0
    assert abs(r["sep_lag_ms"]) < 3.0
    assert abs(r["d_err_pct"]) < 2.0 and abs(r["psi_r_deg"]) < 0.05 and abs(r["yaw_rest_deg"]) < 0.1
    assert abs(r["x_fk"] - r["x_sep"]) < 2e-4
    assert r["track_q_rms"] < 2e-4

"""push.py: the prescribed-paddle harness.  The first two tests need no Drake; the last runs one short push."""
import math
import numpy as np
import pytest
from physics import G, release_from_profile, release_speed
from push import PLATE, Pusher, profile

def test_profile_peaks_at_v_peak_with_zero_acceleration():
    for v, J in ((0.8, 100.0), (1.0, 50.0), (0.6, 30.0), (1.0, 2000.0)):
        p = profile(v, J)
        i = int(np.argmax(p["v"]))
        assert abs(p["v"][i] - v) < 1e-6          # the sample grid straddles the exact peak by up to dt/2
        assert abs(p["t"][i] - p["t_peak"]) < 2e-4 and abs(p["a"][i]) < 0.01
        assert p["v"][-1] == 0.0 and p["a"][-1] == 0.0 and np.all(np.diff(p["t"]) > 0)

def test_profile_brake_matches_release_model():
    """One constant-jerk segment through the peak: release_from_profile on the command reproduces release_speed."""
    for v, J, mu in ((0.8, 100.0, 0.3), (1.0, 50.0, 0.1), (0.6, 30.0, 0.5)):
        p = profile(v, J)
        t_star, v_r = release_from_profile(p["t"], p["v"], p["a"], mu)
        assert abs(v_r - release_speed(v, mu, J)) < 1e-6
        assert abs(t_star - (p["t_peak"] + mu*G/J)) < 2e-4

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

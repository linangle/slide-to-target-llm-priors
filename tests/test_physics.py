"""Tests 1 and 6 of the implementation spec, plus checks on the piecewise slide and the tipping inequalities."""
import math
import numpy as np
import physics as P

def test_release_from_profile_matches_release_speed():
    """Test 1: on a synthetic constant-jerk profile the sampled release matches v_peak - (mu g)^2 / 2J to 1e-6 m/s."""
    for v_peak, J, mu in ((0.8, 100.0, 0.3), (1.1, 300.0, 0.5), (0.5, 50.0, 0.1)):
        t, v, a = P.constant_jerk_profile(v_peak, J, dt=1e-4)
        ts, vs = P.release_from_profile(t, v, a, mu)
        assert abs(vs - P.release_speed(v_peak, mu, J)) < 1e-6
        assert ts > t[int(np.argmax(v))]

def test_release_speed_limits():
    assert P.release_speed(0.8, 0.3, math.inf) == 0.8
    assert P.release_speed(0.1, 0.5, 10.0) == 0.0          # ramp never out-brakes friction: no release
    assert P.release_from_profile(*P.constant_jerk_profile(0.8, 100.0, a_max=1.0), mu=0.3) is None

def test_disk_spin_ratio():
    """Test 6: uniform-pressure disk integrator, d/d0 = 1.065 +- 0.01 at R omega / v = 0.5 and 1.27 +- 0.02 at 1."""
    assert abs(P.disk_spin_ratio(0.5) - 1.065) < 0.01
    assert abs(P.disk_spin_ratio(1.0) - 1.27) < 0.02
    assert abs(P.disk_spin_ratio(0.0) - 1.0) < 1e-3

def test_slide_piecewise_reduces_to_uniform():
    s = P.slide_piecewise(0.0, 0.8, [(-1.0, 0.1, 0.3), (0.1, math.inf, 0.3)])
    assert abs(s["x_rest"] - P.stop_distance(0.8, 0.3)) < 1e-9
    assert abs(s["t_stop"] - P.slide_time(0.8, 0.3)) < 1e-9
    # a stickier strip in the way shortens the slide by the energy it absorbs
    s2 = P.slide_piecewise(0.0, 0.8, [(-1.0, 0.05, 0.3), (0.05, 0.08, 0.6), (0.08, math.inf, 0.3)])
    assert s2["x_rest"] < s["x_rest"]
    assert abs(s2["x_rest"] - (0.05 + (0.8**2/(2*P.G) - 0.05*0.3)/0.6)) < 1e-9   # stops inside the sticky strip
    f = P.slide_field(0.0, 0.8, lambda x: np.full_like(x, 0.3))
    assert abs(f["x_rest"] - P.stop_distance(0.8, 0.3)) < 1e-3

def test_no_tip():
    push_ok, slide_ok = P.no_tip(a=2.0, mu=0.3, h_p=0.025, h_c=0.025, b=0.05)
    assert push_ok and slide_ok
    assert not P.no_tip(a=2.0, mu=0.9, h_p=0.025, h_c=0.025, b=0.02)[1]   # mu h_c = 0.0225 > b
    assert not P.no_tip(a=30.0, mu=0.3, h_p=0.045, h_c=0.025, b=0.05)[0]

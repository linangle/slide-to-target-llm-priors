"""Test 5 of the implementation spec, plus the losses and the practice speed."""
import math
import numpy as np
import belief as B
import decide as D

def test_choose_speed_matches_closed_form_for_abrupt_braking():
    """Test 5: J -> inf, squared relative loss: u* = sqrt(2 g d* exp(m - 1.5 S^2)) to three significant digits.
    The posteriors must sit well inside the grid: the mu^-2 weight of the loss makes the result sensitive to the
    grid's mu = 0.05 floor once m - 3 S approaches ln 0.05 (a wide prior around mu = 0.2 is already off by 0.5%)."""
    g = B.theta_grid()
    for m, S, d_star in ((math.log(0.3), 0.3, 0.3), (math.log(0.15), 0.2, 0.5), (math.log(0.2), 0.3, 0.2)):
        post = B.posterior(g, B.logprior(g, m, S))
        u = D.choose_speed(post, D.Release(J=math.inf), d_star, D.loss_sq_rel, u_max=3.0)["u"]
        want = D.closed_form_speed(d_star, m, S)
        assert abs(u/want - 1) < 5e-4, (u, want)

def test_finite_jerk_raises_commanded_speed():
    g = B.theta_grid()
    post = B.posterior(g, B.logprior(g, math.log(0.3), 0.2))
    u_inf = D.choose_speed(post, D.Release(J=math.inf), 0.3, u_max=3.0)["u"]
    u_100 = D.choose_speed(post, D.Release(J=100.0), 0.3, u_max=3.0)["u"]
    assert u_100 > u_inf

def test_losses_and_practice_speed():
    assert D.loss_miss(np.array([0.3, 0.32, 0.35]), 0.3).tolist() == [0.0, 0.0, 1.0]
    assert D.loss_table_edge(0.5, 0.3) > 10
    g = B.theta_grid()
    post = B.posterior(g, B.logprior(g, math.log(0.3), 0.3))
    u = D.practice_speed(1.2, 0.3, post, D.Release(J=math.inf))
    mu_low = post["mu_q05"]
    assert u < 1.2 and abs(u**2/(2*mu_low*D.G) - 0.3) < 1e-6
    assert D.practice_speed(0.3, 0.6, post, D.Release(J=math.inf)) == 0.3
    for name in D.LOSSES:
        r = D.choose_speed(post, D.Release(), 0.3, D.LOSSES[name], u_max=2.0)
        assert 0 < r["u"] <= 2.0

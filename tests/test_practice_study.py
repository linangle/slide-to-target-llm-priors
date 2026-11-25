"""Offline tests of evaluation/practice_study.py: no Drake run, no API."""
import math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
for _d in ("src", "evaluation"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/_d))
import run_arm_evaluation as E
import practice_study as P
from belief import logprior, posterior, theta_grid, update_with_slides
from observe import simulate_obs
from physics import OBJECT_SIZE


def fake_shot(mu, v=.4):
    t0 = .5; t = np.arange(0, 1.5, .001); tau = np.clip(t - t0, 0, v/(mu*9.81))
    x = .60 + v*tau - .5*mu*9.81*tau**2
    r = dict(x_rest=float(x[-1]), y_rest=0., t_force_zero=t0,
             traj=dict(t=t, x=x, vx=np.where(t > t0, np.maximum(v - mu*9.81*(t - t0), 0), 0), q=np.zeros(len(t))))
    return r, SimpleNamespace(face_x=lambda q: .60 - OBJECT_SIZE[0]/2)


def test_noisy_observation_is_the_evaluations():
    """The study's observation is run_arm_evaluation.make_observation's at the same noise multiple and random state."""
    r, p = fake_shot(.14)
    clean = P.clean_observation(r, p)
    for name, scale in (("low", .5), ("nominal", 1.), ("high", 2.)):
        a = P.noisy(clean, np.random.default_rng(7), scale)
        b = E.make_observation(r, p, np.random.default_rng(7), name)
        np.testing.assert_allclose(a.t, b.t); np.testing.assert_allclose(a.x, b.x, atol=1e-15)
        assert math.isclose(a.x_rest, b.x_rest, abs_tol=1e-15) and math.isclose(a.x_fk, b.x_fk, abs_tol=1e-15)
        assert (a.sigma_f, a.sigma_rest, a.sigma_fk) == (b.sigma_f, b.sigma_rest, b.sigma_fk)
    wide = P.noisy(clean, np.random.default_rng(7), 16.)
    np.testing.assert_allclose(wide.x - clean["x"], 16*(P.noisy(clean, np.random.default_rng(7), 1.).x - clean["x"]), atol=1e-15)


def test_priors_are_the_evaluations():
    truth, g, initials = P.priors()
    assert set(initials) == {"Y01", "Y02", "Y03"} and set(initials["Y01"]) == set(P.METHODS)
    assert len(g) == 2000 and math.isclose(math.exp(g[0]), .005) and math.isclose(math.exp(g[-1]), 5.)
    assert math.isclose(posterior(g, initials["Y01"]["generic"])["mu_median"], P.PRIOR_MEDIAN, rel_tol=2e-3)
    assert truth["Y03"] == .26


def test_transfer_update_limits():
    """No spread: the standard update.  A small spread: nearly the standard update.  A large spread: a precise practice
    slide moves the belief only part of the way from the prior, by the share s0^2 / (s0^2 + s^2), and the belief about the
    next slide is at least as wide as the spread."""
    g = theta_grid(2000, math.log(.005), math.log(5.))
    m0, s0, mu = math.log(.28), .26, .14
    lp = logprior(g, m0, s0)
    obs = [simulate_obs(.8, .9, mu, 60., .002, .0005, .0005, np.random.default_rng(1))]
    plain = update_with_slides(g, lp, obs)
    assert P.transfer_kernel(g, 0.) is None and P.update_with_transfer(g, lp, obs, None)["median"] == plain["median"]
    small = P.update_with_transfer(g, lp, obs, P.transfer_kernel(g, .01))
    assert abs(small["median"] - plain["median"]) < .005 and .012 < small["sd"] < .02       # sqrt(2) * 0.01, near enough
    s = .25
    wide = P.update_with_transfer(g, lp, obs, P.transfer_kernel(g, s))
    w = s0**2/(s0**2 + s**2)
    expect = m0 + w*(math.log(mu) - m0)
    assert abs(wide["mean"] - expect) < .01
    assert math.isclose(wide["sd"], math.sqrt(s**2 + 1/(1/s0**2 + 1/s**2)), rel_tol=.03)
    assert abs(sum(P.transfer_kernel(g, s)[1000]) - 1) < 1e-12


def test_landing_map_drops_seam_losses(tmp_path):
    """A push that lands short at one sampled speed does not pull the map down at its neighbours."""
    from arm import rest_x
    truth, _, _ = P.priors()
    us = np.round(np.arange(.1, 1.0, .01), 6)
    with (tmp_path/"landing_map.csv").open("w") as f:
        f.write("pair,u,x_rest,y_rest,valid,v_frac,tau_frac,a_frac,error\n")
        for u in us:
            x = rest_x(u, truth["Y01"], E.JERK, E.X0) - 2e-4 - (5e-3 if abs(u - .5) < 1e-9 else 0)
            f.write(f"Y01,{u},{x},0,True,0,0,0,\n")
        f.write("Y01,0.05,0.55,0,False,0,0,0,\n")
    lmap = P.load_map(tmp_path)
    for u in (.495, .5, .505, .3):
        assert abs(P.landing(lmap, "Y01", u) - (rest_x(u, truth["Y01"], E.JERK, E.X0) - 2e-4)) < 2e-5
    assert lmap["Y01"][0][0] == .1                                  # the invalid push is not in the map
    assert E.X0 < P.landing(lmap, "Y01", .05) < P.landing(lmap, "Y01", .1)
    import json
    c = json.loads((tmp_path/"map_check.json").read_text())["Y01"]
    assert c["pushes_with_seam_loss"] == 1 and abs(c["seam_loss_worst_mm"] + 5) < .01 and c["seam_loss_worst_at"] == .5

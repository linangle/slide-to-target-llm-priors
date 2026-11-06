"""Test 4 of the implementation spec, plus the slide likelihood on the grid."""
import math
import numpy as np
import pytest
import belief as B
import observe as O

def test_grid_matches_conjugate_closed_form():
    """Test 4: with Gaussian-in-theta observations the grid posterior matches the normal update to 1e-3."""
    g = B.theta_grid()
    m0, s0, s = math.log(0.3), 0.45, 0.1
    ys = [math.log(0.25), math.log(0.28), math.log(0.22)]
    post = B.posterior(g, B.logprior(g, m0, s0), *[B.loglik_gaussian_theta(g, y, s) for y in ys])
    mean, sd = B.conjugate_update(m0, s0, ys, s)
    assert abs(post["mean"] - mean) < 1e-3
    assert abs(post["sd"] - sd) < 1e-3

def test_llm_likelihood_fit():
    rng = np.random.default_rng(0)
    th = np.log(rng.uniform(0.1, 0.6, 40))
    m = 0.2 + 0.8*th + 0.15*rng.standard_normal(40)
    p = B.fit_llm_likelihood(m, th)
    assert abs(p["alpha"] - 0.2) < 0.15 and abs(p["beta"] - 0.8) < 0.15 and abs(p["s"] - 0.15) < 0.05
    loo = B.fit_llm_likelihood_loo(m, th, np.repeat(np.arange(8), 5))
    assert len(loo) == 8 and all(v["n"] == 35 for v in loo.values())

def test_slide_likelihood_peaks_at_truth_and_matches_fit():
    rng = np.random.default_rng(5)
    g = B.theta_grid()
    obs = O.simulate_obs(0.5, 0.8, 0.3, 30, 0.002, 0.0005, 0.0005, rng)
    ll = B.loglik_slide(g, obs, "C")
    fit = O.fit_slide(obs, "C")
    post = B.posterior(g, ll)
    assert abs(post["mu_median"]/fit["mu"] - 1) < 0.02          # flat prior: grid mode ~ least-squares estimate
    assert abs(post["sd"] - fit["se_mu"]/fit["mu"]) < 0.3*fit["se_mu"]/fit["mu"]   # SE in ln mu ~ relative SE
    # profiled nuisance at the true mu is close to the truth
    x_r, v_r, _ = B.profile_nuisance(0.3, obs, "C")
    assert abs(x_r[0] - 0.5) < 0.003 and abs(v_r[0] - 0.8) < 0.05

def test_n0_ranges_as_in_spec():
    """n0 from about 5e-5 slides for slippery long slides to about 0.17 for mu = 0.5 at 0.5 m/s (s0 = 0.45) at the
    60 Hz camera (5 Nov decision 8); at the 31 Oct budget's 30 Hz the range was 1e-4 to 0.3."""
    lo = B.n0(O.crlb(0.1, 1.1, O.CAMERA_FPS, 0.002, 0.0005, 0.0005), 0.45)
    hi = B.n0(O.crlb(0.5, 0.5, O.CAMERA_FPS, 0.002, 0.0005, 0.0005), 0.45)
    assert lo < 1e-3 and 0.1 < hi < 0.3
    assert B.n0(O.crlb(0.5, 0.5, 30, 0.002, 0.0005, 0.0005), 0.45) > hi

def test_lookup_is_an_observation_on_the_prior():
    """Decision 3 (5 Nov): the lookup value enters as one Gaussian observation of theta, so prior + lookup on the
    grid matches the conjugate update; the row is read from friction_lookup.csv by column name."""
    g = B.theta_grid()
    m0, s0 = math.log(0.3), 0.45
    mu_tab, s_tab = 0.25, math.log(2.0)/1.645                      # tier A under the 2 / 2.5 / 3 factors
    post = B.posterior(g, B.logprior(g, m0, s0), B.loglik_lookup(g, mu_tab, s_tab))
    mean, sd = B.conjugate_update(m0, s0, [math.log(mu_tab)], s_tab)
    assert abs(post["mean"] - mean) < 1e-3 and abs(post["sd"] - sd) < 1e-3
    assert B.lookup_observation("V03") == (0.33, 0.421)              # decision 2 value, tier A spread
    assert B.lookup_observation("Y02") == (0.25, 0.357)              # decision 5 pooled entry, factor 1.8
    assert B.lookup_observation("M06") is None                       # tier none: prior alone
    for missing in ("Y04", "m07"):                                    # dropped on 5 Nov (decision 6); a typo
        with pytest.raises(KeyError):
            B.lookup_observation(missing)

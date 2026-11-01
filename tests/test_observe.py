"""Tests 2 and 3 of the implementation spec, plus the residual signatures."""
import math
import numpy as np
import observe as O
import physics as P

# Section 3.5 error-budget table: (mu, v) -> (A, B, C, O1 noise, O1 bias J=100, O1 bias J=300), in percent
TABLE = {(0.1, 0.8): (1.6, 1.1, 0.8, 6.0, 1.2, 0.4), (0.1, 1.1): (0.7, 0.5, 0.4, 6.0, 0.9, 0.3),
         (0.3, 0.5): (20.0, 15.6, 12.1, 7.6, 19.9, 6.0), (0.3, 0.8): (7.0, 5.2, 3.9, 6.3, 11.8, 3.7),
         (0.3, 1.1): (3.3, 2.5, 1.8, 6.1, 8.4, 2.7), (0.5, 0.5): (36.7, 30.6, 25.6, 9.9, 73.4, 18.2),
         (0.5, 0.8): (19.6, 10.3, 8.0, 6.7, 38.5, 10.8), (0.5, 1.1): (8.0, 4.9, 3.7, 6.2, 26.1, 7.7)}

def test_crlb_reproduces_error_budget():
    """Test 2: every cell within 0.1 percentage point."""
    for row in O.error_budget():
        want = TABLE[(row["mu"], row["v"])]
        got = (row["A"], row["B"], row["C"], row["O1"], row["bias_J100"], row["bias_J300"])
        for g, w in zip(got, want):
            assert abs(g*100 - w) <= 0.1 + 1e-9, (row["mu"], row["v"], got, want)

def test_fit_slide_model_c_unbiased_and_efficient():
    """Test 3: 2,000 synthetic slides at mu = 0.3, v = 0.8: bias under 1%, SD within 20% of the CRLB."""
    rng = np.random.default_rng(1)
    mu, v, fps, sf, sr, sk = 0.3, 0.8, 30, 0.002, 0.0005, 0.0005
    est = np.array([O.fit_slide(O.simulate_obs(0.5, v, mu, fps, sf, sr, sk, rng), "C")["mu"] for _ in range(2000)])
    bound = O.crlb(mu, v, fps, sf, sr, sk)
    assert abs(est.mean()/mu - 1) < 0.01
    assert abs(est.std(ddof=1)/mu/bound - 1) < 0.20

def test_fit_models_a_and_b_run():
    rng = np.random.default_rng(2)
    obs = O.simulate_obs(0.5, 1.0, 0.2, 30, 0.002, 0.0005, 0.0005, rng)
    for m in "ABC":
        f = O.fit_slide(obs, m)
        assert f["success"] and abs(f["mu"]/0.2 - 1) < 0.2
    assert O.fit_slide(obs, "C")["se_mu"] <= O.fit_slide(obs, "A")["se_mu"]

def test_o1_bias_and_estimator():
    assert abs(O.fit_o1(0.8, P.stop_distance(0.8, 0.3)) - 0.3) < 1e-12
    assert O.o1_bias(0.5, 0.5, 100.0) > O.o1_bias(0.1, 0.5, 100.0) > 0

def test_residual_signatures():
    rng = np.random.default_rng(3)
    # clean: constant mu, long slide
    obs = O.simulate_obs(0.5, 1.1, 0.1, 60, 0.002, 0.0005, 0.0005, rng)
    assert O.residual_signature(obs)["label"] == "clean"
    # step: friction strip crossed part way (mu 0.1 -> 0.2).  Detectable at 0.5-1 mm frame noise and 60 Hz;
    # at the budget's 2 mm / 30 Hz a single strip crossing reads as a smooth trend (see the docstring)
    s = P.slide_piecewise(0.5, 1.1, [(0.0, 0.8, 0.1), (0.8, math.inf, 0.2)], dt=1/60)
    t = np.arange(0, s["t_stop"], 1/60)
    x = np.interp(t, s["t"], s["x"]) + 0.0005*rng.standard_normal(len(t))
    sig = O.residual_signature(O.Obs(t, x, s["x_rest"], 0.5, 0.0005, 0.0005, 0.0005))
    assert sig["label"] == "step" and sig["mu_after"] > sig["mu_before"]
    # smooth trend: a spinning disk (spin-to-slide ratio evolves)
    d = P.disk_spin_slide(1.0, v0=1.1, mu=0.1)
    t = np.arange(0, d["t"][-1], 1/60)
    x = np.interp(t, d["t"], d["x"]) + 0.002*rng.standard_normal(len(t))
    assert O.residual_signature(O.Obs(t, x, d["d"], 0.0, 0.002, 0.0005, 0.0005))["label"] == "smooth trend"
    # speed-correlated across slides: mu rising with release speed
    fits = []
    for v in (0.5, 0.7, 0.9, 1.1, 1.3):
        mu_v = 0.2*(1 + 0.4*(v - 0.9))
        fits.append(O.fit_slide(O.simulate_obs(0.5, v, mu_v, 60, 0.001, 0.0005, 0.0005, rng), "C"))
    assert O.speed_signature(fits)["label"] == "speed-correlated"

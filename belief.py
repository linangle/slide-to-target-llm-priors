#!/usr/bin/env python3
"""
belief.py -- the grid posterior over theta = ln mu (model_specification.tex, Sec. 3.5-3.6).

    p(theta | m, z_1:k)  ~  p0(theta) p(m | theta) prod_j max_{x_r, v_r} p(z_j | theta, x_r, v_r)

p0 is the population distribution of measured ln mu (normal, fitted on training folds); p(m | theta) is the
LLM observation model, m = alpha + beta theta + eps with eps ~ N(0, s^2), fitted leave-one-object-out; each
practice slide enters through the model-C physics likelihood with its per-slide nuisance (x_r, v_r)
profiled out by least squares at every grid point.  The physics likelihood is Gaussian in mu while the
prior and LLM model are Gaussian in ln mu, so the grid is exact where a conjugate update is not.  The
transition between practice slides is the identity (the parameters do not change between slides; the
object is reset; Yu et al. saw about 0.15% wear per pass).

Baseline C2 (the published lookup table, friction_lookup.csv) enters the same way as the LLM term: the row's
nominal is one Gaussian observation of theta on top of the population prior, ln mu_tab ~ N(theta, s_tab^2),
with s_tab the row's s_tab_log (ln f / 1.645 for the tier factors f = 2 / 2.5 / 3; 5 Nov decision 3).  A pair
with tier none has no such term (loglik_lookup, lookup_observation).

The grid has 500 points over [ln 0.05, ln 1.0].
"""
import csv, math, os
import numpy as np
from physics import G

LOOKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "friction_lookup.csv")

THETA_MIN, THETA_MAX, N_GRID = math.log(0.05), math.log(1.0), 500

def theta_grid(n=N_GRID, lo=THETA_MIN, hi=THETA_MAX):
    return np.linspace(lo, hi, n)

# ----------------------------------------------------------------------------- prior and LLM observation model

def log_normal_pdf(x, m, s):
    return -0.5*((np.asarray(x, float) - m)/s)**2 - math.log(s) - 0.5*math.log(2*math.pi)

def prior_from_measured(mu_train, s_floor=0.05):
    """Normal in ln mu fitted to the training-fold pairs: returns (m0, s0)."""
    th = np.log(np.asarray(mu_train, float))
    s0 = float(np.std(th, ddof=1)) if len(th) > 1 else s_floor
    return float(np.mean(th)), max(s0, s_floor)

def logprior(grid, m0, s0):
    return log_normal_pdf(grid, m0, s0)

def loglik_lookup(grid, mu_tab, s_tab):
    """log p(lookup | theta) for baseline C2 (5 Nov decision 3): the table's nominal is one Gaussian observation
    of theta on top of the population prior, ln mu_tab ~ N(theta, s_tab^2)."""
    return log_normal_pdf(math.log(mu_tab), grid, s_tab)

def lookup_observation(pair_id, path=LOOKUP_PATH):
    """(mu_tab, s_tab) for one pair of friction_lookup.csv, read by column name (mu_k_nominal, s_tab_log); None
    when the row has tier none, in which case C2 is the prior alone.  A pair_id that is not in the table raises
    KeyError, so a typo or an ID-scheme mismatch cannot quietly turn C2 into C0."""
    with open(path, newline="") as f:
        rows = {row["pair_id"]: row for row in csv.DictReader(f)}
    if pair_id not in rows:
        raise KeyError(f"pair {pair_id!r} is not in {path}")
    row = rows[pair_id]
    return None if row["tier"] == "none" else (float(row["mu_k_nominal"]), float(row["s_tab_log"]))

def fit_llm_likelihood(m_llm, theta_true):
    """Least squares of the LLM's median (in ln mu) on the true ln mu: m = alpha + beta theta + eps.
    Returns dict(alpha, beta, s, n).  beta -> 0 means the LLM carries no information."""
    m, th = np.asarray(m_llm, float), np.asarray(theta_true, float)
    H = np.column_stack([np.ones_like(th), th])
    beta, *_ = np.linalg.lstsq(H, m, rcond=None)
    r = m - H @ beta
    s = math.sqrt(r @ r/max(len(m) - 2, 1))
    return {"alpha": float(beta[0]), "beta": float(beta[1]), "s": s, "n": len(m)}

def fit_llm_likelihood_loo(m_llm, theta_true, objects):
    """Leave-one-object-out: for each object, the (alpha, beta, s) fitted on every pair of the OTHER objects.
    objects: label per pair.  Returns {object: params}."""
    m, th, ob = np.asarray(m_llm, float), np.asarray(theta_true, float), np.asarray(objects)
    return {o: fit_llm_likelihood(m[ob != o], th[ob != o]) for o in np.unique(ob)}

def loglik_llm(grid, m, params):
    """log p(m | theta) for the LLM median m under the fitted observation model."""
    return log_normal_pdf(m, params["alpha"] + params["beta"]*grid, params["s"])

def loglik_gaussian_theta(grid, y, s):
    """log p(y | theta) for a direct Gaussian observation of theta (used to test the grid against conjugacy)."""
    return log_normal_pdf(y, grid, s)

# ----------------------------------------------------------------------------- physics likelihood of a slide

def profile_nuisance(mu, obs, model="C", iters=12):
    """For every mu in the array, the least-squares (x_r, v_r) and the resulting chi^2 (sum of squared
    sigma-normalized residuals), vectorized over mu.  Frames and the FK point are linear in (x_r, v_r) while
    the rest pose is quadratic in v_r, so a few Gauss-Newton steps from the linear solution converge."""
    from observe import MODELS
    mu = np.atleast_1d(np.asarray(mu, float))
    g, parts = obs.g, MODELS[model]
    t = obs.t[:, None]                                    # (n_frames, 1)
    # linear start: frames assumed before the stop, plus the FK point
    rows = [np.column_stack([np.ones(len(obs.t)), obs.t])/obs.sigma_f]
    rhs = [(obs.x[:, None] + 0.5*mu*g*t**2)/obs.sigma_f]  # (n_frames, K)
    if "fk" in parts:
        rows.append(np.array([[1.0, 0.0]])/obs.sigma_fk); rhs.append(np.full((1, len(mu)), obs.x_fk/obs.sigma_fk))
    H = np.vstack(rows)
    beta = np.linalg.lstsq(H, np.vstack(rhs), rcond=None)[0]   # (2, K)
    x_r, v_r = beta[0], np.maximum(beta[1], 1e-3)
    for _ in range(iters):
        T = v_r/(mu*g)
        tau = np.minimum(t, T)                                # (n_frames, K)
        stopped = t >= T
        r = [(obs.x[:, None] - (x_r + v_r*tau - 0.5*mu*g*tau**2))/obs.sigma_f]
        Jx = [np.ones_like(tau)/obs.sigma_f]
        Jv = [np.where(stopped, T, tau)/obs.sigma_f]
        if "rest" in parts:
            r.append((obs.x_rest - (x_r + v_r**2/(2*mu*g)))[None]/obs.sigma_rest)
            Jx.append(np.ones((1, len(mu)))/obs.sigma_rest); Jv.append((v_r/(mu*g))[None]/obs.sigma_rest)
        if "fk" in parts:
            r.append((obs.x_fk - x_r)[None]/obs.sigma_fk)
            Jx.append(np.ones((1, len(mu)))/obs.sigma_fk); Jv.append(np.zeros((1, len(mu))))
        r, Jx, Jv = np.vstack(r), np.vstack(Jx), np.vstack(Jv)
        a, b, c = (Jx*Jx).sum(0), (Jx*Jv).sum(0), (Jv*Jv).sum(0)
        gx, gv = (Jx*r).sum(0), (Jv*r).sum(0)
        det = a*c - b*b
        dx, dv = (c*gx - b*gv)/det, (a*gv - b*gx)/det
        x_r, v_r = x_r + dx, np.maximum(v_r + dv, 1e-3)
        if np.max(np.abs(dx)) < 1e-10 and np.max(np.abs(dv)) < 1e-10:
            break
    chi2 = (r*r).sum(0)
    return x_r, v_r, chi2

def loglik_slide(grid, obs, model="C"):
    """Profile log-likelihood of one slide on the theta grid: -chi^2/2 at the profiled (x_r, v_r)."""
    return -0.5*profile_nuisance(np.exp(grid), obs, model)[2]

# ----------------------------------------------------------------------------- posterior on the grid

def posterior(grid, *logterms):
    """Normalize the sum of log terms on the grid.  Returns dict(grid, logp, pmf, density, mean, var, sd,
    median, q05, q95, mu_median, mu_q05, mu_q95); mean/var/sd are of theta = ln mu."""
    logp = np.sum([np.asarray(l, float) for l in logterms], axis=0)
    logp = logp - logp.max()
    pmf = np.exp(logp); pmf /= pmf.sum()
    dth = grid[1] - grid[0]
    mean = float(pmf @ grid)
    var = float(pmf @ (grid - mean)**2)
    cdf = np.cumsum(pmf) - 0.5*pmf                            # midpoint CDF for interpolated quantiles
    q = lambda u: float(np.interp(u, cdf, grid))
    out = {"grid": grid, "logp": logp, "pmf": pmf, "density": pmf/dth, "mean": mean, "var": var, "sd": math.sqrt(var),
           "median": q(0.5), "q05": q(0.05), "q95": q(0.95)}
    out.update(mu_median=math.exp(out["median"]), mu_q05=math.exp(out["q05"]), mu_q95=math.exp(out["q95"]))
    return out

def update_with_slides(grid, logp0, obs_list, model="C"):
    """Posterior after the given practice slides (the prior and any LLM term already summed in logp0)."""
    return posterior(grid, logp0, *[loglik_slide(grid, o, model) for o in obs_list])

def conjugate_update(m0, s0, ys, s):
    """Closed-form normal update for direct Gaussian observations y_i = theta + N(0, s^2): returns (mean, sd)."""
    ys = np.atleast_1d(np.asarray(ys, float))
    prec = 1/s0**2 + len(ys)/s**2
    return float((m0/s0**2 + ys.sum()/s**2)/prec), float(math.sqrt(1/prec))

def n0(sigma_obs, s0):
    """Worth of the prior in practice slides: n0 = sigma_obs^2 / s0^2 (sigma_obs: SE of theta from one slide)."""
    return (sigma_obs/s0)**2

if __name__ == "__main__":
    from observe import crlb, CAMERA_FPS
    g = theta_grid()
    m0, s0 = math.log(0.3), 0.45
    print(f"prior: ln mu ~ N({m0:.3f}, {s0}^2)")
    for mu, v in ((0.1, 1.1), (0.3, 0.8), (0.5, 0.5)):
        print(f"mu={mu} v={v}: n0 = {n0(crlb(mu, v, CAMERA_FPS, 0.002, 0.0005, 0.0005), s0):.2e} slides")
    for pid in ("V03", "Y02", "M06"):
        print(f"lookup observation for {pid}: {lookup_observation(pid)}")

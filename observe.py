#!/usr/bin/env python3
"""
observe.py -- observation models and estimators for one practice slide (model_specification.tex, Sec. 3.5).

The free slide is x(t) = x_r + v_r t - mu g t^2 / 2 for 0 <= t <= T = v_r/(mu g), released at t = 0 (the
arm's separation time).  Each practice slide gives three measurements:
    frames     x_i = x(t_i) + e_i,  t_i = i / fps while t_i <= T,  e_i ~ N(0, sigma_f^2)   (depth camera + ICP)
    rest pose  x_rest = x_r + v_r^2 / (2 mu g) + e_rest                                      (stationary frames averaged)
    FK point   x_r^FK = x_r + e_FK                                                            (arm forward kinematics)
Model A uses the frames only, B adds the rest pose, C adds the release point.  All three are unbiased; the
frame error scales roughly as sigma_f / (sqrt(N) T^2), so short (high-mu, slow) slides are noisy.

O1 (ablation only) is the endpoint estimator mu = v_cmd^2 / (2 g d) with the COMMANDED speed.  It carries
the release-ramp bias (v_cmd / v_r)^2 - 1 = (v_cmd / (v_cmd - (mu g)^2 / 2J))^2 - 1, which grows with mu.

crlb() is the Cramer-Rao relative standard error of mu for models A, B, C; error_budget() reproduces the
Sec. 3.5 table (test 2).  fit_slide() is the least-squares estimator (test 3).  residual_signature() labels
the frame residuals of a fit: clean, step (friction strip), smooth trend (spin or speed-dependent mu);
speed_signature() looks across slides for mu correlated with release speed (Sec. 3.8).

Run `python3 observe.py` to print the error-budget table and write error_budget_rows.tex.
"""
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from physics import G, release_speed, slide_time, slide_trajectory, stop_distance

MODELS = {"A": ("frames",), "B": ("frames", "rest"), "C": ("frames", "rest", "fk")}

@dataclass
class Obs:
    t: np.ndarray            # frame times from release, s
    x: np.ndarray            # frame positions, m
    x_rest: float            # rest pose, m
    x_fk: float              # release point from forward kinematics, m
    sigma_f: float
    sigma_rest: float
    sigma_fk: float
    g: float = G
    truth: dict = field(default_factory=dict)   # (x_r, v_r, mu) when simulated

def frame_times(v_r, mu, fps, g=G):
    """t_i = i / fps for 0 <= t_i <= T (the frame at release counts; the last one is at or before the stop)."""
    n = int(math.floor(slide_time(v_r, mu, g)*fps + 1e-9)) + 1
    return np.arange(n)/fps

def simulate_obs(x_r, v_r, mu, fps, sigma_f, sigma_rest, sigma_fk, rng, g=G):
    """One slide's measurements with Gaussian noise."""
    t = frame_times(v_r, mu, fps, g)
    x = slide_trajectory(t, x_r, v_r, mu, g) + sigma_f*rng.standard_normal(len(t))
    x_rest = x_r + stop_distance(v_r, mu, g) + sigma_rest*rng.standard_normal()
    x_fk = x_r + sigma_fk*rng.standard_normal()
    return Obs(t, x, float(x_rest), float(x_fk), sigma_f, sigma_rest, sigma_fk, g, {"x_r": x_r, "v_r": v_r, "mu": mu})

# ----------------------------------------------------------------------------- least-squares fit (models A, B, C)

def _predict(p, obs, model):
    """Normalized residuals and their Jacobian for parameters p = (x_r, v_r, mu)."""
    x_r, v_r, mu = p
    g, parts = obs.g, MODELS[model]
    T = v_r/(mu*g)
    tau = np.clip(obs.t, 0.0, T)
    stopped = obs.t >= T
    r = [(obs.x - (x_r + v_r*tau - 0.5*mu*g*tau**2))/obs.sigma_f]
    d_v = np.where(stopped, T, tau)                                  # d x / d v_r
    d_mu = np.where(stopped, -v_r**2/(2*mu*mu*g), -0.5*g*tau**2)     # d x / d mu
    J = [np.column_stack([np.ones_like(tau), d_v, d_mu])/obs.sigma_f]
    if "rest" in parts:
        r.append([(obs.x_rest - (x_r + v_r**2/(2*mu*g)))/obs.sigma_rest])
        J.append(np.array([[1.0, v_r/(mu*g), -v_r**2/(2*mu*mu*g)]])/obs.sigma_rest)
    if "fk" in parts:
        r.append([(obs.x_fk - x_r)/obs.sigma_fk])
        J.append(np.array([[1.0, 0.0, 0.0]])/obs.sigma_fk)
    return np.concatenate(r), -np.vstack(J)

def initial_guess(obs, model="C"):
    """(x_r, v_r, mu) from a quadratic fit to the frames, falling back to the rest pose and FK point."""
    if len(obs.t) >= 3 and np.ptp(obs.t) > 0:
        c2, c1, c0 = np.polyfit(obs.t, obs.x, 2)
        mu = float(np.clip(-2*c2/obs.g, 0.02, 2.0))
        return np.array([c0, max(c1, 0.05), mu])
    x_r = obs.x_fk if "fk" in MODELS[model] else float(obs.x[0])
    d = max(obs.x_rest - x_r, 1e-3) if "rest" in MODELS[model] else max(obs.x[-1] - x_r, 1e-3)
    T = max(obs.t[-1], 1/30.0) if len(obs.t) else 0.2
    v_r = 2*d/T if len(obs.t) > 1 else 0.5
    return np.array([x_r, v_r, float(np.clip(v_r**2/(2*obs.g*d), 0.02, 2.0))])

def fit_slide(obs, model="C", init=None):
    """Least squares over (x_r, v_r, mu) for model A, B or C.  Returns dict(mu, se_mu, x_r, v_r, cov, chi2,
    dof, residuals, n_frames, model).  cov is (J^T J)^-1 of the sigma-normalized residuals."""
    p0 = np.asarray(init if init is not None else initial_guess(obs, model), float)
    res = least_squares(lambda p: _predict(p, obs, model)[0], p0, jac=lambda p: _predict(p, obs, model)[1],
                        bounds=([-np.inf, 1e-3, 1e-3], [np.inf, 10.0, 5.0]), x_scale=[0.01, 0.1, 0.1], xtol=1e-12, ftol=1e-12)
    r, J = _predict(res.x, obs, model)
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    return {"mu": float(res.x[2]), "se_mu": float(math.sqrt(max(cov[2, 2], 0.0))), "x_r": float(res.x[0]),
            "v_r": float(res.x[1]), "cov": cov, "chi2": float(r @ r), "dof": len(r) - 3, "residuals": r,
            "n_frames": len(obs.t), "model": model, "success": bool(res.success)}

# ----------------------------------------------------------------------------- O1 ablation

def fit_o1(v_cmd, d, g=G):
    """Endpoint estimator with the commanded speed: mu = v_cmd^2 / (2 g d).  Biased by the release ramp."""
    return np.asarray(v_cmd, float)**2/(2*g*np.asarray(d, float))

def o1_noise(mu, v, speed_err=0.03, endpoint_err=0.002, g=G):
    """Relative standard error of O1 from a relative speed error and an absolute endpoint error."""
    return math.sqrt((2*speed_err)**2 + (endpoint_err/stop_distance(v, mu, g))**2)

def o1_bias(mu, v_peak, J, g=G):
    """Relative bias of O1 when the paddle brakes at jerk J: (v_peak / v_r)^2 - 1."""
    return (v_peak/release_speed(v_peak, mu, J, g))**2 - 1

# ----------------------------------------------------------------------------- Cramer-Rao bound and error budget

def fisher(mu, v, fps, sigma_f, sigma_rest=None, sigma_fk=None, g=G):
    """Fisher information of (x_r, v_r, mu) for a slide released at speed v on friction mu."""
    t = frame_times(v, mu, fps, g)
    H = np.column_stack([np.ones_like(t), t, -0.5*g*t**2])
    F = H.T @ H/sigma_f**2
    if sigma_rest:
        h = np.array([1.0, v/(mu*g), -v**2/(2*mu*mu*g)])
        F += np.outer(h, h)/sigma_rest**2
    if sigma_fk:
        F[0, 0] += 1/sigma_fk**2
    return F

def crlb(mu, v, fps, sigma_f, sigma_rest=None, sigma_fk=None, g=G):
    """Cramer-Rao relative standard error of mu from one slide (model A: frames only; B: + rest pose;
    C: + FK release point, chosen by which sigmas are given)."""
    return math.sqrt(np.linalg.inv(fisher(mu, v, fps, sigma_f, sigma_rest, sigma_fk, g))[2, 2])/mu

BUDGET_CASES = ((0.1, 0.8), (0.1, 1.1), (0.3, 0.5), (0.3, 0.8), (0.3, 1.1), (0.5, 0.5), (0.5, 0.8), (0.5, 1.1))

def error_budget(cases=BUDGET_CASES, fps=30, sigma_f=0.002, sigma_rest=0.0005, sigma_fk=0.0005,
                 speed_err=0.03, endpoint_err=0.002, Js=(100.0, 300.0), g=G):
    """One row per (mu, v): slide time and length, relative SE of models A, B, C, O1's noise and its bias at
    each braking jerk.  Reproduces the Sec. 3.5 table with the defaults."""
    rows = []
    for mu, v in cases:
        row = {"mu": mu, "v": v, "T": float(slide_time(v, mu, g)), "d": float(stop_distance(v, mu, g)),
               "A": crlb(mu, v, fps, sigma_f, g=g), "B": crlb(mu, v, fps, sigma_f, sigma_rest, g=g),
               "C": crlb(mu, v, fps, sigma_f, sigma_rest, sigma_fk, g), "O1": o1_noise(mu, v, speed_err, endpoint_err, g)}
        for J in Js:
            row[f"bias_J{J:g}"] = o1_bias(mu, v, J, g)
        rows.append(row)
    return rows

def budget_table(rows, tex=False):
    """The budget as text or LaTeX rows."""
    Js = [k for k in rows[0] if k.startswith("bias_J")]
    out = []
    for r in rows:
        cells = [f"{r['mu']:.1f}", f"{r['v']:.1f}", f"{r['T']:.2f} s, {r['d']*100:.1f} cm"] + \
                [f"{r[k]*100:.1f}\\%" if tex else f"{r[k]*100:.1f}%" for k in ("A", "B", "C", "O1")] + \
                [f"{r[k]*100:+.1f}\\%" if tex else f"{r[k]*100:+.1f}%" for k in Js]
        out.append(" & ".join(cells) + " \\\\" if tex else "  ".join(f"{c:>14}" for c in cells))
    return "\n".join(out)

# ----------------------------------------------------------------------------- residual signatures (Sec. 3.8)

def _lstsq(H, y):
    beta, *_ = np.linalg.lstsq(H, y, rcond=None)
    r = y - H @ beta
    return beta, float(r @ r)

def residual_signature(obs, fit=None, z=None, strong=6.0):
    """Label the structure of the frame residuals (frames before the fitted stop only).  Closed-form linear
    alternatives to the constant-mu parabola, compared by BIC = chi^2 + k ln N:
      const   x_r, v_r, mu                                   (k = 3)
      smooth  deceleration linear or quadratic in time        (k = 4, 5): spin, or a speed-dependent mu, which
              in a single slide look alike (see speed_signature for the cross-slide test)
      step    deceleration changes once at a frame boundary, x and v continuous: mu_1 before, mu_2 after,
              best change point over the interior frames      (k = 4 + the change point = 5)
    'step' needs BIC lower than both others by `strong` (Kass-Raftery strong evidence), 'smooth trend' needs
    BIC lower than const by `strong`; otherwise 'clean' if the constant fit's reduced chi^2 is below 2.5,
    else 'unexplained'.  Returns dict(label, bic, chi2, dof, mu_fit, t_step, mu_before, mu_after, ...).
    Detectability (synthetic, 1.1 m/s slide, mu 0.1 -> 0.2 part way): the step is labelled on 20 of 20 trials at
    0.5 mm frame noise and 60 Hz, 18 of 20 at 1 mm, but at the budget's 2 mm and 30 Hz it reads as 'smooth
    trend' on 20 of 20.  Telling a strip from spin in ONE slide needs sub-millimetre frames or 60 Hz or more."""
    t_all, g = obs.t, obs.g
    out = {"n_frames": len(t_all)}
    if len(t_all) < 5:
        out.update(label="too few frames", chi2=float("nan"), dof=len(t_all) - 3)
        return out
    y_all = obs.x/obs.sigma_f
    base = np.column_stack([np.ones(len(t_all)), t_all, -0.5*g*t_all**2])/obs.sigma_f
    b0, _ = _lstsq(base, y_all)
    T_hat = b0[1]/(max(b0[2], 1e-3)*g)                         # fitted stop time
    keep = t_all <= T_hat*1.02 + 1e-9
    t, y, N = t_all[keep], y_all[keep], int(keep.sum())
    out["n_used"] = N
    if N < 5:
        out.update(label="too few frames", chi2=float("nan"), dof=N - 3)
        return out
    base = np.column_stack([np.ones(N), t, -0.5*g*t**2])/obs.sigma_f
    b0, chi2_0 = _lstsq(base, y)
    chi2 = {"const": chi2_0}
    chi2["cubic"] = _lstsq(np.column_stack([base, -g*t**3/6/obs.sigma_f]), y)[1]
    chi2["quartic"] = _lstsq(np.column_stack([base, -g*t**3/6/obs.sigma_f, -g*t**4/12/obs.sigma_f]), y)[1]
    best = (np.inf, None, None)
    for k in range(2, N - 1):
        tk = t[k]
        after = t >= tk
        col1 = np.where(after, tk*tk + 2*tk*(t - tk), t*t)*(-0.5*g)
        col2 = np.where(after, (t - tk)**2, 0.0)*(-0.5*g)
        beta, c = _lstsq(np.column_stack([np.ones(N), t, col1, col2])/obs.sigma_f, y)
        if c < best[0]:
            best = (c, float(tk), beta)
    chi2["step"] = best[0]
    k_par = {"const": 3, "cubic": 4, "quartic": 5, "step": 5}
    bic = {m: chi2[m] + k_par[m]*math.log(N) for m in chi2}
    out.update(chi2=chi2_0, dof=N - 3, mu_fit=float(b0[2]), bic=bic, t_step=best[1],
               dchi2_step=chi2_0 - chi2["step"], dchi2_trend=chi2_0 - min(chi2["cubic"], chi2["quartic"]))
    if best[2] is not None:
        out["mu_before"], out["mu_after"] = float(best[2][2]), float(best[2][3])
    smooth = min(bic["cubic"], bic["quartic"])
    if bic["step"] + strong < min(bic["const"], smooth):
        out["label"] = "step"
    elif smooth + strong < bic["const"]:
        out["label"] = "smooth trend"
    elif chi2_0/(N - 3) < 2.5:
        out["label"] = "clean"
    else:
        out["label"] = "unexplained"
    return out

def speed_signature(fits, z=3.0):
    """Across slides: weighted regression of the fitted mu on the fitted release speed.  'speed-correlated'
    if the slope differs from zero by more than z standard errors.  fits: list of fit_slide() results."""
    v = np.array([f["v_r"] for f in fits]); mu = np.array([f["mu"] for f in fits])
    w = 1/np.array([max(f["se_mu"], 1e-9) for f in fits])**2
    if len(fits) < 3 or np.ptp(v) < 1e-6:
        return {"label": "too few slides", "slope": float("nan"), "se_slope": float("nan")}
    H = np.column_stack([np.ones_like(v), v])*np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(H, mu*np.sqrt(w), rcond=None)
    cov = np.linalg.inv(H.T @ H)
    se = math.sqrt(cov[1, 1])
    return {"label": "speed-correlated" if abs(beta[1]) > z*se else "no speed dependence",
            "slope": float(beta[1]), "se_slope": se, "mu_at_mean_v": float(beta[0] + beta[1]*v.mean())}

if __name__ == "__main__":
    rows = error_budget()
    print(f"{'mu':>14}{'v_r':>16}{'slide':>16}{'A: frames':>16}{'B: +rest':>16}{'C: +FK':>16}{'O1 noise':>16}"
          f"{'O1 bias J=100':>16}{'O1 bias J=300':>16}")
    print(budget_table(rows))
    with open("error_budget_rows.tex", "w") as f:
        f.write("% generated by observe.py: relative standard error of mu from one slide (30 Hz, 2 mm frames; 0.5 mm rest and FK)\n")
        f.write(budget_table(rows, tex=True) + "\n")
    print("wrote error_budget_rows.tex")

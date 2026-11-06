#!/usr/bin/env python3
"""
decide.py -- the release-speed decision (model_specification.tex, Sec. 3.7).

Because the release speed depends on mu (physics.release_speed), the closed-form optimum holds only for
abrupt braking.  The commanded peak speed u is chosen by a bounded one-dimensional search on the expected
loss under the grid posterior, with the release model inside the distance:
    u* = argmin_u  int L(d(mu, u) - d*) p(mu | data) dmu,     d(mu, u) = v_r(mu, u)^2 / (2 mu g).
The old closed form, v_opt^2 = 2 g d* exp(m - 1.5 S^2), is kept as the J -> inf limit with squared
relative loss (test 5).  Losses: squared relative error, absolute relative error, a miss outside
max(0.1 d*, 1 cm), and the table-edge penalty beyond L = 1.3 d*.

Practice slides run at the maximum feasible speed: the standard error of mu falls roughly as v^-2.5, so
the information grows roughly as v^5 (valid while mu does not depend on speed, which the rig's mu(v) data
check).  practice_speed() is the fastest speed that keeps the slide on the board for the slipperiest
plausible mu; slide_room() is that length: from the release point to the far edge of the 48 in board
(physics.BOARD_X_MAX, 5 Nov decision 1), less the object's half length.
"""
import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from physics import BOARD_X_MAX, G, release_speed

@dataclass
class Release:
    J: float = 100.0      # braking jerk, m/s^3; math.inf = abrupt braking (v_r = u)
    g: float = G

def distance(u, mu, rel):
    """Landing distance for commanded peak speed u on friction mu under the release model."""
    v_r = release_speed(u, mu, rel.J, rel.g)
    return v_r*v_r/(2*np.asarray(mu, float)*rel.g)

# ----------------------------------------------------------------------------- losses L(d, d*)

def loss_sq_rel(d, d_star):
    return ((d - d_star)/d_star)**2

def loss_abs_rel(d, d_star):
    return np.abs(d - d_star)/d_star

def loss_miss(d, d_star, tol=None):
    """1 outside the success window max(0.1 d*, 1 cm), else 0."""
    tol = max(0.1*d_star, 0.01) if tol is None else tol
    return (np.abs(d - d_star) > tol).astype(float)

def loss_table_edge(d, d_star, L=None, penalty=10.0):
    """Squared relative error plus a penalty when the object would leave the table at L = 1.3 d*."""
    L = 1.3*d_star if L is None else L
    return loss_sq_rel(d, d_star) + penalty*(d > L)

LOSSES = {"sq_rel": loss_sq_rel, "abs_rel": loss_abs_rel, "miss": loss_miss, "edge": loss_table_edge}

# ----------------------------------------------------------------------------- expected loss and its minimizer

def expected_loss(u, post, rel, d_star, loss=loss_sq_rel):
    """E[L] over the grid posterior post (from belief.posterior) for commanded speed u (scalar or array)."""
    mu = np.exp(post["grid"])
    u = np.asarray(u, float)
    d = distance(u[..., None], mu, rel) if u.ndim else distance(u, mu, rel)
    return (loss(d, d_star)*post["pmf"]).sum(axis=-1)

def choose_speed(post, rel, d_star, loss=loss_sq_rel, u_max=1.5, u_min=0.0, n_coarse=600):
    """Bounded 1-D search: a coarse grid over [u_min, u_max], then a golden-section refinement in the best
    bracket.  Returns dict(u, expected_loss, clipped)."""
    us = np.linspace(u_min, u_max, n_coarse)
    el = expected_loss(us, post, rel, d_star, loss)
    k = int(np.argmin(el))
    lo, hi = us[max(k - 1, 0)], us[min(k + 1, n_coarse - 1)]
    if hi > lo:
        res = minimize_scalar(lambda u: float(expected_loss(u, post, rel, d_star, loss)), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-7})
        u, e = (float(res.x), float(res.fun)) if res.fun <= el[k] else (float(us[k]), float(el[k]))
    else:
        u, e = float(us[k]), float(el[k])
    return {"u": u, "expected_loss": e, "clipped": bool(k == n_coarse - 1)}

def closed_form_speed(d_star, m, S, g=G):
    """v_opt = sqrt(2 g d* exp(m - 1.5 S^2)): the squared-relative-loss optimum for ln mu ~ N(m, S^2) and
    abrupt braking."""
    return math.sqrt(2*g*d_star*math.exp(m - 1.5*S*S))

def practice_speed(u_max, length_limit, post, rel, quantile=0.05):
    """Fastest commanded speed, at most u_max, whose slide stays within length_limit for the slipperiest
    plausible friction (the posterior's lower quantile)."""
    mu_low = math.exp(float(np.interp(quantile, np.cumsum(post["pmf"]) - 0.5*post["pmf"], post["grid"])))
    f = lambda u: float(distance(u, mu_low, rel)) - length_limit
    if f(u_max) <= 0:
        return u_max
    return brentq(f, 0.0, u_max)

def slide_room(x_release, half_length=0.05, x_end=BOARD_X_MAX):
    """Free slide length from the release point (object centre at x_release) to the board's far edge, less
    the object's half length so the footprint stays on the board: practice_speed's length_limit."""
    return x_end - half_length - x_release

if __name__ == "__main__":
    from belief import logprior, posterior, theta_grid
    g = theta_grid()
    m, S = math.log(0.3), 0.3
    post = posterior(g, logprior(g, m, S))
    for J in (math.inf, 300.0, 100.0):
        r = choose_speed(post, Release(J=J), 0.3, loss_sq_rel, u_max=2.0)
        print(f"d*=0.3 m, ln mu ~ N({m:.2f}, {S}^2), J={J}: u*={r['u']:.4f} m/s, E[L]={r['expected_loss']:.4f}"
              + (f"   closed form {closed_form_speed(0.3, m, S):.4f}" if math.isinf(J) else ""))
    room = slide_room(0.75)
    print(f"practice speed (release at x = 0.75 m on the 48 in board: {room:.3f} m of room, u_max 1.2): "
          f"{practice_speed(1.2, room, post, Release()):.3f} m/s")

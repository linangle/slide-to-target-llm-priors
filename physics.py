#!/usr/bin/env python3
"""
physics.py -- the analytic push-and-slide model (model_specification.tex, Sec. 3.4).

Push map (release model).  The object rides on the paddle while the paddle decelerates more slowly
than friction would decelerate the free object.  Separation is at
    t* = min{ t > t_peak : a_p(t) < -mu g },        v_r = v_p(t*).
For braking that starts at zero deceleration and ramps at constant jerk J,
    v_r = v_peak - (mu g)^2 / (2 J),
so the release speed depends on mu: at J = 100 m/s^3 and v_peak = 0.8 m/s this costs 11% of the
distance at mu = 0.3 and about 1% at mu = 0.1.

Free slide with uniform friction, released at tau = 0:
    x(tau) = x_r + v_r tau - mu g tau^2 / 2,   0 <= tau <= T = v_r / (mu g),
    x_rest = x_r + v_r^2 / (2 mu g).
Across friction strips the deceleration is mu(x) g, integrated piecewise until the speed is zero.

Spin (diagnostic only).  For a base with symmetric pressure, spin LENGTHENS the slide: sliding and
spinning share the Coulomb friction (Farkas et al., arXiv physics/0210024).  A uniform-pressure disk
gives d/d0 ~ 1 + 0.26 (R omega / v)^2 for R omega / v <= 1 (1.065 at 0.5, 1.27 at 1);
disk_spin_slide() integrates the coupled equations.  A heading error psi shortens the along-axis
distance only by psi^2/2 and adds a lateral miss of psi d.

Tipping (A5): none during the push if (a + mu g) h_p <= g b + a h_c, none during the slide if
mu h_c < b, with b the half-length along the push, h_c the COM height, h_p the paddle contact height.
"""
import math
import numpy as np

G = 9.81   # m/s^2; scene.py uses the same value

# Rig geometry (5 Nov decision 1): the object slides on a 48 in board.  Along the push (+x) it runs from
# BOARD_X_MIN, ahead of the iiwa base at the origin, to BOARD_X_MAX.  scene.Table tiles it with strips and
# decide.slide_room measures the free length from the release point, which moves with the push speed
# (release_push_distance).  The far edge is where the pre-decision
# 1.8 m table ended, so the E0 runs (rest points up to x = 1.35 m) stay on the board.
BOARD_LENGTH = 48*0.0254                 # m, 1.2192
BOARD_X_MIN = 0.3                        # m; the object starts at x0 >= 0.55 (paddle reach), 0.2 m in from the near edge
BOARD_X_MAX = BOARD_X_MIN + BOARD_LENGTH  # m, 1.5192

# ----------------------------------------------------------------------------- free slide

def stop_distance(v, mu, g=G):
    """d = v^2 / (2 mu g)."""
    v, mu = np.asarray(v, float), np.asarray(mu, float)
    return v*v/(2*mu*g)

def slide_time(v_r, mu, g=G):
    """T = v_r / (mu g)."""
    return np.asarray(v_r, float)/(np.asarray(mu, float)*g)

def slide_trajectory(t, x_r, v_r, mu, g=G):
    """x(t) of a free slide released at t = 0, held at the rest pose after the stop (and at x_r before t = 0)."""
    tau = np.clip(np.asarray(t, float), 0.0, slide_time(v_r, mu, g))
    return x_r + v_r*tau - 0.5*mu*g*tau**2

def slide_speed(t, v_r, mu, g=G):
    """v(t) of a free slide released at t = 0; zero after the stop."""
    tau = np.clip(np.asarray(t, float), 0.0, slide_time(v_r, mu, g))
    return v_r - mu*g*tau

def slide_piecewise(x_r, v_r, strips, g=G, dt=1e-3):
    """Free slide across friction strips, each (x_start, x_end, mu), in order and covering the path (the
    last x_end may be inf).  Within a strip the deceleration is mu g, so v^2 drops by 2 mu g per metre.
    Returns dict(t, x, v, x_rest, t_stop, segments) with the trajectory sampled every dt and the exact stop."""
    strips = [(float(a), float(b), float(m)) for a, b, m in strips]
    x, v, t = float(x_r), float(v_r), 0.0
    i = next((k for k, (a, b, _) in enumerate(strips) if a <= x < b), None)
    if i is None:
        raise ValueError(f"release point {x_r} is not inside any strip")
    segments = []   # (t0, x0, v0, mu) at the entry of each strip
    while True:
        a, b, mu = strips[i]
        segments.append((t, x, v, mu))
        v_end_sq = v*v - 2*mu*g*(b - x)
        if v_end_sq <= 0 or i == len(strips) - 1:
            t_stop, x_rest = t + v/(mu*g), x + v*v/(2*mu*g)
            break
        v_end = math.sqrt(v_end_sq)
        t, x, v = t + (v - v_end)/(mu*g), b, v_end
        i += 1
    ts = np.arange(0.0, t_stop + dt, dt)
    xs, vs = np.empty_like(ts), np.empty_like(ts)
    starts = np.array([s[0] for s in segments])
    for k, tk in enumerate(ts):
        t0, x0, v0, mu = segments[min(int(np.searchsorted(starts, tk, side="right")) - 1, len(segments) - 1)]
        tau = min(tk - t0, v0/(mu*g))
        xs[k], vs[k] = x0 + v0*tau - 0.5*mu*g*tau**2, max(v0 - mu*g*tau, 0.0)
    return {"t": ts, "x": xs, "v": vs, "x_rest": x_rest, "t_stop": t_stop, "segments": segments}

def strips_from_edges(x_edges, mus):
    """(x_start, x_end, mu) list from strip boundaries (len = len(mus) + 1), as in scene.Scene.x_edges."""
    return [(x_edges[i], x_edges[i + 1], mus[i]) for i in range(len(mus))]

def slide_field(x_r, v_r, mu_of_x, g=G, dx=1e-4, x_max=None):
    """Free slide through any friction field mu(x) (callable, vectorized): the energy balance
    v(x)^2 = v_r^2 - 2 g int_{x_r}^{x} mu dx', with time from dt = dx / v.  Returns dict(t, x, v, x_rest)."""
    x_max = x_max if x_max is not None else x_r + 1.5*v_r**2/(2*g*min(np.min(mu_of_x(np.linspace(x_r, x_r + 3.0, 301))), 1e-3))
    xs = np.arange(x_r, x_max, dx)
    mu = mu_of_x(xs)
    work = np.concatenate([[0.0], np.cumsum(0.5*(mu[1:] + mu[:-1])*np.diff(xs))])
    v_sq = v_r**2 - 2*g*work
    n = int(np.argmax(v_sq <= 0)) if np.any(v_sq <= 0) else len(xs)
    if n == 0:
        raise ValueError("mu field too high or x_max too short")
    x_rest = float(np.interp(0.0, -v_sq[max(n - 1, 0):n + 1], xs[max(n - 1, 0):n + 1])) if n < len(xs) else float(xs[-1])
    xs, v = xs[:n], np.sqrt(np.maximum(v_sq[:n], 0.0))
    t = np.concatenate([[0.0], np.cumsum(np.diff(xs)/(0.5*(v[1:] + v[:-1])))])
    return {"t": t, "x": xs, "v": v, "x_rest": x_rest}

# ----------------------------------------------------------------------------- push map (release model)

def release_speed(v_peak, mu, J, g=G):
    """v_r = v_peak - (mu g)^2 / (2 J) for constant-jerk braking; J = inf is abrupt braking (v_r = v_peak).
    Clipped at zero: if the ramp never out-brakes friction, the object stays on the paddle and does not slide."""
    v_peak, mu, J = np.asarray(v_peak, float), np.asarray(mu, float), np.asarray(J, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(np.isinf(J), 0.0, (mu*g)**2/(2*J))
    return np.maximum(v_peak - loss, 0.0)

A_PUSH = 2.0   # m/s^2: the push's acceleration before the brake (push.profile, constant_jerk_profile)

def release_push_distance(v_peak, mu, J, a_push=A_PUSH, g=G):
    """Travel of the paddle (and the object riding on it) from rest to the release, under push.profile's command:
    accelerate at a_push (capped at sqrt(2 J v_peak)), then one constant-jerk segment -J through the speed peak
    into the brake.  The object leaves when the deceleration reaches mu g, (a_push + mu g) / J into that segment,
    or stays on until the paddle stops if that comes first (release_speed = 0).  J = inf releases at the peak,
    after v_peak^2 / (2 a_push).  Scalars; the release point is the start plus this distance."""
    if v_peak <= 0:
        return 0.0
    if math.isinf(J):
        return v_peak*v_peak/(2*a_push)
    a = min(a_push, math.sqrt(2*J*v_peak))
    v1 = max(v_peak - a*a/(2*J), 0.0)                     # speed when the jerk segment starts
    tau_stop = (a + math.sqrt(a*a + 2*J*v1))/J            # v1 + a tau - J tau^2 / 2 = 0
    tau = min((a + mu*g)/J, tau_stop)
    return v1*v1/(2*a) + v1*tau + 0.5*a*tau**2 - J*tau**3/6

def release_from_profile(t, v_p, a_p, mu, g=G):
    """Separation (t*, v_r) from any sampled paddle profile: the first time after the speed peak at which
    a_p < -mu g.  The crossing is located by linear interpolation of a_p between samples and v_r by the
    trapezoid rule from the sample before it, which is exact for a constant-jerk ramp.
    Returns None if the paddle never brakes harder than friction."""
    t, v_p, a_p = (np.asarray(z, float) for z in (t, v_p, a_p))
    ip = int(np.argmax(v_p))
    thr = -mu*g
    below = np.nonzero(a_p[ip:] < thr)[0]
    if len(below) == 0:
        return None
    i = ip + int(below[0])
    if i == 0:
        return float(t[0]), float(v_p[0])
    f = (a_p[i - 1] - thr)/(a_p[i - 1] - a_p[i])
    ts = t[i - 1] + f*(t[i] - t[i - 1])
    vs = v_p[i - 1] + 0.5*(a_p[i - 1] + thr)*(ts - t[i - 1])
    return float(ts), float(vs)

def constant_jerk_profile(v_peak, J, a_push=2.0, a_max=math.inf, dt=1e-4):
    """Synthetic paddle profile: accelerate at a_push to v_peak, then brake with deceleration ramping at
    jerk J (capped at a_max) until the paddle stops.  Returns (t, v_p, a_p)."""
    t_acc = v_peak/a_push
    t_list, v_list, a_list = [0.0], [0.0], [a_push]
    t, v, a = 0.0, 0.0, a_push
    while v < v_peak - 1e-12:
        t = min(t + dt, t_acc); v = a_push*t
        t_list.append(t); v_list.append(v); a_list.append(a_push)
    t_peak, tau = t, 0.0
    while v > 0:
        tau += dt
        a = -min(J*tau, a_max)
        v = max(v_peak - (0.5*J*tau**2 if J*tau <= a_max else 0.5*a_max**2/J + a_max*(tau - a_max/J)), 0.0)
        t_list.append(t_peak + tau); v_list.append(v); a_list.append(a if v > 0 else 0.0)
    return np.array(t_list), np.array(v_list), np.array(a_list)

# ----------------------------------------------------------------------------- spin and heading (diagnostics)

def spin_factor(R, omega, v):
    """d/d0 ~ 1 + 0.26 (R omega / v)^2 for a uniform-pressure disk (valid for R omega / v <= 1)."""
    return 1.0 + 0.26*(R*np.asarray(omega, float)/np.asarray(v, float))**2

def heading_effect(psi, d):
    """(along-axis distance, lateral miss) of a slide of length d launched psi rad off the push axis."""
    return d*np.cos(psi), d*np.sin(psi)

def disk_friction(v, omega, R, n=120):
    """Net Coulomb friction on a uniform-pressure disk of radius R translating at v along +x and spinning at
    omega, per unit mu N: (Fx, Fy, Mz) with Mz in metres (torque per unit mu N).  Midpoint rule in r, theta."""
    r = (np.arange(n) + 0.5)/n*R
    th = (np.arange(2*n) + 0.5)/(2*n)*2*math.pi
    rr, tt = np.meshgrid(r, th, indexing="ij")
    x, y = rr*np.cos(tt), rr*np.sin(tt)
    vx, vy = v - omega*y, omega*x
    s = np.hypot(vx, vy) + 1e-15
    dA = rr*(R/n)*(math.pi/n)/(math.pi*R*R)          # uniform pressure: weight = area fraction
    return -np.sum(vx/s*dA), -np.sum(vy/s*dA), -np.sum((x*vy - y*vx)/s*dA)

def disk_spin_slide(eps, v0=1.0, R=0.05, mu=0.3, g=G, m=1.0, n=120):
    """Uniform-pressure disk launched with R omega / v = eps: integrates m dv/dt = mu m g Fx and
    I domega/dt = mu m g Mz (I = m R^2 / 2) to the stop.  Returns dict(t, x, v, omega, d, d0, ratio)."""
    from scipy.integrate import solve_ivp
    I = 0.5*m*R*R
    def rhs(_, y):
        _, v, w = y
        Fx, _, Mz = disk_friction(v, w, R, n)
        return [v, mu*g*Fx, mu*m*g*Mz/I]
    def stopped(_, y):
        return y[1] - 1e-4*v0
    stopped.terminal = True
    sol = solve_ivp(rhs, [0.0, 10*v0/(mu*g)], [0.0, v0, eps*v0/R], events=stopped, max_step=0.002*v0/(mu*g), rtol=1e-7)
    d, d0 = float(sol.y[0, -1]), v0**2/(2*mu*g)
    return {"t": sol.t, "x": sol.y[0], "v": sol.y[1], "omega": sol.y[2], "d": d, "d0": d0, "ratio": d/d0}

def disk_spin_ratio(eps, **kw):
    """d/d0 for a uniform-pressure disk with R omega / v = eps at launch (dimensionless: independent of R, mu, v0)."""
    return disk_spin_slide(eps, **kw)["ratio"]

# ----------------------------------------------------------------------------- tipping (A5)

def no_tip(a, mu, h_p, h_c, b, g=G):
    """A5 checks.  Push: (a + mu g) h_p <= g b + a h_c.  Slide: mu h_c < b.  Returns (push_ok, slide_ok)."""
    return bool((a + mu*g)*h_p <= g*b + a*h_c), bool(mu*h_c < b)

def max_push_acceleration(mu, h_p, h_c, b, g=G):
    """Largest push acceleration that satisfies the push inequality (inf if h_p <= h_c)."""
    if h_p <= h_c:
        return math.inf
    return (g*b - mu*g*h_p)/(h_p - h_c)

if __name__ == "__main__":
    for mu in (0.1, 0.3, 0.5):
        v = release_speed(0.8, mu, 100.0)
        print(f"mu={mu}: v_r={v:.4f} m/s at J=100, distance loss {1 - stop_distance(v, mu)/stop_distance(0.8, mu):.1%}")
    for eps in (0.5, 1.0):
        print(f"disk R omega/v = {eps}: d/d0 = {disk_spin_ratio(eps):.4f} (quadratic {spin_factor(1, eps, 1):.3f})")

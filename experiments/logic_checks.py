#!/usr/bin/env python3
"""
logic_checks.py -- three checks of the model's logic that the unit tests do not make (25 Nov).

A. Release bound.  Which friction does a push of peak speed u release?  The commanded brake (push.profile) ramps at jerk J
   from zero deceleration at the speed peak until the paddle stops, so its deceleration peaks at
       min(a_brake, sqrt(2 J u)),
   and the release model (physics.release_speed) says the object leaves when that exceeds mu g, i.e. v_r > 0.  This part
   prints the profile's peak deceleration and then scans friction in Drake, at each speed, for the largest value that
   still passes run_arm_evaluation.released() (contact force zero before the commanded stop, apart at the end).
   Drake's bound comes out below the profile's: close to the bound the object does leave, at the model's speed, but the gap
   it opens before the paddle stops is J t^3 / 6, tens of micrometres, and the paddle's settling after the commanded stop
   closes it again.

B. Calibration of the belief.  Friction is drawn from a prior, one practice slide is simulated from observation model C,
   the grid posterior is computed with the same prior, and the truth's position in the posterior is recorded.  If the
   likelihood, the profiling of the nuisance (x_r, v_r) and the grid are consistent, the 90% interval holds the truth 90%
   of the time and the posterior CDF at the truth is uniform.  Run where the slide dominates (fast, nominal noise) and
   where the prior matters (slow, 8x noise), with the profile likelihood the evaluations use and with the marginal one (a
   flat prior on the nuisance), which differ by a factor that tilts the belief by about its variance in ln mu.

C. Grid resolution at the camera's measured noise.  evaluation/practice_study.py checked the 2,000-point grid down to 0.5x
   the nominal noise.  The rendered camera of E0-5 is about 20x better than nominal; this compares the planned speed and
   the landing it implies on 2,000 and 32,000 points at 0.05x.

    python3 experiments/logic_checks.py            # ~2 min -> results/logic_checks/summary.txt, release_bound.csv
    python3 experiments/logic_checks.py --part B   # one part only (B and C need no Drake)

No file that the evaluations hash is edited or patched.
"""
import argparse, csv, math
from multiprocessing import Pool
import numpy as np
from paths import RESULTS
from physics import G, release_speed, release_push_distance, stop_distance

J = 100.0
SPEEDS = (0.20, 0.26, 0.33, 0.45, 0.60)

# ----------------------------------------------------------------------------- A: release bound

def _released(job):
    kind, mu, u = job
    from push import push_slide
    pusher = None
    if kind != "plate":
        from arm import ArmPusher
        pusher = ArmPusher(a_brake_max=float(kind.split("_")[1]), **({"posture": "limits"} if kind.endswith("6") else {}))
    r = push_slide(mu, u, J, x0=0.55, pusher=pusher, keep_traj=False)
    return dict(pusher=kind, mu=mu, u=u, released=bool(r["separated"]) and float(r["t_force_zero"]) < float(r["t_stop"]),
                lead_ms=round(1000*(float(r["t_stop"]) - float(r["t_force_zero"])), 1), v_at_force_zero=round(float(r["v_sep_force"]), 4),
                v_r_model=round(float(release_speed(u, mu, J)), 4))

def part_a(out, procs):
    from push import profile
    lines = ["A. RELEASE BOUND (J = 100 m/s^3)", "",
             "peak deceleration of the commanded profile, m/s^2 (push.profile; sqrt(2 J u) uncapped):",
             "   u m/s   uncapped   cap 12   cap 6   sqrt(2 J u)   sqrt(J u)"]
    for u in (0.18, 0.20, 0.26, 0.33, 0.36, 0.45, 0.60, 0.72, 1.00):
        pk = [float((-profile(u, J, a_max=c)["a"]).max()) for c in (math.inf, 12.0, 6.0)]
        lines.append(f"   {u:5.2f}   {pk[0]:8.2f}   {pk[1]:6.2f}   {pk[2]:5.2f}   {math.sqrt(2*J*u):11.2f}   {math.sqrt(J*u):9.2f}")
    lines += ["   -> the 12 m/s^2 cap is reached from u = 12^2 / (2 J) = 0.72 m/s, the 6 m/s^2 cap from 0.18 m/s.", ""]
    jobs = []
    for u in SPEEDS:
        top = math.sqrt(2*J*u)/G
        jobs += [("plate", round(mu, 3), u) for mu in np.arange(0.30, top + 0.011, 0.01)]
    for kind, speeds in (("arm_12", (0.26, 0.33, 0.45)), ("arm_6", (0.26, 0.33, 0.45))):
        for u in speeds:
            jobs += [(kind, round(mu, 3), u) for mu in np.arange(0.44, min(math.sqrt(2*J*u)/G, 0.86) + 0.011, 0.02)]
    with Pool(procs) as pool:
        rows = pool.map(_released, jobs, chunksize=4)
    with (out/"release_bound.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    lines += ["largest friction released in Drake (scan in steps of 0.01 for the plate, 0.02 for the arm):",
              "   pusher   u m/s   every mu up to   first not released   any release above it   model: sqrt(2Ju)/g (cap/g)   sqrt(Ju)/g"]
    for kind in ("plate", "arm_12", "arm_6"):
        for u in sorted({r["u"] for r in rows if r["pusher"] == kind}):
            rs = sorted((r for r in rows if r["pusher"] == kind and r["u"] == u), key=lambda r: r["mu"])
            fail = next((r["mu"] for r in rs if not r["released"]), None)
            ok = max((r["mu"] for r in rs if fail is None or r["mu"] < fail), default=float("nan"))
            above = [r["mu"] for r in rs if fail is not None and r["mu"] > fail and r["released"]]
            cap = {"plate": math.inf, "arm_12": 12.0, "arm_6": 6.0}[kind]
            model = min(cap, math.sqrt(2*J*u))/G
            lines.append(f"   {kind:7s}  {u:5.2f}   {ok:14.2f}   {'none' if fail is None else format(fail, '.2f'):>18}   "
                         f"{(str(min(above)) + '-' + str(max(above))) if above else 'no':>21}   {model:26.3f}   {math.sqrt(J*u)/G:10.3f}")
    lines += ["", "   The object leaves at the model's speed whenever the commanded brake passes mu g; close to the bound the gap opened",
              "   before the stop is tens of micrometres and the settling paddle closes it, so the push fails released().", ""]
    return lines

# ----------------------------------------------------------------------------- B: calibration of the belief

def _logdet_nuisance(mu, v_r, obs):
    """ln det of the (x_r, v_r) normal matrix of model C at each grid point: the factor a flat prior on the nuisance would
    add to the profile likelihood as -0.5 ln det (Laplace; the nuisance enters almost linearly)."""
    T = v_r/(mu*obs.g)
    tau = np.minimum(obs.t[:, None], T)
    jv = np.where(obs.t[:, None] >= T, T, tau)
    a = len(obs.t)/obs.sigma_f**2 + 1/obs.sigma_rest**2 + 1/obs.sigma_fk**2
    b = jv.sum(0)/obs.sigma_f**2 + T/obs.sigma_rest**2
    c = (jv*jv).sum(0)/obs.sigma_f**2 + T*T/obs.sigma_rest**2
    return np.log(np.maximum(a*c - b*b, 1e-300))

def _sbc(job):
    name, m0, s0, u, scale, n, seed, n_grid = job
    from belief import theta_grid, logprior, posterior, profile_nuisance
    from observe import simulate_obs
    rng = np.random.default_rng(seed)
    g = theta_grid(n_grid, math.log(0.005), math.log(5.0))
    lp = logprior(g, m0, s0)
    out = {k: ([], [], [], []) for k in ("profile", "marginal")}           # PIT, inside, posterior SD, median - truth
    while len(out["profile"][0]) < n:
        th = rng.normal(m0, s0)
        mu = math.exp(th)
        v_r = float(release_speed(u, mu, J))
        if v_r < 0.05:                                  # no slide to observe; rare under these priors
            continue
        x_r = 0.55 + float(release_push_distance(u, mu, J)) + rng.normal(0, 0.002)
        obs = simulate_obs(x_r, v_r, mu, 60.0, 0.002*scale, 0.0005*scale, 0.0005*scale, rng)
        if len(obs.t) < 3:
            continue
        _, v_hat, chi2 = profile_nuisance(np.exp(g), obs, "C")
        for kind, ll in (("profile", -0.5*chi2), ("marginal", -0.5*chi2 - 0.5*_logdet_nuisance(np.exp(g), v_hat, obs))):
            post = posterior(g, lp, ll)
            cdf = np.cumsum(post["pmf"]) - 0.5*post["pmf"]
            o = out[kind]
            o[0].append(float(np.interp(th, g, cdf))); o[1].append(post["q05"] <= th <= post["q95"]); o[2].append(post["sd"])
            o[3].append(post["median"] - th)
    return name, out

def part_b(procs, n=2400):
    cases = [("slide dominates: practice 1.01 m/s, nominal noise, prior N(ln 0.3, 0.3^2)", math.log(0.3), 0.3, 1.01, 1.0),
             ("in between:      practice 0.60 m/s, 4x noise,      prior N(ln 0.2, 0.25^2)", math.log(0.2), 0.25, 0.60, 4.0),
             ("prior matters:   practice 0.45 m/s, 8x noise,      prior N(ln 0.3, 0.3^2)", math.log(0.3), 0.3, 0.45, 8.0)]
    per = n//procs + 1
    jobs = [(c[0], c[1], c[2], c[3], c[4], per, 1000*i + k, 2000) for i, c in enumerate(cases) for k in range(procs)]
    with Pool(procs) as pool:
        parts = pool.map(_sbc, jobs, chunksize=1)
    lines = ["B. CALIBRATION OF THE BELIEF (truth drawn from the prior, one simulated slide, model C, 2,000-point grid)", "",
             "   'profile' is belief.loglik_slide (the nuisance x_r, v_r maximised out, as the evaluations use it); 'marginal' adds",
             "   -0.5 ln det of the nuisance's normal matrix, which is what a flat prior on (x_r, v_r) would give.", "",
             "   case / likelihood                                                         draws   in 90% interval   CDF at truth: mean (0.5 +- SE)   median - truth, ln mu   posterior SD"]
    for c in cases:
        for kind in ("profile", "marginal"):
            pit, inside, sd, bias = (np.concatenate([p[1][kind][k] for p in parts if p[0] == c[0]]) for k in range(4))
            se = math.sqrt(0.9*0.1/len(pit))
            lines.append(f"   {(c[0] if kind == 'profile' else '') :72s} {kind:8s} {len(pit):5d}   {100*inside.mean():5.1f}% (+-{100*se:.1f})   "
                         f"{pit.mean():.3f} +- {0.2887/math.sqrt(len(pit)):.3f}{'':14s}{bias.mean():+.4f} +- {bias.std()/math.sqrt(len(bias)):.4f}{'':6s}{np.median(sd):.4f}")
    lines += ["", "   Calibrated if the interval holds the truth 90% of the time and the mean CDF is 0.5, each within about two standard",
              "   errors.  A mean CDF above 0.5 says the belief sits low in friction.", ""]
    return lines

# ----------------------------------------------------------------------------- C: grid resolution at low noise

def part_c(n=40, seed=3):
    from belief import theta_grid, logprior, update_with_slides
    from decide import Release, choose_speed, point_target
    from observe import simulate_obs
    rng = np.random.default_rng(seed)
    rel, target, x0, u_prac = Release(J=J), 0.77, 0.55, 1.01
    lines = ["C. GRID RESOLUTION (one practice slide at 1.01 m/s, target 0.77 m, generic prior; landing by the release model)", "",
             "   noise   posterior SD ln mu   grid spacing 2,000 / 32,000   |landing(2,000) - landing(32,000)| mm: mean, max"]
    def landing(u, mu):
        return x0 - 0.001 + float(release_push_distance(u, mu, J)) + float(stop_distance(release_speed(u, mu, J), mu))
    for scale in (1.0, 0.5, 0.2, 0.05):
        diff, sds = [], []
        for _ in range(n):
            mu = float(rng.uniform(0.12, 0.45))
            v_r = float(release_speed(u_prac, mu, J))
            obs = simulate_obs(x0 + float(release_push_distance(u_prac, mu, J)), v_r, mu, 60.0, 0.002*scale, 0.0005*scale, 0.0005*scale, rng)
            lands = []
            for n_grid in (2000, 32000):
                g = theta_grid(n_grid, math.log(0.005), math.log(5.0))
                post = update_with_slides(g, logprior(g, math.log(0.29665), 0.7967), [obs], model="C")
                u = choose_speed(post, rel, point_target(target, x0), u_max=1.158)["u"]
                lands.append(landing(u, mu))
                if n_grid == 32000:
                    sds.append(post["sd"])
            diff.append(1000*abs(lands[0] - lands[1]))
        lines.append(f"   x{scale:<5g}  {np.median(sds):.5f}              {math.log(1000)/1999:.5f} / {math.log(1000)/31999:.5f}            {np.mean(diff):.3f}, {np.max(diff):.3f}")
    lines += ["", "   The 2,000-point grid is adequate while the posterior SD is at least about its spacing (0.0035).", ""]
    return lines

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=str(RESULTS/"logic_checks"))
    a = ap.parse_args()
    from pathlib import Path
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = ["logic_checks.py, 25 Nov 2026.  Simulator and analytic model only.", ""]
    if a.part in ("A", "all"):
        lines += part_a(out, a.procs)
    if a.part in ("B", "all"):
        lines += part_b(a.procs)
    if a.part in ("C", "all"):
        lines += part_c()
    text = "\n".join(lines)
    print(text)
    name = "summary.txt" if a.part == "all" else f"summary_part_{a.part}.txt"
    (out/name).write_text(text + "\n")

if __name__ == "__main__":
    main()

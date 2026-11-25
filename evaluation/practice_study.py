#!/usr/bin/env python3
"""
practice_study.py -- how informative does a practice slide have to be before the prior stops mattering?

The side-base evaluation (results/side_base/evaluation/ten_seeds_*) found that one practice slide at 1.01 m/s puts the
generic, lookup and LLM methods within 0.1 mm of each other: the comparison between priors exists only before any
practice.  This study asks where that changes.  It keeps the evaluation's setup (side base, computed speed cap, derived
targets, the same three priors, belief update and planner) and varies how much one practice slide says about friction:

  speed      the practice push's peak speed, from 0.15 m/s to the default (a slower push slides a shorter way, so the
             camera sees fewer frames and the rest point says less)
  noise      the observation noise as a multiple of the evaluation's nominal (2 mm frames, 0.5 mm rest point, 0.5 mm
             release point); the likelihood is told the same multiple, as in run_arm_evaluation.py
  transfer   friction that differs from slide to slide: every slide's ln mu is the pair's plus N(0, s^2).  The simulator's
             friction is uniform, so this part runs in the release model's world (physics.py) instead of Drake, with two
             kinds of inference: "unaware" is the pipeline as it is (it assumes the practice friction is the scored
             shot's), "aware" knows s and updates a pair-level friction (update_with_transfer)

Practice pushes run in Drake (one per pair and speed; the simulator is deterministic, so repeated slides differ only in
their noise draws).  Scored shots are read off a landing map: Drake pushes every 0.01 m/s from 0.06 m/s to the cap for
each pair, with the board's seam losses taken out (load_map) and interpolated linearly.  --verify re-runs a sample of
scored shots in Drake to measure what the map leaves out.
The planner's loss is its default (decide.loss_sq_rel).  Nothing in the evaluation runner is changed; its functions are
imported.

    python3 evaluation/practice_study.py                 # everything, about two hours on 8 processes -> results/side_base/practice_study/
    python3 evaluation/practice_study.py --stage map     # one stage: map, sweep, transfer, grid, verify, summarize
    python3 evaluation/practice_study.py --summarize     # rewrite the summary and figure from the saved cells
"""
import argparse, csv, functools, gzip, json, math, pickle, time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paths import RESULTS
import run_arm_evaluation as E
from belief import loglik_lookup, loglik_slide, logprior, lookup_observation, posterior, theta_grid, update_with_slides
from demo_published_comparison import OBJECT, SURFACES, raw_llm_logprior
from llm_prior import build_prompt, cached_response, parse_response
from observe import Obs
from run_arm_comparison_v2 import PRIOR_LOG_SD, PRIOR_MEDIAN

OUT = RESULTS/"side_base"/"practice_study"
SPEEDS = (0.15, 0.2, 0.3, 0.45, 0.6, 0.8)        # practice peak speeds, m/s; the default practice speed is added to them
NOISE = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)          # multiples of the nominal observation noise
SLIDES = (1, 2, 3)
METHODS = ("generic", "lookup", "llm_raw")
TRANSFER = (0.0, 0.03, 0.06, 0.12, 0.25)         # SD of ln mu from slide to slide (Yu et al.'s spatial SD / mu: 0.11-0.25)
SIGMA = (0.002, 0.0005, 0.0005)                  # nominal SDs: frames, rest point, release point (run_arm_evaluation.py)
MAP_STEP, MAP_MIN = 0.01, 0.06
CELL_FIELDS = ["pair", "practice_peak", "noise", "seed", "slides", "method", "target_x", "mu_median", "post_sd", "sigma_data",
               "peak_speed", "x_rest", "error_mm"]
TRANSFER_FIELDS = ["pair", "transfer_sd", "inference", "seed", "slides", "method", "target_x", "mu_test", "mu_median", "post_sd",
                   "peak_speed", "x_rest", "error_mm"]

# ----------------------------------------------------------------------------- the evaluation's setup

@functools.lru_cache(maxsize=None)
def priors(grid_points=2000):
    """(truth, grid, {pair: {method: log prior}}) exactly as run_arm_evaluation.run builds them.  No Drake."""
    with E._file("ground_truth_published.csv").open(newline="") as f:
        truth = {r["pair_id"]: float(r["mu_k_nominal"]) for r in csv.DictReader(f)}
    g = theta_grid(grid_points, math.log(.005), math.log(5.))
    generic = logprior(g, math.log(PRIOR_MEDIAN), PRIOR_LOG_SD)
    initials = {}
    for pair in SURFACES:
        raw = cached_response(build_prompt(dict(description_object=OBJECT, description_surface=SURFACES[pair])),
                              E._file("llm_cache_published_pilot.json"))
        if raw is None:
            raise ValueError(f"Cache miss for {pair}; no request made")
        table = lookup_observation(pair, str(E._file("friction_lookup.csv")))
        initials[pair] = dict(generic=generic, lookup=generic.copy() if table is None else generic + loglik_lookup(g, *table),
                              llm_raw=raw_llm_logprior(g, parse_response(raw)))
    return truth, g, initials

def arm_defaults():
    """The default evaluation's arm setup (side base, computed cap and practice speed, derived targets)."""
    truth, g, initials = priors()
    args = SimpleNamespace(layout="side", peak_cap="auto", practice_peak="auto", targets=["auto"])
    s = E.arm_setup(args, truth, posterior(g, initials[next(iter(SURFACES))]["generic"]))
    return dict(cap=s["cap"], practice=s["practice"], targets=s["targets"], options=s["options"], record=s["record"])

# ----------------------------------------------------------------------------- observations

def clean_observation(result, pusher):
    """The noise-free measurements of one Drake practice slide, taken as run_arm_evaluation.make_observation takes them:
    60 Hz frames while the object moves after the contact force drops to zero, the release point from the arm's forward
    kinematics, and the rest point."""
    from physics import OBJECT_SIZE
    tr, t0 = result["traj"], result["t_force_zero"]
    moving = np.flatnonzero((tr["t"] > t0) & (np.abs(tr["vx"]) > .005))
    if not len(moving):
        raise ValueError("No measurable free slide")
    ts = np.arange(1, int((tr["t"][moving[-1]] - t0)*60) + 1)/60.
    if len(ts) < 2:
        raise ValueError("Fewer than two free-slide frames")
    fk = float(pusher.face_x(float(np.interp(t0, tr["t"], tr["q"]))) + OBJECT_SIZE[0]/2)
    return dict(t=ts, x=np.interp(t0 + ts, tr["t"], tr["x"]), fk=fk, rest=float(result["x_rest"]))

def noisy(clean, rng, scale):
    """One noisy observation of a clean slide: the same draws, in the same order (frames, release point, rest point), and
    the same likelihood SDs as run_arm_evaluation.make_observation at noise multiple `scale` (tests check they agree)."""
    z_frame, z_fk, z_rest = rng.standard_normal(len(clean["t"])), rng.standard_normal(), rng.standard_normal()
    return Obs(clean["t"], clean["x"] + SIGMA[0]*scale*z_frame, float(clean["rest"] + SIGMA[1]*scale*z_rest),
               float(clean["fk"] + SIGMA[2]*scale*z_fk), SIGMA[0]*scale, SIGMA[1]*scale, SIGMA[2]*scale)

# ----------------------------------------------------------------------------- friction that varies between slides

def transfer_kernel(g, s):
    """K[i, j] = P(a slide's ln mu is g[j] | the pair's ln mu is g[i]) for slide = pair + N(0, s^2), on the grid.  None when
    s = 0 (the slides share the pair's friction)."""
    if s <= 0:
        return None
    d = (g[:, None] - g[None, :])/s
    K = np.exp(-0.5*d*d)
    return K/K.sum(axis=1, keepdims=True)

def update_with_transfer(g, logp0, obs_list, K):
    """Belief about the NEXT slide's ln mu when every slide's friction is the pair's plus the kernel's spread: each practice
    slide's likelihood is smoothed by K before it updates the pair-level belief, and the pair-level posterior is spread by
    K once more for the slide to come.  With K = None this is belief.update_with_slides."""
    if K is None:
        return update_with_slides(g, logp0, obs_list)
    terms = []
    for o in obs_list:
        ll = loglik_slide(g, o)
        terms.append(np.log(np.maximum(K @ np.exp(ll - ll.max()), 1e-300)))
    pred = posterior(g, logp0, *terms)["pmf"] @ K
    return posterior(g, np.log(np.maximum(pred, 1e-300)))

# ----------------------------------------------------------------------------- stage: landing map and practice slides

def _push(job):
    pair, mu, u, options, keep = job
    out = dict(pair=pair, u=u, x_rest=math.nan, y_rest=math.nan, valid=False, v_frac=math.nan, tau_frac=math.nan, a_frac=math.nan, error="")
    try:
        r, p, lim, valid = E.checked_execute(mu, u, **options)
        out.update(x_rest=float(r["x_rest"]), y_rest=float(r["y_rest"]), valid=bool(valid), v_frac=lim["v"], tau_frac=lim["tau"], a_frac=lim["a"])
        if keep:
            out["clean"] = clean_observation(r, p)
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out

def stage_map(out, procs):
    arm = arm_defaults()
    truth, _, _ = priors()
    us = np.round(np.append(np.arange(MAP_MIN, arm["cap"], MAP_STEP), arm["cap"]), 6)
    speeds = list(SPEEDS) + [arm["practice"]]
    jobs = [(pair, truth[pair], float(u), arm["options"], False) for pair in SURFACES for u in us]
    jobs += [(pair, truth[pair], float(u), arm["options"], True) for pair in SURFACES for u in speeds]
    with Pool(procs) as pool:
        rows = pool.map(_push, jobs, chunksize=2)
    n_map = len(SURFACES)*len(us)                       # pool.map keeps the order of the jobs
    lmap, practice = rows[:n_map], {(r["pair"], r["u"]): r for r in rows[n_map:]}
    bad = [(r["pair"], r["u"], r["error"] or "invalid") for r in practice.values() if "clean" not in r or not r["valid"]]
    if bad:
        raise RuntimeError(f"practice pushes failed: {bad}")
    with (out/"landing_map.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "u", "x_rest", "y_rest", "valid", "v_frac", "tau_frac", "a_frac", "error"], lineterminator="\n")
        w.writeheader(); w.writerows(lmap)
    with (out/"practice_slides.pkl").open("wb") as f:
        pickle.dump({k: v["clean"] for k, v in practice.items()}, f)
    E.write_json(out/"settings.json", dict(
        study="practice informativeness", arm=arm["record"], peak_cap=arm["cap"], default_practice_peak=arm["practice"], targets=arm["targets"],
        practice_speeds=speeds, noise_multiples=list(NOISE), slides=list(SLIDES), methods=list(METHODS), transfer_sd=list(TRANSFER),
        nominal_sd=list(SIGMA), x0=E.X0, jerk=E.JERK, fps=60, map_step=MAP_STEP, map_min=MAP_MIN, generic_median=PRIOR_MEDIAN,
        generic_log_sd=PRIOR_LOG_SD, loss="decide.loss_sq_rel (the planner's default)",
        scored_shots="Drake landing map (landing_map.csv) without the board's seam losses, interpolated linearly; see map_check.json and verify.csv",
        transfer_world="release model (physics.py), not Drake; nominal noise; default practice speed"))
    return load_map(out)

def load_map(out):
    """{pair: (u, x_rest)}: the landing map with the board's seam losses taken out, and what was taken out (map_check.json).

    push_slide's uniform table is 24 equal-friction strips (scene.Table), and the object loses up to 8 mm of slide at single
    time steps as it crosses a seam; which pushes are hit depends on the speed to the centimetre per second, so the landing
    is a smooth curve with isolated short landings under it (a one-piece board gives the smooth curve alone).  A short
    landing at one sampled speed says nothing about its neighbours, so the map is the release model plus the upper envelope
    (running maximum over 11 points) of the pushes' residual against it; the seam losses are reported, not interpolated."""
    from arm import rest_x
    truth, _, _ = priors()
    by = {}
    with (out/"landing_map.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["valid"] == "True":
                by.setdefault(r["pair"], []).append((float(r["u"]), float(r["x_rest"])))
    lmap, check = {}, {}
    for pair, pts in by.items():
        u, x = np.array(sorted(pts)).T
        model = np.array([rest_x(ui, truth[pair], E.JERK, E.X0) for ui in u])
        res = x - model
        base = np.array([res[max(i - 5, 0):i + 6].max() for i in range(len(res))])    # seam losses only shorten the slide
        lmap[pair] = (u, model + base)
        seam = res - base                                   # the seam losses (and nothing else, to within the baseline's ripple)
        hit = seam < -2e-4
        check[pair] = dict(points=len(u), u_min=float(u[0]), u_max=float(u[-1]),
                           baseline_vs_release_model_mm=[round(float(1e3*base.max()), 3), round(float(1e3*base.min()), 3)],
                           pushes_with_seam_loss=int(hit.sum()), seam_loss_rms_mm=round(float(1e3*np.sqrt(np.mean(seam**2))), 3),
                           seam_loss_worst_mm=round(float(1e3*seam.min()), 3), seam_loss_worst_at=float(u[int(np.argmin(seam))]),
                           seam_loss_rms_above_0p6_mm=round(float(1e3*np.sqrt(np.mean(seam[u >= 0.6]**2))), 3))
    E.write_json(out/"map_check.json", check)
    return lmap

def landing(lmap, pair, u):
    us, xs = lmap[pair]
    return float(np.interp(u, us, xs)) if u >= us[0] else float(E.X0 + (xs[0] - E.X0)*(u/us[0])**2)   # below the map: scaled as u^2

# ----------------------------------------------------------------------------- stage: speed x noise sweep (Drake practice, landing map)

def _plan(target, post, cap):
    from arm import plan_peak_speed
    return plan_peak_speed(target, post, J=E.JERK, x0=E.X0, u_max=cap)

def _sweep(job):
    pair, k_speed, u_practice, clean, lmap, seeds, cap, targets, grid_points, noise_levels = job
    _, g, initials = priors(grid_points)
    rows = []
    for scale in noise_levels:
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(pair[1:]), k_speed]))   # the same draws at every noise level
            obs = [noisy(clean, rng, scale) for _ in range(max(SLIDES))]
            sigma_data = posterior(g, loglik_slide(g, obs[0]))["sd"]           # what one slide alone says: SD of ln mu, flat prior
            for n in SLIDES:
                for method in METHODS:
                    post = update_with_slides(g, initials[pair][method], obs[:n])
                    for target in targets:
                        u = _plan(target, post, cap)
                        x = landing(lmap, pair, u)
                        rows.append(dict(pair=pair, practice_peak=u_practice, noise=scale, seed=seed, slides=n, method=method, target_x=target,
                                         mu_median=round(post["mu_median"], 6), post_sd=round(post["sd"], 5), sigma_data=round(sigma_data, 5),
                                         peak_speed=round(u, 6), x_rest=round(x, 6), error_mm=round(1e3*(x - target), 3)))
    return rows

def reference_rows(lmap, cap, targets, grid_points=2000):
    """Rows that do not depend on the practice slide: every method with no practice, and the true-friction plan."""
    truth, g, initials = priors(grid_points)
    rows = []
    for pair in SURFACES:
        for target in targets:
            for method in METHODS + ("oracle",):
                post = None if method == "oracle" else posterior(g, initials[pair][method])
                u = _plan(target, truth[pair] if post is None else post, cap)
                x = landing(lmap, pair, u)
                rows.append(dict(pair=pair, practice_peak="", noise="", seed="", slides=0 if post else "", method=method, target_x=target,
                                 mu_median=truth[pair] if post is None else round(post["mu_median"], 6), post_sd="" if post is None else round(post["sd"], 5),
                                 sigma_data="", peak_speed=round(u, 6), x_rest=round(x, 6), error_mm=round(1e3*(x - target), 3)))
    return rows

def write_cells(path, fields, rows):
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)

def read_cells(path):
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))

def stage_sweep(out, procs, seeds, grid_points=2000):
    lmap, arm = load_map(out), json.loads((out/"settings.json").read_text())
    with (out/"practice_slides.pkl").open("rb") as f:
        practice = pickle.load(f)
    speeds = arm["practice_speeds"]
    jobs = [(pair, k, u, practice[(pair, u)], lmap, seeds, arm["peak_cap"], arm["targets"], grid_points, NOISE)
            for pair in SURFACES for k, u in enumerate(speeds)]
    with Pool(procs) as pool:
        rows = [r for part in pool.map(_sweep, jobs, chunksize=1) for r in part]
    write_cells(out/"cells.csv.gz", CELL_FIELDS, reference_rows(lmap, arm["peak_cap"], arm["targets"], grid_points) + rows)
    frames = {f"{pair} at {u:g} m/s": len(practice[(pair, u)]["t"]) for pair in SURFACES for u in speeds}
    E.write_json(out/"practice_frames.json", frames)

# ----------------------------------------------------------------------------- stage: friction that varies between slides (model world)

def _transfer(job):
    seeds, cap, targets, practice_peak, grid_points = job
    from arm import rest_x
    from observe import simulate_obs
    from physics import release_push_distance, release_speed
    truth, g, initials = priors(grid_points)
    kernels = {s: transfer_kernel(g, s) for s in TRANSFER}
    rows = []
    for seed in seeds:
        for pair in SURFACES:
            z = np.random.default_rng(np.random.SeedSequence([seed, int(pair[1:]), 99])).standard_normal(max(SLIDES) + 1)   # shared by every level
            for s in TRANSFER:
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(pair[1:]), 98]))
                mus = truth[pair]*np.exp(s*z)                                 # practice slides 1..3, then the scored shot
                obs = []
                for mu in mus[:-1]:
                    x_r = E.X0 - 0.001 + float(release_push_distance(practice_peak, mu, E.JERK))
                    obs.append(simulate_obs(x_r, float(release_speed(practice_peak, mu, E.JERK)), mu, 60., *SIGMA, rng))
                for inference in ("unaware", "aware") if s > 0 else ("unaware",):
                    K = kernels[s] if inference == "aware" else None
                    for n in (0,) + SLIDES:
                        for method in METHODS:
                            post = update_with_transfer(g, initials[pair][method], obs[:n], K)
                            for target in targets:
                                u = _plan(target, post, cap)
                                x = rest_x(u, mus[-1], E.JERK, E.X0)
                                rows.append(dict(pair=pair, transfer_sd=s, inference=inference, seed=seed, slides=n, method=method, target_x=target,
                                                 mu_test=round(float(mus[-1]), 6), mu_median=round(post["mu_median"], 6), post_sd=round(post["sd"], 5),
                                                 peak_speed=round(u, 6), x_rest=round(x, 6), error_mm=round(1e3*(x - target), 3)))
    return rows

def stage_transfer(out, procs, seeds, grid_points=2000):
    arm = json.loads((out/"settings.json").read_text())
    chunks = [list(c) for c in np.array_split(seeds, max(1, min(len(seeds), procs*2))) if len(c)]
    jobs = [([int(s) for s in c], arm["peak_cap"], arm["targets"], arm["default_practice_peak"], grid_points) for c in chunks]
    with Pool(procs) as pool:
        rows = [r for part in pool.map(_transfer, jobs, chunksize=1) for r in part]
    write_cells(out/"transfer_cells.csv.gz", TRANSFER_FIELDS, rows)

# ----------------------------------------------------------------------------- stage: does the belief grid limit the result?

def stage_grid(out, procs, seeds, grids=(2000, 8000, 32000)):
    """The same cells at the default practice speed with finer belief grids: at 1.01 m/s and nominal noise one slide gives
    ln mu to about the grid's spacing (0.0035 at 2000 points), so the posterior sits on one or two grid points."""
    lmap, arm = load_map(out), json.loads((out/"settings.json").read_text())
    with (out/"practice_slides.pkl").open("rb") as f:
        practice = pickle.load(f)
    u = arm["default_practice_peak"]
    k = arm["practice_speeds"].index(u)
    jobs = [(pair, k, u, practice[(pair, u)], lmap, seeds, arm["peak_cap"], arm["targets"], n, (0.5, 1.0, 2.0)) for n in grids for pair in SURFACES]
    with Pool(procs) as pool:
        parts = pool.map(_sweep, jobs, chunksize=1)
    rows = [dict(r, grid_points=job[8]) for job, part in zip(jobs, parts) for r in part]
    oracle = np.mean([abs(float(r["error_mm"])) for r in reference_rows(lmap, arm["peak_cap"], arm["targets"]) if r["method"] == "oracle"])
    table = {}
    for n in grids:
        for scale in (0.5, 1.0, 2.0):
            for slides in SLIDES:
                e = [abs(r["error_mm"]) for r in rows if r["grid_points"] == n and r["noise"] == scale and r["slides"] == slides]
                table[f"grid {n}, noise x{scale:g}, {slides} slide(s)"] = round(float(np.mean(e)), 3)
    step = {n: round(math.log(1000.)/(n - 1), 5) for n in grids}
    E.write_json(out/"grid_check.json", dict(mean_abs_error_mm=table, oracle_mean_abs_error_mm=round(float(oracle), 3), grid_spacing_ln_mu=step,
                                             practice_peak=u, seeds=list(seeds)))

# ----------------------------------------------------------------------------- stage: scored shots re-run in Drake

def stage_verify(out, procs, n=60, seed=0):
    lmap, arm = load_map(out), json.loads((out/"settings.json").read_text())
    truth, _, _ = priors()
    cells = [r for r in read_cells(out/"cells.csv.gz") if r["method"] != "oracle"]
    pick = np.random.default_rng(seed).choice(len(cells), size=n, replace=False)
    options = {"base": tuple(arm["arm"]["base"])}
    jobs = [(cells[i]["pair"], truth[cells[i]["pair"]], float(cells[i]["peak_speed"]), options, False) for i in pick]
    with Pool(procs) as pool:
        res = pool.map(_push, jobs, chunksize=2)
    with (out/"verify.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pair", "peak_speed", "x_rest_map", "x_rest_drake", "difference_mm", "valid"])
        for i, r in zip(pick, res):
            w.writerow([cells[i]["pair"], cells[i]["peak_speed"], cells[i]["x_rest"], round(r["x_rest"], 6),
                        round(1e3*(float(cells[i]["x_rest"]) - r["x_rest"]), 3), r["valid"]])

# ----------------------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["all", "map", "sweep", "transfer", "grid", "verify", "summarize"], default="all")
    ap.add_argument("--summarize", action="store_true", help="the same as --stage summarize")
    ap.add_argument("--seeds", type=int, default=20, help="noise seeds per cell of the speed x noise sweep")
    ap.add_argument("--transfer-seeds", type=int, default=100)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    stage = "summarize" if a.summarize else a.stage
    a.out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    if stage in ("all", "map"):
        stage_map(a.out, a.procs); print(f"landing map and practice slides: {time.time() - t:.0f} s", flush=True)
    if stage in ("all", "sweep"):
        stage_sweep(a.out, a.procs, list(range(a.seeds))); print(f"sweep: {time.time() - t:.0f} s", flush=True)
    if stage in ("all", "transfer"):
        stage_transfer(a.out, a.procs, list(range(a.transfer_seeds))); print(f"transfer: {time.time() - t:.0f} s", flush=True)
    if stage in ("all", "grid"):
        stage_grid(a.out, a.procs, list(range(a.seeds))); print(f"grid check: {time.time() - t:.0f} s", flush=True)
    if stage in ("all", "verify"):
        stage_verify(a.out, a.procs); print(f"verify: {time.time() - t:.0f} s", flush=True)
    if stage in ("all", "summarize"):
        from practice_study_summary import summarize
        print(summarize(a.out))

if __name__ == "__main__":
    main()

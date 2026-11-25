#!/usr/bin/env python3
"""
prior_study.py -- the parts of the proposal's plan that need no new language-model answers, on any pair set.

For every pair and every prior, scored with the release model (no Drake push; the arm model is used only for the speed
cap and the targets, with the rerun's setup: 6 m/s^2 brake, limits posture, acceleration checked):

  priors      generic (wide, 0.08-1.1); lookup; one per LLM cache (--llm NAME=CACHE, as the other runners); for each LLM
              the width-matched control generic@NAME; and three that did not exist before:
                population      the other pairs' friction, leaving this pair out (normal in ln mu): the third baseline of D17
                robust@SOURCE   0.8 x the source's prior + 0.2 x the generic prior, for the lookup and each LLM: a confident
                                prior that is wrong keeps a wide component for the data to fall back on
                recal_NAME      the LLM's median as an observation of ln mu, m = alpha + beta theta + eps, fitted on the other
                                pairs (belief.fit_llm_likelihood), on top of the population prior: the specification's
                                recalibration, leave-one-pair-out
  intervals   does the prior's 90% interval hold the true friction; its width; the interval score and the log score (H4)
  first shot  plug-in (plan with the median as if it were the friction) against uncertainty-aware (minimum expected loss
              under the whole belief), under the relative loss and under the squared miss (H2)
  table edge  the same shot with an edge 30% beyond the target, counting from the object's start: how often the object
              falls off under plug-in, uncertainty-aware and edge-aware planning (the loss with a penalty past the edge)
  weak slide  one practice slide that leaves the prior in play (0.60 m/s, 4x nominal noise), then a shot: where a robust
              prior is supposed to help

Differences between priors are reported over pairs with bootstrap intervals and signed-rank tests (paired_stats.py).

    python3 evaluation/prior_study.py                                   # the 23 pairs with the stand-in LLM, ~1 min
    python3 evaluation/prior_study.py --llm claude=data/llm_cache_handbook_claude.json gpt=data/llm_cache_handbook_gpt.json --out results/side_base/prior_study_api

On L01-L20 the truth is the handbook value: the lookup holds the answer and an LLM may recall the same tables.
"""
import argparse, csv, math
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import decide
import run_arm_evaluation as E
import run_pair_set as R
from belief import fit_llm_likelihood, loglik_llm, loglik_lookup, logprior, lookup_observation, posterior, prior_from_measured, theta_grid, update_with_slides
from demo_published_comparison import raw_llm_logprior
from observe import simulate_obs
from paired_stats import line, paired
from paths import DATA, RESULTS
from physics import release_push_distance, release_speed
from run_arm_comparison_v2 import PRIOR_LOG_SD, PRIOR_MEDIAN

Z90 = 1.6448536269514722
ROBUST_WEIGHT = 0.8                  # weight of the informative component in a robust prior
EDGE = 0.3                           # the table edge is this fraction of (target - start) beyond the target
EDGE_PENALTY = 10.0                  # decide.loss_table_edge's penalty for leaving the table
WEAK = dict(peak=0.60, noise=4.0)    # the weak practice slide: slow and noisy enough that the prior still counts (a slide
                                     # at 0.30 m/s and 8x noise is shorter than its own noise and says nothing at all)
FIELDS = ["source", "pair", "stage", "rule", "target", "seed", "miss_mm", "fell"]

def normalised(g, logp):
    """A log density on the grid (so that two priors can be mixed with stated weights)."""
    logp = np.asarray(logp, float) - np.max(logp)
    return logp - math.log(np.exp(logp).sum()*(g[1] - g[0]))

def mixture(g, informative, wide, w=ROBUST_WEIGHT):
    return np.logaddexp(math.log(w) + normalised(g, informative), math.log(1 - w) + normalised(g, wide))

def priors_for(pair, ctx, g):
    """{source: log prior on the grid} for one pair."""
    generic = logprior(g, math.log(PRIOR_MEDIAN), PRIOR_LOG_SD)
    out = {"generic": generic}
    others = [q for q in ctx["truth"] if q != pair]
    m0, s0 = prior_from_measured([ctx["truth"][q] for q in others])
    out["population"] = logprior(g, m0, s0)
    table = ctx["lookup"].get(pair)
    if table is not None:
        out["lookup"] = generic + loglik_lookup(g, *table)
        out["robust@lookup"] = mixture(g, out["lookup"], generic)
    for name, answers in ctx["llm"].items():
        est = answers.get(pair)
        if est is None:
            continue
        out[f"llm_{name}"] = raw_llm_logprior(g, est)
        out[f"generic@{name}"] = logprior(g, math.log(PRIOR_MEDIAN), math.log(est["high"]/est["low"])/(2*Z90))
        out[f"robust@{name}"] = mixture(g, out[f"llm_{name}"], generic)
        fit = [q for q in others if q in answers]
        if len(fit) >= 4:
            params = fit_llm_likelihood([math.log(answers[q]["median"]) for q in fit], [math.log(ctx["truth"][q]) for q in fit])
            out[f"recal_{name}"] = out["population"] + loglik_llm(g, math.log(est["median"]), params)
    return out

def interval_row(g, post, mu):
    """Calibration of one prior against the true friction, in ln mu: inside the 90% interval, its width, the interval score
    (width plus 2/alpha times the distance outside; lower is better) and the log score (-ln density at the truth)."""
    th, lo, hi = math.log(mu), post["q05"], post["q95"]
    score = (hi - lo) + 20.0*max(lo - th, 0.0) + 20.0*max(th - hi, 0.0)
    dens = float(np.interp(th, g, post["density"]))
    return dict(inside=int(lo <= th <= hi), width=hi - lo, interval_score=score, log_score=-math.log(max(dens, 1e-300)),
                median_err=post["median"] - th, prior_sd=post["sd"])

def _pair(job):
    pair, ctx = job
    from arm import plan_peak_speed, rest_x
    g = theta_grid(ctx["grid_points"], math.log(.005), math.log(5.))
    mu, cap, J, X0 = ctx["truth"][pair], ctx["cap"], E.JERK, E.X0
    land = lambda u: rest_x(u, mu, J, X0)
    rows, cal = [], {}
    weak_obs = []
    v_r = float(release_speed(WEAK["peak"], mu, J))
    for seed in ctx["seeds"]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, sum(map(ord, pair))]))
        s = WEAK["noise"]
        weak_obs.append(simulate_obs(X0 - 0.001 + float(release_push_distance(WEAK["peak"], mu, J)), v_r, mu, 60.0, .002*s, .0005*s, .0005*s, rng)
                        if v_r > 0.05 else None)
    for source, lp in priors_for(pair, ctx, g).items():
        post = posterior(g, lp)
        cal[source] = interval_row(g, post, mu)
        cal[source]["weak_sd"] = float(np.mean([update_with_slides(g, lp, [o], model="C")["sd"] for o in weak_obs if o is not None] or [float("nan")]))
        for t in ctx["targets"]:
            x_edge = t + EDGE*(t - X0)
            edge_loss = lambda d, ds: decide.loss_sq_rel(d, ds) + EDGE_PENALTY*(d > ds + (x_edge - t))
            shots = {"plug-in": plan_peak_speed(t, post["mu_median"], J=J, x0=X0, u_max=cap, clip=True),
                     "aware": plan_peak_speed(t, post, J=J, x0=X0, u_max=cap),
                     "aware_sq_abs": plan_peak_speed(t, post, J=J, x0=X0, u_max=cap, loss=decide.loss_sq_abs),
                     "edge-aware": plan_peak_speed(t, post, J=J, x0=X0, u_max=cap, loss=edge_loss)}
            for rule, u in shots.items():
                x = land(u)
                rows.append(dict(source=source, pair=pair, stage="first", rule=rule, target=t, seed=-1, miss_mm=round(1e3*(x - t), 3), fell=int(x > x_edge)))
            for seed, obs in zip(ctx["seeds"], weak_obs):
                if obs is None:
                    continue
                p1 = update_with_slides(g, lp, [obs], model="C")
                for rule, u in (("aware", plan_peak_speed(t, p1, J=J, x0=X0, u_max=cap)),
                                ("plug-in", plan_peak_speed(t, p1["mu_median"], J=J, x0=X0, u_max=cap, clip=True))):
                    x = land(u)
                    rows.append(dict(source=source, pair=pair, stage="weak1", rule=rule, target=t, seed=seed, miss_mm=round(1e3*(x - t), 3), fell=int(x > x_edge)))
    return pair, rows, cal

def setup(llms, truth_files, only, grid_points, seeds):
    pairs = R.load_pairs(truth_files, only)
    truth = {p: d["mu"] for p, d in pairs.items()}
    g = theta_grid(grid_points, math.log(.005), math.log(5.))
    from arm import target_grid
    arm = E.arm_setup(SimpleNamespace(layout="side", peak_cap="auto", practice_peak="auto", targets=[0.6], a_brake=6.0, check_accel=True, posture="limits"),
                      {**truth, **{p: 0. for p in R.SURFACES if p not in truth}}, posterior(g, logprior(g, math.log(PRIOR_MEDIAN), PRIOR_LOG_SD)))
    lookup = {p: lookup_observation(p, str(E._file("friction_lookup.csv"))) for p in pairs}
    answers = {name: {p: est for p, d in pairs.items() if (est := R.llm_answer(d["prompt"], path)) is not None} for name, path in llms.items()}
    return dict(pairs=pairs, truth=truth, lookup=lookup, llm=answers, cap=arm["cap"], targets=target_grid(arm["cap"], max(truth.values()), 3, E.JERK, E.X0),
                grid_points=grid_points, seeds=list(seeds), models={name: R.cache_model(path) for name, path in llms.items()})

def mean_abs(rows, **where):
    """{pair: mean |miss|} over the rows that match."""
    acc = {}
    for r in rows:
        if all(r[k] == v for k, v in where.items()):
            acc.setdefault(r["pair"], []).append(abs(float(r["miss_mm"])))
    return {p: float(np.mean(v)) for p, v in acc.items()}

def rmse(rows, **where):
    e = [float(r["miss_mm"]) for r in rows if all(r[k] == v for k, v in where.items())]
    return math.sqrt(float(np.mean(np.square(e)))) if e else float("nan")

def contrast(rows, a, b, label=None):
    """Paired difference a - b over the pairs both cover; a, b are dicts of row filters."""
    A, B = mean_abs(rows, **a), mean_abs(rows, **b)
    common = sorted(set(A) & set(B))
    if len(common) < 2:
        return None
    return line(*(label or (a["source"], b["source"])), paired([A[p] for p in common], [B[p] for p in common]))

def summarize(ctx, rows, cal):
    sources = list(dict.fromkeys(r["source"] for r in rows))
    pairs = list(ctx["truth"])
    avg = lambda d: float(np.mean(list(d.values()))) if d else float("nan")
    out = [f"prior study: {len(pairs)} pairs, friction {min(ctx['truth'].values()):.2f}-{max(ctx['truth'].values()):.2f}; targets {ctx['targets']} m; cap {ctx['cap']} m/s;",
           "scored with the release model (no Drake push).  LLMs: " + ("; ".join(f"{n} = {m}" for n, m in ctx["models"].items()) or "none"),
           "On L01-L20 the lookup holds the answer and an LLM may recall the same tables.", "",
           "1. THE PRIORS' 90% INTERVALS AGAINST THE TRUE FRICTION (ln mu; interval and log scores: lower is better)",
           "   source                pairs  inside 90%   RMS median error   mean width   interval score   log score"]
    for s in sources:
        c = [cal[p][s] for p in pairs if s in cal[p]]
        out.append(f"   {s:20s}  {len(c):4d}   {sum(x['inside'] for x in c):3d} ({100*np.mean([x['inside'] for x in c]):3.0f}%)        "
                   f"{math.sqrt(np.mean([x['median_err']**2 for x in c])):.3f}           {np.mean([x['width'] for x in c]):.2f}          "
                   f"{np.mean([x['interval_score'] for x in c]):.2f}          {np.mean([x['log_score'] for x in c]):+.2f}")
    out += ["", "2. FIRST SHOT, NO PRACTICE: mean |miss| (RMSE), mm.  aware minimises expected squared error under its own belief",
            "   source                plug-in           aware, relative loss   aware, squared miss     aware - plug-in in mean |miss| over pairs"]
    for s in sources:
        m = {rule: avg(mean_abs(rows, source=s, stage="first", rule=rule)) for rule in ("plug-in", "aware", "aware_sq_abs")}
        A, B = mean_abs(rows, source=s, stage="first", rule="aware"), mean_abs(rows, source=s, stage="first", rule="plug-in")
        st = paired([A[p] for p in sorted(A)], [B[p] for p in sorted(A)])
        r = {rule: rmse(rows, source=s, stage="first", rule=rule) for rule in m}
        out.append(f"   {s:20s}  {m['plug-in']:6.1f} ({r['plug-in']:6.1f})   {m['aware']:6.1f} ({r['aware']:6.1f})        {m['aware_sq_abs']:6.1f} ({r['aware_sq_abs']:6.1f})       "
                   f"{st['diff']:+6.1f} [{st['lo']:+6.1f}, {st['hi']:+6.1f}]  p = {st['p']:.3f}")
    out += ["", f"3. TABLE EDGE {100*EDGE:.0f}% BEYOND THE TARGET, NO PRACTICE: share of shots that leave the table; mean |miss| mm",
            "   source                plug-in            aware              edge-aware"]
    for s in sources:
        cells = []
        for rule in ("plug-in", "aware", "edge-aware"):
            rs = [r for r in rows if r["source"] == s and r["stage"] == "first" and r["rule"] == rule]
            cells.append(f"{100*np.mean([int(r['fell']) for r in rs]):3.0f}%  {np.mean([abs(float(r['miss_mm'])) for r in rs]):6.1f}")
        out.append(f"   {s:20s}  " + "       ".join(cells))
    out += ["", f"4. AFTER ONE WEAK PRACTICE SLIDE ({WEAK['peak']} m/s, {WEAK['noise']:g}x nominal noise, {len(ctx['seeds'])} seeds): mean |miss| (RMSE), mm",
            "   source                aware              plug-in           SD of ln mu: prior -> after the slide"]
    for s in sources:
        c = [cal[p][s] for p in pairs if s in cal[p]]
        out.append(f"   {s:20s}  {avg(mean_abs(rows, source=s, stage='weak1', rule='aware')):6.1f} ({rmse(rows, source=s, stage='weak1', rule='aware'):6.1f})    "
                   f"{avg(mean_abs(rows, source=s, stage='weak1', rule='plug-in')):6.1f} ({rmse(rows, source=s, stage='weak1', rule='plug-in'):6.1f})     "
                   f"{np.mean([x['prior_sd'] for x in c]):.2f} -> {np.nanmean([x['weak_sd'] for x in c]):.2f}")
    out += ["", "5. DIFFERENCES OVER PAIRS (a - b in mean |miss|, mm; bootstrap 95% interval over pairs; Wilcoxon signed-rank p)"]
    for title, stage in (("first shot, uncertainty-aware", "first"), ("after one weak practice slide, uncertainty-aware", "weak1")):
        out.append(f"   {title}:")
        base = dict(stage=stage, rule="aware")
        todo = [(s, "generic") for s in sources if s != "generic"]
        for name in ctx["models"]:
            todo += [(f"llm_{name}", f"generic@{name}"), (f"llm_{name}", "lookup"), (f"robust@{name}", f"llm_{name}"), (f"recal_{name}", f"llm_{name}"),
                     (f"llm_{name}", "population")]
        todo += [("robust@lookup", "lookup")]
        for a, b in todo:
            text = contrast(rows, dict(source=a, **base), dict(source=b, **base))
            if text:
                out.append("   " + text)
        out.append("")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", nargs="+", default=[f"claude={DATA/'llm_standin_claude_haiku.json'}"], metavar="NAME=CACHE")
    ap.add_argument("--truth", nargs="+", type=Path, default=list(R.TRUTH))
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--grid-points", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RESULTS/"side_base"/"prior_study")
    a = ap.parse_args()
    llms = dict(item.split("=", 1) for item in a.llm)
    ctx = setup(llms, a.truth, a.pairs, a.grid_points, a.seeds)
    job_ctx = {k: v for k, v in ctx.items() if k != "pairs"}
    with Pool(a.procs) as pool:
        parts = pool.map(_pair, [(p, job_ctx) for p in ctx["truth"]], chunksize=1)
    rows = [r for _, rs, _ in parts for r in rs]
    cal = {p: c for p, _, c in parts}
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/"rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    with (a.out/"intervals.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "source", "mu_true", "inside", "width", "interval_score", "log_score", "median_err", "prior_sd", "weak_sd"], lineterminator="\n")
        w.writeheader()
        w.writerows(dict(pair=p, source=s, mu_true=ctx["truth"][p], **{k: round(v, 5) for k, v in c.items()}) for p in cal for s, c in cal[p].items())
    text = summarize(ctx, rows, cal)
    (a.out/"summary.txt").write_text(text + "\n")
    print(text)

if __name__ == "__main__":
    main()

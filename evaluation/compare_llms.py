#!/usr/bin/env python3
"""
compare_llms.py -- the LLMs against the truth and against each other, on the pairs they have both answered.

run_pair_set.py scores each LLM as its own method in Drake.  This script compares the LLMs themselves, from their cached
answers alone, so a new model can be looked at in seconds, before any simulation:

  per LLM      how far its median is from the true friction (ln ratio: mean = bias, RMS = error), how often the truth is
               inside its stated 90% interval, how wide that interval is, and the first-shot miss its prior gives
  per pair     every LLM's median and interval next to the truth
  control      for each LLM, the generic prior's median with that LLM's own interval width (generic@<llm>): what its
               confidence alone would buy, without knowing the pair
  between two  on the pairs both answered: how closely their medians agree (ln ratio, correlation), which is nearer the
               truth more often, and the difference in first-shot miss

The first-shot miss here is planned with the evaluation's planner and prior (the raw lognormal read of the answer) and
scored with the release model (arm.rest_x), not Drake: the two agree to about 0.5 mm away from the board's seams
(evaluation/practice_study.py), which is small against first-shot misses of tens of millimetres.  The generic prior and
the lookup table are listed as reference rows.

Caches are run_pair_set.py's two formats; a pair an LLM has not answered is left out of that LLM's rows, and "between
two" uses only the pairs both have.  On the handbook pairs the lookup row holds the truth, and an LLM may recall the same
handbooks (friction/make_handbook_truth.py).

    python3 evaluation/compare_llms.py --llm claude=data/llm_standin_claude_haiku.json gpt=data/llm_cache_published_pilot.json
    python3 evaluation/compare_llms.py --llm ... --out results/side_base/pair_sets/llm_comparison.txt
"""
import argparse, itertools, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_arm_evaluation as E
import run_pair_set as R
from belief import loglik_lookup, logprior, lookup_observation, posterior, theta_grid
from demo_published_comparison import raw_llm_logprior
from run_arm_comparison_v2 import PRIOR_LOG_SD, PRIOR_MEDIAN

Z90 = 1.6448536269514722

def matched(name):
    """Name of the control row for an LLM: the generic prior's median with that LLM's own interval width, pair by pair."""
    return f"generic@{name}"

def first_shot(post, mu, targets, cap):
    """Mean |miss| over the targets, mm, of the shot planned under belief `post` when the friction is mu (release model)."""
    from arm import plan_peak_speed, rest_x
    return float(np.mean([abs(rest_x(plan_peak_speed(t, post, J=E.JERK, x0=E.X0, u_max=cap), mu, E.JERK, E.X0) - t) for t in targets]))*1e3

def collect(llms, truth_files=R.TRUTH, only=None, grid_points=2000):
    """dict(pairs, targets, cap, rows): rows[name][pair] = dict(median, low, high, miss) for every source that covers the
    pair; the sources are 'generic', 'lookup' and one per LLM."""
    from arm import target_grid
    pairs = R.load_pairs(truth_files, only)
    g = theta_grid(grid_points, math.log(.005), math.log(5.))
    generic = logprior(g, math.log(PRIOR_MEDIAN), PRIOR_LOG_SD)
    truth = {p: d["mu"] for p, d in pairs.items()}
    setup = E.arm_setup(SimpleNamespace(layout="side", peak_cap="auto", practice_peak="auto", targets=[0.6]),
                        {**truth, **{p: 0. for p in R.SURFACES if p not in truth}}, posterior(g, generic))
    cap, targets = setup["cap"], target_grid(setup["cap"], max(truth.values()), 3, E.JERK, E.X0)
    rows = {"generic": {}, "lookup": {}, **{name: {} for name in llms}, **{matched(name): {} for name in llms}}
    for p, d in pairs.items():
        post = posterior(g, generic)
        rows["generic"][p] = dict(median=PRIOR_MEDIAN, low=post["mu_q05"], high=post["mu_q95"], miss=first_shot(post, d["mu"], targets, cap))
        table = lookup_observation(p, str(E._file("friction_lookup.csv")))
        if table is not None:
            post = posterior(g, generic + loglik_lookup(g, *table))
            rows["lookup"][p] = dict(median=table[0], low=post["mu_q05"], high=post["mu_q95"], miss=first_shot(post, d["mu"], targets, cap))
        for name, path in llms.items():
            est = R.llm_answer(d["prompt"], path)
            if est is not None:
                post = posterior(g, raw_llm_logprior(g, est))
                rows[name][p] = dict(median=est["median"], low=est["low"], high=est["high"], miss=first_shot(post, d["mu"], targets, cap))
                sd = math.log(est["high"]/est["low"])/(2*Z90)       # the control: as confident as the LLM, knowing nothing about the pair
                post = posterior(g, logprior(g, math.log(PRIOR_MEDIAN), sd))
                rows[matched(name)][p] = dict(median=PRIOR_MEDIAN, low=PRIOR_MEDIAN*math.exp(-Z90*sd), high=PRIOR_MEDIAN*math.exp(Z90*sd),
                                              miss=first_shot(post, d["mu"], targets, cap))
    return dict(pairs=pairs, targets=targets, cap=cap, rows=rows, models={name: R.cache_model(path) for name, path in llms.items()})

def _stats(rows, pairs, ps):
    e = np.array([math.log(rows[p]["median"]/pairs[p]["mu"]) for p in ps])
    inside = sum(rows[p]["low"] <= pairs[p]["mu"] <= rows[p]["high"] for p in ps)
    width = float(np.mean([math.log(rows[p]["high"]/rows[p]["low"]) for p in ps]))
    return dict(n=len(ps), bias=float(e.mean()), rms=float(np.sqrt((e**2).mean())), worst=float(e[np.argmax(np.abs(e))]), inside=inside, width=width,
                miss=float(np.mean([rows[p]["miss"] for p in ps])))

def report(c):
    pairs, rows = c["pairs"], c["rows"]
    llms = [n for n in rows if n != "lookup" and not n.startswith("generic")]
    hb = [p for p in pairs if pairs[p]["file"] == "ground_truth_handbook.csv"]
    L = [f"{len(pairs)} pairs, friction {min(d['mu'] for d in pairs.values())}-{max(d['mu'] for d in pairs.values())}; first shots at targets {c['targets']} m, cap {c['cap']} m/s, "
         "scored with the release model", "LLMs: " + "; ".join(f"{n} = {c['models'][n]} ({len(rows[n])} pairs answered)" for n in llms), ""]

    # ---- per pair
    L.append(f"{'pair':5} {'object on surface':44} {'mu':>5} {'lookup':>7} " + " ".join(f"{n + ': median [90%]':>24}" for n in llms))
    for p, d in pairs.items():
        cells = []
        for n in llms:
            a = rows[n].get(p)
            cells.append(("-" if a is None else f"{a['median']:.2f} [{a['low']:.2f}, {a['high']:.2f}]{'' if a['low'] <= d['mu'] <= a['high'] else ' x':2}").rjust(24))
        look = rows["lookup"].get(p)
        L.append(f"{p:5} {(d['object'] + ' on ' + d['surface'])[:44]:44} {d['mu']:>5.2f} {('-' if look is None else format(look['median'], '.2f')):>7} " + " ".join(cells))
    L += ["   x: the truth is outside the stated 90% interval", ""]

    # ---- per source
    def block(title, ps_all):
        L.append(title)
        L.append(f"{'':16} {'pairs':>5} {'bias':>7} {'RMS':>6} {'worst':>7} {'truth in 90%':>13} {'width':>6} {'first shot, mm':>15}")
        for n in rows:
            ps = [p for p in ps_all if p in rows[n]]
            if not ps:
                continue
            s = _stats(rows[n], pairs, ps)
            cov = f"{s['inside']} of {s['n']}"
            L.append(f"{n:16} {s['n']:>5} {s['bias']:>+7.2f} {s['rms']:>6.2f} {s['worst']:>+7.2f} {cov:>13} {s['width']:>6.2f} {s['miss']:>15.1f}")
        L.append("")
    L.append("against the truth.  bias, RMS, worst: ln(median / true friction).  width: ln(high / low) of the 90% interval (generic and lookup: of the prior used)")
    L.append("generic@<llm>: the generic prior's median (0.30) with that LLM's interval width, pair by pair: as confident as the LLM, knowing nothing about the pair.")
    L.append("   An LLM's knowledge of the pair is the gap between its row and this one; the gap from 'generic' to this one is only the generic prior being wide.")
    block("  all pairs each source covers", list(pairs))
    if hb and len(hb) < len(pairs):
        block("  handbook pairs (the lookup row holds the truth; an LLM may recall the same handbooks)", hb)
        block("  other pairs", [p for p in pairs if p not in hb])
    common = [p for p in pairs if all(p in rows[n] for n in llms)]
    if len(llms) > 1 and 0 < len(common) < len(pairs):
        block(f"  the {len(common)} pairs every LLM answered ({', '.join(common)})", common)

    # ---- between LLMs
    for a, b in itertools.combinations(llms, 2):
        ps = [p for p in pairs if p in rows[a] and p in rows[b]]
        if not ps:
            L.append(f"{a} and {b}: no pair answered by both"); continue
        d = np.array([math.log(rows[a][p]["median"]/rows[b][p]["median"]) for p in ps])
        ea = np.array([abs(math.log(rows[a][p]["median"]/pairs[p]["mu"])) for p in ps])
        eb = np.array([abs(math.log(rows[b][p]["median"]/pairs[p]["mu"])) for p in ps])
        dm = np.array([rows[a][p]["miss"] - rows[b][p]["miss"] for p in ps])
        corr = float(np.corrcoef([math.log(rows[a][p]["median"]) for p in ps], [math.log(rows[b][p]["median"]) for p in ps])[0, 1]) if len(ps) > 2 else math.nan
        L.append(f"{a} and {b}, {len(ps)} pairs both answered: medians differ by ln ratio mean {d.mean():+.2f}, RMS {np.sqrt((d**2).mean()):.2f}"
                 + (f", correlation of ln medians {corr:.2f}" if math.isfinite(corr) else "")
                 + f"; {a} nearer the truth on {int((ea < eb).sum())}, {b} on {int((eb < ea).sum())}, tied {int((ea == eb).sum())}; "
                 f"first-shot miss {a} minus {b}: mean {dm.mean():+.1f} mm, {a} smaller on {int((dm < 0).sum())} of {len(ps)}")
        tm = np.array([math.log(pairs[p]["mu"]) for p in ps])
        if len(ps) > 2:
            L.append(f"   correlation of ln median with ln true friction: {a} {np.corrcoef([math.log(rows[a][p]['median']) for p in ps], tm)[0, 1]:.2f}, "
                     f"{b} {np.corrcoef([math.log(rows[b][p]['median']) for p in ps], tm)[0, 1]:.2f}")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llm", nargs="+", required=True, metavar="NAME=CACHE")
    ap.add_argument("--truth", nargs="+", type=Path, default=list(R.TRUTH))
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None, help="also write the report here")
    a = ap.parse_args()
    text = report(collect(dict(item.split("=", 1) for item in a.llm), a.truth, a.pairs))
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")

if __name__ == "__main__":
    main()

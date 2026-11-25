#!/usr/bin/env python3
"""
run_pair_set.py -- the method comparison on any set of pairs, with one or more LLMs.

run_arm_evaluation.py is written around Y01-Y03 (one object description, three surfaces, one LLM cache).  This runner
does the same comparison for whatever pairs its truth files list, so it can take the handbook pairs
(data/ground_truth_handbook.csv, friction/make_handbook_truth.py) and, later, the measured pairs.  It reuses the
evaluation's pieces unchanged: the generic prior, the lookup observation, the lognormal read of an LLM answer, the arm
setup rules (side base, computed speed cap, practice speed, targets derived from the highest-friction pair), the
observation model (run_arm_evaluation.make_observation at nominal noise), the belief update and the planner.  Every
scored shot is a Drake push.

  truth        one or more CSVs in the measured-friction format (docs/measured_friction_interface.md).  A row's prompt
               comes from its description_object / description_surface; Y01-Y03, whose file has no descriptions, take
               the ones the evaluation uses (demo_published_comparison).
  methods      generic; lookup (generic prior + the pair's row of friction_lookup.csv; generic alone when the row has
               no value); one llm_<name> per --llm NAME=CACHE; oracle (the true friction).
  LLM caches   either the project's format (llm_prior.cached_response: gpt-5.4-mini answers keyed by prompt and request
               settings; any OpenAI model, one file per model: --query-llm CACHE --model NAME) or a stand-in file
               {"model", "method", "entries": {sha256(prompt): {"prompt", "response"}}} holding another model's answers
               to the same prompts.  A pair missing from a cache stops the run; nothing is requested unless --query-llm
               is given.  Several --llm entries run side by side, each as its own method.
  practice     0, 1 and 2 slides; the practice push is one Drake push per pair at the practice speed, observed with
               fresh noise per seed.

On pairs whose truth IS the lookup value (the handbook pairs) the lookup method holds the answer: read it as a ceiling.

    python3 evaluation/run_pair_set.py --llm claude=data/llm_standin_claude_haiku.json --out results/side_base/pair_sets/handbook
    python3 evaluation/run_pair_set.py --query-llm data/llm_cache_handbook_gpt.json  # paid: fills a gpt-5.4-mini cache (needs OPENAI_API_KEY)
    python3 evaluation/run_pair_set.py --llm claude=data/llm_standin_claude_haiku.json gpt=data/llm_cache_handbook_gpt.json --out ...
    python3 evaluation/run_pair_set.py --summarize results/side_base/pair_sets/handbook
"""
import argparse, csv, hashlib, json, math, zlib
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paths import DATA
import run_arm_evaluation as E
from belief import loglik_lookup, logprior, lookup_observation, posterior, theta_grid, update_with_slides
from demo_published_comparison import OBJECT, SURFACES, raw_llm_logprior
from llm_prior import MODEL, build_prompt, cached_response, parse_response, query_llm
from run_arm_comparison_v2 import PRIOR_LOG_SD, PRIOR_MEDIAN

TRUTH = (DATA/"ground_truth_handbook.csv", DATA/"ground_truth_published.csv")
VERSION = "pair_set_v1"
SLIDES = (0, 1, 2)
FIELDS = ["pair", "seed", "target_x", "method", "slides", "mu_true", "mu_median", "mu_low", "mu_high", "peak_speed", "at_cap", "x_rest",
          "y_rest", "error_x_mm", "valid", "separated_or_limits"]
FAMILY = {**{f"L{i:02d}": "metal on metal" for i in (1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15)},
          **{f"L{i:02d}": "plastic on steel" for i in (5, 6, 7, 8)}, **{f"L{i:02d}": "wood" for i in (16, 17, 18)},
          **{f"L{i:02d}": "glass plate" for i in (19, 20)}, **{f"Y{i:02d}": "stainless on sheet (Yu)" for i in (1, 2, 3)}}

# ----------------------------------------------------------------------------- pairs, prompts, LLM answers

def load_pairs(paths=TRUTH, only=None):
    """{pair_id: dict(mu, object, surface, prompt, file)} from the truth files, in file order."""
    pairs = {}
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                pid = r["pair_id"]
                if only and pid not in only:
                    continue
                if pid in pairs:
                    raise ValueError(f"{pid} is in two truth files")
                d_obj, d_surf = r.get("description_object", ""), r.get("description_surface", "")
                if not (d_obj and d_surf):
                    if pid not in SURFACES:
                        raise ValueError(f"{pid} has no prompt description in {path}")
                    d_obj, d_surf = OBJECT, SURFACES[pid]
                pairs[pid] = dict(mu=float(r["mu_k_nominal"]), object=r["object_material"], surface=r["surface_material"],
                                  prompt=build_prompt(dict(description_object=d_obj, description_surface=d_surf)), file=Path(path).name)
    if only and set(only) - set(pairs):
        raise KeyError(f"not in the truth files: {sorted(set(only) - set(pairs))}")
    return pairs

def standin_key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()

def _api_entries(data, prompt):
    """Entries of a project-format cache (llm_prior.cached_response) that answer `prompt`, whatever model made them."""
    return [e for e in data.values() if isinstance(e, dict) and e.get("prompt") == prompt and "settings" in e]

def llm_answer(prompt, cache_path):
    """The validated answer (median, low, high, reasoning) to `prompt` in a cache of either format, or None.  A
    project-format cache is read by prompt, so it may hold any model's answers, but only one model per file."""
    data = json.loads(Path(cache_path).read_text())
    if "entries" in data:                                          # a stand-in file
        entry = data["entries"].get(standin_key(prompt))
        return None if entry is None else parse_response(entry["response"])
    hits = _api_entries(data, prompt)
    if len({e["settings"]["model"] for e in hits}) > 1:
        raise ValueError(f"{cache_path} holds answers from more than one model to the same prompt; keep one cache file per model")
    return parse_response(hits[-1]["response"]) if hits else None

def cache_model(cache_path):
    data = json.loads(Path(cache_path).read_text())
    if "entries" in data:
        return data.get("model", "unnamed stand-in")
    models = sorted({e["settings"]["model"] for e in data.values() if isinstance(e, dict) and "settings" in e})
    if len(models) != 1:
        raise ValueError(f"{cache_path}: expected answers from exactly one model, found {models or 'none'}")
    return models[0]

def query_missing(cache_path, pairs, model=None):
    """Fill a cache in the project's format: one paid request per pair that is not in it.  model: an OpenAI model name
    in place of llm_prior.MODEL (the request settings, and so the cache identity, carry the name)."""
    import llm_prior
    if model:
        llm_prior.MODEL = model
    if Path(cache_path).exists() and cache_model(cache_path) != llm_prior.MODEL:
        raise ValueError(f"{cache_path} holds {cache_model(cache_path)} answers; use another file for {llm_prior.MODEL}")
    missing = [p for p, d in pairs.items() if not Path(cache_path).exists() or llm_answer(d["prompt"], cache_path) is None]
    print(f"{len(pairs) - len(missing)} of {len(pairs)} pairs already cached in {cache_path}; requesting {len(missing)} from {llm_prior.MODEL}", flush=True)
    for p in missing:
        est = parse_response(cached_response(pairs[p]["prompt"], cache_path, query=query_llm))
        print(f"  {p}: median {est['median']}, 90% interval {est['low']}-{est['high']}", flush=True)

# ----------------------------------------------------------------------------- one pair, in one process

def _pair(job):
    pair, info, initials, cap, practice_peak, targets, options, seeds, grid_points = job
    from arm import plan_peak_speed
    g = theta_grid(grid_points, math.log(.005), math.log(5.))
    mu, rows, cache = info["mu"], [], {}

    def shot(u):
        if u not in cache:
            try:
                r, _, lim, valid = E.checked_execute(mu, u, **options)
                cache[u] = (float(r["x_rest"]), float(r["y_rest"]), bool(valid), f"separated={bool(r['separated'])} v={lim['v']:.2f} tau={lim['tau']:.2f}")
            except Exception as ex:
                cache[u] = (math.nan, math.nan, False, f"{type(ex).__name__}: {ex}")
        return cache[u]

    def score(seed, n, posts):
        for target in targets:
            for method in list(posts) + ["oracle"]:
                post = posts.get(method)
                u = plan_peak_speed(target, mu, J=E.JERK, x0=E.X0, u_max=cap, clip=True) if post is None else \
                    plan_peak_speed(target, post, J=E.JERK, x0=E.X0, u_max=cap)
                x, y, valid, note = shot(float(u))
                rows.append(dict(pair=pair, seed=seed, target_x=target, method=method, slides=n, mu_true=mu,
                                 mu_median=mu if post is None else round(post["mu_median"], 6), mu_low=mu if post is None else round(post["mu_q05"], 6),
                                 mu_high=mu if post is None else round(post["mu_q95"], 6), peak_speed=round(float(u), 6), at_cap=bool(u >= cap - 1e-4),
                                 x_rest=round(x, 6), y_rest=round(y, 6), error_x_mm=round(1e3*(x - target), 3), valid=valid, separated_or_limits=note))

    score(-1, 0, {m: posterior(g, lp) for m, lp in initials.items()})
    r, p, lim, valid = E.checked_execute(mu, practice_peak, **options)
    if not valid:
        raise RuntimeError(f"{pair}: invalid practice push at {practice_peak} m/s ({lim})")
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(pair.encode())]))
        obs = [E.make_observation(r, p, rng, "nominal") for _ in range(max(SLIDES))]
        for n in SLIDES[1:]:
            score(seed, n, {m: update_with_slides(g, lp, obs[:n]) for m, lp in initials.items()})
    return rows

def run(a):
    pairs = load_pairs(a.truth, a.pairs)
    g = theta_grid(a.grid_points, math.log(.005), math.log(5.))
    generic = logprior(g, math.log(PRIOR_MEDIAN), PRIOR_LOG_SD)
    llms = dict(item.split("=", 1) for item in a.llm)
    initials, answers, tables = {}, {}, {}
    for pid, d in pairs.items():
        table = lookup_observation(pid, str(E._file("friction_lookup.csv")))
        tables[pid] = table
        initials[pid] = dict(generic=generic, lookup=generic.copy() if table is None else generic + loglik_lookup(g, *table))
        for name, path in llms.items():
            est = llm_answer(d["prompt"], path)
            if est is None:
                raise ValueError(f"{pid} is not in the {name} cache ({path}); no request made")
            answers[(pid, name)] = {k: est[k] for k in ("median", "low", "high")}
            initials[pid][f"llm_{name}"] = raw_llm_logprior(g, est)
    truth = {p: d["mu"] for p, d in pairs.items()}
    setup = E.arm_setup(SimpleNamespace(layout="side", peak_cap="auto", practice_peak="auto", targets=[0.6]), {**truth, **{p: 0. for p in SURFACES if p not in truth}},
                        posterior(g, generic))
    from arm import target_grid
    mu_hi = max(truth.values())
    targets = target_grid(setup["cap"], mu_hi, a.n_targets, E.JERK, E.X0)
    files = ["run_pair_set.py", "run_arm_evaluation.py", "run_arm_comparison_v2.py", "demo_published_comparison.py", "llm_prior.py", "belief.py",
             "observe.py", "decide.py", "arm.py", "push.py", "physics.py", "scene.py", "drake_config.py", "sim_config.yaml", "friction_lookup.csv"]
    config = dict(version=VERSION, pairs=list(pairs), truth_files=[str(Path(p).name) for p in a.truth], seeds=a.seeds, slides=list(SLIDES),
                  targets=targets, targets_rule=f"arm.target_grid: x0 + 5 cm to the reach of mu = {mu_hi} (the highest in the set) at 90% of the cap",
                  x0=E.X0, peak_cap=setup["cap"], practice_peak=setup["practice"], jerk=E.JERK, arm=setup["record"], grid_points=a.grid_points,
                  generic_median=PRIOR_MEDIAN, generic_log_sd=PRIOR_LOG_SD, noise="nominal (2 mm frames, 0.5 mm rest and release points)",
                  loss="decide.loss_sq_rel (the planner's default)",
                  llm={name: dict(cache=str(Path(path).name), model=cache_model(path), read="raw approximate lognormal, uncalibrated") for name, path in llms.items()},
                  llm_answers={f"{p}:{n}": v for (p, n), v in answers.items()},
                  lookup={p: None if t is None else dict(mu_tab=t[0], s_tab_log=t[1]) for p, t in tables.items()},
                  pair_info={p: dict(mu=d["mu"], object=d["object"], surface=d["surface"], truth_file=d["file"], prompt=d["prompt"]) for p, d in pairs.items()},
                  scored_shots="Drake pushes; a push that does not release or breaks a planned joint limit is kept and flagged (valid = False)")
    config["sha256"] = {name: hashlib.sha256(E._file(name).read_bytes()).hexdigest() for name in files}
    config["sha256"].update({Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in list(a.truth) + list(llms.values())})
    if a.check_only:
        print(f"{len(pairs)} pairs, friction {min(truth.values())}-{max(truth.values())}; cap {setup['cap']} m/s, practice {setup['practice']} m/s, targets {targets}; "
              f"methods generic, lookup, {', '.join('llm_' + n for n in llms)}, oracle.  No Drake run, nothing written.")
        return
    a.out.mkdir(parents=True, exist_ok=False)
    E.write_json(a.out/"settings.json", config)
    jobs = [(p, pairs[p], initials[p], setup["cap"], setup["practice"], targets, setup["options"], a.seeds, a.grid_points) for p in pairs]
    with Pool(a.procs) as pool:
        rows = [r for part in pool.map(_pair, jobs, chunksize=1) for r in part]
    with (a.out/"results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    E.write_json(a.out/"status.json", dict(status="complete", rows=len(rows), invalid_rows=sum(not r["valid"] for r in rows)))
    print(summarize(a.out))

# ----------------------------------------------------------------------------- summary

def _mae(rows):
    return float(np.mean([abs(float(r["error_x_mm"])) for r in rows])) if rows else math.nan

def _rmse(rows):
    return float(np.sqrt(np.mean([float(r["error_x_mm"])**2 for r in rows]))) if rows else math.nan

def summarize(out):
    out = Path(out)
    s = json.loads((out/"settings.json").read_text())
    with (out/"results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    methods = [m for m in dict.fromkeys(r["method"] for r in rows) if m != "oracle"]
    llm = [m for m in methods if m.startswith("llm_")]
    pairs, info = s["pairs"], s["pair_info"]
    first = [r for r in rows if r["slides"] == "0"]
    sel = lambda rs, **kw: [r for r in rs if all(r[k] == v for k, v in kw.items())]
    L = [f"pair set: {len(pairs)} pairs, friction {min(i['mu'] for i in info.values())}-{max(i['mu'] for i in info.values())}; {s['arm']['layout']} base, cap {s['peak_cap']} m/s, "
         f"practice {s['practice_peak']} m/s, targets {s['targets']} m; seeds {s['seeds']}",
         "LLM: " + "; ".join(f"llm_{n} = {v['model']} ({v['cache']})" for n, v in s["llm"].items()),
         "truth files: " + ", ".join(s["truth_files"]) + ".  On pairs from ground_truth_handbook.csv the lookup method holds the truth: a ceiling, not a competitor.", ""]

    # ---- per pair
    head = f"{'pair':5} {'object on surface':46} {'mu':>5} {'lookup':>7} " + " ".join(f"{m[4:] + ' median [90%]':>24}" for m in llm) + " | first shot |error|, mm: " + " ".join(f"{m:>10}" for m in methods)
    L += ["before any practice, mean over the targets", head]
    for p in pairs:
        look = s["lookup"][p]
        est = " ".join(f"{a['median']:>7.2f} [{a['low']:.2f}, {a['high']:.2f}]{'' if a['low'] <= info[p]['mu'] <= a['high'] else ' x':2}".rjust(24)
                       for a in (s["llm_answers"][f"{p}:{m[4:]}"] for m in llm))
        L.append(f"{p:5} {(info[p]['object'] + ' on ' + info[p]['surface'])[:46]:46} {info[p]['mu']:>5.2f} {('-' if look is None else format(look['mu_tab'], '.2f')):>7} {est} |"
                 + " "*26 + " ".join(f"{_mae(sel(first, pair=p, method=m)):>10.1f}" for m in methods))
    L.append("   x: the truth is outside the LLM's stated 90% interval")
    L.append("")

    # ---- pooled, by group
    groups = {"all pairs": pairs}
    for fam in dict.fromkeys(FAMILY.get(p, "other") for p in pairs):
        groups[fam] = [p for p in pairs if FAMILY.get(p, "other") == fam]
    hb = [p for p in pairs if info[p]["truth_file"] == "ground_truth_handbook.csv"]
    if hb and len(hb) < len(pairs):
        groups = {"all pairs": pairs, "handbook pairs": hb, **{k: v for k, v in groups.items() if k != "all pairs"}}
    L.append("before any practice, by group: mean |error| mm (RMSE mm; mean |error| as % of the slide asked for)")
    L.append(f"{'group':26} {'pairs':>5} " + " ".join(f"{m:>26}" for m in methods + ["oracle"]))
    rel = lambda rs: float(np.mean([abs(float(r["error_x_mm"]))/(1e3*(float(r["target_x"]) - s["x0"])) for r in rs]))*100 if rs else math.nan
    for name, ps in groups.items():
        cells = []
        for m in methods + ["oracle"]:
            rs = [r for r in first if r["method"] == m and r["pair"] in ps]
            cells.append(f"{_mae(rs):7.1f} ({_rmse(rs):6.1f}; {rel(rs):5.1f}%)".rjust(26))
        L.append(f"{name:26} {len(ps):>5} " + " ".join(cells))
    L.append("")

    # ---- prior quality
    L.append("the priors against the truth (ln of the prior's median over the true friction)")
    L.append(f"{'':26} {'pairs':>5} {'mean':>8} {'RMS':>8} {'worst':>8}   {'truth inside stated 90% interval':>34}")
    def quality(label, med, inside, ps):
        e = np.array([math.log(med(p)/info[p]["mu"]) for p in ps])
        cov = "" if inside is None else f"{sum(inside(p) for p in ps)} of {len(ps)}"
        L.append(f"{label:26} {len(ps):>5} {e.mean():>+8.2f} {np.sqrt((e**2).mean()):>8.2f} {e[np.argmax(np.abs(e))]:>+8.2f}   {cov:>34}")
    for name, ps in (("all pairs", pairs), ("handbook pairs", hb)) if hb and len(hb) < len(pairs) else (("all pairs", pairs),):
        L.append(f"  {name}")
        quality("  generic (median 0.30)", lambda p: s["generic_median"], None, ps)
        tab = [p for p in ps if s["lookup"][p] is not None]
        quality("  lookup nominal", lambda p: s["lookup"][p]["mu_tab"], None, tab)
        for m in llm:
            a = lambda p, m=m: s["llm_answers"][f"{p}:{m[4:]}"]
            quality(f"  {m}", lambda p: a(p)["median"], lambda p: a(p)["low"] <= info[p]["mu"] <= a(p)["high"], ps)
    L.append("")

    # ---- paired: who is closer
    for m in llm:
        for other in ("generic",):
            for name, ps in groups.items():
                if name not in ("all pairs", "handbook pairs"):
                    continue
                d = np.array([_mae(sel(first, pair=p, method=m)) - _mae(sel(first, pair=p, method=other)) for p in ps])
                L.append(f"{m} against {other}, {name}: closer on {int((d < 0).sum())} of {len(ps)} pairs; mean change {d.mean():+.1f} mm, median {np.median(d):+.1f} mm")
    L.append("")

    # ---- after practice
    L.append("after practice (nominal noise): mean |error| mm, all pairs; and how far apart the methods' shots land")
    L.append(f"{'slides':>6} " + " ".join(f"{m:>12}" for m in methods + ["oracle"]) + f" {'shots apart':>12}")
    for n in s["slides"]:
        rs = sel(rows, slides=str(n))
        cells = {}
        for r in rs:
            if r["method"] != "oracle":
                cells.setdefault((r["pair"], r["seed"], r["target_x"]), []).append(float(r["x_rest"]))
        spread = float(np.mean([1e3*np.ptp(v) for v in cells.values()]))
        L.append(f"{n:>6} " + " ".join(f"{_mae(sel(rs, method=m)):>12.2f}" for m in methods + ["oracle"]) + f" {spread:>12.2f}")
    L.append("")
    bad = [r for r in rows if r["valid"] != "True"]
    capped = [r for r in rows if r["at_cap"] == "True"]
    L.append(f"{len(rows)} scored shots; {len(capped)} at the speed cap ({len([r for r in capped if r['slides'] == '0'])} before practice); {len(bad)} flagged invalid"
             + (": " + "; ".join(sorted({f"{r['pair']} {r['method']} at {float(r['peak_speed']):.3f} m/s ({r['separated_or_limits']})" for r in bad})[:8]) if bad else ""))
    text = "\n".join(L)
    (out/"summary.txt").write_text(text + "\n")
    return text

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", nargs="+", type=Path, default=list(TRUTH), help="truth files (measured-friction format)")
    ap.add_argument("--pairs", nargs="+", default=None, help="only these pair ids")
    ap.add_argument("--llm", nargs="+", default=[], metavar="NAME=CACHE", help="one LLM method per entry")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-targets", type=int, default=3)
    ap.add_argument("--grid-points", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--query-llm", type=Path, default=None, metavar="CACHE", help=f"paid: request answers for pairs missing from this cache, then stop")
    ap.add_argument("--model", default=None, help=f"OpenAI model for --query-llm (default {MODEL}); one cache file per model")
    ap.add_argument("--summarize", type=Path, default=None, metavar="RUN", help="rewrite and print the summary of a saved run")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.summarize:
        print(summarize(a.summarize)); return
    if a.query_llm:
        query_missing(a.query_llm, load_pairs(a.truth, a.pairs), a.model); return
    if a.out is None and not a.check_only:
        ap.error("--out is required for a run")
    run(a)

if __name__ == "__main__":
    main()

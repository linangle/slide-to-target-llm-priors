#!/usr/bin/env python3
"""
patchy_table_study.py -- the transition model, tested in Drake: what if the board's friction is not uniform?

The belief update assumes the friction a practice slide measures is the friction the scored shot will meet (the identity
transition of model_specification.tex, Sec. 3.6).  On a real board friction varies from place to place (Yu et al. report
a spatial SD of about 9-12% of the mean for the Y pairs' surfaces; 22% for polyurethane, which is not simulated), and the practice slide and the scored shot do not cross the same stretch: at
the default practice speed the practice slide runs from about x = 0.83 to 1.19 m, the scored shots from 0.57 to 0.94 m.

The scene already models this: scene.Table cuts the board into 2 in strips and gives each its own friction,
mu_i = mu exp(s z_i), z_i ~ N(0, 1), one draw per board (scene.strip_targets).  push_slide always builds the uniform
table, so this script swaps the scene builder it calls for one with a patchy board (patchy_board), without editing push.py;
with s = 0 the push is the evaluation's own, bit for bit.

For each pair, strip spread s and board (a seed):
  no practice      every method's first shot on that board
  practice         one Drake practice slide on the same board, observed at nominal noise, at the default practice speed
                   and at a slower one whose slide lies on the stretch the scored shots use
  unaware          the pipeline as it is: belief.update_with_slides (identity transition)
  told             the belief is told that friction differs between the practice slide and the shot by a log-normal
                   spread (practice_study.update_with_transfer): the spread measured across the boards of this very
                   study (transfer_sd.json: the SD of ln(shot friction / practice estimate)), 1.4 times that, and the
                   strip spread s for each slide, which is what a point-by-point friction survey of the board would
                   suggest (told_kernels)
  oracle           the pair's nominal friction: the average of the board, not the board

The check of the transition model itself is that SD: under the identity transition the friction a shot meets should
equal the practice estimate to within the posterior's width (about 0.006 in ln mu).  A shot's friction is read off its
landing with the release model (arm.rest_x inverted).

Every scored shot is a Drake push on the board in question.  The board's seams also shorten some slides by about 1 mm
RMS (evaluation/practice_study.py); the s = 0 row is that floor.

A second rule (--aimed) puts the practice slide where the shot goes: the practice push is the method's own first shot at
the target, and the next shot at that target is planned from it; a third shot from both (or from the second alone).  It
runs on the saved run's boards and is written beside it (aimed.csv, aimed_summary.txt); the saved results are not touched.

    python3 evaluation/patchy_table_study.py              # about an hour on 8 processes -> results/side_base/patchy_table/
    python3 evaluation/patchy_table_study.py --aimed      # about a quarter of an hour more
    python3 evaluation/patchy_table_study.py --aimed-pairs   # the aimed rule on run_pair_set.py's 23 pairs -> patchy_table/pair_set/
    python3 evaluation/patchy_table_study.py --summarize
"""
import argparse, contextlib, csv, json, math, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from paths import DATA, RESULTS
import practice_study as P
import run_arm_evaluation as E
from belief import posterior, update_with_slides

OUT = RESULTS/"side_base"/"patchy_table"
SPREADS = (0.0, 0.06, 0.12, 0.25)       # SD of ln mu from strip to strip
SLOW_PRACTICE = 0.7                     # m/s: a practice slide on the stretch the scored shots use
METHODS = P.METHODS
FIELDS = ["pair", "strip_sd", "board", "practice_peak", "slides", "inference", "method", "target_x", "mu_true", "mu_median", "post_sd",
          "peak_speed", "x_rest", "error_mm", "mu_shot", "valid"]

@contextlib.contextmanager
def patchy_board(s, board):
    """Within the block, push.push_slide builds the board with strip friction mu exp(s z_i) drawn from seed `board`
    (scene.strip_targets) instead of the uniform table.  s = 0 leaves the push exactly as it is."""
    import push, scene
    original = push.build_scene
    def build(builder, pair, obj, table, contact, **kw):
        return scene.build_scene(builder, {**pair, "s_tab_log": float(s)}, obj, scene.Table(patchiness=1.0), contact, seed=int(board), **kw)
    push.build_scene = build if s > 0 else original
    try:
        yield
    finally:
        push.build_scene = original

def board_seed(pair, board):
    """The scene seed of a pair's board number `board`: every pair gets its own boards (Y01 and Y02 have the same friction,
    and would otherwise slide on identical boards)."""
    return 1000*int(pair[1:]) + int(board)

def shot_friction(u, x_rest, lo=0.02, hi=3.0):
    """The uniform friction under which a push at u would have landed at x_rest (release model): the friction the shot met."""
    from arm import rest_x
    f = lambda mu: rest_x(u, mu, E.JERK, E.X0) - x_rest
    try:
        return float(brentq(f, lo, hi, xtol=1e-7))
    except ValueError:
        return math.nan

def _plan(target, post, cap):
    from arm import plan_peak_speed
    return plan_peak_speed(target, post, J=E.JERK, x0=E.X0, u_max=cap, **({} if isinstance(post, dict) else dict(clip=True)))

def _cell(job):
    """One pair on one board: first shots, then for each practice speed the unaware shots (stage 1) or the told shots (stage 2)."""
    pair, s, board, speeds, cap, targets, options, grid_points, kernels_sd = job
    truth, g, initials = P.priors(grid_points)
    mu, rows, cache = truth[pair], [], {}

    seed = board_seed(pair, board)

    def shot(u):
        if u not in cache:
            with patchy_board(s, seed):
                r, _, lim, valid = E.checked_execute(mu, u, **options)
            cache[u] = (float(r["x_rest"]), bool(valid))
        return cache[u]

    def score(practice_peak, n, inference, posts):
        for target in targets:
            for method, post in posts.items():
                u = float(_plan(target, post, cap))
                x, valid = shot(u)
                isd = isinstance(post, dict)
                rows.append(dict(pair=pair, strip_sd=s, board=board, practice_peak=practice_peak, slides=n, inference=inference, method=method, target_x=target,
                                 mu_true=mu, mu_median=round(post["mu_median"], 6) if isd else mu, post_sd=round(post["sd"], 5) if isd else "",
                                 peak_speed=round(u, 6), x_rest=round(x, 6), error_mm=round(1e3*(x - target), 3), mu_shot=round(shot_friction(u, x), 6), valid=valid))

    if kernels_sd is None:                                            # stage 1
        score("", 0, "none", {**{m: posterior(g, lp) for m, lp in initials[pair].items()}, "oracle": mu})
    for speed in speeds:
        with patchy_board(s, seed):
            r, p, lim, valid = E.checked_execute(mu, speed, **options)
        if not valid:
            raise RuntimeError(f"{pair} s={s} board {board}: invalid practice push at {speed} m/s ({lim})")
        clean = P.clean_observation(r, p)
        rng = np.random.default_rng(np.random.SeedSequence([board, zlib.crc32(pair.encode()), int(round(1e3*speed))]))
        obs = [P.noisy(clean, rng, 1.0)]
        if kernels_sd is None:
            score(speed, 1, "unaware", {m: update_with_slides(g, lp, obs) for m, lp in initials[pair].items()})
        else:
            for label, sd in kernels_sd[f"{s:g}|{speed:g}"].items():
                K = P.transfer_kernel(g, sd)
                score(speed, 1, label, {m: P.update_with_transfer(g, lp, obs, K) for m, lp in initials[pair].items()})
    return rows

def transfer_sd(rows):
    """{'s|speed': SD of ln(shot friction / practice estimate)} over the unaware shots: what the identity transition gets wrong."""
    out = {}
    for s in sorted({r["strip_sd"] for r in rows}):
        for speed in sorted({r["practice_peak"] for r in rows if r["practice_peak"] != ""}):
            d = [math.log(r["mu_shot"]/r["mu_median"]) for r in rows if r["strip_sd"] == s and r["practice_peak"] == speed and r["inference"] == "unaware"
                 and r["method"] == "generic" and math.isfinite(r["mu_shot"])]
            if d:
                out[f"{s:g}|{speed:g}"] = dict(sd=float(np.std(d)), mean=float(np.mean(d)), n=len(d))
    return out

def told_kernels(sd, only=None):
    """Per-slide spreads for the told beliefs, by 's|speed'.  update_with_transfer applies the spread twice (pair -> practice
    slide, pair -> shot), so a practice-to-shot spread of sigma is a per-slide spread of sigma / sqrt(2)."""
    out = {}
    for key, v in sd.items():
        s = float(key.split("|")[0])
        if s > 0:
            k = {"told: measured spread": v["sd"]/math.sqrt(2), "told: 1.4x measured": v["sd"], "told: strip spread per slide": s}
            out[key] = {name: k[name] for name in (only or k)}
    return out

def add_told(a, only=("told: measured spread",)):
    """Run told variants that a saved run lacks and append them to its results (the practice pushes are repeated; they
    are deterministic).  Rows saved under the labels of the first run are renamed to what they were."""
    old = {"told: measured spread": "told: 1.4x measured", "told: strip spread": "told: strip spread per slide"}
    s = json.loads((a.out/"settings.json").read_text())
    sd = json.loads((a.out/"transfer_sd.json").read_text())
    with (a.out/"results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if any(r["inference"] == "told: strip spread" for r in rows):      # a first-run file
        for r in rows:
            r["inference"] = old.get(r["inference"], r["inference"])
    have = {r["inference"] for r in rows}
    need = [name for name in only if name not in have]
    if need:
        arm = P.arm_defaults()
        cells = [(pair, x, b) for pair in s["pairs"] for x in s["strip_sd"] if x > 0 for b in range(s["boards_per_pair"])]
        jobs = [(*c, s["practice_speeds"], arm["cap"], arm["targets"], arm["options"], s["grid_points"], told_kernels(sd, need)) for c in cells]
        with Pool(a.procs) as pool:
            rows += [r for part in pool.map(_cell, jobs, chunksize=1) for r in part]
    with (a.out/"results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(summarize(a.out))

def run(a):
    arm = P.arm_defaults()
    speeds = [arm["practice"], SLOW_PRACTICE]
    boards = lambda s: [0] if s == 0 else list(range(a.boards))
    cells = [(pair, s, b) for pair in P.SURFACES for s in SPREADS for b in boards(s)]
    a.out.mkdir(parents=True, exist_ok=False)
    E.write_json(a.out/"settings.json", dict(
        study="patchy table: the transition model in Drake", pairs=list(P.SURFACES), strip_sd=list(SPREADS), boards_per_pair=a.boards, strip_length_m=0.0508,
        practice_speeds=speeds, peak_cap=arm["cap"], targets=arm["targets"], arm=arm["record"], noise="nominal", grid_points=a.grid_points,
        board_model="scene.strip_targets: mu_i = mu exp(s z_i), z_i ~ N(0, 1), 24 strips of 2 in, one draw per board seed; same board for practice and shots",
        loss="decide.loss_sq_rel (the planner's default)", shot_friction="arm.rest_x inverted at the push's commanded speed and its landing"))
    job = lambda c, k: (*c, speeds, arm["cap"], arm["targets"], arm["options"], a.grid_points, k)
    with Pool(a.procs) as pool:
        rows = [r for part in pool.map(_cell, [job(c, None) for c in cells], chunksize=1) for r in part]
        sd = transfer_sd(rows)
        E.write_json(a.out/"transfer_sd.json", sd)
        told = [job(c, told_kernels(sd)) for c in cells if c[1] > 0]
        rows += [r for part in pool.map(_cell, told, chunksize=1) for r in part]
    with (a.out/"results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(summarize(a.out))

# ----------------------------------------------------------------------------- practice aimed at the shot's own target

AIMED_FIELDS = ["pair", "strip_sd", "board", "attempt", "inference", "method", "target_x", "mu_true", "mu_median", "post_sd", "peak_speed",
                "x_rest", "error_mm", "mu_shot", "valid", "observed"]
AIMED = ("none", "1 aimed slide", "2 aimed slides", "2 aimed slides, last only")

def _aimed_rows(pair, mu, initials, s, seed, board, cap, targets, options, g, fixed=()):
    """The aimed rule for one pair on one board.  For every target and method (initials: {method: log prior}): attempt 1
    is the method's first shot, and is observed like a practice slide; attempt 2 is planned from that one slide; attempt 3
    from both slides (identity transition) or from the second alone (the slide nearest the stretch the shot will cross).
    So the practice slide crosses the strips the shot crosses, at the price of a shorter and so less informative slide for
    the near targets, and of every method practising with its own push.

    fixed: practice speeds for reference rows on the same board (generic prior only: after a fixed-speed slide the methods
    make the same shot), together with the plan made with the pair's nominal friction (method 'oracle')."""
    rows, cache = [], {}

    def push(u):
        if u not in cache:
            with patchy_board(s, seed):
                r, p, lim, valid = E.checked_execute(mu, u, **options)
            try:
                clean = P.clean_observation(r, p) if valid else None
            except ValueError:                                        # too short a slide to observe
                clean = None
            cache[u] = (float(r["x_rest"]), bool(valid), clean)
        return cache[u]

    def attempt(n, inference, method, target, post, seen):
        u = float(_plan(target, post, cap))
        x, valid, clean = push(u)
        isd = isinstance(post, dict)
        rows.append(dict(pair=pair, strip_sd=s, board=board, attempt=n, inference=inference, method=method, target_x=target, mu_true=mu,
                         mu_median=round(post["mu_median"], 6) if isd else mu, post_sd=round(post["sd"], 5) if isd else "", peak_speed=round(u, 6),
                         x_rest=round(x, 6), error_mm=round(1e3*(x - target), 3), mu_shot=round(shot_friction(u, x), 6), valid=valid, observed=seen))
        return clean

    for ti, target in enumerate(targets):
        for mi, (method, lp) in enumerate(initials.items()):
            rng = np.random.default_rng(np.random.SeedSequence([board, zlib.crc32(pair.encode()), ti, mi, 7]))
            belief = lambda obs: update_with_slides(g, lp, obs) if obs else posterior(g, lp)
            c1 = attempt(1, AIMED[0], method, target, belief([]), 0)
            o1 = [P.noisy(c1, rng, 1.0)] if c1 else []
            c2 = attempt(2, AIMED[1], method, target, belief(o1), len(o1))
            o2 = [P.noisy(c2, rng, 1.0)] if c2 else []
            attempt(3, AIMED[2], method, target, belief(o1 + o2), len(o1 + o2))
            attempt(3, AIMED[3], method, target, belief(o2 or o1), len(o2 or o1))
    if fixed:
        for target in targets:
            attempt(0, "nominal plan", "oracle", target, mu, 0)
    for speed in fixed:
        clean = push(float(speed))[2]
        if clean is None:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([board, zlib.crc32(pair.encode()), int(round(1e3*speed))]))
        post = update_with_slides(g, initials["generic"], [P.noisy(clean, rng, 1.0)])
        for target in targets:
            attempt(2, fixed_label(speed), "generic", target, post, 1)
    return rows

def fixed_label(speed):
    return f"fixed {float(speed):g} m/s"

def _aimed_cell(job):
    """One of the saved run's pairs on one of its boards (--aimed)."""
    pair, s, board, cap, targets, options, grid_points = job
    truth, g, initials = P.priors(grid_points)
    return _aimed_rows(pair, truth[pair], initials[pair], s, board_seed(pair, board), board, cap, targets, options, g)

def run_aimed(a):
    """Add the aimed-practice rule to a saved run, on the same boards: aimed.csv, aimed_settings.json, aimed_summary.txt."""
    s = json.loads((a.out/"settings.json").read_text())
    arm = P.arm_defaults()
    if [arm["cap"], arm["targets"]] != [s["peak_cap"], s["targets"]]:
        raise SystemExit("the arm setup differs from the saved run's")
    cells = [(pair, x, b) for pair in s["pairs"] for x in s["strip_sd"] for b in ([0] if x == 0 else range(s["boards_per_pair"]))]
    with Pool(a.procs) as pool:
        rows = [r for part in pool.map(_aimed_cell, [(*c, arm["cap"], arm["targets"], arm["options"], s["grid_points"]) for c in cells], chunksize=1) for r in part]
    E.write_json(a.out/"aimed_settings.json", dict(
        study="patchy table: practice aimed at the shot's own target", boards="those of settings.json", noise="nominal",
        rule="attempt 1 = the method's first shot at the target, observed as a practice slide; attempt 2 planned from it; attempt 3 from both slides "
             "(identity transition) or from the second alone", inference="belief.update_with_slides", attempts=list(AIMED)))
    with (a.out/"aimed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AIMED_FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(summarize_aimed(a.out))

def summarize_aimed(out):
    """aimed_summary.txt: the aimed rule next to the saved run's rows on the same boards."""
    out = Path(out)
    s = json.loads((out/"settings.json").read_text())
    with (out/"results.csv").open(newline="") as f:
        base = list(csv.DictReader(f))
    with (out/"aimed.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    mae = lambda rs: float(np.mean([abs(float(r["error_mm"])) for r in rs])) if rs else math.nan
    trip = lambda rs: " / ".join(f"{mae([r for r in rs if r['method'] == m]):.1f}" for m in METHODS)
    sel = lambda rs, **kw: [r for r in rs if all(r[k] == str(v) for k, v in kw.items())]
    spreads = sorted({r["strip_sd"] for r in rows}, key=float)
    speeds = sorted({r["practice_peak"] for r in base if r["practice_peak"]}, key=float, reverse=True)
    fixed = [(f"1 slide at {float(u):g} m/s", lambda x, u=u, **kw: sel(base, strip_sd=x, practice_peak=u, inference="unaware", **kw)) for u in speeds]
    aimed = [(k, lambda x, k=k, **kw: sel(rows, strip_sd=x, inference=k, **kw)) for k in AIMED[1:]]
    cols = fixed + aimed
    L = [f"practice aimed at the shot's own target, on the boards of the saved run: {len(s['pairs'])} pairs ({'/'.join(s['pairs'])}), {s['boards_per_pair']} boards per pair and "
         f"strip spread; targets {s['targets']} m, nominal noise",
         "aimed: the method's first shot at the target is observed as its practice slide, the next shot is planned from it; '2 aimed slides' observes that shot as well.",
         "'1 slide at ...': the saved run's fixed-speed practice (one push shared by all methods and targets).  mean |miss|, mm, as generic / lookup / LLM", ""]
    head = lambda first: f"{first:>12} | " + " | ".join(f"{name:>26}" for name, _ in cols)
    L.append(f"{'strip spread':>12} | {'no practice':>20} {'oracle (nominal mu)':>20} | " + " | ".join(f"{name:>26}" for name, _ in cols))
    for x in spreads:
        L.append(f"{float(x):>12g} | {trip(sel(rows, strip_sd=x, attempt=1)):>20} {mae(sel(base, strip_sd=x, slides=0, method='oracle')):>20.1f} | "
                 + " | ".join(f"{trip(get(x)):>26}" for _, get in cols))
    L.append("")

    L.append("by target (the three methods together): a near target means a short practice slide")
    for x in spreads:
        for t in sorted({r["target_x"] for r in rows}, key=float):
            L.append(f"{float(x):>12g} {float(t):>5g} m | {mae(sel(rows, strip_sd=x, attempt=1, target_x=t)):>20.1f} {mae(sel(base, strip_sd=x, slides=0, method='oracle', target_x=t)):>20.1f} | "
                     + " | ".join(f"{mae(get(x, target_x=t)):>26.1f}" for _, get in cols))
    L.append("")

    def ratio(rs):
        d = [math.log(float(r["mu_shot"])/float(r["mu_median"])) for r in rs if r["method"] == "generic" and r["mu_shot"] != "nan"]
        return f"{np.mean(d):+.3f}, {np.std(d):.3f}" if d else "-"
    L.append("the transition model: ln(friction the shot met / the belief's median before it), generic prior: mean, SD.  The identity transition says 0")
    L.append(head("strip spread"))
    for x in spreads:
        L.append(f"{float(x):>12g} | " + " | ".join(f"{ratio(get(x)):>26}" for _, get in cols))
    L.append("")
    def sd_only(rs):
        d = [math.log(float(r["mu_shot"])/float(r["mu_median"])) for r in rs if r["method"] == "generic" and r["mu_shot"] != "nan"]
        return f"{np.std(d):.3f}" if len(d) > 1 else "-"
    L.append("the same by target, SD only.  At the near target the aimed slide is a few centimetres long and says little, so the SD there is the estimate's noise, not the board")
    for x in spreads:
        for t in sorted({r["target_x"] for r in rows}, key=float):
            L.append(f"{float(x):>12g} {float(t):>5g} m | " + " | ".join(f"{sd_only(get(x, target_x=t)):>26}" for _, get in cols))
    L.append("")

    def by_board(rs):
        c = {}
        for r in rs:
            c.setdefault((r["pair"], r["board"]), []).append(abs(float(r["error_mm"])))
        return {k: float(np.mean(v)) for k, v in c.items()}
    def diff(a, b):
        v = np.array([a[k] - b[k] for k in sorted(set(a) & set(b))])
        return f"{v.mean():+.1f} +/- {v.std(ddof=1)/math.sqrt(len(v)):.1f}" if len(v) > 1 else "-"
    L.append("paired by board: difference in mean |miss|, mm, +/- its standard error over boards (negative: the first is better)")
    L.append(f"{'strip spread':>12} {'rule':>26} | " + " ".join(f"{'minus ' + name:>26}" for name, _ in fixed) + f" {'minus nominal-mu plan':>22} | {'lookup minus generic':>21} {'LLM minus generic':>18}")
    for x in spreads:
        for name, get in aimed:
            mine = by_board(get(x))
            L.append(f"{float(x):>12g} {name:>26} | " + " ".join(f"{diff(mine, by_board(g2(x))):>26}" for _, g2 in fixed)
                     + f" {diff(mine, by_board(sel(base, strip_sd=x, slides=0, method='oracle'))):>22} | "
                     + f"{diff(by_board(get(x, method='lookup')), by_board(get(x, method='generic'))):>21} {diff(by_board(get(x, method='llm_raw')), by_board(get(x, method='generic'))):>18}")
    L.append("")

    L.append("by pair (generic / lookup / LLM): after aimed practice the methods differ")
    L.append(head("spread, pair"))
    for x in spreads:
        for p in s["pairs"]:
            L.append(f"{float(x):>12g} {p:>7} | " + " | ".join(f"{trip(get(x, pair=p)):>26}" for _, get in cols))
    L.append("")

    def coverage(rs):
        ok = [abs(math.log(float(r["mu_shot"])/float(r["mu_median"]))) <= 1.6448536*float(r["post_sd"]) for r in rs
              if r["method"] == "generic" and r["post_sd"] and r["mu_shot"] != "nan"]
        return f"{100*np.mean(ok):.0f}%" if ok else "-"
    L.append("is the belief honest?  share of shots whose friction lies inside the belief's 90% interval (generic prior)")
    L.append(head("strip spread"))
    for x in spreads:
        L.append(f"{float(x):>12g} | " + " | ".join(f"{coverage(get(x)):>26}" for _, get in cols))
    L.append("")

    def apart(rs):
        cells = {}
        for r in rs:
            if r["method"] in METHODS:
                cells.setdefault((r["pair"], r["board"], r["target_x"]), []).append(float(r["x_rest"]))
        return f"{np.mean([1e3*np.ptp(v) for v in cells.values()]):.1f}" if cells else "-"
    L.append("how far apart the three methods' shots land, mm (mean over boards and targets)")
    L.append(head("strip spread"))
    for x in spreads:
        L.append(f"{float(x):>12g} | " + " | ".join(f"{apart(get(x)):>26}" for _, get in cols))
    L.append("")

    first = {(r["pair"], r["strip_sd"], r["board"], r["method"], r["target_x"]): r["x_rest"] for r in sel(base, slides=0)}
    same = sum(first.get((r["pair"], r["strip_sd"], r["board"], r["method"], r["target_x"])) == r["x_rest"] for r in sel(rows, attempt=1))
    L.append(f"check: {same} of {len(sel(rows, attempt=1))} first shots land where the saved run's no-practice shots landed")
    unseen = [r for r in rows if int(r["observed"]) < {"none": 0, AIMED[1]: 1, AIMED[2]: 2, AIMED[3]: 1}[r["inference"]]]
    L.append(f"{len(rows)} shots ({len(rows) - len(sel(rows, attempt=1))} after practice), {sum(r['valid'] != 'True' for r in rows)} flagged invalid, "
             f"{len(unseen)} planned with a practice slide too short to observe")
    text = "\n".join(L)
    (out/"aimed_summary.txt").write_text(text + "\n")
    return text

# ----------------------------------------------------------------------------- the aimed rule on the 23-pair set

SET_OUT = OUT/"pair_set"
SET_SPREADS = (0.0, 0.12, 0.25)
Z90 = 1.6448536269514722

def set_seed(pair, board):
    """Board seeds for the pair set: the Y pairs keep the saved run's boards, the handbook pairs get their own."""
    return board_seed(pair, board) + (0 if pair.startswith("Y") else 100000)

def set_priors(llm, cache, grid_points=2000, truth_files=None, only=None):
    """The pair set of run_pair_set.py with its priors: dict(pairs, g, initials, cap, targets, options, answers).  The
    methods are generic, lookup, llm_<name> and the control generic@<name>: the generic prior's median with that LLM's own
    interval width, pair by pair (as confident as the LLM, knowing nothing about the pair; evaluation/compare_llms.py)."""
    from types import SimpleNamespace
    import run_pair_set as R
    from arm import target_grid
    from belief import loglik_lookup, logprior, lookup_observation, theta_grid
    from demo_published_comparison import raw_llm_logprior
    from run_arm_comparison_v2 import PRIOR_LOG_SD, PRIOR_MEDIAN
    pairs = R.load_pairs(truth_files or R.TRUTH, only)
    g = theta_grid(grid_points, math.log(.005), math.log(5.))
    generic = logprior(g, math.log(PRIOR_MEDIAN), PRIOR_LOG_SD)
    initials, answers = {}, {}
    for pid, d in pairs.items():
        est = R.llm_answer(d["prompt"], cache)
        if est is None:
            raise ValueError(f"{pid} is not in the {llm} cache ({cache}); no request made")
        table = lookup_observation(pid, str(E._file("friction_lookup.csv")))
        answers[pid] = dict(median=est["median"], low=est["low"], high=est["high"], lookup=None if table is None else table[0])
        initials[pid] = {"generic": generic, "lookup": generic.copy() if table is None else generic + loglik_lookup(g, *table),
                         f"llm_{llm}": raw_llm_logprior(g, est),
                         f"generic@{llm}": logprior(g, math.log(PRIOR_MEDIAN), math.log(est["high"]/est["low"])/(2*Z90))}
    truth = {pid: d["mu"] for pid, d in pairs.items()}
    setup = E.arm_setup(SimpleNamespace(layout="side", peak_cap="auto", practice_peak="auto", targets=[0.6]),
                        {**truth, **{pid: 0. for pid in R.SURFACES if pid not in truth}}, posterior(g, generic))
    return dict(pairs=pairs, g=g, initials=initials, cap=setup["cap"], practice=setup["practice"], options=setup["options"], answers=answers,
                targets=target_grid(setup["cap"], max(truth.values()), 3, E.JERK, E.X0), model=R.cache_model(cache))

def _set_cell(job):
    pair, mu, initials, s, board, cap, targets, options, fixed, grid_points = job
    from belief import theta_grid
    return _aimed_rows(pair, mu, initials, s, set_seed(pair, board), board, cap, targets, options, theta_grid(grid_points, math.log(.005), math.log(5.)), fixed)

def run_aimed_set(a):
    """The aimed rule on run_pair_set.py's pairs (--aimed-pairs): does a prior that knows the pair win when practice has
    to be on the shot's own stretch?  Written to its own folder."""
    llm, cache = a.llm.split("=", 1)
    c = set_priors(llm, cache, a.grid_points)
    boards, fixed = a.boards or 8, [c["practice"], SLOW_PRACTICE]
    cells = [(pid, s, b) for pid in c["pairs"] for s in SET_SPREADS for b in ([0] if s == 0 else range(boards))]
    a.set_out.mkdir(parents=True, exist_ok=False)
    E.write_json(a.set_out/"settings.json", dict(
        study="patchy table: practice aimed at the shot's own target, on the pair set of run_pair_set.py", pairs=list(c["pairs"]),
        pair_info={pid: dict(mu=d["mu"], object=d["object"], surface=d["surface"], truth_file=d["file"]) for pid, d in c["pairs"].items()},
        answers=c["answers"], llm=dict(name=llm, cache=Path(cache).name, model=c["model"]), methods=list(next(iter(c["initials"].values()))),
        strip_sd=list(SET_SPREADS), boards_per_pair=boards, peak_cap=c["cap"], targets=c["targets"], fixed_practice=fixed, noise="nominal", grid_points=a.grid_points,
        attempts=list(AIMED), board_seeds="set_seed: the Y pairs' boards are the saved run's first boards",
        reference="generic prior after one fixed-speed practice slide, and the plan with the pair's nominal friction (oracle), same boards"))
    with Pool(a.procs) as pool:
        jobs = [(pid, c["pairs"][pid]["mu"], c["initials"][pid], s, b, c["cap"], c["targets"], c["options"], fixed, a.grid_points) for pid, s, b in cells]
        rows = []
        for i, part in enumerate(pool.imap(_set_cell, jobs, chunksize=1), 1):          # in order, with a progress line
            rows += part
            if i % 20 == 0 or i == len(jobs):
                print(f"{i} of {len(jobs)} boards done", file=sys.stderr, flush=True)
    with (a.set_out/"aimed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AIMED_FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(summarize_aimed_set(a.set_out))

def summarize_aimed_set(out):
    out = Path(out)
    s = json.loads((out/"settings.json").read_text())
    with (out/"aimed.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    methods, info = s["methods"], s["pair_info"]
    llm, ctrl = f"llm_{s['llm']['name']}", f"generic@{s['llm']['name']}"
    index = {}
    for r in rows:
        index.setdefault((r["strip_sd"], r["inference"], r["method"], r["pair"]), []).append(r)
    spreads = sorted({r["strip_sd"] for r in rows}, key=float)
    def miss(x, kind, method, pairs):
        v = [abs(float(r["error_mm"])) for p in pairs for r in index.get((x, kind, method, p), [])]
        return float(np.mean(v)) if v else math.nan
    def by_pair(x, kind, method, pairs):
        return np.array([miss(x, kind, method, [p]) for p in pairs])
    def diff(x, kind, a, b, pairs):
        v = by_pair(x, kind, a, pairs) - by_pair(x, kind, b, pairs)
        v = v[np.isfinite(v)]
        return f"{v.mean():+.1f} +/- {v.std(ddof=1)/math.sqrt(len(v)):.1f} ({int((v < 0).sum())}/{len(v)})" if len(v) > 1 else "-"
    def apart(x, kind, pairs):
        cells = {}
        for m in methods:
            for p in pairs:
                for r in index.get((x, kind, m, p), []):
                    cells.setdefault((p, r["board"], r["target_x"]), []).append(float(r["x_rest"]))
        return float(np.mean([1e3*np.ptp(v) for v in cells.values()])) if cells else math.nan
    hb = [p for p in s["pairs"] if info[p]["truth_file"] == "ground_truth_handbook.csv"]
    other = [p for p in s["pairs"] if p not in hb]
    mus = [info[p]["mu"] for p in s["pairs"]]
    L = [f"practice aimed at the shot's own target, pair set: {len(s['pairs'])} pairs (friction {min(mus)}-{max(mus)}), {s['boards_per_pair']} boards per pair and strip spread "
         f"(one at spread 0); targets {s['targets']} m, cap {s['peak_cap']} m/s, nominal noise; LLM {s['llm']['name']} = {s['llm']['model']}",
         f"{ctrl}: the generic prior's median with the LLM's interval width, pair by pair.  On the handbook pairs the lookup table holds the simulator's friction.",
         "mean |miss|, mm.  apart: how far apart the methods' shots land.  reference rows: generic prior after one fixed-speed slide; plan with the pair's nominal friction", ""]
    def block(title, pairs):
        L.append(title)
        L.append(f"{'strip spread':>12} {'rule':>27} | " + " ".join(f"{m:>15}" for m in methods) + f" | {'apart':>6}")
        for x in spreads:
            for kind in AIMED:
                L.append(f"{float(x):>12g} {('no practice' if kind == AIMED[0] else kind):>27} | " + " ".join(f"{miss(x, kind, m, pairs):>15.1f}" for m in methods)
                         + f" | {apart(x, kind, pairs):>6.1f}")
            L.append(f"{'':>12} {'reference':>27} | " + "; ".join(f"{fixed_label(u)} {miss(x, fixed_label(u), 'generic', pairs):.1f}" for u in s["fixed_practice"])
                     + f"; nominal plan {miss(x, 'nominal plan', 'oracle', pairs):.1f}")
        L.append("")
    block(f"all {len(s['pairs'])} pairs", s["pairs"])
    if hb and other:
        block(f"the {len(hb)} handbook pairs", hb)
        block(f"the other {len(other)} pairs ({', '.join(other)})", other)

    L.append("paired by pair: difference in mean |miss|, mm, +/- its standard error over pairs (pairs on which the first is better / pairs).  negative: the first is better")
    L.append(f"{'strip spread':>12} {'rule':>27} | {'lookup - generic':>24} {llm + ' - generic':>28} {ctrl + ' - generic':>30} {llm + ' - ' + ctrl:>36} "
             f"{'lookup - ' + ctrl:>32} {llm + ' - lookup':>26}")
    for x in spreads:
        for kind in AIMED:
            L.append(f"{float(x):>12g} {('no practice' if kind == AIMED[0] else kind):>27} | {diff(x, kind, 'lookup', 'generic', s['pairs']):>24} "
                     f"{diff(x, kind, llm, 'generic', s['pairs']):>28} {diff(x, kind, ctrl, 'generic', s['pairs']):>30} {diff(x, kind, llm, ctrl, s['pairs']):>36} "
                     f"{diff(x, kind, 'lookup', ctrl, s['pairs']):>32} {diff(x, kind, llm, 'lookup', s['pairs']):>26}")
    L.append("")
    near = lambda p: abs(math.log(s["answers"][p]["median"]/info[p]["mu"]))
    L.append(f"how near the medians are to the pair's friction, mean |ln(median / mu)|: {s['llm']['name']} {np.mean([near(p) for p in s['pairs']]):.2f}, "
             f"the generic median {np.mean([abs(math.log(s.get('generic_median', 0.2966)/info[p]['mu'])) for p in s['pairs']]):.2f}")
    L.append("")

    targets = sorted({r["target_x"] for r in rows}, key=float)
    def at(x, kind, method, target):
        v = [abs(float(r["error_mm"])) for p in s["pairs"] for r in index.get((x, kind, method, p), []) if r["target_x"] == target]
        return float(np.mean(v)) if v else math.nan
    def width(x, method, target):
        v = [float(r["post_sd"]) for p in s["pairs"] for r in index.get((x, AIMED[1], method, p), []) if r["target_x"] == target]
        return float(np.mean(v)) if v else math.nan
    L.append("by target, all pairs: mean |miss|, mm, per method; last columns: the belief's SD of ln mu after one aimed slide, per method")
    L.append(f"{'strip spread':>12} {'target':>7} {'rule':>27} | " + " ".join(f"{m:>15}" for m in methods))
    for x in spreads:
        for target in targets:
            for kind in AIMED[1:]:
                L.append(f"{float(x):>12g} {float(target):>5g} m {kind:>27} | " + " ".join(f"{at(x, kind, m, target):>15.1f}" for m in methods)
                         + (" | " + " ".join(f"{width(x, m, target):.2f}" for m in methods) if kind == AIMED[1] else ""))
    L.append("")

    for x in spreads:
        L.append(f"by pair, strip spread {float(x):g}: mean |miss|, mm, after one aimed slide, then after two (generic / lookup / LLM / control)")
        L.append(f"{'pair':5} {'object on surface':44} {'mu':>5} {'lookup':>7} {'LLM median [90%]':>19} | {'one aimed slide':>26} | {'two aimed slides':>26}")
        for p in s["pairs"]:
            a = s["answers"][p]
            look = "-" if a["lookup"] is None else format(a["lookup"], ".2f")
            L.append(f"{p:5} {(info[p]['object'] + ' on ' + info[p]['surface'])[:44]:44} {info[p]['mu']:>5.2f} {look:>7} "
                     f"{format(a['median'], '.2f') + ' [' + format(a['low'], '.2f') + ', ' + format(a['high'], '.2f') + ']':>19} | "
                     + " | ".join(f"{' / '.join(format(miss(x, kind, m, [p]), '.1f') for m in methods):>26}" for kind in AIMED[1:3]))
        L.append("")
    after = [r for r in rows if r["inference"] in AIMED[1:]]
    need = {AIMED[1]: 1, AIMED[2]: 2, AIMED[3]: 1}
    short = [r for r in after if int(r["observed"]) < need[r["inference"]]]
    count = lambda rs: ", ".join(f"{m} {sum(r['method'] == m for r in rs)}" for m in methods)
    L.append(f"{len(rows)} shots, {sum(r['valid'] != 'True' for r in rows)} flagged invalid by the push check ({count([r for r in rows if r['valid'] != 'True'])})")
    L.append(f"{len(short)} of {len(after)} shots after aimed practice were planned with a practice slide too short to observe, so from an unchanged belief ({count(short)})")
    text = "\n".join(L)
    (out/"summary.txt").write_text(text + "\n")
    return text

# ----------------------------------------------------------------------------- summary

def summarize(out):
    out = Path(out)
    s = json.loads((out/"settings.json").read_text())
    sd = json.loads((out/"transfer_sd.json").read_text())
    with (out/"results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    mae = lambda rs: float(np.mean([abs(float(r["error_mm"])) for r in rs])) if rs else math.nan
    sel = lambda **kw: [r for r in rows if all(r[k] == str(v) for k, v in kw.items())]
    spreads = sorted({r["strip_sd"] for r in rows}, key=float)
    speeds = sorted({r["practice_peak"] for r in rows if r["practice_peak"]}, key=float, reverse=True)
    L = [f"patchy table: {len(s['pairs'])} pairs ({'/'.join(s['pairs'])}), {s['boards_per_pair']} boards per pair and strip spread, strips of 2 in; cap {s['peak_cap']} m/s, targets {s['targets']} m, nominal noise",
         "strip spread: SD of ln mu from strip to strip (Yu et al.'s spatial SD over the mean for these pairs: about 0.09-0.12).  mean |miss|, mm, as generic / lookup / LLM", ""]
    trip = lambda rs: " / ".join(f"{mae([r for r in rs if r['method'] == m]):.1f}" for m in METHODS)
    L.append(f"{'strip spread':>12} | {'no practice':>20} {'oracle (nominal mu)':>20} | " + " | ".join(f"{'1 slide at ' + format(float(u), 'g') + ' m/s, unaware':>32}" for u in speeds))
    for x in spreads:
        L.append(f"{float(x):>12g} | {trip(sel(strip_sd=x, slides=0)):>20} {mae(sel(strip_sd=x, slides=0, method='oracle')):>20.1f} | "
                 + " | ".join(f"{trip(sel(strip_sd=x, practice_peak=u, inference='unaware')):>32}" for u in speeds))
    L.append("")
    L.append("the transition model: ln(friction the shot met / practice estimate), over boards, pairs and targets.  The identity transition says 0, to within about 0.006")
    L.append(f"{'strip spread':>12} | " + " | ".join(f"{'practice at ' + format(float(u), 'g') + ' m/s: mean, SD':>34}" for u in speeds))
    for x in spreads:
        cells = []
        for u in speeds:
            v = sd.get(f"{float(x):g}|{float(u):g}")
            cells.append(("-" if v is None else f"{v['mean']:+.3f}, {v['sd']:.3f}").rjust(34))
        L.append(f"{float(x):>12g} | " + " | ".join(cells))
    L.append("")
    told = sorted({r["inference"] for r in rows if r["inference"].startswith("told")})
    if told:
        L.append("telling the belief that friction differs between the practice slide and the shot (mean |miss|, mm, generic / lookup / LLM; shots apart, mm)")
        L.append(f"{'strip spread':>12} {'practice':>9} | {'unaware':>30} | " + " | ".join(f"{t:>36}" for t in told))
        def spread_mm(rs):
            cells = {}
            for r in rs:
                cells.setdefault((r["pair"], r["board"], r["target_x"]), []).append(float(r["x_rest"]))
            return float(np.mean([1e3*np.ptp(v) for v in cells.values()])) if cells else math.nan
        for x in spreads:
            if float(x) == 0:
                continue
            for u in speeds:
                base = [r for r in sel(strip_sd=x, practice_peak=u, inference="unaware") if r["method"] != "oracle"]
                L.append(f"{float(x):>12g} {float(u):>9g} | {trip(base) + f'; {spread_mm(base):.1f}':>30} | "
                         + " | ".join(f"{trip(sel(strip_sd=x, practice_peak=u, inference=t)) + f'; {spread_mm(sel(strip_sd=x, practice_peak=u, inference=t)):.1f}':>36}" for t in told))
        L.append("")
    L.append("is the belief honest?  share of shots whose friction lies inside the belief's 90% interval after one practice slide (generic prior; lognormal read: median, SD of ln mu)")
    def coverage(rs):
        ok = [abs(math.log(float(r["mu_shot"])/float(r["mu_median"]))) <= 1.6448536*float(r["post_sd"]) for r in rs if r["post_sd"] and r["mu_shot"] != "nan"]
        return f"{100*np.mean(ok):.0f}%" if ok else "-"
    kinds = ["unaware"] + told
    L.append(f"{'strip spread':>12} {'practice':>9} | " + " | ".join(f"{k:>30}" for k in kinds))
    for x in spreads:
        for u in speeds:
            L.append(f"{float(x):>12g} {float(u):>9g} | " + " | ".join(f"{coverage(sel(strip_sd=x, practice_peak=u, inference=k, method='generic')):>30}" for k in kinds))
    L.append("")
    L.append("by pair, one practice slide at the default speed, unaware (mean |miss|, mm, the three methods together) against the oracle's on the same boards")
    for x in spreads:
        L.append(f"   strip spread {float(x):g}: " + "; ".join(
            f"{p} {mae([r for r in sel(strip_sd=x, practice_peak=speeds[0], inference='unaware', pair=p)]):.1f} (oracle {mae(sel(strip_sd=x, slides=0, method='oracle', pair=p)):.1f})"
            for p in s["pairs"]))
    bad = [r for r in rows if r["valid"] != "True"]
    L.append("")
    L.append(f"{len(rows)} scored shots, {len(bad)} flagged invalid")
    text = "\n".join(L)
    (out/"summary.txt").write_text(text + "\n")
    return text

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boards", type=int, default=None, help="boards (seeds) per pair and strip spread (default 16; 8 for --aimed-pairs)")
    ap.add_argument("--grid-points", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--add-told", action="store_true", help="add the told variants a saved run lacks (stage 2 only)")
    ap.add_argument("--aimed", action="store_true", help="add the rule that aims the practice push at the shot's own target (same boards)")
    ap.add_argument("--aimed-pairs", action="store_true", help="the aimed rule on run_pair_set.py's 23 pairs, with the LLM of --llm")
    ap.add_argument("--llm", default=f"claude={DATA/'llm_standin_claude_haiku.json'}", metavar="NAME=CACHE")
    ap.add_argument("--set-out", type=Path, default=SET_OUT)
    a = ap.parse_args()
    if a.summarize:
        print(summarize(a.out))
        if (a.out/"aimed.csv").exists():
            print("\n" + summarize_aimed(a.out))
        if (a.set_out/"aimed.csv").exists():
            print("\n" + summarize_aimed_set(a.set_out))
        return
    if a.aimed_pairs:
        run_aimed_set(a); return
    a.boards = a.boards or 16
    if a.aimed:
        run_aimed(a); return
    if a.add_told:
        add_told(a); return
    run(a)

if __name__ == "__main__":
    main()

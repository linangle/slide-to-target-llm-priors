#!/usr/bin/env python3
"""
practice_study_summary.py -- tables and figure for evaluation/practice_study.py.

Reads the saved cells of a study folder and writes summary.txt, results.csv (one row per practice speed, noise multiple,
slide count and method) and summary.png.  Run through `python3 evaluation/practice_study.py --summarize`.

Two measures are reported side by side:
  mean |error|   the miss of the scored shot, mm, averaged over pairs, targets and seeds
  shot spread    how far apart the three methods' shots land when they are given the SAME observations (largest minus
                 smallest rest point over generic / lookup / LLM), averaged the same way.  It says whether the prior
                 still changes the decision, whichever method happens to be right on these three pairs.
"""
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from practice_study import METHODS, SLIDES, read_cells

NAMES = {"generic": "generic", "lookup": "lookup", "llm_raw": "LLM"}
SEPARATE_MM = 1.0        # the methods' shots count as apart when they land more than this far from each other on average

def _f(v):
    return float(v) if v not in ("", None) else math.nan

def _mean_abs(rows):
    return float(np.mean([abs(float(r["error_mm"])) for r in rows])) if rows else math.nan

def _group(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out

def _spread(rows, extra=()):
    """Mean over (pair, seed, target, ...) of the largest minus the smallest rest point among the methods, mm."""
    cells = _group(rows, "pair", "seed", "target_x", *extra)
    return float(np.mean([1e3*np.ptp([float(r["x_rest"]) for r in c]) for c in cells.values()])) if cells else math.nan

def _table(title, row_labels, col_labels, value, fmt="{:8.1f}", corner=""):
    L = [title, f"{corner:>12} |" + "".join(f"{c:>9}" for c in col_labels)]
    for r in row_labels:
        L.append(f"{r:>12} |" + "".join(" " + fmt.format(value(r, c)) for c in col_labels))
    return L

def summarize(out):
    out = Path(out)
    s = json.loads((out/"settings.json").read_text())
    cells = read_cells(out/"cells.csv.gz")
    ref = [r for r in cells if r["practice_peak"] == ""]
    sweep = [r for r in cells if r["practice_peak"] != ""]
    speeds = sorted({r["practice_peak"] for r in sweep}, key=float)
    noises = sorted({r["noise"] for r in sweep}, key=float)
    pairs = sorted({r["pair"] for r in sweep})
    n_seeds = len({r["seed"] for r in sweep})
    by = _group(sweep, "practice_peak", "noise", "slides")
    L = [f"practice informativeness study: {s['arm']['layout']} base, cap {s['peak_cap']} m/s, targets {s['targets']} m, pairs {'/'.join(pairs)}, "
         f"{n_seeds} noise seeds per cell",
         f"practice speeds {', '.join(f'{float(u):g}' for u in speeds)} m/s (the last is the default); noise in multiples of "
         f"{1e3*s['nominal_sd'][0]:g} mm frames, {1e3*s['nominal_sd'][1]:g} mm rest point, {1e3*s['nominal_sd'][2]:g} mm release point", ""]

    # ---- reference rows
    oracle = _mean_abs([r for r in ref if r["method"] == "oracle"])
    L.append("no practice (mean |error|, mm): " + ", ".join(f"{NAMES[m]} {_mean_abs([r for r in ref if r['method'] == m]):.1f}" for m in METHODS)
             + f"; shot spread {_spread([dict(r, seed=0) for r in ref if r['method'] != 'oracle']):.1f}; true-friction plan {oracle:.2f}")
    for p in pairs:
        L.append(f"   {p}: " + ", ".join(f"{NAMES[m]} {_mean_abs([r for r in ref if r['method'] == m and r['pair'] == p]):.1f}" for m in METHODS))
    L.append("")

    # ---- what one slide says, and the methods after one slide
    sd = lambda u, k: float(np.median([_f(r["sigma_data"]) for r in by[(u, k, "1")]]))
    cols = [f"x{float(k):g}" for k in noises]
    lab = {f"{float(u):g} m/s": u for u in speeds}
    col = dict(zip(cols, noises))
    L += _table("what one practice slide says on its own: SD of ln mu (flat prior), median over pairs and seeds   [prior SDs: generic "
                f"{s['generic_log_sd']:.2f}, lookup 0.33-0.51, LLM 0.26-0.30]", lab, cols, lambda r, c: sd(lab[r], col[c]), "{:8.3f}", "speed \\ noise") + [""]
    for n in SLIDES:
        L += _table(f"shot spread after {n} practice slide(s), mm (how far apart the three methods' shots land on the same observations)",
                    lab, cols, lambda r, c: _spread(by[(lab[r], col[c], str(n))]), "{:8.2f}", "speed \\ noise") + [""]
    for m in METHODS:
        L += _table(f"mean |error| after 1 practice slide, mm: {NAMES[m]}", lab, cols,
                    lambda r, c: _mean_abs([x for x in by[(lab[r], col[c], "1")] if x["method"] == m]), "{:8.2f}", "speed \\ noise") + [""]

    # ---- slides needed
    def needed(u, k):
        for n in SLIDES:
            if _spread(by[(u, k, str(n))]) < SEPARATE_MM:
                return n
        return math.inf
    L += _table(f"practice slides needed before the methods' shots land within {SEPARATE_MM:g} mm of each other (inf: more than {max(SLIDES)})",
                lab, cols, lambda r, c: needed(lab[r], col[c]), "{:8.0f}", "speed \\ noise") + [""]

    # ---- one axis: informativeness
    edges = [0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.8, math.inf]
    one = [r for r in sweep if r["slides"] == "1"]
    L.append("the two axes on one: cells binned by what the slide says on its own (SD of ln mu), after 1 practice slide")
    L.append(f"{'SD of ln mu':>14} | {'cells':>6} {'spread':>8} " + " ".join(f"{NAMES[m]:>8}" for m in METHODS) + "   (mean |error|, mm)")
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [r for r in one if lo <= _f(r["sigma_data"]) < hi]
        if not b:
            continue
        row = dict(lo=lo, hi=hi, n=len(b)//(3*len(s["targets"])), spread=_spread(b, ("practice_peak", "noise")),
                   sd=float(np.exp(np.mean(np.log([_f(r["sigma_data"]) for r in b])))),
                   **{m: _mean_abs([r for r in b if r["method"] == m]) for m in METHODS})
        bins.append(row)
        L.append(f"{lo:>6g}-{hi:<7g} | {row['n']:>6} {row['spread']:>8.2f} " + " ".join(f"{row[m]:>8.2f}" for m in METHODS))
    L.append("")

    # ---- per pair, three places on the grid
    L.append("by pair, mean |error| after 1 slide (generic / lookup / LLM), mm")
    for u, k in ((speeds[-1], "1.0"), (speeds[0], "1.0"), (speeds[0], noises[-1]), (speeds[-1], noises[-1])):
        L.append(f"   {float(u):g} m/s, noise x{float(k):g}: " + "; ".join(
            f"{p} " + " / ".join(f"{_mean_abs([r for r in by[(u, k, '1')] if r['method'] == m and r['pair'] == p]):.1f}" for m in METHODS) for p in pairs))
    L.append("")

    # ---- results.csv
    with (out/"results.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["practice_peak", "noise", "slides", "method", "mean_abs_error_mm", "mean_error_mm", "median_abs_error_mm",
                    "sd_ln_mu_one_slide", "posterior_sd_ln_mu", "shot_spread_mm", "cells"])
        for (u, k, n), rows in sorted(by.items(), key=lambda kv: (float(kv[0][0]), float(kv[0][1]), int(kv[0][2]))):
            for m in METHODS:
                r = [x for x in rows if x["method"] == m]
                e = np.array([float(x["error_mm"]) for x in r])
                w.writerow([u, k, n, m, round(float(np.abs(e).mean()), 3), round(float(e.mean()), 3), round(float(np.median(np.abs(e))), 3),
                            round(float(np.median([_f(x["sigma_data"]) for x in r])), 5), round(float(np.mean([_f(x["post_sd"]) for x in r])), 5),
                            round(_spread(rows), 3), len(r)])

    # ---- friction that varies between slides
    tr = None
    if (out/"transfer_cells.csv.gz").exists():
        tr = read_cells(out/"transfer_cells.csv.gz")
        tby = _group(tr, "transfer_sd", "inference", "slides")
        levels = sorted({r["transfer_sd"] for r in tr}, key=float)
        L.append(f"friction that differs from slide to slide (release-model world, {len({r['seed'] for r in tr})} seeds, practice at "
                 f"{s['default_practice_peak']} m/s, nominal noise): mean |error|, mm, as generic / lookup / LLM")
        L.append(f"{'SD of ln mu':>12} {'inference':>9} |" + "".join(f"{f'{n} slide(s)':>24}" for n in (0,) + tuple(SLIDES)) + f"{'spread, 1 slide':>18}")
        for lv in levels:
            for inf in ("unaware", "aware"):
                if (lv, inf, "1") not in tby:
                    continue
                L.append(f"{float(lv):>12g} {inf:>9} |" + "".join(
                    f"{' / '.join(f'{_mean_abs([r for r in tby[(lv, inf, str(n))] if r['method'] == m]):.1f}' for m in METHODS):>24}"
                    for n in (0,) + tuple(SLIDES)) + f"{_spread(tby[(lv, inf, '1')]):>18.2f}")
        L.append("   unaware: the pipeline as it is (the practice friction is taken to be the scored shot's); aware: the belief is told the spread")
        L.append("")

    # ---- checks
    if (out/"grid_check.json").exists():
        g = json.loads((out/"grid_check.json").read_text())
        L.append(f"belief-grid check at {g['practice_peak']} m/s (mean |error|, mm; true-friction plan {g['oracle_mean_abs_error_mm']}): "
                 + "; ".join(f"{k} {v}" for k, v in g["mean_abs_error_mm"].items() if "1 slide" in k))
        L.append(f"   grid spacing in ln mu: {g['grid_spacing_ln_mu']}")
    if (out/"map_check.json").exists():
        c = json.loads((out/"map_check.json").read_text())
        L.append("board seams (push_slide's uniform table is 24 equal strips): " + "; ".join(
            f"{p} {v['pushes_with_seam_loss']} of {v['points']} map pushes land short, RMS {v['seam_loss_rms_mm']} mm, worst {v['seam_loss_worst_mm']} mm at "
            f"{v['seam_loss_worst_at']} m/s" for p, v in c.items()))
    if (out/"verify.csv").exists():
        with (out/"verify.csv").open(newline="") as f:
            d = np.array([float(r["difference_mm"]) for r in csv.DictReader(f)])
        L.append(f"scored shots re-run in Drake ({len(d)} sampled): map minus Drake mean {d.mean():+.2f} mm, RMS {np.sqrt(np.mean(d**2)):.2f}, worst {d[np.argmax(np.abs(d))]:+.2f}; "
                 f"{int((np.abs(d) < 0.1).sum())} within 0.1 mm (the rest crossed a seam)")
    text = "\n".join(L)
    (out/"summary.txt").write_text(text + "\n")
    figure(out, s, speeds, noises, by, bins, tr, oracle)
    return text

# ----------------------------------------------------------------------------- figure

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COLOR = {"generic": "#2a78d6", "lookup": "#eb6834", "llm_raw": "#1baf7a"}      # validated together (dataviz palette, slots 1-3)
MARK = {"generic": "o", "lookup": "s", "llm_raw": "^"}
BLUES = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

def figure(out, s, speeds, noises, by, bins, tr, oracle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    # (a) shot spread after one slide over speed x noise
    a = ax[0, 0]
    Z = np.array([[_spread(by[(u, k, "1")]) for k in noises] for u in speeds])
    bounds = [0, 0.1, 0.3, 1, 3, 10, 30, 100, 1000]
    a.imshow(Z, cmap=ListedColormap(BLUES), norm=BoundaryNorm(bounds, len(BLUES)), origin="lower", aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            a.text(j, i, f"{Z[i, j]:.1f}" if Z[i, j] >= 0.95 else f"{Z[i, j]:.2f}", ha="center", va="center",
                   color="#ffffff" if Z[i, j] >= 3 else INK, fontsize=8.5)
    a.set_xticks(range(len(noises)), [f"×{float(k):g}" for k in noises]); a.set_yticks(range(len(speeds)), [f"{float(u):g}" for u in speeds])
    for e in np.arange(-0.5, len(noises)): a.axvline(e, color=SURFACE, lw=2)
    for e in np.arange(-0.5, len(speeds)): a.axhline(e, color=SURFACE, lw=2)
    a.set_xlabel("observation noise (multiple of nominal)"); a.set_ylabel("practice peak speed, m/s")
    a.set_title("How far apart the three methods' shots land\nafter one practice slide, mm")
    a.tick_params(length=0)
    # (b) error against what the slide says on its own
    b = ax[0, 1]
    x = [r["sd"] for r in bins]
    for m in METHODS:
        y = [r[m] for r in bins]
        b.plot(x, y, color=COLOR[m], marker=MARK[m], ms=6, lw=2, label=NAMES[m], markeredgecolor=SURFACE, markeredgewidth=1)
    b.plot(x, [r["spread"] for r in bins], color=MUTED, marker="D", ms=5, lw=2, ls=(0, (4, 2)), label="shots apart (largest − smallest)",
           markeredgecolor=SURFACE, markeredgewidth=1)
    b.axhline(oracle, color=MUTED, lw=1, ls=(0, (1, 2)))
    b.text(x[-1], oracle*1.15, f"true-friction plan, {oracle:.2f} mm", color=MUTED, fontsize=8, va="bottom", ha="right")
    b.set_xscale("log"); b.set_yscale("log")
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_xlabel("what one practice slide says on its own: SD of ln μ"); b.set_ylabel("after one practice slide, mm")
    b.set_title("Mean |error| follows what the slide says; the methods'\nshots move apart, their average error does not")
    b.legend(frameon=False, loc="upper left")
    # (c, d) friction that differs between slides
    for a_, inf, title in ((ax[1, 0], "unaware", "Friction differs between slides; the belief assumes it\ndoes not (the pipeline as it is)"),
                           (ax[1, 1], "aware", "Friction differs between slides and the belief\nis told by how much")):
        if tr is None:
            a_.axis("off"); continue
        tby = _group([r for r in tr if r["slides"] == "1"], "transfer_sd", "inference")
        levels = sorted({r["transfer_sd"] for r in tr}, key=float)
        ends = []
        for m in METHODS:
            xs, ys = [], []
            for lv in levels:
                rows = tby.get((lv, inf)) or (tby.get((lv, "unaware")) if float(lv) == 0 else None)
                if rows:
                    xs.append(float(lv)); ys.append(_mean_abs([r for r in rows if r["method"] == m]))
            a_.plot(xs, ys, color=COLOR[m], marker=MARK[m], ms=6, lw=2, label=NAMES[m], markeredgecolor=SURFACE, markeredgewidth=1)
            ends.append(ys[-1])
        a_.set_xlabel("slide-to-slide SD of ln μ"); a_.set_ylabel("mean |error| after one slide, mm")
        a_.set_title(title); a_.set_ylim(bottom=0); a_.margins(x=0.06)
        a_.text(0.97, 0.06, f"at {xs[-1]:g} the three methods are within\n{max(ends) - min(ends):.1f} mm of each other", transform=a_.transAxes,
                ha="right", va="bottom", color=MUTED, fontsize=8.5)
        a_.legend(frameon=False, loc="upper left")
    hi = max(a_.get_ylim()[1] for a_ in ax[1])
    for a_ in ax[1]:
        a_.set_ylim(0, hi)
    for a_ in (ax[0, 1], ax[1, 0], ax[1, 1]):
        a_.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True)
        for side in ("top", "right"): a_.spines[side].set_visible(False)
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    fig.savefig(out/"summary.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    from practice_study import OUT
    print(summarize(OUT))

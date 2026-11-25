#!/usr/bin/env python3
"""
paired_stats.py -- the statistics the proposal named: bootstrap confidence intervals over pairs and Wilcoxon signed-rank tests.

Every comparison in this project is paired: the methods see the same pairs, targets, seeds and practice slides.  The unit
that was sampled is the object-surface pair, so the pair is what is resampled and what the signed-rank test ranks.  Seeds
and targets are averaged inside a pair first; they are repeats of the same pair, not new evidence.

    paired(a, b)         mean of a - b over pairs, its bootstrap 95% interval (pairs resampled with replacement), the
                         Wilcoxon signed-rank p-value, and how many pairs each side wins
    from a saved run     python3 evaluation/paired_stats.py results/side_base/trials_nov21/pairs23_10seeds/onepiece_rel/results.csv
                         (a pair-set results.csv: mean |miss| per pair, method and number of practice slides, then every
                         method against the reference, --ref, by default the generic prior)

With n pairs the smallest two-sided signed-rank p-value is 2 / 2^n: 0.25 for the three Yu pairs, so three pairs cannot
reach significance whatever the data; 23 pairs can.
"""
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

def bootstrap_mean(x, n_boot=10000, level=0.95, seed=0):
    """Percentile bootstrap interval of the mean of x, resampling its entries (pairs) with replacement."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return tuple(float(q) for q in np.quantile(means, [(1 - level)/2, (1 + level)/2]))

def paired(a, b, n_boot=10000, seed=0):
    """a, b: one number per pair, in the same order (lower is better for both).  Returns dict(n, mean_a, mean_b, diff, lo, hi,
    p, a_wins, b_wins, ties): diff = mean(a - b), so a negative diff says a is better."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be one number per pair, in the same order")
    d = a - b
    lo, hi = bootstrap_mean(d, n_boot, seed=seed)
    nonzero = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 1 else 1.0
    return dict(n=len(d), mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(d.mean()), lo=lo, hi=hi, p=p,
                a_wins=int((d < -1e-12).sum()), b_wins=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()))

def line(name_a, name_b, s, unit="mm"):
    """One row of a comparison table."""
    ties = f" (ties {s['ties']})" if s["ties"] else ""
    return (f"{name_a:>18s} - {name_b:<18s} {s['diff']:+8.2f} {unit}  [{s['lo']:+7.2f}, {s['hi']:+7.2f}]   p = {s['p']:.4f}"
            f"   pairs won {s['a_wins']:2d} / {s['b_wins']:2d}{ties}")

def per_pair(rows, value="error_x_mm"):
    """{(method, slides): {pair: mean |value|}} from a pair-set results.csv, valid rows only."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if str(r.get("valid", "True")) == "True":
            acc[(r["method"], int(r["slides"]))][r["pair"]].append(abs(float(r[value])))
    return {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in acc.items()}

def report(path, ref="generic", value="error_x_mm"):
    with Path(path).open(newline="") as f:
        table = per_pair(list(csv.DictReader(f)), value)
    methods = sorted({m for m, _ in table}, key=lambda m: (m != ref, m))
    out = [f"{path}", f"mean |{value}| per pair, then method - {ref} over pairs: mean difference, bootstrap 95% interval, signed-rank p", ""]
    for slides in sorted({s for _, s in table}):
        if (ref, slides) not in table:
            continue
        pairs = sorted(table[(ref, slides)])
        out.append(f"{slides} practice slide(s), {len(pairs)} pairs; {ref} {np.mean([table[(ref, slides)][p] for p in pairs]):.2f} mm"
                   f"   (smallest possible p: {2/2**len(pairs):.2g})")
        for m in methods:
            if m == ref or (m, slides) not in table or m == "oracle":
                continue
            common = [p for p in pairs if p in table[(m, slides)]]
            s = paired([table[(m, slides)][p] for p in common], [table[(ref, slides)][p] for p in common])
            out.append("  " + line(m, ref, s))
        out.append("")
    return "\n".join(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="+", help="pair-set results.csv files")
    ap.add_argument("--ref", default="generic")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    text = "\n".join(report(p, a.ref) for p in a.results)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")

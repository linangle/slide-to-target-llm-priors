#!/usr/bin/env python3
"""
aggregate_pairs.py -- per-pair friction from the rig's per-run results and mu(x) profiles.

Input: the per-run results CSV(s) written by friction_analysis.py --out, the profiles CSV
written by --profiles, optionally the run log (temp_C, RH_pct per session) and a pair
metadata file whose identity columns (object_id, surface_id, materials, description_*,
photo_*) are copied over; ground_truth_placeholder.csv carries the identities.  Output:
measured_friction.csv in the contract of measured_friction_interface.md: the 21 original
columns, then the appended ones, one row per pair that has accepted runs.

Order of averaging (the protocol's):
  1. Within each lane and speed setting, forward and reverse orientations are averaged first,
         m[lane, speed] = (mean mu_fwd + mean mu_rev)/2,
     so the phone's mounting tilt, which biases the two orientations by +delta and -delta,
     cancels slide by slide.  A group with one orientation only is kept and reported.
  2. mu_k_nominal = mean of m[lane, speed] over the groups (equal weight per group).
  3. s_tab_log = sample SD of ln m[lane, speed] over the groups: the log-spread a sled sees
     between parts of the board and speeds, which is what the simulated patchy table draws
     from (strip mu = mu*exp(s*z)).  With fewer than two groups it falls back to the spatial
     log-SD of the pass-averaged profile.  interval90_low/high = mu*exp(-+1.645*s_tab_log).
  4. s_lane = sample SD of ln(lane means) over lanes, lane mean = mean over speeds of m.
  5. speed_effect = (m_fastest - m_slowest)/mu_k_nominal, speeds averaged over lanes,
     settings ordered by their number (s1 < s2 < ...).
  6. mu_slope_per_mps = mean over accepted runs of the per-run within-slide slope d mu/d v.
  7. se = SD of the group means/sqrt(n_groups); with one group, SD of the runs/sqrt(n_runs).
  8. mu_v2d = mean per-run endpoint estimate v_start^2/(2 g d_tape); v2d_ratio = mu_v2d/mu_k
     per run, its mean and SD being the stopping-law check and the endpoint noise.
  9. profile_repeatability = mean Pearson correlation between the mu(x) profiles of passes on
     the same lane over their common bins (at least 5), averaged over lanes.
 10. strip_scale_mm = lag at which the pass-averaged autocorrelation of the mean-removed
     profile first falls to 1/e, floored at the 100 mm coupon length (the coupon's own
     averaging correlates everything within one length) and capped at half the profile.
 11. n_rejected, reject_reasons = rejected runs and a "reason:count;..." list.

Usage:
    python3 aggregate_pairs.py --results results.csv --profiles profiles.csv \
        --runlog run_log.csv --meta ground_truth_placeholder.csv --out measured_friction.csv
"""
import argparse, csv, math, re
import numpy as np

COUPON_MM = 100.0
BASE_COLUMNS = ["pair_id", "object_id", "surface_id", "object_material", "surface_material", "mu_k_nominal",
                "s_tab_log", "interval90_low", "interval90_high", "sources", "n_runs", "se", "speed_effect",
                "load_effect", "temp_C", "RH_pct", "in_sim_range", "description_object", "description_surface",
                "photo_object", "photo_surface"]
NEW_COLUMNS = ["mu_v2d", "v2d_ratio_mean", "v2d_ratio_sd", "mu_slope_per_mps", "strip_scale_mm",
               "profile_repeatability", "n_rejected", "reject_reasons", "s_lane"]
META_COLUMNS = ["object_id", "surface_id", "object_material", "surface_material", "description_object",
                "description_surface", "photo_object", "photo_surface"]
SIM_RANGE = (0.10, 0.50)

def read_csv(path):
    """Rows as dicts; lines starting with # (the profiles note) are skipped."""
    with open(path, newline="") as f:
        return list(csv.DictReader([l for l in f if not l.startswith("#")]))

def ffloat(s):
    try: return float(s)
    except (TypeError, ValueError): return float("nan")

def fmt(x, nd):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

def speed_key(s):
    m = re.search(r"\d+", s or "")
    return (int(m.group()) if m else 0, s or "")

def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")

def orientation(r):
    return (r.get("orientation") or r.get("direction") or "fwd").strip().lower()

def pair_runs(runs):
    """Steps 1-8 on one pair's result rows.  Returns a dict of the friction columns."""
    acc = [r for r in runs if (r.get("accepted") or "").strip() == "1"]
    rej = [r for r in runs if r not in acc]
    out = {"n_runs": len(acc), "n_rejected": len(rej)}
    reasons = {}
    for r in rej:
        for s in re.split(r"[;,]", r.get("reject_reason") or r.get("flags") or "unknown"):
            if s.strip(): reasons[s.strip()] = reasons.get(s.strip(), 0) + 1
    out["reject_reasons"] = ";".join(f"{k}:{n}" for k, n in sorted(reasons.items(), key=lambda kv: -kv[1]))
    if not acc:
        return out
    groups = {}                                   # (lane, speed) -> orientation -> [mu]
    for r in acc:
        g = ((r.get("lane") or "").strip(), (r.get("speed_setting") or "").strip())
        groups.setdefault(g, {}).setdefault(orientation(r), []).append(float(r["mu_k"]))
    m = {g: float(np.mean([np.mean(v) for v in d.values()])) for g, d in groups.items()}   # step 1
    out["one_orientation"] = [g for g, d in groups.items() if len(d) < 2]
    mus = np.array(list(m.values()))
    mu = float(np.mean(mus))                                                                 # step 2
    out["mu_k_nominal"] = mu
    out["s_tab_log"] = sd(np.log(mus))                                                       # step 3
    lanes = {}
    for (lane, speed), v in m.items():
        lanes.setdefault(lane, []).append(v)
    out["s_lane"] = sd(np.log([np.mean(v) for v in lanes.values()]))                        # step 4
    speeds = {}
    for (lane, speed), v in m.items():
        speeds.setdefault(speed, []).append(v)
    if len(speeds) >= 2:                                                                     # step 5
        order = sorted(speeds, key=speed_key)
        out["speed_effect"] = (np.mean(speeds[order[-1]]) - np.mean(speeds[order[0]]))/mu
    else:
        out["speed_effect"] = float("nan")
    slopes = [ffloat(r.get("mu_slope_per_mps")) for r in acc]
    slopes = [s for s in slopes if not math.isnan(s)]
    out["mu_slope_per_mps"] = float(np.mean(slopes)) if slopes else float("nan")           # step 6
    run_mu = [float(r["mu_k"]) for r in acc]
    out["se"] = sd(mus)/math.sqrt(len(mus)) if len(mus) > 1 else sd(run_mu)/math.sqrt(len(run_mu))  # step 7
    v2d = [ffloat(r.get("mu_v2d")) for r in acc]
    ratio = [ffloat(r.get("v2d_ratio")) for r in acc]
    v2d = [x for x in v2d if not math.isnan(x)]; ratio = [x for x in ratio if not math.isnan(x)]
    out["mu_v2d"] = float(np.mean(v2d)) if v2d else float("nan")                            # step 8
    out["v2d_ratio_mean"] = float(np.mean(ratio)) if ratio else float("nan")
    out["v2d_ratio_sd"] = sd(ratio)
    out["sessions"] = sorted({(r.get("session") or "").strip() for r in acc} - {""})
    return out

def load_profiles(rows):
    """pair_id -> list of passes, each dict(lane, orientation, speed_setting, file, bin, mu) sorted by bin."""
    by = {}
    for r in rows:
        by.setdefault((r["pair_id"], r["file"]), {"lane": r.get("lane", ""), "orientation": orientation(r),
                                                   "speed_setting": r.get("speed_setting", ""), "file": r["file"],
                                                   "bin": [], "mu": []})
        by[(r["pair_id"], r["file"])]["bin"].append(float(r["bin_mm"]))
        by[(r["pair_id"], r["file"])]["mu"].append(float(r["mu"]))
    out = {}
    for (pid, _), p in by.items():
        o = np.argsort(p["bin"])
        p["bin"], p["mu"] = np.array(p["bin"])[o], np.array(p["mu"])[o]
        out.setdefault(pid, []).append(p)
    return out

def profile_repeatability(passes, min_common=5):
    """Step 9: mean pairwise correlation of passes on the same lane over their common bins."""
    rs = []
    lanes = {}
    for p in passes:
        lanes.setdefault(p["lane"], []).append(p)
    for ps in lanes.values():
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                common = np.intersect1d(ps[i]["bin"], ps[j]["bin"])
                if len(common) < min_common:
                    continue
                a = ps[i]["mu"][np.searchsorted(ps[i]["bin"], common)]
                b = ps[j]["mu"][np.searchsorted(ps[j]["bin"], common)]
                if np.std(a) > 0 and np.std(b) > 0:
                    rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")

def bin_width(passes):
    d = np.concatenate([np.diff(p["bin"]) for p in passes if len(p["bin"]) > 1])
    d = d[d > 0]
    return float(d.min()) if len(d) else 20.0

def strip_scale(passes, floor_mm=COUPON_MM):
    """Step 10: 1/e lag of the pass-averaged autocorrelation of the mean-removed profile, floored at the coupon."""
    acfs = []
    bw = bin_width(passes)
    for p in passes:
        if len(p["bin"]) < 6:
            continue
        grid = np.arange(p["bin"][0], p["bin"][-1] + bw/2, bw)     # regular grid, missing bins interpolated
        y = np.interp(grid, p["bin"], p["mu"]); y = y - y.mean()
        if np.sum(y*y) == 0:
            continue
        n = len(y)
        acfs.append(np.array([np.sum(y[:n-k]*y[k:]) for k in range(n//2 + 1)])/np.sum(y*y))
    if not acfs:
        return float("nan")
    L = min(len(a) for a in acfs)
    acf = np.mean([a[:L] for a in acfs], axis=0)
    below = np.flatnonzero(acf < 1/math.e)
    if len(below) == 0:
        lag = L - 1                                                  # never decorrelates: cap at half the profile
    else:
        k = int(below[0]); lag = k - 1 + (acf[k-1] - 1/math.e)/(acf[k-1] - acf[k])
    return max(floor_mm, lag*bw)

def profile_log_sd(passes):
    """Spatial log-SD of the pass-averaged profile, the fallback for s_tab_log with a single lane/speed group."""
    allb = np.unique(np.concatenate([p["bin"] for p in passes]))
    prof = np.array([np.mean([p["mu"][p["bin"] == b][0] for p in passes if b in p["bin"]]) for b in allb])
    prof = prof[prof > 0]
    return sd(np.log(prof))

def aggregate(results, profiles=None, runlog=None, meta=None):
    """All pairs -> list of output rows (dicts over BASE_COLUMNS + NEW_COLUMNS) and a notes list."""
    profiles = profiles or {}; runlog = runlog or []; meta = meta or {}
    rows, notes = [], []
    pids = sorted({(r.get("pair_id") or "").strip() for r in results} - {""})
    for pid in pids:
        p = pair_runs([r for r in results if (r.get("pair_id") or "").strip() == pid])
        if not p.get("n_runs"):
            notes.append(f"{pid}: no accepted runs ({p['n_rejected']} rejected: {p['reject_reasons']}), no row written")
            continue
        passes = profiles.get(pid, [])
        if math.isnan(p["s_tab_log"]):
            p["s_tab_log"] = profile_log_sd(passes) if passes else float("nan")
            notes.append(f"{pid}: one lane/speed group, s_tab_log from the profile's spatial log-SD")
        if p["one_orientation"]:
            notes.append(f"{pid}: one orientation only in lane/speed {p['one_orientation']}, tilt not cancelled")
        mu, s = p["mu_k_nominal"], p["s_tab_log"]
        lg = [r for r in runlog if (r.get("pair_id") or "").strip() == pid]
        env = {k: [ffloat(r.get(k)) for r in lg] for k in ("temp_C", "RH_pct")}
        env = {k: float(np.nanmean(v)) if v and not all(math.isnan(x) for x in v) else float("nan") for k, v in env.items()}
        md = meta.get(pid, {})
        row = {c: md.get(c, "") for c in META_COLUMNS}
        row.update({
            "pair_id": pid, "mu_k_nominal": fmt(mu, 3), "s_tab_log": fmt(s, 3),
            "interval90_low": fmt(mu*math.exp(-1.645*s), 3) if not math.isnan(s) else "",
            "interval90_high": fmt(mu*math.exp(1.645*s), 3) if not math.isnan(s) else "",
            "sources": "friction rig (phone-accelerometer sled)" + (", sessions " + " ".join(p["sessions"]) if p["sessions"] else ""),
            "n_runs": p["n_runs"], "se": fmt(p["se"], 4), "speed_effect": fmt(p["speed_effect"], 3), "load_effect": "",
            "temp_C": fmt(env["temp_C"], 1), "RH_pct": fmt(env["RH_pct"], 0),
            "in_sim_range": int(SIM_RANGE[0] <= mu <= SIM_RANGE[1]),
            "mu_v2d": fmt(p["mu_v2d"], 3), "v2d_ratio_mean": fmt(p["v2d_ratio_mean"], 3),
            "v2d_ratio_sd": fmt(p["v2d_ratio_sd"], 3), "mu_slope_per_mps": fmt(p["mu_slope_per_mps"], 4),
            "strip_scale_mm": fmt(strip_scale(passes), 0) if passes else "",
            "profile_repeatability": fmt(profile_repeatability(passes), 2) if passes else "",
            "n_rejected": p["n_rejected"], "reject_reasons": p["reject_reasons"], "s_lane": fmt(p["s_lane"], 3)})
        rows.append(row)
    missing = sorted(set(meta) - set(pids))
    if missing:
        notes.append("pairs in the metadata file without results: " + " ".join(missing))
    return rows, notes

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, help="per-run results CSV(s) from friction_analysis.py --out")
    ap.add_argument("--profiles", default="", help="profiles CSV from friction_analysis.py --profiles")
    ap.add_argument("--runlog", default="", help="run log (temp_C, RH_pct per pair); EXAMPLE rows skipped")
    ap.add_argument("--meta", default="", help="file with pair identities to copy (ground_truth_placeholder.csv)")
    ap.add_argument("--out", default="measured_friction.csv")
    a = ap.parse_args()
    results = [r for path in a.results for r in read_csv(path)]
    profiles = load_profiles(read_csv(a.profiles)) if a.profiles else {}
    runlog = [r for r in read_csv(a.runlog) if (r.get("session") or "").strip().upper() != "EXAMPLE"] if a.runlog else []
    meta = {r["pair_id"]: r for r in read_csv(a.meta)} if a.meta else {}
    rows, notes = aggregate(results, profiles, runlog, meta)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BASE_COLUMNS + NEW_COLUMNS)
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['pair_id']}: mu_k={r['mu_k_nominal']} s_tab_log={r['s_tab_log']} n={r['n_runs']} rejected={r['n_rejected']} "
              f"speed_effect={r['speed_effect'] or '-'} mu_v2d={r['mu_v2d'] or '-'} ratio={r['v2d_ratio_mean'] or '-'}"
              f"+-{r['v2d_ratio_sd'] or '-'} strip={r['strip_scale_mm'] or '-'}mm repeat={r['profile_repeatability'] or '-'}")
    for n in notes:
        print("note:", n)
    print(f"wrote {a.out}: {len(rows)} pairs, {len(BASE_COLUMNS)} + {len(NEW_COLUMNS)} columns")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_patchy_trial.py -- patchy_table_study.py with the arm setup of decision D9 (brake cap, acceleration check, posture
rule) set from outside, to see what the softer brake does on a board whose friction varies from strip to strip.  A brake
of a m/s^2 releases the object only where friction is below a / g (6 m/s^2: 0.61); on a patchy board a strip can be above
that although the pair's nominal friction is not.  No file an evaluation hashes is edited.

    python3 evaluation/run_patchy_trial.py [switches] -- <patchy_table_study.py options, with --out or --set-out>
    python3 evaluation/run_patchy_trial.py --census [--a-brake 6]        # which saved boards have a strip above the bound
    python3 evaluation/run_patchy_trial.py --compare OLD.csv NEW.csv      # the same boards under two arm setups, row by row
    python3 evaluation/run_patchy_trial.py --release-audit AIMED.csv --out FILE.csv [--strip-sd S ...] [--pairs ID ...]
                                           [--a-brake A] [--check-accel] [--posture NAME] [--procs N]
        # replays the distinct pushes behind a saved run's rows under the arm setup given and records, for each, whether the
        # object left the paddle before the paddle stopped (run_arm_evaluation.released); FILE_summary.txt counts the saved
        # rows that were scored although the object never released

--a-brake A      brake cap in m/s^2 (default: the arm's 12)
--check-accel    planned joint accelerations must be inside the SDF limits
--check-release  a push is valid only if the object left the paddle before the paddle stopped (run_arm_evaluation.released)
--posture NAME   speed (the arm's default) or limits
--peak-cap U     a fixed speed cap in m/s in place of the computed one, so that the cap, the practice speed and the
                 targets equal a saved run's (1.019 for the runs of 17 Nov) and only the brake and the posture differ
--pairs ID ...   with --aimed-pairs: only these pairs of the pair set (the targets are still derived from --target-mu)
--target-mu MU   with --pairs: the friction the targets are derived from (default 0.49, the pair set's highest), so a
                 subset keeps the full set's targets

The arm setup is computed in the parent process and travels to the workers inside each job, so nothing needs patching
there.  The run folder gets trial_override.json beside the study's own files.
"""
import argparse, csv, json, math, sys
from pathlib import Path
from types import SimpleNamespace

def census(a_brake, x_lo=0.55, x_hi=1.0):
    """Strips above the release bound on the boards of the saved studies (Y pairs: 16 boards; pair set: 8 boards)."""
    import numpy as np
    import patchy_table_study as S
    import run_pair_set as R
    import scene
    bound = a_brake/9.81
    table = scene.Table(patchiness=1.0)
    edges = table.edges()
    mid = (edges[:-1] + edges[1:])/2
    zone = (edges[1:] > x_lo) & (edges[:-1] < x_hi)          # strips the object can be on when the paddle brakes
    pairs = {p: d["mu"] for p, d in R.load_pairs(R.TRUTH).items()}
    L = [f"Strips above the release bound of a {a_brake:g} m/s^2 brake (friction {bound:.3f}), on the saved studies' boards.",
         f"A board counts if a strip between x = {x_lo:g} and {x_hi:g} m (where pushes release) is above the bound; 24 strips of 2 in per board.", "",
         f"{'pair':5} {'mu':>5} {'s':>5} {'boards':>6} {'boards with a strip above, in the release zone':>48} {'anywhere':>9} {'highest strip':>14}"]
    total = {}
    for pair, mu in sorted(pairs.items(), key=lambda kv: kv[1]):
        for s in (0.06, 0.12, 0.25):
            if pair.startswith("Y"):
                seeds = [S.board_seed(pair, b) for b in range(16)]
            elif s in S.SET_SPREADS:
                seeds = [S.set_seed(pair, b) for b in range(8)]
            else:
                continue
            strips = np.array([scene.strip_targets(mu, s, table, seed) for seed in seeds])
            in_zone = int(((strips > bound) & zone).any(axis=1).sum())
            anywhere = int((strips > bound).any(axis=1).sum())
            total[s] = total.get(s, 0) + in_zone
            if anywhere or mu >= 0.35:
                L.append(f"{pair:5} {mu:>5.2f} {s:>5.2f} {len(seeds):>6} {in_zone:>48} {anywhere:>9} {strips.max():>14.3f}")
    L += ["", "pairs not listed have no strip above the bound on any board.",
          "boards with a strip above the bound in the release zone, all pairs: " + ", ".join(f"s = {s:g}: {n}" for s, n in sorted(total.items()))]
    return "\n".join(L) + "\n"

def compare(old, new):
    """Row by row on the same boards: how many pushes become invalid, and how far the landings move."""
    def load(path):
        rows = list(csv.DictReader(open(path, newline="")))
        key = [k for k in ("pair", "strip_sd", "board", "practice_peak", "slides", "attempt", "inference", "method", "target_x") if k in rows[0]]
        return {tuple(r[k] for k in key): r for r in rows}, key
    A, key = load(old)
    B, _ = load(new)
    both = [k for k in A if k in B]
    L = [f"{old}  ->  {new}", f"{len(A)} and {len(B)} rows; {len(both)} on the same board, method, target and step", ""]
    i_s, i_pair = key.index("strip_sd"), key.index("pair")
    L.append(f"{'strip SD':>8} {'rows':>6} {'invalid before':>14} {'invalid now':>12} {'RMSE before':>12} {'RMSE now':>10} {'landing moved, RMS mm':>22} {'largest mm':>11}   (RMSE and moves over rows valid in both)")
    for s in sorted({k[i_s] for k in both}, key=float):
        ks = [k for k in both if k[i_s] == s]
        ok = [k for k in ks if A[k]["valid"] == "True" and B[k]["valid"] == "True"]
        rm = lambda R: math.sqrt(sum(float(R[k]["error_mm"])**2 for k in ok)/len(ok)) if ok else math.nan
        d = [1e3*(float(B[k]["x_rest"]) - float(A[k]["x_rest"])) for k in ok]
        L.append(f"{float(s):>8g} {len(ks):>6} {sum(A[k]['valid'] != 'True' for k in ks):>14} {sum(B[k]['valid'] != 'True' for k in ks):>12} {rm(A):>12.2f} {rm(B):>10.2f} "
                 f"{math.sqrt(sum(x*x for x in d)/len(d)) if d else math.nan:>22.3f} {max((abs(x) for x in d), default=math.nan):>11.2f}")
    bad = {}
    for k in both:
        if B[k]["valid"] != "True":
            c = bad.setdefault((k[i_pair], k[i_s]), dict(rows=0, boards=set(), mu=B[k]["mu_true"]))
            c["rows"] += 1; c["boards"].add(k[key.index("board")])
    if bad:
        L += ["", "invalid now, by pair and strip SD: rows (boards)"]
        for (pair, s), c in sorted(bad.items(), key=lambda kv: (float(kv[1]["mu"]), kv[0])):
            L.append(f"  {pair} (mu {float(c['mu']):.2f}), s = {float(s):g}: {c['rows']} rows on {len(c['boards'])} boards ({', '.join(sorted(c['boards'], key=int))})")
    else:
        L += ["", "no row is invalid under the new setup"]
    # rows scored in both runs whose landing moved by more than 5 mm, and practice slides that could not be observed
    L += ["", "rows valid in both whose landing moved by more than 5 mm (shorter / longer), and their boards by pair"]
    for s in sorted({k[i_s] for k in both}, key=float):
        ok = [k for k in both if k[i_s] == s and A[k]["valid"] == "True" and B[k]["valid"] == "True"]
        d = {k: 1e3*(float(B[k]["x_rest"]) - float(A[k]["x_rest"])) for k in ok}
        big = {k: v for k, v in d.items() if abs(v) > 5}
        boards = {}
        for k in big:
            boards.setdefault(k[i_pair], set()).add(k[key.index("board")])
        L.append(f"  s = {float(s):g}: {len(big)} of {len(ok)} ({sum(v < 0 for v in big.values())} / {sum(v > 0 for v in big.values())})"
                 + ("; " + ", ".join(f"{p} ({', '.join(sorted(b, key=int))})" for p, b in sorted(boards.items())) if boards else ""))
    if "observed" in next(iter(A.values())) and "inference" in key:
        i_inf, i_m = key.index("inference"), key.index("method")
        aimed = [k for k in both if "aimed" in k[i_inf]]
        L += ["", "shots after aimed practice planned from a practice slide that could not be observed: before, now, of"]
        for s in sorted({k[i_s] for k in aimed}, key=float):
            ks = [k for k in aimed if k[i_s] == s]
            L.append(f"  s = {float(s):g}: {sum(A[k]['observed'] == '0' for k in ks)}, {sum(B[k]['observed'] == '0' for k in ks)}, {len(ks)}")
        methods = [m for m in dict.fromkeys(k[i_m] for k in both) if m != "oracle"]
        L += ["", "mean |miss| mm over each run's valid rows, before -> now: " + " / ".join(methods)]
        for s in sorted({k[i_s] for k in both}, key=float):
            for inf in ("none", "1 aimed slide", "2 aimed slides"):
                def mean(R, m):
                    v = [abs(float(R[k]["error_mm"])) for k in both if k[i_s] == s and k[i_inf] == inf and k[i_m] == m and R[k]["valid"] == "True"]
                    return sum(v)/len(v) if v else math.nan
                if any(k[i_s] == s and k[i_inf] == inf for k in both):
                    L.append(f"  s = {float(s):g}, {inf:15}: " + " / ".join(f"{mean(A, m):.1f}" for m in methods) + "  ->  " + " / ".join(f"{mean(B, m):.1f}" for m in methods))
    return "\n".join(L) + "\n"

AUDIT_FIELDS = ["pair", "strip_sd", "board", "peak_speed", "mu_true", "x_rest", "valid", "separated", "released", "t_stop", "t_force_zero",
                "sep_lag_ms", "v_sep_force", "v_r_cmd", "lim_v", "lim_tau", "lim_a"]

def _audit_push(job):
    pair, mu, s, board, u, options = job
    import patchy_table_study as S
    import run_arm_evaluation as E
    with S.patchy_board(s, S.set_seed(pair, board)):
        r, _, lim, valid = E.checked_execute(mu, u, **options)
    return dict(pair=pair, strip_sd=s, board=board, peak_speed=u, mu_true=mu, x_rest=round(float(r["x_rest"]), 6), valid=bool(valid),
                separated=bool(r["separated"]), released=E.released(r), t_stop=round(float(r["t_stop"]), 5), t_force_zero=round(float(r["t_force_zero"]), 5),
                sep_lag_ms=round(float(r["sep_lag_ms"]), 2), v_sep_force=round(float(r["v_sep_force"]), 5), v_r_cmd=round(float(r["v_r_cmd"]), 5),
                lim_v=round(lim["v"], 3), lim_tau=round(lim["tau"], 3), lim_a=round(lim["a"], 3))

def release_audit(aimed, out, a, procs):
    """Replay every distinct push (pair, strip SD, board, peak speed) behind the rows of a saved aimed.csv."""
    import multiprocessing
    from types import SimpleNamespace as NS
    import practice_study as P
    import run_arm_evaluation as E
    rows = [r for r in csv.DictReader(open(aimed, newline="")) if (not a.strip_sd or float(r["strip_sd"]) in a.strip_sd) and (not a.pairs or r["pair"] in a.pairs)]
    truth, g, initials = P.priors()
    ns = NS(layout="side", peak_cap=1.0, practice_peak=1.0, targets=[0.6])
    if a.a_brake is not None: ns.a_brake = a.a_brake
    if a.check_accel: ns.check_accel = True
    if a.posture is not None: ns.posture = a.posture
    options = E.arm_setup(ns, truth, P.posterior(g, initials[next(iter(E.SURFACES))]["generic"]))["options"]
    jobs = sorted({(r["pair"], float(r["mu_true"]), float(r["strip_sd"]), int(r["board"]), float(r["peak_speed"])) for r in rows})
    print(f"{len(rows)} rows, {len(jobs)} distinct pushes, options {options}", flush=True)
    with multiprocessing.get_context("spawn").Pool(procs) as pool, out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS); w.writeheader()
        for i, row in enumerate(pool.imap(_audit_push, [(*j, options) for j in jobs], chunksize=4), 1):
            w.writerow(row)
            if i % 250 == 0:
                f.flush(); print(f"{i} of {len(jobs)} pushes", flush=True)
    text = audit_summary(aimed, out, options)
    out.with_name(out.stem + "_summary.txt").write_text(text)
    print(text, end="")

def audit_summary(aimed, audit, options=None):
    """The saved rows joined to the replayed pushes: how many were scored although the object never left the paddle."""
    P = {(r["pair"], float(r["strip_sd"]), int(r["board"]), round(float(r["peak_speed"]), 6)): r for r in csv.DictReader(open(audit, newline=""))}
    rows = [(r, P[k]) for r in csv.DictReader(open(aimed, newline=""))
            if (k := (r["pair"], float(r["strip_sd"]), int(r["board"]), round(float(r["peak_speed"]), 6))) in P]
    yes = lambda v: v == "True"
    L = [f"Release audit of {aimed}", f"replayed pushes: {audit}" + (f"; arm options {options}" if options else ""),
         "released = the paddle-object contact force reached zero before the commanded stop (run_arm_evaluation.released)", ""]
    dx = [abs(float(p["x_rest"]) - float(r["x_rest"])) for r, p in rows if r["x_rest"] not in ("", "nan")]
    L.append(f"{len(P)} distinct pushes behind {len(rows)} saved rows; the replay lands within {1e3*max(dx):.3f} mm of the saved rest position on every row"
             if dx else f"{len(P)} distinct pushes")
    L += ["", f"{'strip SD':>8} {'pushes':>7} {'not released':>13} {'rows':>6} {'saved valid':>12} {'saved valid, not released':>26} {'saved invalid, released':>24} {'saved invalid, not released':>28}"]
    for s in sorted({float(p["strip_sd"]) for p in P.values()}):
        ps = [p for p in P.values() if float(p["strip_sd"]) == s]
        rs = [(r, p) for r, p in rows if float(p["strip_sd"]) == s]
        n = lambda v, rel: sum(yes(r["valid"]) == v and yes(p["released"]) == rel for r, p in rs)
        L.append(f"{s:>8g} {len(ps):>7} {sum(not yes(p['released']) for p in ps):>13} {len(rs):>6} {sum(yes(r['valid']) for r, _ in rs):>12} {n(True, False):>26} {n(False, True):>24} {n(False, False):>28}")
    rel = [p for p in P.values() if yes(p["released"])]
    not_rel = [p for p in P.values() if not yes(p["released"])]
    lag = lambda p: 1e3*(float(p["t_force_zero"]) - float(p["t_stop"]))
    L += ["", "how far apart the two classes are (contact force zero relative to the commanded stop, ms; object speed at that moment, m/s)"]
    if rel:
        L.append(f"  released ({len(rel)}): force zero {min(map(lag, rel)):.0f} to {max(map(lag, rel)):.0f} ms; speed {min(float(p['v_sep_force']) for p in rel):.3f} to {max(float(p['v_sep_force']) for p in rel):.3f}")
    if not_rel:
        L.append(f"  not released ({len(not_rel)}): force zero {min(map(lag, not_rel)):+.0f} to {max(map(lag, not_rel)):+.0f} ms; speed {min(float(p['v_sep_force']) for p in not_rel):.4f} to {max(float(p['v_sep_force']) for p in not_rel):.4f}; "
                 f"of these {sum(yes(p['separated']) for p in not_rel)} have separated = True (apart at the end)")
        by = {}
        for p in not_rel:
            c = by.setdefault((p["pair"], float(p["strip_sd"])), dict(n=0, boards=set(), mu=float(p["mu_true"]))); c["n"] += 1; c["boards"].add(int(p["board"]))
        L += ["", "pushes not released, by pair and strip SD: pushes (boards)"]
        L += [f"  {pair} (mu {c['mu']:.2f}), s = {s:g}: {c['n']} on {len(c['boards'])} boards ({', '.join(map(str, sorted(c['boards'])))})" for (pair, s), c in sorted(by.items(), key=lambda kv: (kv[1]["mu"], kv[0]))]
    if rows and "inference" in rows[0][0]:
        methods = [m for m in dict.fromkeys(r["method"] for r, _ in rows) if m != "oracle"]
        L += ["", "mean |miss| mm: saved valid rows -> the same without the rows that never released: " + " / ".join(methods)]
        for s in sorted({float(p["strip_sd"]) for p in P.values()}):
            for inf in ("none", "1 aimed slide", "2 aimed slides"):
                def mean(m, need_release):
                    v = [abs(float(r["error_mm"])) for r, p in rows if float(p["strip_sd"]) == s and r["inference"] == inf and r["method"] == m and yes(r["valid"])
                         and (yes(p["released"]) or not need_release)]
                    return sum(v)/len(v) if v else math.nan
                L.append(f"  s = {s:g}, {inf:15}: " + " / ".join(f"{mean(m, False):.1f}" for m in methods) + "  ->  " + " / ".join(f"{mean(m, True):.1f}" for m in methods))
    return "\n".join(L) + "\n"

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--release-audit" in argv:
        p = argparse.ArgumentParser()
        p.add_argument("--release-audit", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--strip-sd", nargs="+", type=float, default=None)
        p.add_argument("--pairs", nargs="+", default=None)
        p.add_argument("--a-brake", type=float, default=None)
        p.add_argument("--check-accel", action="store_true")
        p.add_argument("--posture", choices=["speed", "limits"], default=None)
        p.add_argument("--procs", type=int, default=8)
        p.add_argument("--summary-only", action="store_true")
        a = p.parse_args(argv)
        if a.summary_only:
            text = audit_summary(a.release_audit, a.out)
            a.out.with_name(a.out.stem + "_summary.txt").write_text(text); print(text, end="")
        else:
            release_audit(a.release_audit, a.out, a, a.procs)
        return
    if "--census" in argv or "--compare" in argv:
        p = argparse.ArgumentParser()
        p.add_argument("--census", action="store_true")
        p.add_argument("--compare", nargs=2, type=Path)
        p.add_argument("--a-brake", type=float, default=6.0)
        p.add_argument("--out", type=Path, default=None)
        a = p.parse_args(argv)
        text = census(a.a_brake) if a.census else compare(*a.compare)
        print(text, end="")
        if a.out:
            a.out.write_text(text)
        return
    if "--" not in argv:
        sys.exit(__doc__)
    cut = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-brake", type=float, default=None)
    p.add_argument("--check-accel", action="store_true")
    p.add_argument("--check-release", action="store_true")
    p.add_argument("--posture", choices=["speed", "limits"], default=None)
    p.add_argument("--peak-cap", type=float, default=None)
    p.add_argument("--pairs", nargs="+", default=None)
    p.add_argument("--target-mu", type=float, default=0.49)
    a = p.parse_args(argv[:cut])
    rest = argv[cut + 1:]

    import arm
    import run_arm_evaluation as E
    import run_pair_set as R
    setup, load, grid = E.arm_setup, R.load_pairs, arm.target_grid

    def arm_setup(args, truth, post):
        ns = SimpleNamespace(**vars(args))
        if a.a_brake is not None: ns.a_brake = a.a_brake
        if a.check_accel: ns.check_accel = True
        if a.check_release: ns.check_release = True
        if a.posture is not None: ns.posture = a.posture
        if a.peak_cap is not None: ns.peak_cap = a.peak_cap
        return setup(ns, truth, post)

    E.arm_setup = arm_setup
    if a.pairs:
        R.load_pairs = lambda paths=R.TRUTH, only=None: load(paths, a.pairs)
        arm.target_grid = lambda cap, mu, n, J, x0: grid(cap, a.target_mu, n, J, x0)
    sys.argv = [sys.argv[0]] + rest
    import patchy_table_study as S
    try:
        S.main()
    finally:
        E.arm_setup, R.load_pairs, arm.target_grid = setup, load, grid
        for flag in ("--set-out", "--out"):
            if flag in rest:
                out = Path(rest[rest.index(flag) + 1])
                if out.is_dir():
                    (out/"trial_override.json").write_text(json.dumps(dict(
                        a_brake=a.a_brake, check_accel=a.check_accel, check_release=a.check_release, posture=a.posture, peak_cap=a.peak_cap, pairs=a.pairs,
                        target_mu=a.target_mu if a.pairs else None, study_options=rest,
                        note="set by evaluation/run_patchy_trial.py from outside patchy_table_study.py; boards (seeds) are the saved study's"), indent=2) + "\n")
                break

if __name__ == "__main__":
    main()

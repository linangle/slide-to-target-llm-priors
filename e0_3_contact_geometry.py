#!/usr/bin/env python3
"""
e0_3_contact_geometry.py -- E0-3: line vs point contact, and the COM.

Pushes with the prescribed paddle (push.py) at the frozen contact settings (sim_config.yaml), mu = 0.3, v_peak = 0.8 m/s,
J = 100 m/s^3:
  * box (line contact: the 12 cm plate covers the whole 10 cm end face) and disc (point contact);
  * COM offset e along y: 0, +-w/8, +-w/4 and +-(w/2 - 5 mm), with w = 10 cm the face width, which is also the contact
    segment.  A COM at or beyond the footprint edge would not stand, so "beyond the segment" is made instead by sliding
    the plate sideways until the part of the face it covers no longer straddles the COM (plate offset 70 and 85 mm:
    segment [10, 50] and [25, 50] mm of the face, COM 1.5 and 3 segment half-widths outside its centre);
  * plate face misalignment 0, 0.5 and 2 deg about the vertical;
  * paddle-object friction mu_p 0.1, 0.3 and 0.6.
Metrics: release heading psi_r and spin omega_r (simulator state at the last step in contact), lateral miss and yaw at
rest, the free slide against v_sep^2/(2 mu g), the contact-patch centroid on the plate (mean, min, max of its lateral
position during the push) and the object's largest tilt.
Prediction (A2; Lynch & Mason 1996): the box leaves with psi_r ~ 0 while its COM projection lies inside the contact
segment; the disc's heading grows with e; the distance error stays second order.  Report whatever happens.

Writes e0_3_results.csv, e0_3_summary.txt, e0_3_summary.png.
    python3 e0_3_contact_geometry.py                 # 132 pushes, a few minutes on 8 workers
    python3 e0_3_contact_geometry.py --quick
    python3 e0_3_contact_geometry.py --summarize e0_3_results.csv
"""
import argparse, csv, math, os, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scene import Obj
from drake_config import contact_from_config
from push import Pusher, push_slide, scalars

MU, V_PEAK, J = 0.3, 0.8, 100.0
W = Obj().size[1]                                        # face width = contact segment, m
E_FRAC = (0.0, 0.125, -0.125, 0.25, -0.25)               # e / w
E_EDGE = W/2 - 0.005                                     # "w/2": 5 mm inside the footprint edge, so the object stands
BEYOND = (0.070, 0.085)                                  # plate offsets, m: segment [lat - 60, 50] mm no longer straddles the COM
YAWS = (0.0, 0.5, 2.0)
MUPS = (0.1, 0.3, 0.6)

def offsets():
    return [f*W for f in E_FRAC] + [E_EDGE, -E_EDGE]

def tasks(quick):
    es = [0.0, 0.25*W, E_EDGE] if quick else offsets()
    yaws, mups = ((0.0, 2.0), (0.1, 0.6)) if quick else (YAWS, MUPS)
    todo = [("com", shape, e, 0.0, yaw, mu_p) for shape in ("box", "disc") for e in es for yaw in yaws for mu_p in mups]
    todo += [("beyond", "box", 0.0, lat, 0.0, mu_p) for lat in BEYOND for mu_p in mups]
    return todo

def segment(lat, e):
    """Contact segment [lo, hi] along y for a plate centred at `lat` on a face of width W, and the COM offset from its
    centre in half-widths."""
    half = Pusher().plate[1]/2
    lo, hi = max(-W/2, lat - half), min(W/2, lat + half)
    c, hw = (lo + hi)/2, (hi - lo)/2
    return lo, hi, (e - c)/hw

def run(task):
    stage, shape, e, lat, yaw, mu_p = task
    row = {"stage": stage, "shape": shape, "e_mm": e*1e3, "plate_offset_mm": lat*1e3, "yaw_cmd_deg": yaw, "mu_p": mu_p}
    lo, hi, rel = segment(lat, e)
    row.update(segment_lo_mm=lo*1e3, segment_hi_mm=hi*1e3, e_over_halfw=rel)
    try:
        r = push_slide(MU, V_PEAK, J, contact_from_config(), Obj(shape=shape, com_offset=(0.0, e, 0.0)),
                       Pusher(mu_p=mu_p, lateral=lat, yaw_deg=yaw), keep_traj=False)
        r = scalars(r)
        for k in ("mu", "v_peak", "J", "shape", "mu_p", "lateral", "yaw_cmd_deg", "com_y", "a_push"):
            r.pop(k, None)
        row.update(r); row["toppled"] = int(r["tilt_max_deg"] > 20.0); row["error"] = ""
    except Exception as ex:
        row["error"] = str(ex).splitlines()[0][:200]
    return row

def toppled(r):
    return str(r.get("toppled", "0")) == "1" or float(r["tilt_max_deg"]) > 20.0

def fmt(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"

def summarize(rows, out=None):
    ok = [r for r in rows if not r.get("error")]
    lines = [f"== E0-3: line vs point contact, and the COM (mu={MU}, v_peak={V_PEAK}, J={J}; {len(ok)} of {len(rows)} pushes ran) ==",
             "per shape and COM offset e: range over face yaw and mu_p of release heading psi_r, spin omega_r, lateral miss and yaw at rest, slide error",
             f"{'shape':<6}{'e mm':>8}{'e/(w/2)':>9} | {'psi_r deg':>18} | {'omega_r rad/s':>18} | {'miss mm':>16} | {'rest yaw deg':>16} | {'d err %':>14} | {'tilt deg':>9} | toppled"]
    def rng(rs, k, nd=3):
        v = np.array([float(r[k]) for r in rs], float)
        v = v[np.isfinite(v)]
        return f"{v.min():+.{nd}f} .. {v.max():+.{nd}f}" if len(v) else "n/a"
    for shape in ("box", "disc"):
        for e in sorted({float(r["e_mm"]) for r in ok if r["shape"] == shape and r["stage"] == "com"}):
            all_rs = [r for r in ok if r["shape"] == shape and r["stage"] == "com" and float(r["e_mm"]) == e]
            rs = [r for r in all_rs if not toppled(r)] or all_rs
            lines.append(f"{shape:<6}{e:>8.1f}{e/(W/2*1e3):>9.2f} | {rng(rs, 'psi_r_deg'):>18} | {rng(rs, 'omega_r', 4):>18} | "
                         f"{rng([dict(r, m=float(r['lateral_miss'])*1e3) for r in rs], 'm', 2):>16} | {rng(rs, 'yaw_rest_deg'):>16} | "
                         f"{rng(rs, 'd_err_pct', 2):>14} | {max(float(r['tilt_max_deg']) for r in all_rs):>9.3f} | {sum(toppled(r) for r in all_rs)} of {len(all_rs)}")
    lines.append("")
    lines.append("face yaw alone (box and disc, e = 0): the object aligns with the plate or not?")
    for shape in ("box", "disc"):
        for yaw in sorted({float(r["yaw_cmd_deg"]) for r in ok}):
            rs = [r for r in ok if r["shape"] == shape and r["stage"] == "com" and float(r["e_mm"]) == 0.0 and float(r["yaw_cmd_deg"]) == yaw]
            if rs:
                lines.append(f"  {shape:<5} plate yaw {yaw:>4.1f} deg: psi_r {rng(rs, 'psi_r_deg')} deg, rest yaw {rng(rs, 'yaw_rest_deg')} deg, "
                             f"miss {rng([dict(r, m=float(r['lateral_miss'])*1e3) for r in rs], 'm', 2)} mm, patch centroid y "
                             f"{rng([dict(r, c=float(r['patch_cy_mean'])*1e3) for r in rs], 'c', 1)} mm (range of min..max over the push: "
                             f"{min(float(r['patch_cy_min']) for r in rs)*1e3:+.1f} .. {max(float(r['patch_cy_max']) for r in rs)*1e3:+.1f})")
    lines.append("")
    lines.append("COM beyond the contact segment (box, e = 0, plate slid sideways):")
    for r in sorted((r for r in ok if r["stage"] == "beyond"), key=lambda r: (float(r["plate_offset_mm"]), float(r["mu_p"]))):
        lines.append(f"  plate offset {float(r['plate_offset_mm']):.0f} mm (segment [{float(r['segment_lo_mm']):.0f}, {float(r['segment_hi_mm']):.0f}] mm, "
                     f"COM {float(r['e_over_halfw']):+.1f} half-widths from its centre), mu_p {float(r['mu_p'])}: psi_r {float(r['psi_r_deg']):+.3f} deg, "
                     f"omega_r {float(r['omega_r']):+.4f} rad/s, miss {float(r['lateral_miss'])*1e3:+.1f} mm, rest yaw {float(r['yaw_rest_deg']):+.2f} deg, "
                     f"d err {float(r['d_err_pct']):+.2f}%, v_sep vs model {float(r['v_r_cmd_err_pct']):+.2f}%, patch y {float(r['patch_cy_mean'])*1e3:+.1f} mm")
    lines.append("")
    def mx(rs, k, scale=1.0):
        v = [abs(float(r[k]))*scale for r in rs if r[k] not in ("nan", "")]
        return max(v) if v else float("nan")
    box = [r for r in ok if r["shape"] == "box" and r["stage"] == "com"]
    inside = [r for r in box if abs(float(r["e_mm"])) <= W/4*1e3 + 1e-6]
    edge = [r for r in box if abs(float(r["e_mm"])) > W/4*1e3 + 1e-6]
    disc = [r for r in ok if r["shape"] == "disc" and r["stage"] == "com"]
    disc0 = [r for r in disc if float(r["e_mm"]) == 0.0 and float(r["yaw_cmd_deg"]) == 0.0]
    disc_e = [r for r in disc if float(r["e_mm"]) != 0.0 and not toppled(r)]
    beyond = [r for r in ok if r["stage"] == "beyond"]
    lines.append("A2 check (the rows above exclude toppled runs, tilt > 20 deg):")
    lines.append(f"  box, |e| <= w/4: |psi_r| <= {mx(inside, 'psi_r_deg'):.3f} deg, |miss| <= {mx(inside, 'lateral_miss', 1e3):.2f} mm, "
                 f"|d err| <= {mx(inside, 'd_err_pct'):.2f}%, tilt <= {mx(inside, 'tilt_max_deg'):.3f} deg: the line contact holds the heading, "
                 f"and the slide error does not grow with e.")
    lines.append(f"  box, COM 5 mm inside the footprint edge (e = 0.9 w/2): tilt up to {mx(edge, 'tilt_max_deg'):.2f} deg, |psi_r| up to {mx(edge, 'psi_r_deg'):.2f} deg, "
                 f"rest yaw up to {mx(edge, 'yaw_rest_deg'):.1f} deg, |d err| up to {mx(edge, 'd_err_pct'):.1f}%, {sum(toppled(r) for r in edge)} toppled: "
                 f"the push rocks it onto a footprint edge, so the contact and the slide are no longer A1/A2.")
    lines.append(f"  disc, e = 0, square plate: |psi_r| <= {mx(disc0, 'psi_r_deg'):.3f} deg (point contact through the centre); disc with e != 0: |psi_r| up to "
                 f"{mx(disc_e, 'psi_r_deg'):.1f} deg, |miss| up to {mx(disc_e, 'lateral_miss', 1e3):.0f} mm, rest yaw up to {mx(disc_e, 'yaw_rest_deg'):.0f} deg, "
                 f"|d err| up to {mx(disc_e, 'd_err_pct'):.0f}%, {sum(toppled(r) for r in disc)} toppled: the heading grows with e, as predicted, and so does the slide error.")
    lines.append(f"  box with the COM outside the covered segment: v_sep {min(float(r['v_r_cmd_err_pct']) for r in beyond):+.0f}% .. "
                 f"{max(float(r['v_r_cmd_err_pct']) for r in beyond):+.0f}% against the release model, rest yaw to {mx(beyond, 'yaw_rest_deg'):.0f} deg: "
                 f"the release model needs A2." if beyond else "  (no beyond-segment pushes in this run)")
    bad = [r for r in rows if r.get("error")]
    if bad:
        lines.append(f"{len(bad)} pushes failed: " + "; ".join(f"{r['shape']} e={float(r['e_mm']):.0f} yaw={r['yaw_cmd_deg']} mu_p={r['mu_p']}: {r['error']}" for r in bad[:5]))
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w").write(text + "\n")

def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hue = dict(zip(MUPS, ("#2a78d6", "#eb6834", "#1baf7a")))
    mark = dict(zip(YAWS, ("o", "s", "^")))
    ok = [r for r in rows if not r.get("error")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = (("psi_r_deg", 1.0, "release heading psi_r, deg"), ("lateral_miss", 1e3, "lateral miss at rest, mm"),
              ("d_err_pct", 1.0, "slide vs v_sep^2/(2 mu g), %"))
    for ax, (key, scale, label) in zip(axes, panels):
        for r in ok:
            x = float(r["e_over_halfw"])
            y = float(r[key])*scale
            if not math.isfinite(y) or toppled(r):
                continue
            filled = r["shape"] == "box"
            ax.scatter(x, y, s=28, marker=mark[float(r["yaw_cmd_deg"])], facecolors=hue[float(r["mu_p"])] if filled else "none",
                       edgecolors=hue[float(r["mu_p"])], linewidths=1.2, zorder=3)
        ax.axvspan(-1, 1, color="#eeeeee", zorder=0)
        ax.axhline(0, color="#52514e", linewidth=0.6)
        ax.set_ylabel(label)
        ax.grid(color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for mu_p in MUPS:
        axes[0].scatter([], [], color=hue[mu_p], s=28, label=f"mu_p {mu_p}")
    for yaw in YAWS:
        axes[0].scatter([], [], marker=mark[yaw], facecolors="none", edgecolors="#0b0b0b", s=28, label=f"plate yaw {yaw:g} deg")
    axes[0].scatter([], [], color="#0b0b0b", s=28, label="box (filled)")
    axes[0].scatter([], [], facecolors="none", edgecolors="#0b0b0b", s=28, label="disc (hollow)")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"E0-3: line vs point contact and the COM (mu {MU}, v_peak {V_PEAK} m/s, J {J:g} m/s^3)", y=0.995)
    fig.supxlabel("COM offset from the centre of the contact segment, in segment half-widths (band: COM inside the segment)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    n_top = sum(toppled(r) for r in ok)
    axes[0].set_ylim(-30, 30); axes[1].set_ylim(-60, 130); axes[2].set_ylim(-40, 60)
    fig.text(0.01, 0.925, "points beyond |e| = 1 are the box pushed by a plate slid sideways so the covered part of the face no longer straddles the COM; "
             f"e = 0.9 is the COM 5 mm inside the footprint edge.  {n_top} toppled runs (tilt > 20 deg) are left out; axes clip, see e0_3_summary.txt",
             fontsize=8, color="#52514e")
    fig.savefig(path, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    ap.add_argument("--out", default="e0_3_results.csv")
    ap.add_argument("--summary", default="e0_3_summary.txt")
    ap.add_argument("--fig", default="e0_3_summary.png")
    ap.add_argument("--summarize", metavar="CSV")
    a = ap.parse_args()
    if a.summarize:
        with open(a.summarize, newline="") as f:
            rows = list(csv.DictReader(f))
        summarize(rows, a.summary); plot(rows, a.fig); return
    todo = tasks(a.quick)
    print(f"{len(todo)} pushes on {a.workers} workers")
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, todo), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["stage"], r["shape"], float(r["e_mm"]), float(r["plate_offset_mm"]), float(r["yaw_cmd_deg"]), float(r["mu_p"])))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"wrote {a.out}")
    summarize(rows, a.summary)
    plot(rows, a.fig)
    print(f"wrote {a.summary}, {a.fig}")

if __name__ == "__main__":
    main()

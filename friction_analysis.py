#!/usr/bin/env python3
"""
friction_analysis.py -- kinetic friction from a phone-accelerometer sled.

Reads phyphox CSV exports ("Acceleration with g": columns Time (s), X, Y, Z in m/s^2,
optionally with gyroscope columns if a custom experiment is used), finds the free-slide
segment, and computes mu_k from the ratio of the along-slide to the normal specific force.

Physics.  An accelerometer reads the specific force f = a - g.  For a body sliding on a
plane, the component of f along the slide direction is -mu*N/m = -mu*g*cos(phi) and the
component normal to the plane is +g*cos(phi), whatever the plane's tilt phi.  So
    mu_k = -f_along / f_normal
during the slide, independent of the tilt, the launch speed, and g.  The only geometric
error is misalignment delta between the phone axes and the plane, which adds about
delta*(1+mu^2); sliding in both directions and averaging cancels it to first order.

Usage:
    python3 friction_analysis.py RUN1.csv RUN2.csv ... --pair P01 --object "PLA block" \
        --surface "melamine" --method flat --direction fwd --out results.csv
    python3 friction_analysis.py --selftest      # synthetic check, no files needed

Axis convention: by default the phone's long axis (phyphox Y) is the slide direction and
Z is normal to the screen.  Change with --along/--normal if the phone is mounted differently.
"""
import argparse, csv, math, sys
import numpy as np

G = 9.80665

def load_phyphox(path):
    """Return t, ax, ay, az (and gx, gy, gz if present) as float arrays."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip().lower() for h in rows[0]]
    data = np.array([[float(x) for x in r[:len(header)]] for r in rows[1:] if r and r[0] != ""])
    def col(*names):
        for n in names:
            for i, h in enumerate(header):
                if h.startswith(n):
                    return data[:, i]
        return None
    t = col("time")
    ax, ay, az = col("x (m/s"), col("y (m/s"), col("z (m/s")
    if ax is None:  # phyphox sometimes labels as "acceleration x"
        ax, ay, az = col("acceleration x"), col("acceleration y"), col("acceleration z")
    gy = col("gyroscope z", "z (rad/s")  # yaw rate if a custom experiment logged it
    return t, ax, ay, az, gy


def calibrate(files6):
    """
    Six-position tumble calibration.  files6: dict with keys '+x','-x','+y','-y','+z','-z',
    each a phyphox CSV recorded for ~10 s with that phone axis pointing straight UP and the
    phone still.  Returns {axis: (bias, scale)} so that corrected = (raw - bias)/scale.
    """
    out = {}
    for ax_name, idx in (("x", 1), ("y", 2), ("z", 3)):
        rp = np.mean(load_phyphox(files6["+" + ax_name])[idx])
        rm = np.mean(load_phyphox(files6["-" + ax_name])[idx])
        bias = (rp + rm) / 2
        scale = (rp - rm) / (2 * G)
        out[ax_name] = (float(bias), float(scale))
    return out

def apply_calib(ax, ay, az, cal):
    if not cal:
        return ax, ay, az
    (bx, sx), (by, sy), (bz, sz) = cal["x"], cal["y"], cal["z"]
    return (ax - bx) / sx, (ay - by) / sy, (az - bz) / sz

def find_slide(t, f_along, f_norm, f_lat, min_len=0.12, thresh_frac=0.5):
    """
    Locate the free-slide window.  Strategy: the push produces a large positive f_along
    (sled accelerated by the paddle); the slide is the following stretch where f_along is
    negative and roughly constant; the stop is where |f_along| drops to rest level.
    Returns (i0, i1) indices, or None.
    """
    dt = np.median(np.diff(t))
    # smooth lightly (5-sample moving average)
    k = 5
    fa = np.convolve(f_along, np.ones(k)/k, mode="same")
    # rest level and noise from the first 0.5 s
    n0 = max(5, int(0.5/dt))
    rest = np.median(fa[:n0]); noise = np.std(fa[:n0]) + 0.02
    # push = largest positive excursion
    ipush = int(np.argmax(fa))
    if fa[ipush] - rest < 1.0:
        return None
    # after the push, find where fa goes clearly negative (sliding)
    i = ipush
    while i < len(fa) and fa[i] > rest - 3*noise:
        i += 1
    i0 = i
    if i0 >= len(fa):
        return None
    # plateau level: median of the next 0.1 s
    nplat = max(3, int(0.1/dt))
    plateau = np.median(fa[i0:i0+nplat])
    # end of slide: fa returns toward rest (above half the plateau depth)
    j = i0 + nplat
    while j < len(fa) and fa[j] < rest + thresh_frac*(plateau - rest):
        j += 1
    i1 = j
    # trim the edges (release transient, stop transient)
    trim = max(2, int(0.03/dt))
    i0 += trim; i1 -= trim
    if (i1 - i0)*dt < min_len:
        return None
    return i0, i1

def analyse_run(t, ax, ay, az, gyaw=None, along="y", normal="z", lateral="x", direction=+1):
    comp = {"x": ax, "y": ay, "z": az}
    f_al = direction*comp[along]      # positive = slide direction
    f_no = comp[normal]
    f_la = comp[lateral]
    dt = np.median(np.diff(t))
    # rest calibration (first 0.5 s): axis misalignment estimate
    n0 = max(5, int(0.5/dt))
    rest_al, rest_no = np.mean(f_al[:n0]), np.mean(f_no[:n0])
    delta_rest = math.atan2(rest_al, rest_no)   # tilt+misalignment seen at rest (rad)
    win = find_slide(t, f_al, f_no, f_la)
    if win is None:
        return {"ok": False, "reason": "no slide segment found"}
    i0, i1 = win
    seg_al, seg_no, seg_la = f_al[i0:i1], f_no[i0:i1], f_la[i0:i1]
    mu_raw = -np.mean(seg_al)/np.mean(seg_no)
    # speed profile by backward integration from the stop (v=0 at i1):
    # true acceleration along the slide = f_al + g_al, where g_al = -rest_al (gravity
    # component along the plane, from the rest reading) -- valid when delta is mostly tilt.
    a_true = seg_al - rest_al
    v = -np.cumsum(a_true[::-1])[::-1]*dt   # v(t) = -integral_t^stop a dt'  (v=0 at the stop)
    v_start, v_end = float(v[0]), float(v[max(0, len(v)-1)])
    # per-sample mu against speed (for speed dependence)
    mu_t = -seg_al/seg_no
    # quality flags
    lat_rms = float(np.sqrt(np.mean(seg_la**2)))
    yaw_max = float(np.max(np.abs(gyaw[i0:i1]))*180/math.pi) if gyaw is not None else float("nan")
    dur = float(t[i1]-t[i0])
    return {"ok": True, "mu": float(mu_raw), "mu_sd_within": float(np.std(mu_t)/math.sqrt(len(mu_t))),
            "dur_s": dur, "n_samples": int(i1-i0), "v_start": v_start, "v_mean": float(np.mean(v)),
            "lat_rms": lat_rms, "yaw_max_degps": yaw_max, "delta_rest_deg": delta_rest*180/math.pi,
            "mu_vs_v": list(zip(v.tolist(), mu_t.tolist()))}

def quality(r, max_lat=0.5, max_yaw=20.0, min_dur=0.12):
    bad = []
    if r["dur_s"] < min_dur: bad.append("short slide")
    if r["lat_rms"] > max_lat: bad.append("lateral force")
    if not math.isnan(r["yaw_max_degps"]) and r["yaw_max_degps"] > max_yaw: bad.append("yaw")
    return bad

def synth_run(mu=0.30, tilt_deg=0.0, delta_deg=0.0, v0=1.2, fs=100, noise=0.03, seed=0, reverse=False):
    """Synthetic phyphox-like record: 1 s rest, push 0.15 s, slide to rest, 1 s rest."""
    rng = np.random.default_rng(seed)
    dt = 1/fs
    phi = math.radians(tilt_deg)
    # along/normal specific force in plane axes
    rest = (-G*math.sin(phi), G*math.cos(phi))
    t_rest, t_push = 1.0, 0.15
    a_push = v0/t_push
    a_slide = -(mu*G*math.cos(phi)) + G*math.sin(phi)*0  # true accel along (level: -mu g)
    # sliding: true accel = g sin(phi) - mu g cos(phi) (downhill positive)
    a_true = G*math.sin(phi) - mu*G*math.cos(phi)
    t_slide = v0/(-a_true) if a_true < 0 else 1.0
    T = t_rest + t_push + t_slide + 1.0
    t = np.arange(0, T, dt)
    f_al = np.full_like(t, rest[0]); f_no = np.full_like(t, rest[1])
    push = (t >= t_rest) & (t < t_rest+t_push)
    slide = (t >= t_rest+t_push) & (t < t_rest+t_push+t_slide)
    f_al[push] = rest[0] + a_push
    f_al[slide] = -mu*G*math.cos(phi)      # specific force along plane during slide
    if reverse:                            # same phone mounting, sled travelling toward -along
        f_al = 2*rest[0] - f_al            # push and friction flip sign about the rest level
    # misalignment rotation by delta about the lateral axis
    d = math.radians(delta_deg)
    fa = f_al*math.cos(d) + f_no*math.sin(d)
    fn = -f_al*math.sin(d) + f_no*math.cos(d)
    fa += rng.normal(0, noise, len(t)); fn += rng.normal(0, noise, len(t))
    fl = rng.normal(0, noise, len(t))
    return t, fl, fa, fn   # x=lateral, y=along, z=normal

def selftest():
    print("self-test: synthetic slides (mu=0.30)")
    for tilt, delta in [(0,0),(2,0),(0,1),(0,-1),(3,1)]:
        t, ax, ay, az = synth_run(mu=0.30, tilt_deg=tilt, delta_deg=delta)
        r = analyse_run(t, ax, ay, az)
        print(f" tilt={tilt:+.0f} deg  misalign={delta:+.0f} deg -> mu={r['mu']:.4f}  dur={r['dur_s']:.2f}s  v0~{r['v_start']:.2f} m/s")
    # two-direction average cancels misalignment
    t1, ax1, ay1, az1 = synth_run(mu=0.30, delta_deg=1.0, seed=1)
    t2, ax2, ay2, az2 = synth_run(mu=0.30, delta_deg=1.0, seed=2, reverse=True)
    r1 = analyse_run(t1, ax1, ay1, az1, direction=+1)
    r2 = analyse_run(t2, ax2, ay2, az2, direction=-1)
    print(f" misalign 1 deg: +x {r1['mu']:.4f}, -x {r2['mu']:.4f}, mean {(r1['mu']+r2['mu'])/2:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pair", default=""); ap.add_argument("--object", default=""); ap.add_argument("--surface", default="")
    ap.add_argument("--method", default="flat", choices=["flat","incline"])
    ap.add_argument("--direction", default="fwd", choices=["fwd","rev"], help="fwd: phone top (+y) points along the slide; rev: sled turned 180 deg")
    ap.add_argument("--load_g", default="")
    ap.add_argument("--along", default="y"); ap.add_argument("--normal", default="z"); ap.add_argument("--lateral", default="x")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calib", default="", help="JSON from --make-calib; applied to every run")
    ap.add_argument("--make-calib", nargs=6, metavar=("PX","MX","PY","MY","PZ","MZ"),
                    help="six static CSVs with +x,-x,+y,-y,+z,-z pointing up; writes calib.json")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.make_calib:
        import json
        keys = ["+x","-x","+y","-y","+z","-z"]
        cal = calibrate(dict(zip(keys, a.make_calib)))
        json.dump(cal, open("calib.json","w"), indent=2)
        for k,(b,sc) in cal.items():
            print(f"axis {k}: bias {b:+.4f} m/s^2, scale {sc:.4f}")
        print("wrote calib.json"); return
    cal = None
    if a.calib:
        import json
        cal = {k: tuple(v) for k, v in json.load(open(a.calib)).items()}
    rows = []
    for fpath in a.files:
        t, ax, ay, az, gy = load_phyphox(fpath)
        ax, ay, az = apply_calib(ax, ay, az, cal)
        r = analyse_run(t, ax, ay, az, gy, a.along, a.normal, a.lateral, +1 if a.direction=="fwd" else -1)
        if not r["ok"]:
            print(f"{fpath}: REJECT ({r['reason']})"); continue
        bad = quality(r)
        print(f"{fpath}: mu={r['mu']:.4f} dur={r['dur_s']:.2f}s n={r['n_samples']} v0~{r['v_start']:.2f} "
              f"lat_rms={r['lat_rms']:.3f} yaw={r['yaw_max_degps']:.1f} {'REJECT: '+','.join(bad) if bad else 'ok'}")
        rows.append({"pair_id": a.pair, "object": a.object, "surface": a.surface, "method": a.method,
                     "direction": a.direction, "load_g": a.load_g, "file": fpath, "mu_k": f"{r['mu']:.4f}",
                     "dur_s": f"{r['dur_s']:.3f}", "n_samples": r["n_samples"], "v_start": f"{r['v_start']:.3f}",
                     "lat_rms": f"{r['lat_rms']:.3f}", "yaw_max_degps": f"{r['yaw_max_degps']:.1f}",
                     "delta_rest_deg": f"{r['delta_rest_deg']:.2f}", "accepted": int(not bad), "flags": ";".join(bad)})
    acc = [float(r["mu_k"]) for r in rows if r["accepted"]]
    if acc:
        print(f"\n{a.pair} {a.object} on {a.surface} [{a.method} {a.direction}]: n={len(acc)} "
              f"mean mu_k={np.mean(acc):.4f} sd={np.std(acc, ddof=1) if len(acc)>1 else float('nan'):.4f}")
    if a.out and rows:
        new = not __import__("os").path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new: w.writeheader()
            w.writerows(rows)

if __name__ == "__main__":
    main()

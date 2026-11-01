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

Speed and distance.  The true along acceleration is a = f_along - f_along(rest): the rest
reading removes the gravity component along a tilted board.  With v = 0 at the true stop,
v(t) = -int_t^stop a dt' gives the release speed v_start, and a second integral from the
release gives the position of every sample and the slide distance d_acc_m.  Both integrals
run over the untrimmed signal from the release to the true stop; the trimmed window serves
the friction ratio only.  A residual bias of 0.005 m/s^2 over a 1 s slide moves d_acc_m by
about 2.5 mm.

Usage:
    python3 friction_analysis.py M07_*.csv --runlog run_log.csv --calib calib.json \
        --out results.csv --profiles profiles.csv
    python3 friction_analysis.py --selftest      # synthetic checks, no files needed; exit 1 on failure

Pair, lane, orientation and speed setting come from the run log when the file is listed
there, else from the file name <pairID>_<lane>_<orient>_<speed>_<n>.csv (M07_L_fwd_s1_1.csv);
--pair and --direction override both.  Rejected runs are written to the results CSV too
(accepted=0 with reject_reason), so the rejection rate is on record.

Axis convention: by default the phone's long axis (phyphox Y) is the slide direction and
Z is normal to the screen.  Change with --along/--normal if the phone is mounted differently.
"""
import argparse, csv, math, os, re, sys
import numpy as np

G = 9.80665
COUPON_MM = 100.0   # sled coupon length along the slide: the profile cannot resolve shorter features
PROFILE_NOTE = ("# mu along the lane by board position (sled leading edge, mm from the release point, "
                "bin lower edge). The 100 mm coupon averages friction over its own length, so features "
                "shorter than about 10 cm are not resolvable.")
RESULT_FIELDS = ["pair_id", "object", "surface", "method", "direction", "load_g", "file", "mu_k", "dur_s",
                 "n_samples", "v_start", "lat_rms", "yaw_max_degps", "delta_rest_deg", "accepted", "flags",
                 # appended (6.2/6.4); the 16 columns above keep their order
                 "lane", "orientation", "speed_setting", "rest_mm", "d_acc_m", "mu_v2d", "v2d_ratio",
                 "mu_slope_per_mps", "stop_ms", "warnings", "reject_reason", "session"]
PROFILE_FIELDS = ["pair_id", "lane", "orientation", "speed_setting", "file", "bin_mm", "mu", "n"]

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

def find_slide(t, f_along, f_norm, f_lat, min_len=0.12, thresh_frac=0.5, quiet_s=0.05):
    """
    Locate the free slide.  The push produces a large positive f_along (sled accelerated by
    the paddle); the slide is the following stretch where f_along is negative and roughly
    constant; the stop is where f_along returns to rest level.  Returns None if no slide is
    found, else a dict:
      i0, i1   trimmed window for the friction ratio (30 ms off each end)
      i_rel    release: first sample after the push whose f_along is clearly below rest level
      j        where the smoothed f_along has come halfway back from the plateau to rest
      i_stop   true stop: start of the first stretch of at least quiet_s during which the
               smoothed f_along stays inside the rest noise band.  At j the sled is still
               moving (half the deceleration remains, spread over a 20-50 ms stop transient),
               so the speed integral must run to i_stop, not j.  Detecting the stop late is
               harmless, since the extra samples integrate to zero-mean noise; early is not.
      rest, noise, plateau   levels of the smoothed signal; flags: list of warnings
    """
    dt = np.median(np.diff(t))
    k = 5                                        # 5-sample moving average, for event detection only
    fa = np.convolve(f_along, np.ones(k)/k, mode="same")
    n0 = max(5, int(0.5/dt))                     # rest level and noise from the first 0.5 s
    rest = np.median(fa[:n0]); noise = np.std(fa[:n0]) + 0.02
    ipush = int(np.argmax(fa))                   # push = largest positive excursion
    if fa[ipush] - rest < 1.0:
        return None
    i = ipush                                    # after the push, where fa goes clearly negative
    while i < len(fa) and fa[i] > rest - 3*noise:
        i += 1
    if i >= len(fa):
        return None
    i_rel = i                                    # step back over samples the smoothing smeared
    rest_raw = np.median(f_along[:n0]); noise_raw = np.std(f_along[:n0]) + 0.02
    while i_rel > ipush + 1 and f_along[i_rel-1] < rest_raw - 3*noise_raw:
        i_rel -= 1
    nplat = max(3, int(0.1/dt))                  # plateau level: median of the next 0.1 s
    plateau = np.median(fa[i:i+nplat])
    j = i + nplat                                # halfway back toward rest
    while j < len(fa) and fa[j] < rest + thresh_frac*(plateau - rest):
        j += 1
    trim = max(2, int(0.03/dt))                  # trim release and stop transients for the ratio
    i0, i1 = i + trim, j - trim
    if (i1 - i0)*dt < min_len:
        return None
    # true stop: first run of >= quiet_s samples inside the rest band, searched from j on.  The
    # band is 3 sigma of the rest noise or 5% of the plateau depth, whichever is wider; if the
    # signal never settles that close (rest level shifted), retry at 10%, then give up at j.
    flags = []
    nq = max(2, int(round(quiet_s/dt)))
    i_stop = None
    for frac, band in ((0.05, max(3*noise, 0.05*(rest - plateau))), (0.10, 0.10*(rest - plateau))):
        quiet = np.concatenate([np.abs(fa[j:] - rest) <= band, np.ones(nq, bool)])  # padded: rest continues
        run = np.convolve(quiet.astype(int), np.ones(nq, int), mode="valid") == nq  # quiet for the next nq
        hits = np.flatnonzero(run)
        if len(hits) and j + hits[0] < len(fa):
            i_stop = j + int(hits[0])
            if frac > 0.05: flags.append("rest band widened")
            break
    if i_stop is None:
        i_stop = j; flags.append("no stop found")
    elif i_stop > len(fa) - nq:
        flags.append("stop near record end")
    return dict(i0=i0, i1=i1, i_rel=i_rel, j=j, i_stop=i_stop, rest=float(rest), noise=float(noise),
                plateau=float(plateau), flags=flags)

def analyse_run(t, ax, ay, az, gyaw=None, along="y", normal="z", lateral="x", direction=+1):
    comp = {"x": ax, "y": ay, "z": az}
    f_al = direction*comp[along]      # positive = slide direction
    f_no = comp[normal]
    f_la = comp[lateral]
    dt = float(np.median(np.diff(t)))
    # rest calibration (first 0.5 s): axis misalignment estimate
    n0 = max(5, int(0.5/dt))
    rest_al, rest_no = np.mean(f_al[:n0]), np.mean(f_no[:n0])
    delta_rest = math.atan2(rest_al, rest_no)   # tilt+misalignment seen at rest (rad)
    w = find_slide(t, f_al, f_no, f_la)
    if w is None:
        return {"ok": False, "reason": "no slide segment found"}
    i0, i1, i_rel, i_stop = w["i0"], w["i1"], w["i_rel"], w["i_stop"]
    seg_al, seg_no, seg_la = f_al[i0:i1], f_no[i0:i1], f_la[i0:i1]
    mu_raw = -np.mean(seg_al)/np.mean(seg_no)
    # speed and position: trapezoid integrals of the untrimmed true acceleration a = f_al - rest_al
    # (g_al = -rest_al is the gravity component along the plane -- valid when delta is mostly
    # tilt), backward from the true stop (v = 0 there), then forward from the release (x = 0).
    a_true = f_al[i_rel:i_stop+1] - rest_al
    dv = 0.5*(a_true[:-1] + a_true[1:])*dt
    v = -np.concatenate([np.cumsum(dv[::-1])[::-1], [0.0]])      # v(t) = -integral_t^stop a dt'
    x = np.concatenate([[0.0], np.cumsum(0.5*(v[:-1] + v[1:])*dt)])
    sl = slice(i0 - i_rel, i1 - i_rel)
    v_win, x_win = v[sl], x[sl]
    # per-sample mu against speed and position (speed dependence, friction profile)
    mu_t = -seg_al/seg_no
    slope = float(np.polyfit(v_win, mu_t, 1)[0]) if np.ptp(v_win) > 0.1 else float("nan")
    # quality flags (lateral relative to its rest level, so board tilt and bias don't count)
    lat_rms = float(np.sqrt(np.mean((seg_la - np.mean(f_la[:n0]))**2)))
    yaw_max = float(np.max(np.abs(gyaw[i0:i1]))*180/math.pi) if gyaw is not None else float("nan")
    dur = float(t[i1]-t[i0])
    return {"ok": True, "mu": float(mu_raw), "mu_sd_within": float(np.std(mu_t)/math.sqrt(len(mu_t))),
            "dur_s": dur, "n_samples": int(i1-i0), "v_start": float(v[0]), "v_mean": float(np.mean(v_win)),
            "d_acc_m": float(x[-1]), "mu_slope_per_mps": slope, "stop_ms": (i_stop - w["j"])*dt*1000,
            "lat_rms": lat_rms, "yaw_max_degps": yaw_max, "delta_rest_deg": delta_rest*180/math.pi,
            "warnings": w["flags"], "x_mm": (x_win*1000).tolist(), "mu_t": mu_t.tolist(),
            "mu_vs_v": list(zip(v_win.tolist(), mu_t.tolist()))}

def quality(r, max_lat=0.5, max_yaw=20.0, min_dur=0.12):
    bad = []
    if r["dur_s"] < min_dur: bad.append("short slide")
    if r["lat_rms"] > max_lat: bad.append("lateral force")
    if not math.isnan(r["yaw_max_degps"]) and r["yaw_max_degps"] > max_yaw: bad.append("yaw")
    return bad

def profile_bins(x_mm, mu_t, bin_mm=20.0):
    """
    Friction profile along the lane: the per-sample ratio binned by board position (leading
    edge, mm from the release point).  Returns [(bin lower edge mm, mean mu, n samples)].  The
    coupon covers [x - COUPON_MM, x], so the profile is the local mu smoothed over a coupon
    length: a step shows as a ramp of one coupon length and shorter features are not resolved.
    """
    b = np.floor(np.asarray(x_mm)/bin_mm)*bin_mm
    mu_t = np.asarray(mu_t)
    return [(float(bb), float(np.mean(mu_t[b == bb])), int(np.sum(b == bb))) for bb in np.unique(b)]

def step_position(prof, n_edge=3):
    """
    Where a binned profile crosses the midpoint between its first and last n_edge bins (mm,
    linear interpolation between bin centres).  For a sharp step in mu at x_s the leading-edge
    profile ramps from x_s to x_s + COUPON_MM, so the crossing sits half a coupon past the step.
    """
    b = np.array([p[0] for p in prof]) + 0.5*np.median(np.diff([p[0] for p in prof]))
    m = np.array([p[1] for p in prof])
    lo, hi = m[:n_edge].mean(), m[-n_edge:].mean(); mid = 0.5*(lo + hi)
    k = int(np.argmax((m - mid)*(hi - lo) > 0))          # first bin on the far side
    if k == 0:
        return float(b[0])
    f = (mid - m[k-1])/(m[k] - m[k-1])
    return float(b[k-1] + f*(b[k] - b[k-1]))

def synth_run(mu=0.30, tilt_deg=0.0, delta_deg=0.0, v0=1.2, fs=100, noise=0.03, seed=0, reverse=False,
              stop_tau=0.0, mu_x=None, mu_v=None, coupon_m=COUPON_MM/1000, truth=False):
    """
    Synthetic phyphox-like record: 1 s rest, 0.15 s push at constant acceleration, free slide
    to rest, 1 s rest.  The motion is integrated on a grid 50x finer than the sample rate and
    read at the phone's sample instants, so a sharp stop falls at an arbitrary sub-sample time,
    as on the rig (the analysis cannot place it closer than half a sample).
      mu        kinetic friction; the base value when mu_x or mu_v are given
      mu_x      mu as a function of board position (m, leading edge from the release point),
                vectorised; the coupon averages it over its own length coupon_m
      mu_v      mu as a function of sled speed (m/s); with mu_x too, mu = mu_x*mu_v/mu
      stop_tau  stop transient (s): the deceleration ramps linearly to zero over the last
                stop_tau seconds instead of switching off at once (real stops take 20-50 ms)
      reverse   sled turned 180 deg, same travel direction on the board: along readings flip
      truth     also return dict(v0, d, t_rel, t_stop) of the simulated motion
    Returns t, fl, fa, fn (x=lateral, y=along, z=normal), plus the truth dict if asked.
    """
    rng = np.random.default_rng(seed)
    dt = 1/fs; sub = 50; h = dt/sub
    phi = math.radians(tilt_deg); c, s = math.cos(phi), math.sin(phi)
    t_rest, t_push = 1.0, 0.15
    a = [0.0]*int(round(t_rest/h)) + [v0/t_push]*int(round(t_push/h))   # true along acceleration, fine grid
    def mu_at(x, v):
        m = mu
        if mu_x is not None:
            m = float(np.mean(mu_x(np.linspace(x - coupon_m, x, 21))))
        if mu_v is not None:
            m = m*mu_v(v)/mu
        return m
    v, x = v0, 0.0
    while v > 0:
        dec = mu_at(x, v)*G*c - G*s                      # true deceleration along the slide
        if dec <= 0:
            raise ValueError("sled does not stop on this slope")
        if stop_tau > 0 and v <= 0.5*dec*stop_tau:      # ramp whose area equals the remaining speed
            n = max(1, int(round(stop_tau/h)))
            for k in range(n):
                ak = -dec*(1 - (k + 0.5)/n); a.append(ak)
                vn = max(v + ak*h, 0.0); x += 0.5*(v + vn)*h; v = vn
            break
        a.append(-dec)
        vn = v - dec*h
        if vn <= 0:                                      # stops inside this step
            x += 0.5*v*(v/dec); v = 0.0; break
        x += 0.5*(v + vn)*h; v = vn
    t_stop = len(a)*h
    a = np.array(a + [0.0]*int(round(1.0/h)))
    f_al = a[::sub] - G*s                                # specific force along the plane
    f_no = np.full(len(f_al), G*c)
    t = np.arange(len(f_al))*dt
    if reverse:                            # phone's along axis points against travel, so every reading flips
        f_al = -f_al
    # misalignment rotation by delta about the lateral axis
    d = math.radians(delta_deg)
    fa = f_al*math.cos(d) + f_no*math.sin(d)
    fn = -f_al*math.sin(d) + f_no*math.cos(d)
    fa += rng.normal(0, noise, len(t)); fn += rng.normal(0, noise, len(t))
    fl = rng.normal(0, noise, len(t))
    if truth:
        return t, fl, fa, fn, {"v0": v0, "d": x, "t_rel": t_rest + t_push, "t_stop": t_stop}
    return t, fl, fa, fn   # x=lateral, y=along, z=normal

def selftest_checks():
    """
    The three synthetic checks of the spec, as [(name, passed, detail)]:
      1. known distance: d_acc_m and v_start within 2%, for an instantaneous stop and a 30 ms one;
      2. a step in mu at a known board position: the profile places it within one coupon length;
      3. a speed-dependent mu: the fitted slope within 10%.
    """
    out = []
    for tau in (0.0, 0.03):
        t, ax, ay, az, tr = synth_run(mu=0.30, v0=1.2, stop_tau=tau, truth=True, seed=3)
        r = analyse_run(t, ax, ay, az)
        ev, ed = r["v_start"]/tr["v0"] - 1, r["d_acc_m"]/tr["d"] - 1
        out.append((f"distance, stop transient {tau*1000:.0f} ms", abs(ev) < 0.02 and abs(ed) < 0.02,
                    f"v_start {r['v_start']:.3f} vs {tr['v0']:.3f} m/s ({ev:+.1%}); d_acc {r['d_acc_m']:.4f} vs "
                    f"{tr['d']:.4f} m ({ed:+.1%}); stop found {r['stop_ms']:.0f} ms after the half-return"))
    x_step = 0.15
    t, ax, ay, az, tr = synth_run(mu=0.30, v0=1.5, stop_tau=0.03, truth=True, seed=4,
                                  mu_x=lambda x: np.where(x < x_step, 0.25, 0.40))
    r = analyse_run(t, ax, ay, az)
    xs = step_position(profile_bins(r["x_mm"], r["mu_t"]))
    out.append(("mu step position", abs(xs - 1000*x_step) <= COUPON_MM,
                f"profile crosses the midpoint at {xs:.0f} mm, step at {1000*x_step:.0f} mm "
                f"(expected about half a coupon past it; tolerance {COUPON_MM:.0f} mm)"))
    k = 0.06
    t, ax, ay, az, tr = synth_run(mu=0.30, v0=1.4, stop_tau=0.03, truth=True, seed=5, mu_v=lambda v: 0.30 + k*v)
    r = analyse_run(t, ax, ay, az)
    out.append(("mu(v) slope", abs(r["mu_slope_per_mps"]/k - 1) < 0.10,
                f"fitted {r['mu_slope_per_mps']:.4f} per m/s vs {k} ({r['mu_slope_per_mps']/k - 1:+.1%})"))
    return out

def selftest():
    print("self-test: synthetic slides (mu=0.30)")
    for tilt, delta in [(0,0),(2,0),(0,1),(0,-1),(3,1)]:
        t, ax, ay, az = synth_run(mu=0.30, tilt_deg=tilt, delta_deg=delta)
        r = analyse_run(t, ax, ay, az)
        print(f" tilt={tilt:+.0f} deg  misalign={delta:+.0f} deg -> mu={r['mu']:.4f}  dur={r['dur_s']:.2f}s  "
              f"v0~{r['v_start']:.2f} m/s  d~{r['d_acc_m']:.3f} m")
    # two-direction average cancels misalignment (on a tilted board too)
    t1, ax1, ay1, az1 = synth_run(mu=0.30, tilt_deg=2, delta_deg=1.0, seed=1)
    t2, ax2, ay2, az2 = synth_run(mu=0.30, tilt_deg=2, delta_deg=1.0, seed=2, reverse=True)
    r1 = analyse_run(t1, ax1, ay1, az1, direction=+1)
    r2 = analyse_run(t2, ax2, ay2, az2, direction=-1)
    print(f" tilt 2 deg, misalign 1 deg: fwd {r1['mu']:.4f}, rev {r2['mu']:.4f}, mean {(r1['mu']+r2['mu'])/2:.4f}")
    print("checks:")
    checks = selftest_checks()
    for name, ok, detail in checks:
        print(f" {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not all(ok for _, ok, _ in checks):
        print("self-test FAILED"); return 1
    print("self-test passed"); return 0

# ---------------------------------------------------------------- run log, file names, CSV output

def load_runlog(path):
    """file basename -> row of the run log.  Rows whose session is EXAMPLE (the template's) are skipped."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name = (r.get("file") or "").strip()
            if not name or (r.get("session") or "").strip().upper() == "EXAMPLE":
                continue
            out[os.path.basename(name)] = r
    return out

def parse_filename(path):
    """<pairID>_<lane>_<orient>_<speed>_<n>.csv -> dict(pair_id, lane, orientation, speed_setting, rep); best effort."""
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    d = {"pair_id": parts[0], "lane": "", "orientation": "", "speed_setting": "", "rep": ""}
    if len(parts) >= 5 and parts[2].lower() in ("fwd", "rev"):
        d.update(lane=parts[1], orientation=parts[2].lower(), speed_setting=parts[3], rep=parts[4])
        return d
    for i, p in enumerate(parts[1:], 1):
        if p.lower() in ("fwd", "rev"):
            d["orientation"] = p.lower()
            if i == 2: d["lane"] = parts[1]
        elif re.fullmatch(r"[sS]\d+", p):
            d["speed_setting"] = p
    return d

def run_meta(fpath, runlog, a):
    """Lane, orientation, speed, rest_mm and manual rejects for a file: CLI flags > run log > file name."""
    base = os.path.basename(fpath)
    lg = runlog.get(base) or runlog.get(os.path.splitext(base)[0]) or {}
    g = lambda key: (lg.get(key) or "").strip()
    fn = parse_filename(fpath)
    try: rest_mm = float(g("rest_mm"))
    except ValueError: rest_mm = float("nan")
    reason = g("reject_reason")
    return {"pair_id": a.pair or g("pair_id") or fn["pair_id"], "lane": g("lane") or fn["lane"],
            "orientation": a.direction or g("orientation").lower() or fn["orientation"] or "fwd",
            "speed_setting": g("speed_setting") or fn["speed_setting"], "rest_mm": rest_mm,
            "manual_reject": g("accepted").lower() in ("0", "n", "no", "false") or bool(reason),
            "reject_reason": reason, "session": g("session"), "in_runlog": bool(lg)}

def fmt(x, nd):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

def write_rows(path, rows, fields, comment=None):
    """
    Append rows to a CSV.  If the file exists with a different header (an older results file),
    it is rewritten with the union of columns, old rows padded with blanks, so nothing is lost.
    """
    if not rows:
        return
    old, old_fields = [], None
    if os.path.exists(path):
        with open(path, newline="") as f:
            lines = [l for l in f if not l.startswith("#")]
        rd = csv.DictReader(lines); old = list(rd); old_fields = rd.fieldnames
    if old_fields is not None and old_fields != fields:
        fields = list(old_fields) + [k for k in fields if k not in old_fields]
    mode = "a" if old_fields == fields else "w"
    with open(path, mode, newline="") as f:
        if mode == "w":
            if comment: f.write(comment + "\n")
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(old)
        else:
            w = csv.DictWriter(f, fieldnames=fields)
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pair", default="", help="pair ID; default from the run log or the file name")
    ap.add_argument("--object", default=""); ap.add_argument("--surface", default="")
    ap.add_argument("--method", default="flat", choices=["flat","incline"])
    ap.add_argument("--direction", default="", choices=["", "fwd", "rev"],
                    help="fwd: phone top (+y) points along the slide; rev: sled turned 180 deg. Default: run log, then file name")
    ap.add_argument("--load_g", default="")
    ap.add_argument("--along", default="y"); ap.add_argument("--normal", default="z"); ap.add_argument("--lateral", default="x")
    ap.add_argument("--runlog", default="", help="run log CSV (run_log_template.csv columns): rest_mm, lane, orientation, speed_setting, manual rejects")
    ap.add_argument("--out", default="", help="per-run results CSV (appended)")
    ap.add_argument("--profiles", default="", help="per-run mu(x) profiles CSV (appended), accepted runs only")
    ap.add_argument("--bin-mm", type=float, default=20.0, help="profile bin width along the lane")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calib", default="", help="JSON from --make-calib; applied to every run")
    ap.add_argument("--make-calib", nargs=6, metavar=("PX","MX","PY","MY","PZ","MZ"),
                    help="six static CSVs with +x,-x,+y,-y,+z,-z pointing up; writes calib.json")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
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
    runlog = load_runlog(a.runlog) if a.runlog else {}
    rows, prows = [], []
    for fpath in a.files:
        m = run_meta(fpath, runlog, a)
        if a.runlog and not m["in_runlog"]:
            print(f"{fpath}: not in the run log, lane/orientation/speed from the file name, no rest_mm")
        t, ax, ay, az, gy = load_phyphox(fpath)
        ax, ay, az = apply_calib(ax, ay, az, cal)
        r = analyse_run(t, ax, ay, az, gy, a.along, a.normal, a.lateral, +1 if m["orientation"] != "rev" else -1)
        row = {k: "" for k in RESULT_FIELDS}
        row.update({"pair_id": m["pair_id"], "object": a.object, "surface": a.surface, "method": a.method,
                    "direction": m["orientation"], "load_g": a.load_g, "file": fpath, "lane": m["lane"],
                    "orientation": m["orientation"], "speed_setting": m["speed_setting"],
                    "rest_mm": fmt(m["rest_mm"], 0), "session": m["session"]})
        manual = [m["reject_reason"] or "rejected in run log"] if m["manual_reject"] else []
        if not r["ok"]:
            reasons = [r["reason"]] + manual
            print(f"{fpath}: REJECT ({'; '.join(reasons)})")
            row.update({"accepted": 0, "flags": r["reason"], "reject_reason": "; ".join(reasons)})
            rows.append(row); continue
        bad = quality(r)
        reasons = bad + manual
        # endpoint estimate from the tape distance, as the robot would see it: mu = v^2/(2 g d)
        mu_v2d = float("nan")
        if m["rest_mm"] > 0 and "no stop found" not in r["warnings"]:
            mu_v2d = r["v_start"]**2/(2*G*m["rest_mm"]/1000)
        print(f"{fpath}: mu={r['mu']:.4f} dur={r['dur_s']:.2f}s n={r['n_samples']} v0~{r['v_start']:.2f} "
              f"d_acc={r['d_acc_m']:.3f}m" + (f" d_tape={m['rest_mm']/1000:.3f}m mu_v2d={mu_v2d:.4f}" if mu_v2d == mu_v2d else "") +
              f" lat_rms={r['lat_rms']:.3f} yaw={r['yaw_max_degps']:.1f} {'REJECT: '+', '.join(reasons) if reasons else 'ok'}"
              + (f" [warn: {', '.join(r['warnings'])}]" if r["warnings"] else ""))
        row.update({"mu_k": f"{r['mu']:.4f}", "dur_s": f"{r['dur_s']:.3f}", "n_samples": r["n_samples"],
                    "v_start": f"{r['v_start']:.3f}", "lat_rms": f"{r['lat_rms']:.3f}",
                    "yaw_max_degps": f"{r['yaw_max_degps']:.1f}", "delta_rest_deg": f"{r['delta_rest_deg']:.2f}",
                    "accepted": int(not reasons), "flags": ";".join(bad), "d_acc_m": f"{r['d_acc_m']:.4f}",
                    "mu_v2d": fmt(mu_v2d, 4), "v2d_ratio": fmt(mu_v2d/r["mu"], 4),
                    "mu_slope_per_mps": fmt(r["mu_slope_per_mps"], 4), "stop_ms": f"{r['stop_ms']:.0f}",
                    "warnings": ";".join(r["warnings"]), "reject_reason": "; ".join(reasons)})
        rows.append(row)
        if not reasons and a.profiles:
            for b, mu_b, n in profile_bins(r["x_mm"], r["mu_t"], a.bin_mm):
                prows.append({"pair_id": m["pair_id"], "lane": m["lane"], "orientation": m["orientation"],
                              "speed_setting": m["speed_setting"], "file": fpath, "bin_mm": f"{b:.0f}",
                              "mu": f"{mu_b:.4f}", "n": n})
    for pid in sorted({r["pair_id"] for r in rows}):
        acc = [float(r["mu_k"]) for r in rows if r["pair_id"] == pid and r["accepted"] == 1]
        rat = [float(r["v2d_ratio"]) for r in rows if r["pair_id"] == pid and r["accepted"] == 1 and r["v2d_ratio"]]
        nrej = sum(1 for r in rows if r["pair_id"] == pid and r["accepted"] != 1)
        if acc:
            print(f"\n{pid} {a.object} on {a.surface} [{a.method}]: n={len(acc)} rejected={nrej} "
                  f"mean mu_k={np.mean(acc):.4f} sd={np.std(acc, ddof=1) if len(acc)>1 else float('nan'):.4f}"
                  + (f"  mu_v2d/mu_k={np.mean(rat):.3f}+-{np.std(rat, ddof=1) if len(rat)>1 else float('nan'):.3f}" if rat else ""))
    if a.out:
        write_rows(a.out, rows, RESULT_FIELDS)
    if a.profiles:
        write_rows(a.profiles, prows, PROFILE_FIELDS, comment=PROFILE_NOTE)

if __name__ == "__main__":
    main()

"""Checks for friction_analysis.py (rig analysis) and aggregate_pairs.py, on synthetic slides."""
import csv, os, subprocess, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import friction_analysis as fa
import aggregate_pairs as ag


def test_spec_checks_pass():
    """6.2e: known distance within 2%, mu step within one coupon, mu(v) slope within 10%."""
    checks = fa.selftest_checks()
    assert len(checks) == 4
    for name, ok, detail in checks:
        assert ok, f"{name}: {detail}"


def test_two_orientation_mean_cancels_misalignment():
    t1, ax1, ay1, az1 = fa.synth_run(mu=0.30, tilt_deg=2, delta_deg=1.0, seed=1)
    t2, ax2, ay2, az2 = fa.synth_run(mu=0.30, tilt_deg=2, delta_deg=1.0, seed=2, reverse=True)
    r1 = fa.analyse_run(t1, ax1, ay1, az1, direction=+1)
    r2 = fa.analyse_run(t2, ax2, ay2, az2, direction=-1)
    assert abs(r1["mu"] - 0.30) > 0.01 and abs(r2["mu"] - 0.30) > 0.01      # each orientation is biased
    assert abs((r1["mu"] + r2["mu"])/2 - 0.30) < 0.003                      # the mean is not


def test_true_stop_is_after_half_return():
    """
    The bug of 6.2a.  The old code integrated the trimmed window [i0, j): v_start was the speed
    30 ms after release (short by mu g * 0.03 s = 0.09 m/s at mu = 0.3) and the integral ended at
    the half-return j, where the sled still moves.  The true stop lies after j; integrating the
    untrimmed signal from release to there recovers the release speed.
    """
    t, ax, ay, az, tr = fa.synth_run(mu=0.30, v0=1.2, stop_tau=0.03, truth=True, seed=7)
    w = fa.find_slide(t, ay, az, ax)
    dt = float(np.median(np.diff(t)))
    assert w["i_stop"] > w["j"] > w["i1"]
    assert w["i_rel"] < w["i0"]
    v_old = -np.sum(ay[w["i0"]:w["j"]] - np.mean(ay[:50]))*dt            # the old v_start
    r = fa.analyse_run(t, ax, ay, az)
    assert v_old < 0.95*tr["v0"], f"old estimate {v_old:.3f} should read low by about mu g * 30 ms"
    assert abs(r["v_start"]/tr["v0"] - 1) < 0.02
    assert abs(r["d_acc_m"]/tr["d"] - 1) < 0.02


@pytest.mark.parametrize("tau", [0.0, 0.02, 0.05])
@pytest.mark.parametrize("v0", [0.8, 1.1])
def test_distance_across_stop_transients(tau, v0):
    """Sharp and 20-50 ms stops at speeds the robot uses; the sharp case carries a half-sample ambiguity."""
    t, ax, ay, az, tr = fa.synth_run(mu=0.25, v0=v0, stop_tau=tau, truth=True, seed=11)
    r = fa.analyse_run(t, ax, ay, az)
    tol = 0.03 if tau == 0 else 0.02
    assert abs(r["v_start"]/tr["v0"] - 1) < tol
    assert abs(r["d_acc_m"]/tr["d"] - 1) < 1.5*tol


def test_parse_filename():
    d = fa.parse_filename("data/raw/S1/M07_L_fwd_s1_1.csv")
    assert d == {"pair_id": "M07", "lane": "L", "orientation": "fwd", "speed_setting": "s1", "rep": "1"}
    d = fa.parse_filename("V03_rev_s2.csv")
    assert d["pair_id"] == "V03" and d["orientation"] == "rev" and d["speed_setting"] == "s2" and d["lane"] == ""


def test_profile_bins_and_step():
    x = np.arange(0, 400, 5.0)
    mu = np.where(x < 200, 0.2, 0.4)
    prof = fa.profile_bins(x, mu, 20.0)
    assert [p[0] for p in prof] == list(np.arange(0, 400, 20.0))
    assert all(p[2] == 4 for p in prof)
    assert abs(fa.step_position(prof) - 200) <= 20


def write_phyphox(path, t, ax, ay, az):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time (s)", "Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"])
        w.writerows(zip(t, ax, ay, az))


def test_cli_results_profiles_and_aggregate(tmp_path):
    """End to end: synthetic files + run log -> results/profiles CSVs (rejects included) -> measured_friction.csv."""
    mu_x = lambda x: 0.30 + 0.01*np.sin(2*np.pi*x/0.4)     # a gentle patchiness along the lane
    runs, log = [], []
    k = 0
    for lane in ("L", "R"):
        for speed, v0 in (("s1", 0.9), ("s2", 1.2)):
            for orient in ("fwd", "rev"):
                k += 1
                # board levelled to 0.3 deg, phone mounted 0.8 deg off: each orientation's mu_k is biased by
                # about +-delta(1+mu^2) = 0.015, which the fwd/rev average cancels
                t, ax, ay, az, tr = fa.synth_run(mu=0.30, v0=v0, tilt_deg=0.3, delta_deg=0.8, seed=k, stop_tau=0.03,
                                                 reverse=(orient == "rev"), mu_x=mu_x, truth=True)
                name = f"M05_{lane}_{orient}_{speed}_1.csv"
                write_phyphox(tmp_path/name, t, ax, ay, az)
                runs.append(str(tmp_path/name))
                log.append(dict(session="S1", date="2026-11-04", operator="T", temp_C="21", RH_pct="40", calib_file="",
                                pair_id="M05", file=name, lane=lane, orientation=orient, speed_setting=speed,
                                rest_mm=f"{tr['d']*1000:.0f}", accepted="1", reject_reason="", notes=""))
    # a run with no slide (phone left at rest) and a manual reject
    t = np.arange(0, 3, 0.01); write_phyphox(tmp_path/"M05_L_fwd_s1_9.csv", t, 0*t, 0*t, 0*t + fa.G)
    runs.append(str(tmp_path/"M05_L_fwd_s1_9.csv"))
    log.append(dict(log[0], file="M05_L_fwd_s1_9.csv"))
    t, ax, ay, az = fa.synth_run(mu=0.30, v0=1.0, seed=99, reverse=True)
    write_phyphox(tmp_path/"M05_R_rev_s1_9.csv", t, ax, ay, az); runs.append(str(tmp_path/"M05_R_rev_s1_9.csv"))
    log.append(dict(log[0], file="M05_R_rev_s1_9.csv", lane="R", orientation="rev", accepted="0", reject_reason="touched paddle"))
    log.append(dict(log[0], session="EXAMPLE", file="M05_L_fwd_s1_1.csv", rest_mm="1"))   # template row, must be skipped
    with open(tmp_path/"run_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    py = sys.executable
    cmd = [py, os.path.join(ROOT, "friction_analysis.py"), *runs, "--runlog", str(tmp_path/"run_log.csv"),
           "--out", str(tmp_path/"results.csv"), "--profiles", str(tmp_path/"profiles.csv")]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    with open(tmp_path/"results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [c for c in rows[0]] == fa.RESULT_FIELDS
    assert len(rows) == 10
    rej = {r["file"].split("/")[-1]: r for r in rows if r["accepted"] != "1"}
    assert set(rej) == {"M05_L_fwd_s1_9.csv", "M05_R_rev_s1_9.csv"}
    assert rej["M05_L_fwd_s1_9.csv"]["reject_reason"] == "no slide segment found"
    assert rej["M05_R_rev_s1_9.csv"]["reject_reason"] == "touched paddle"
    acc = [r for r in rows if r["accepted"] == "1"]
    assert all(r["lane"] in ("L", "R") and r["speed_setting"] in ("s1", "s2") for r in acc)
    ratios = [float(r["v2d_ratio"]) for r in acc]
    # per run, the tape estimate v^2/(2 g d) is unbiased by the mounting tilt but mu_k is not (+-5%), and the
    # 0.3 deg board tilt lowers it by sin(0.3 deg)/mu = 1.7%; the pair mean (checked below) is within 3%
    assert all(abs(x - 1) < 0.08 for x in ratios), ratios
    assert all(float(r["d_acc_m"])*1000/float(r["rest_mm"]) - 1 < 0.02 for r in acc)
    with open(tmp_path/"profiles.csv") as f:
        first = f.readline()
    assert first.startswith("#") and "100 mm" in first
    prof = ag.read_csv(tmp_path/"profiles.csv")
    assert set(r["file"].split("/")[-1] for r in prof) == {os.path.basename(r) for r in runs[:8]}
    # aggregate
    cmd = [py, os.path.join(ROOT, "aggregate_pairs.py"), "--results", str(tmp_path/"results.csv"), "--profiles",
           str(tmp_path/"profiles.csv"), "--runlog", str(tmp_path/"run_log.csv"), "--meta",
           os.path.join(ROOT, "ground_truth_placeholder.csv"), "--out", str(tmp_path/"measured_friction.csv")]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    with open(tmp_path/"measured_friction.csv", newline="") as f:
        mf = list(csv.DictReader(f))
    assert list(mf[0]) == ag.BASE_COLUMNS + ag.NEW_COLUMNS
    assert len(mf) == 1 and mf[0]["pair_id"] == "M05"
    r = mf[0]
    assert abs(float(r["mu_k_nominal"]) - 0.30) < 0.01       # mounting tilt cancelled by the fwd/rev average
    assert r["object_id"] == "O5" and r["surface_id"] == "S1"  # identities copied from the metadata file
    assert r["n_runs"] == "8" and r["n_rejected"] == "2"
    assert "touched paddle:1" in r["reject_reasons"] and "no slide segment found:1" in r["reject_reasons"]
    assert r["temp_C"] == "21.0" and r["RH_pct"] == "40"
    assert float(r["strip_scale_mm"]) >= 100
    assert -1 <= float(r["profile_repeatability"]) <= 1
    assert abs(float(r["v2d_ratio_mean"]) - 1) < 0.03
    assert r["in_sim_range"] == "1"
    assert float(r["interval90_low"]) <= float(r["mu_k_nominal"]) <= float(r["interval90_high"])   # equal when s is tiny
    assert float(r["s_tab_log"]) < 0.02                      # four lane/speed means of one synthetic board


def test_results_file_with_old_header_is_upgraded(tmp_path):
    """An existing results.csv with the 16 original columns gets the new columns appended, old rows kept."""
    old = fa.RESULT_FIELDS[:16]
    with open(tmp_path/"results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old); w.writeheader()
        w.writerow({k: "x" for k in old})
    fa.write_rows(tmp_path/"results.csv", [{k: "y" for k in fa.RESULT_FIELDS}], fa.RESULT_FIELDS)
    with open(tmp_path/"results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == fa.RESULT_FIELDS and len(rows) == 2
    assert rows[0]["mu_k"] == "x" and rows[0]["d_acc_m"] == "" and rows[1]["d_acc_m"] == "y"

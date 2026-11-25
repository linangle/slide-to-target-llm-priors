"""The 21 Nov trial scripts, without Drake pushes: the pair-set wrapper's start gap and measures, the patchy-board
census and comparison, and the single-posture search's scoring."""
import csv, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
import numpy as np
import posture_search as PS
import run_pair_set_trial as RT
import run_patchy_trial as PT
import run_trial as T

def test_a_pushs_start_gap_depends_only_on_seed_friction_and_speed():
    g = RT.gap_of(7, 0.26, 0.6)
    assert g == RT.gap_of(7, 0.26, 0.6)                       # whichever worker runs the pair
    assert T.GAP_RANGE[0] <= g <= T.GAP_RANGE[1]
    assert g != RT.gap_of(7, 0.26, 0.6001) and g != RT.gap_of(8, 0.26, 0.6)

def test_both_measures_joins_rounded_results_to_the_push_log(tmp_path):
    mu, u, J, target = 0.26, 0.612345678, 100.0, 0.77
    x_fk = T.release_model_x(u, mu, J) - 0.0015               # the paddle started 1.5 mm further back than planned
    x_rest = 0.7702 - 0.0015
    fields = ["pair", "seed", "target_x", "method", "slides", "mu_true", "peak_speed", "x_rest", "error_x_mm", "valid"]
    with (tmp_path/"results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerow(dict(pair="Y03", seed=0, target_x=target, method="generic", slides=1, mu_true=mu, peak_speed=round(u, 6), x_rest=round(x_rest, 6),
                        error_x_mm=round(1e3*(x_rest - target), 3), valid=True))
        w.writerow(dict(pair="Y03", seed=0, target_x=target, method="lookup", slides=1, mu_true=mu, peak_speed=0.5, x_rest="nan", error_x_mm="nan", valid=False))
    with (tmp_path/"release_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=T.LOG_FIELDS); w.writeheader()
        w.writerow(dict(mu=repr(mu), v_peak=repr(u), J=J, gap_mm=2.5, x_fk=repr(x_fk), x_sep=repr(x_fk), v_sep=0.5, x_rest=repr(x_rest), y_rest=0.0, separated=True))
    rows = RT.both_measures(tmp_path, J)
    assert math.isclose(rows[0]["miss_rest_mm"], -1.3, abs_tol=1e-3)           # includes the 1.5 mm placement error
    assert math.isclose(rows[0]["miss_slide_mm"], 0.2, abs_tol=1e-6)          # the slide itself was 0.2 mm long
    assert rows[1]["miss_slide_mm"] == ""                                      # an invalid push has no slide measure

def test_the_strip_census_finds_no_yu_board_above_a_6_brake_and_some_handbook_boards():
    text = PT.census(6.0)
    assert not any(line.startswith("Y0") for line in text.splitlines())
    assert any(line.startswith("L16") for line in text.splitlines())
    assert "s = 0.06: 0" in text

def test_compare_counts_pushes_that_stop_releasing(tmp_path):
    fields = ["pair", "strip_sd", "board", "attempt", "inference", "method", "target_x", "mu_true", "x_rest", "error_mm", "valid"]
    def write(name, rows):
        with (tmp_path/name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    row = lambda board, x, valid: dict(pair="L16", strip_sd=0.25, board=board, attempt=1, inference="none", method="generic", target_x=0.78, mu_true=0.49,
                                       x_rest=x, error_mm=round(1e3*(x - 0.78), 3), valid=valid)
    write("old.csv", [row(0, 0.781, True), row(1, 0.779, True)])
    write("new.csv", [row(0, 0.782, True), row(1, 0.60, False)])
    text = PT.compare(tmp_path/"old.csv", tmp_path/"new.csv")
    line = next(l for l in text.splitlines() if l.strip().startswith("0.25"))
    assert line.split()[1:4] == ["2", "0", "1"]                               # rows, invalid before, invalid now
    assert "L16 (mu 0.49), s = 0.25: 1 rows on 1 boards (1)" in text

def test_a_postures_worst_ratio_is_over_all_speeds_and_limits():
    sc = {0.2: (0.3, 0.2, 0.8, 4, 2, 1), 1.0: (0.9, 0.3, 0.7, 4, 2, 4)}
    assert PS.worst(sc) == (0.9, 1.0, "speed", 4)
    sc[1.1] = None                                                             # the path stops short of this push
    assert PS.worst(sc)[0] == math.inf and PS.worst(sc)[2] == "reach"

def test_distinct_drops_near_duplicate_postures():
    q = np.zeros(7)
    kept = PS.distinct([("ik", q.tolist()), ("sweep", (q + 0.01).tolist()), ("sweep", (q + 0.05).tolist())])
    assert [k for k, _ in kept] == ["ik", "sweep"]

def test_released_means_the_object_left_before_the_paddle_stopped():
    import run_arm_evaluation as E
    assert E.released(dict(separated=True, t_force_zero=0.707, t_stop=0.7606))
    assert not E.released(dict(separated=True, t_force_zero=0.789, t_stop=0.7865))      # carried to the stop, apart at the end
    assert not E.released(dict(separated=False, t_force_zero=0.5, t_stop=0.9))

def test_the_release_test_is_off_unless_asked_for(monkeypatch):
    import run_arm_evaluation as E
    carried = dict(separated=True, t_force_zero=0.789, t_stop=0.7865, x_rest=0.72, y_rest=0.0)
    seen = {}
    def execute(mu, u, **options):
        seen.update(options)
        return carried, None, dict(q=0.8, v=0.5, tau=0.3, a=0.8), True
    monkeypatch.setattr(E, "execute", execute)
    assert E.checked_execute(0.44, 0.67, a_brake=6.0)[3] is True                # as every saved run: scored
    assert E.checked_execute(0.44, 0.67, a_brake=6.0, require_release=True)[3] is False
    assert seen == {"a_brake": 6.0}                                              # the switch does not reach the arm
    post = {"grid": np.linspace(math.log(0.02), math.log(2.0), 2000)}
    post["pmf"] = np.exp(-0.5*((post["grid"] - math.log(0.3))/0.8)**2); post["pmf"] /= post["pmf"].sum()
    truth = {"Y01": 0.14, "Y02": 0.14, "Y03": 0.26}
    from types import SimpleNamespace as NS
    off = E.arm_setup(NS(layout="side", a_brake=6.0, peak_cap=1.0, practice_peak=1.0, targets=[0.6]), truth, post)
    on = E.arm_setup(NS(layout="side", a_brake=6.0, peak_cap=1.0, practice_peak=1.0, targets=[0.6], check_release=True), truth, post)
    assert "require_release" not in off["options"] and "release_check" not in off["record"]
    assert on["options"] == {**off["options"], "require_release": True} and "release_check" in on["record"]

def test_the_release_audit_counts_scored_rows_that_never_released(tmp_path):
    fields = ["pair", "strip_sd", "board", "attempt", "inference", "method", "target_x", "mu_true", "peak_speed", "x_rest", "error_mm", "valid"]
    with (tmp_path/"aimed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for board, u, x, valid in [(0, 0.7, 0.871, True), (1, 0.669779, 0.7243, True), (3, 0.908584, 0.8596, False)]:
            w.writerow(dict(pair="L04", strip_sd=0.25, board=board, attempt=2, inference="1 aimed slide", method="generic", target_x=0.87, mu_true=0.44,
                            peak_speed=u, x_rest=x, error_mm=round(1e3*(x - 0.87), 3), valid=valid))
    with (tmp_path/"audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PT.AUDIT_FIELDS); w.writeheader()
        for board, u, x, valid, rel, t0 in [(0, 0.7, 0.871, True, True, 0.72), (1, 0.669779, 0.7243, True, False, 0.789), (3, 0.908584, 0.8596, False, False, 0.996)]:
            w.writerow(dict(pair="L04", strip_sd=0.25, board=board, peak_speed=u, mu_true=0.44, x_rest=x, valid=valid, separated=valid, released=rel,
                            t_stop=0.7865, t_force_zero=t0, sep_lag_ms=0, v_sep_force=0.5 if rel else 0.0002, v_r_cmd=0.57, lim_v=0.5, lim_tau=0.3, lim_a=0.8))
    text = PT.audit_summary(tmp_path/"aimed.csv", tmp_path/"audit.csv")
    line = next(l for l in text.splitlines() if l.strip().startswith("0.25"))
    assert line.split() == ["0.25", "3", "2", "3", "2", "1", "0", "1"]          # one row was scored although the object never released
    assert "s = 0.25, 1 aimed slide  : 73.3  ->  1.0" in text                  # the generic mean with and without that row

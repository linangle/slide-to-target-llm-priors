"""Tests of evaluation/patchy_table_study.py that need no Drake simulation (one scene is built, none is stepped)."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
for _d in ("src", "evaluation"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/_d))
import patchy_table_study as S
import run_arm_evaluation as E


def test_patchy_board_swaps_the_scene_builder_and_restores_it():
    """Inside the block push.push_slide's scene builder makes a board whose strips differ; outside, and for s = 0, it is
    the original.  Two boards differ; the same board is the same."""
    import push, scene
    from pydrake.all import DiagramBuilder
    original = push.build_scene
    with S.patchy_board(0.0, 5):
        assert push.build_scene is original
    def strips(s, board):
        with S.patchy_board(s, board):
            assert push.build_scene is not original
            sc = push.build_scene(DiagramBuilder(), {"mu_k_nominal": 0.14, "s_tab_log": 0.0}, scene.Obj(), scene.Table(patchiness=0.0),
                                  push.contact_from_config(), arm=False, x0=0.55, pusher=push.Pusher())
        assert push.build_scene is original
        return sc.mu_strips
    a, b, c = strips(0.12, 5), strips(0.12, 5), strips(0.12, 6)
    assert len(a) == 24 and np.array_equal(a, b) and not np.array_equal(a, c)
    assert 0.08 < np.std(np.log(a/0.14)) < 0.18 and abs(np.mean(np.log(a/0.14))) < 0.08
    try:
        with S.patchy_board(0.12, 5):
            raise RuntimeError
    except RuntimeError:
        pass
    assert push.build_scene is original


def test_board_seeds_differ_by_pair():
    seeds = {S.board_seed(p, b) for p in ("Y01", "Y02", "Y03") for b in range(16)}
    assert len(seeds) == 48


def test_shot_friction_inverts_the_release_model():
    from arm import rest_x
    for mu, u in ((0.14, 0.8), (0.26, 0.5), (0.14, 0.3)):
        assert math.isclose(S.shot_friction(u, rest_x(u, mu, E.JERK, E.X0)), mu, rel_tol=1e-5)
    assert math.isnan(S.shot_friction(0.5, 5.0))                     # no friction lands a 0.5 m/s push 4.4 m away


def test_transfer_sd_and_summary(tmp_path):
    """transfer_sd is the spread of ln(shot friction / practice estimate) over the unaware shots; the summary prints it."""
    rng = np.random.default_rng(0)
    rows = []
    for s in (0.0, 0.12):
        for board in range(40 if s else 1):
            for target in (0.6, 0.77):
                shot = 0.14*math.exp(0.05*rng.standard_normal() if s else 0.0)
                for m, e in (("generic", 40.), ("lookup", 41.), ("llm_raw", 48.), ("oracle", 5. if s else .3)):
                    rows.append(dict(pair="Y01", strip_sd=s, board=board, practice_peak="", slides=0, inference="none", method=m, target_x=target, mu_true=.14,
                                     mu_median=.3, post_sd=.8, peak_speed=.5, x_rest=target + e/1e3, error_mm=e, mu_shot=.14, valid=True))
                for speed in (1.01, 0.7):
                    for m in S.METHODS:
                        rows.append(dict(pair="Y01", strip_sd=s, board=board, practice_peak=speed, slides=1, inference="unaware", method=m, target_x=target,
                                         mu_true=.14, mu_median=.14, post_sd=.006, peak_speed=.5, x_rest=target + .004, error_mm=4., mu_shot=shot, valid=True))
    sd = S.transfer_sd(rows)
    assert set(sd) == {"0|1.01", "0|0.7", "0.12|1.01", "0.12|0.7"}
    assert sd["0|1.01"]["sd"] == 0 and 0.04 < sd["0.12|1.01"]["sd"] < 0.06 and sd["0.12|1.01"]["n"] == 80
    (tmp_path/"settings.json").write_text(json.dumps(dict(pairs=["Y01"], boards_per_pair=40, peak_cap=1.019, targets=[.6, .77], strip_sd=[0., .12])))
    (tmp_path/"transfer_sd.json").write_text(json.dumps(sd))
    with (tmp_path/"results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=S.FIELDS); w.writeheader(); w.writerows(rows)
    text = S.summarize(tmp_path)
    assert "40.0 / 41.0 / 48.0" in text and "4.0 / 4.0 / 4.0" in text
    line = [l for l in text.splitlines() if l.strip().startswith("0.12 |") and "+0." in l or l.strip().startswith("0.12 |") and "-0." in l][0]
    assert f"{sd['0.12|1.01']['sd']:.3f}" in line
    assert "strip spread 0.12: Y01 4.0 (oracle 5.0)" in text


def test_aimed_summary_reads_both_files(tmp_path):
    """summarize_aimed puts the aimed rule's rows next to the saved run's fixed-speed rows, by spread and by target, and
    counts the first shots that land where the saved run's no-practice shots landed."""
    base, aimed = [], []
    for s in (0.0, 0.12):
        for board in range(3 if s else 1):
            for target in (0.6, 0.77):
                for m, e in (("generic", 40.), ("lookup", 41.), ("llm_raw", 48.), ("oracle", 5.)):
                    base.append(dict(pair="Y01", strip_sd=s, board=board, practice_peak="", slides=0, inference="none", method=m, target_x=target, mu_true=.14,
                                     mu_median=.3, post_sd=.8, peak_speed=.5, x_rest=target + e/1e3, error_mm=e, mu_shot=.14, valid=True))
                for speed, e in ((1.01, 8.), (0.7, 6.)):
                    for m in S.METHODS:
                        base.append(dict(pair="Y01", strip_sd=s, board=board, practice_peak=speed, slides=1, inference="unaware", method=m, target_x=target,
                                         mu_true=.14, mu_median=.14, post_sd=.006, peak_speed=.5, x_rest=target + e/1e3, error_mm=e, mu_shot=.154, valid=True))
                for m, first in (("generic", 40.), ("lookup", 41.), ("llm_raw", 48.)):
                    for n, kind, e, seen in ((1, S.AIMED[0], first, 0), (2, S.AIMED[1], 3. if target > .7 else 7., 1), (3, S.AIMED[2], 2., 2), (3, S.AIMED[3], 1., 1)):
                        aimed.append(dict(pair="Y01", strip_sd=s, board=board, attempt=n, inference=kind, method=m, target_x=target, mu_true=.14, mu_median=.14,
                                          post_sd=.05, peak_speed=.5, x_rest=target + e/1e3, error_mm=e, mu_shot=.147, valid=True, observed=seen))
    (tmp_path/"settings.json").write_text(json.dumps(dict(pairs=["Y01"], boards_per_pair=3, peak_cap=1.019, targets=[.6, .77], strip_sd=[0., .12])))
    for name, fields, rows in (("results.csv", S.FIELDS, base), ("aimed.csv", S.AIMED_FIELDS, aimed)):
        with (tmp_path/name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    text = S.summarize_aimed(tmp_path)
    row = [l for l in text.splitlines() if l.strip().startswith("0.12 |")][0]
    assert [c.strip() for c in row.split("|")[1:]] == ["40.0 / 41.0 / 48.0                  5.0", "8.0 / 8.0 / 8.0", "6.0 / 6.0 / 6.0", "5.0 / 5.0 / 5.0",
                                                         "2.0 / 2.0 / 2.0", "1.0 / 1.0 / 1.0"]
    near = [l for l in text.splitlines() if l.strip().startswith("0.12   0.6 m")][0]
    assert [c.strip() for c in near.split("|")[2:]] == ["8.0", "6.0", "7.0", "2.0", "1.0"]
    assert "check: 24 of 24 first shots land where the saved run's no-practice shots landed" in text
    assert "96 shots (72 after practice), 0 flagged invalid, 0 planned with a practice slide too short to observe" in text
    assert "+0.049, 0.000" in text and "100%" in text and "0%" in text      # ln(.147/.14); |ln| inside 1.645*.05, outside 1.645*.006
    assert (tmp_path/"aimed_summary.txt").read_text() == text + "\n"


def test_pair_set_priors_carry_a_width_control():
    """The pair set's methods: generic, lookup, the LLM, and the generic median at the LLM's width; own boards per pair."""
    from belief import posterior
    from paths import DATA
    c = S.set_priors("claude", DATA/"llm_standin_claude_haiku.json")
    assert len(c["pairs"]) == 23 and c["targets"] == [0.6, 0.735, 0.87]
    for pid in ("L06", "L15", "Y01"):
        ini = c["initials"][pid]
        assert list(ini) == ["generic", "lookup", "llm_claude", "generic@claude"]
        gen, llm, ctrl = (posterior(c["g"], ini[m]) for m in ("generic", "llm_claude", "generic@claude"))
        assert math.isclose(ctrl["mu_median"], gen["mu_median"], rel_tol=2e-3) and math.isclose(ctrl["sd"], llm["sd"], rel_tol=2e-2)
        assert math.isclose(llm["mu_median"], c["answers"][pid]["median"], rel_tol=2e-3)
    assert c["answers"]["L15"]["lookup"] == c["pairs"]["L15"]["mu"] == 0.15           # the table holds the handbook truth
    seeds = {S.set_seed(p, b) for p in c["pairs"] for b in range(8)}
    assert len(seeds) == 23*8 and S.set_seed("Y02", 3) == S.board_seed("Y02", 3)


def test_pair_set_summary(tmp_path):
    """summarize_aimed_set: misses by method and rule, the reference line, the paired differences over pairs."""
    methods = ["generic", "lookup", "llm_claude", "generic@claude"]
    info = {"L01": dict(mu=.42, object="hard steel block", surface="mild steel plate", truth_file="ground_truth_handbook.csv"),
            "L02": dict(mu=.30, object="brass block", surface="mild steel plate", truth_file="ground_truth_handbook.csv"),
            "Y01": dict(mu=.14, object="stainless steel block", surface="ABS sheet", truth_file="ground_truth_published.csv")}
    answers = {"L01": dict(median=.45, low=.3, high=.65, lookup=.42), "L02": dict(median=.3, low=.2, high=.45, lookup=.30), "Y01": dict(median=.35, low=.2, high=.5, lookup=None)}
    (tmp_path/"settings.json").write_text(json.dumps(dict(pairs=list(info), pair_info=info, answers=answers, llm=dict(name="claude", cache="c.json", model="m"),
                                                          methods=methods, strip_sd=[0., .12], boards_per_pair=2, peak_cap=1.019, targets=[.6, .7], fixed_practice=[1.01, .7])))
    err = {S.AIMED[0]: dict(zip(methods, (40., 4., 14., 20.))), S.AIMED[1]: dict(zip(methods, (6., 3., 4., 5.))),
           S.AIMED[2]: dict(zip(methods, (3., 2., 2., 3.))), S.AIMED[3]: dict(zip(methods, (3., 3., 3., 3.)))}
    rows = []
    def add(pair, s, board, n, kind, m, target, e, seen):
        rows.append(dict(pair=pair, strip_sd=s, board=board, attempt=n, inference=kind, method=m, target_x=target, mu_true=info[pair]["mu"], mu_median=.3,
                         post_sd=.05, peak_speed=.5, x_rest=target + e/1e3, error_mm=e, mu_shot=.3, valid=True, observed=seen))
    for pair in info:
        for s in (0.0, 0.12):
            for board in range(2 if s else 1):
                for target in (.6, .7):
                    for kind, n, seen in zip(S.AIMED, (1, 2, 3, 3), (0, 1, 2, 1)):
                        for m in methods:
                            add(pair, s, board, n, kind, m, target, err[kind][m] + (1. if pair == "Y01" and m == "llm_claude" else 0.), seen)
                    add(pair, s, board, 0, "nominal plan", "oracle", target, 5., 0)
                    add(pair, s, board, 2, S.fixed_label(1.01), "generic", target, 8., 1)
                    add(pair, s, board, 2, S.fixed_label(.7), "generic", target, 6., 1)
    with (tmp_path/"aimed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=S.AIMED_FIELDS); w.writeheader(); w.writerows(rows)
    text = S.summarize_aimed_set(tmp_path)
    lines = text.splitlines()
    first = lines.index("all 3 pairs")
    one = [l for l in lines[first:] if l.split("|")[0].split()[:1] == ["0.12"] and "1 aimed slide" in l][0]
    assert one.split("|")[1].split() == ["6.0", "3.0", "4.3", "5.0"]                    # the LLM is 1 mm worse on one of three pairs
    assert "fixed 1.01 m/s 8.0; fixed 0.7 m/s 6.0; nominal plan 5.0" in text
    assert "the 2 handbook pairs" in text and "the other 1 pairs (Y01)" in text
    paired = [l for l in lines if l.split("|")[0].split()[:1] == ["0.12"] and "1 aimed slide" in l and "+/-" in l][0]
    assert "-3.0 +/- 0.0 (3/3)" in paired and "-1.7 +/- 0.3 (3/3)" in paired and "-0.7 +/- 0.3 (2/3)" in paired
    assert "L01   hard steel block on mild steel plate" in text and "0.45 [0.30, 0.65]" in text
    assert "-2.0 +/- 0.0 (3/3)" in paired                                               # lookup against the control
    assert f"{len(rows)} shots, 0 flagged invalid" in text and "0 of 216 shots after aimed practice" in text
    assert "by target, all pairs" in text and "how near the medians are" in text
    assert (tmp_path/"summary.txt").read_text() == text + "\n"

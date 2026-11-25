"""Tests of experiments/base_trial.py's limit-aware posture choice (planning only; no simulation)."""
import pytest
import arm
import base_trial as B

BASE, CAP = (0.6, 0.65, 0.0), 6.0


def _worst(u):
    r = arm.feasibility(u, 100.0, BASE, 0.55, CAP)
    assert r["ok"]
    return max(r[k] for k in B.KEYS)


def test_limit_aware_is_a_patch_that_is_taken_off_again():
    """Outside the block, and with on=False, ArmPusher plans as arm.py does."""
    original = arm.ArmPusher.make_plan
    with B.limit_aware(False):
        assert arm.ArmPusher.make_plan is original
    with B.limit_aware():
        assert arm.ArmPusher.make_plan is not original
    assert arm.ArmPusher.make_plan is original
    with pytest.raises(ZeroDivisionError):
        with B.limit_aware():
            1/0
    assert arm.ArmPusher.make_plan is original


def test_limit_aware_posture_brings_a_fast_push_inside_the_limits():
    """At the default base with a 6 m/s^2 brake, a 0.9 m/s push is over the acceleration limit with arm.py's posture (lowest
    joint speed per m/s) and inside all three limits with the posture chosen for the lowest worst ratio; the choice is never
    worse than arm.py's, and arm.py's result is the same before and after."""
    before = _worst(0.9)
    with B.limit_aware():
        aware = _worst(0.9)
        slow = _worst(0.2)
    assert before > 1.0 > aware
    assert slow < 1.0
    assert _worst(0.9) == before


# ---------------------------------------------------------------- the option in arm.py (ArmPusher.posture, 19 Nov)

def _plan(u, **options):
    from push import profile
    pu = arm.ArmPusher(base=BASE, a_brake_max=CAP, **options)
    pu.x_touch0, pu.model = 0.499, arm.ArmModel(pu.base)
    pu.X0 = arm.link7_pose(pu.x_touch0, 0.0, 0.0, pu.clearance)
    return pu.make_plan(profile(u, 100.0, a_max=CAP))


def test_the_posture_option_defaults_to_the_old_choice():
    assert arm.POSTURE == "speed" == arm.ArmPusher().posture and arm.POSTURES == ("speed", "limits")
    with pytest.raises(ValueError):
        arm.ArmPusher(posture="fastest")
    assert _plan(0.9)["posture"] == "speed"


def test_the_posture_option_plans_what_the_patch_planned():
    """ArmPusher(posture="limits") gives the joint trajectory the trial's patch gave, at a slow and a fast push, so the trial's
    results in results/side_base/base_trial/ describe the option."""
    import numpy as np
    for u in (0.2, 0.9):
        option = _plan(u, posture="limits")
        with B.limit_aware():
            patch = _plan(u)
        for k in ("q_d", "qd_d", "qdd_d", "tau_ff"):
            assert np.array_equal(option[k], patch[k]), (u, k)
        assert max(option[k] for k in B.KEYS) < 1.0
    assert max(_plan(0.9)[k] for k in B.KEYS) > 1.0             # arm.py's default posture at the same push


def test_limit_scan_and_the_evaluators_posture_option():
    """limit_scan reports every speed (limited_peak_speed's bisection only finds the fastest).  The evaluator passes the
    posture on and records it; without the option its setup and record are the ones saved runs have."""
    import math
    from types import SimpleNamespace
    import numpy as np
    import run_arm_evaluation as E
    scan = dict(arm.limit_scan((0.2, 0.9), base=BASE, a_max=CAP, posture="limits"))
    assert max(scan.values()) < 1.0 and dict(arm.limit_scan((0.9,), base=BASE, a_max=CAP))[0.9] > 1.0
    post = {"grid": np.linspace(math.log(0.02), math.log(2.0), 2000)}
    post["pmf"] = np.exp(-0.5*((post["grid"] - math.log(0.3))/0.8)**2); post["pmf"] /= post["pmf"].sum()
    truth = {"Y01": 0.14, "Y02": 0.14, "Y03": 0.26}
    v3 = E.arm_setup(SimpleNamespace(targets=[0.60, 0.62, 0.64]), truth, post)
    assert v3["options"] == {} and "posture" not in v3["record"] and v3["version"] == E.VERSION
    new = E.arm_setup(SimpleNamespace(layout="side", a_brake=6.0, check_accel=True, posture="limits", peak_cap=1.1, practice_peak=1.0,
                                      targets=[0.6, 0.8]), truth, post)
    assert new["options"] == {"base": BASE, "a_brake": 6.0, "check": ("q", "v", "tau", "a"), "posture": "limits"}
    assert new["record"]["posture"] == "limits" and new["version"] == E.VERSION_LAYOUT

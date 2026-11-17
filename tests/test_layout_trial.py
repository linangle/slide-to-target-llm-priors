"""The 16 Nov layout trial: named base positions, the acceleration check, the computed speed cap, targets re-derived from
it, and the evaluator's arm setup.  Planning only (no simulated push); the evaluator's defaults must stay the v3 protocol."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from physics import G

SIDE = (0.6, 0.65, 0.0)

def test_layouts_and_the_acceleration_report():
    """Beside the board joint 1 carries the push instead of the elbow.  The planned joint accelerations are set by the brake
    cap: over the SDF limit at the arm's 12 m/s^2, inside it at 4 m/s^2, at either base."""
    from arm import LAYOUTS, ArmModel, ArmPusher, feasibility, joint_limit_report, layout_base, link7_pose
    from push import profile
    assert layout_base("origin") == (0.0, 0.0, 0.0) and layout_base("side") == SIDE == LAYOUTS["side"] and layout_base((0.3, 0.5)) == (0.3, 0.5, 0.0)
    hard, soft = feasibility(0.45, 100.0, a_max=12.0), feasibility(0.45, 100.0, a_max=4.0)
    assert hard["qd_joint"] == 4 and hard["qdd_ratio"] > 1.5 and soft["qdd_ratio"] < 1.0 and abs(hard["qd_ratio"] - soft["qd_ratio"]) < 0.02
    side = feasibility(1.0, 100.0, SIDE, a_max=4.0)
    assert side["ok"] and side["qd_joint"] == 1 and max(side["qd_ratio"], side["tau_ratio"], side["qdd_ratio"]) < 1.0
    assert feasibility(1.0, 100.0, SIDE, a_max=12.0)["qdd_ratio"] > 1.5
    pu = ArmPusher(base=SIDE, a_brake_max=4.0)                    # joint_limit_report carries the same number under "a"
    pu.x_touch0, pu.model = 0.499, ArmModel(pu.base)
    pu.X0 = link7_pose(pu.x_touch0, 0.0, 0.0, pu.clearance)
    pu.make_plan(profile(1.0, 100.0, a_max=4.0))
    jl = joint_limit_report(pu.traj)
    assert set(jl) == {"q", "v", "tau", "a"} and jl["a"].shape == (7,) and abs(jl["a"].max() - side["qdd_ratio"]) < 1e-9

def test_speed_cap_within_speed_torque_and_acceleration():
    from arm import limited_peak_speed
    v_only = limited_peak_speed(100.0, base=SIDE)
    assert 0.95 < v_only < 1.08                                            # joint 1 at 85 deg/s on a radius of about 0.65 m
    assert abs(limited_peak_speed(100.0, which=("v", "tau", "a"), base=SIDE, a_max=4.0) - v_only) < 0.02    # a soft brake costs no speed
    assert limited_peak_speed(100.0, which=("v", "tau", "a"), base=SIDE, a_max=12.0) < 0.5                    # the 12 m/s^2 brake does

def test_targets_follow_from_the_cap_and_requests_beyond_it_are_clipped():
    from arm import affordable, plan_peak_speed, rest_x, target_grid
    a = affordable(1.0, 0.3)
    v_r = 1.0 - (0.3*G)**2/200.0
    assert abs(a["slide"] - v_r**2/(2*0.3*G)) < 1e-12 and abs(a["x_rest"] - rest_x(1.0, 0.3, 100.0)) < 1e-12
    assert abs(a["x_rest"] - a["x_release"] - a["slide"]) < 1e-12
    targets = target_grid(1.017, 0.26)
    assert targets == [0.6, 0.77, 0.94]
    for mu in (0.14, 0.26):                              # every pair reaches every target below the cap
        assert all(plan_peak_speed(t, mu, u_max=1.017) <= 0.9*1.017 + 1e-9 for t in targets)
    assert target_grid(0.45, 0.26)[-1] < 0.66 and len(target_grid(1.017, 0.26, n=5)) == 5
    with pytest.raises(ValueError):
        target_grid(0.2, 0.5)                             # nothing beyond x0 + 5 cm is in reach
    with pytest.raises(ValueError):
        plan_peak_speed(0.94, 0.5, u_max=0.45)
    assert plan_peak_speed(0.94, 0.5, u_max=0.45, clip=True) == 0.45      # pushed at the cap; the miss is scored
    assert plan_peak_speed(0.60, 0.5, u_max=0.45, clip=True) < 0.45       # within reach: unchanged

def test_the_evaluators_arm_setup():
    import run_arm_evaluation as E
    post = {"grid": np.linspace(math.log(0.02), math.log(2.0), 2000)}
    post["pmf"] = np.exp(-0.5*((post["grid"] - math.log(0.3))/0.8)**2); post["pmf"] /= post["pmf"].sum()
    truth = {"Y01": 0.14, "Y02": 0.14, "Y03": 0.26}
    v3 = E.arm_setup(SimpleNamespace(targets=[0.60, 0.62, 0.64]), truth, post)          # no options: the saved protocol
    assert (v3["cap"], v3["practice"], v3["targets"], v3["options"]) == (0.45, 0.45, [0.60, 0.62, 0.64], {})
    assert v3["record"]["layout"] == "origin" and v3["record"]["brake_cap"] == 12.0 and v3["record"]["checked_limits"] == ["q", "v", "tau"]
    trial = E.arm_setup(SimpleNamespace(layout="side", a_brake=4.0, check_accel=True, peak_cap=1.0, practice_peak="auto", targets=["auto"],
                                        clip_unreachable=True), truth, post)
    assert trial["options"] == {"base": SIDE, "a_brake": 4.0, "check": ("q", "v", "tau", "a")}
    assert trial["cap"] == 1.0 and 0.3 < trial["practice"] <= 1.0 and trial["record"]["clip_unreachable"]
    assert trial["targets"][0] == 0.6 and len(trial["targets"]) == 3 and 0.85 < trial["targets"][-1] < 1.0

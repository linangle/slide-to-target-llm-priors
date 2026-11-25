"""e1_motion_generators.py: Drake's differential IK and kinematic trajectory optimisation as the push's motion generator.
Plans only (no contact simulation); a few seconds each."""
import numpy as np
import pytest
import e1_motion_generators as E
from arm import ArmModel, ArmPusher, LAYOUTS, link7_pose
from physics import OBJECT_SIZE
from push import profile

pytestmark = pytest.mark.slow

def test_current_is_the_arm_py_plan_unchanged():
    plan, prof = E.plan_only("side", "current", 0.5)
    pu = ArmPusher(base=tuple(LAYOUTS["side"]), a_brake_max=6.0, posture="limits")
    pu.x_touch0 = E.X0_OBJ - OBJECT_SIZE[0]/2 - pu.gap
    pu.model = ArmModel(pu.base)
    pu.X0 = link7_pose(pu.x_touch0, 0.0, 0.0, pu.clearance)
    ref = pu.make_plan(profile(0.5, E.J_BRAKE, a_max=6.0))
    assert np.array_equal(plan["q_d"], ref["q_d"]) and np.array_equal(plan["tau_ff"], ref["tau_ff"])
    assert plan["gen"]["path_x_err_mm"] < 1e-3 and plan["gen"]["path_yz_err_mm"] < 1e-3      # the current plan is on the line by construction

def test_diffik_follows_the_command_where_no_limit_binds():
    plan, prof = E.plan_only("side", "diffik", 0.5)
    g = plan["gen"]
    assert not g["clipped"] and g["alpha_min"] > 0.98 and g["stuck_steps"] == 0
    assert abs(g["peak_paddle_speed"] - 0.5) < 0.005 and g["path_x_err_mm"] < 0.05 and g["path_yz_err_mm"] < 0.05
    assert plan["qd_ratio"] <= 1.001 and len(plan["t"]) == len(plan["q_d"]) == len(plan["tau_ff"])
    ref, _ = E.plan_only("side", "current", 0.5)                 # the same start posture and the same line: the same joint motion
    assert abs(plan["qdd_ratio"]/ref["qdd_ratio"] - 1) < 0.05 and abs(plan["qd_ratio"]/ref["qd_ratio"] - 1) < 0.02
    assert plan["t"][-1] >= prof["t_stop"] + E.TAIL and np.all(np.diff(plan["t"]) > 0)       # the plan's own grid runs past the commanded stop

def test_diffik_is_clipped_at_the_origin_base_and_says_nothing():
    """With the base on the push line joint 4's speed limit caps the paddle near 0.49 m/s.  Asked for 0.7, Drake's differential IK
    returns a solution at every step and delivers a slower push."""
    plan, _ = E.plan_only("origin", "diffik", 0.7)
    g = plan["gen"]
    assert g["clipped"] and g["stuck_steps"] == 0 and g["alpha_min"] < 0.9
    assert 0.40 < g["peak_paddle_speed"] < 0.56 and g["lag_max_mm"] > 5.0
    assert plan["qd_ratio"] <= 1.001                                                           # the velocity limit is a hard bound

def test_trajopt_is_inside_all_three_limits_and_on_the_commanded_line():
    plan, prof = E.plan_only("side", "trajopt", 0.8)
    g = plan["gen"]
    assert max(plan["qd_ratio"], plan["qdd_ratio"], plan["tau_ratio"]) <= 1.001
    assert g["path_x_err_mm"] < 0.2 and g["path_yz_err_mm"] < 0.2 and g["path_tilt_deg"] < 0.06
    assert np.allclose(plan["qd_d"][0], 0) and np.allclose(plan["qd_d"][-1], 0)                # at rest at both ends

def test_trajopt_refuses_a_push_it_cannot_make_inside_the_bounds():
    with pytest.raises(RuntimeError, match="no trajectory inside the bounds"):
        E.plan_only("origin", "trajopt", 0.7)

def test_generator_name_is_checked():
    with pytest.raises(ValueError, match="generator is one of"):
        E.plan_only("side", "spline", 0.5)

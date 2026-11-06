"""arm.py: target-point planning, the paddle line, and one arm push against the plate's bars (tests/test_push.py)."""
import math
import numpy as np
import pytest
from arm import P_FACE, R_PUSH, ArmPusher, _arm_model, joint_trajectory, plan_peak_speed, rest_x, slide_to_target, solve_ik
from push import profile

def test_plan_lands_on_target_under_the_model():
    for x_t, mu, J in ((0.75, 0.3, 100.0), (0.70, 0.5, 300.0), (0.9, 0.15, 100.0)):
        u = plan_peak_speed(x_t, mu, J)
        assert abs(rest_x(u, mu, J) - x_t) < 1e-6

def test_plan_rejects_unreachable_targets():
    with pytest.raises(ValueError):
        plan_peak_speed(0.50, 0.3)            # behind the start
    with pytest.raises(ValueError):
        plan_peak_speed(1.50, 0.5)            # farther than the arm's stroke can send it

def test_plan_with_a_belief_matches_a_point_belief():
    """A belief concentrated on one mu gives the same speed as that mu; a spread belief stays close."""
    grid = np.linspace(math.log(0.05), 0.0, 500)
    u0 = plan_peak_speed(0.8, 0.3)
    point = {"grid": grid, "pmf": np.exp(-0.5*((grid - math.log(0.3))/0.002)**2)}
    point["pmf"] /= point["pmf"].sum()
    assert abs(plan_peak_speed(0.8, point) - u0) < 2e-3

def test_ik_puts_the_face_square_on_the_line():
    plant, _, paddle = _arm_model()
    ctx = plant.CreateDefaultContext()
    for x in (0.50, 0.70, 0.90):
        plant.SetPositions(ctx, solve_ik([x, 0.0, 0.093]))
        X = plant.CalcRelativeTransform(ctx, plant.world_frame(), paddle.body_frame())
        assert np.allclose(X @ P_FACE, [x, 0.0, 0.093], atol=1e-5)
        assert X.rotation().IsNearlyEqualTo(R_PUSH, 1e-4)

def test_joint_trajectory_follows_the_profile():
    tr = joint_trajectory([0.499, 0.0, 0.093], profile(0.4, 100.0))
    assert tr["face_err"].max() < 2e-5
    assert np.isfinite(tr["tau"]).all()

@pytest.mark.slow
def test_arm_push_meets_the_plate_bars():
    """test_push.py's case with the arm in place of the plate: mu 0.3, v_peak 0.8, J 100."""
    from push import push_slide
    r = push_slide(0.3, 0.8, 100.0, pusher=ArmPusher(), keep_traj=False)
    assert r["n_point_contacts"] == 0 and r["separated"]
    assert abs(r["v_r_cmd_err_pct"]) < 1.0 and abs(r["v_r_act_err_pct"]) < 1.0
    assert abs(r["sep_lag_ms"]) < 3.0
    assert abs(r["d_err_pct"]) < 2.0 and abs(r["psi_r_deg"]) < 0.2 and abs(r["yaw_rest_deg"]) < 0.1
    assert abs(r["x_fk"] - r["x_sep"]) < 2e-4
    assert r["track_q_rms"] < 2e-4

@pytest.mark.slow
def test_slide_to_target_with_true_mu():
    r = slide_to_target(0.75, 0.3)
    assert abs(r["miss"]) < 0.005 and abs(r["miss_y"]) < 0.002

"""The layouts (16 Nov): named base positions with the side base as the default, the board shifted clear of it, the camera's
coverage, the acceleration check, the computed speed cap, targets derived from it, and the evaluator's arm setup.  Planning
only (no simulated push).  The evaluator called with a bare namespace, and with --v3, must stay the saved v3 protocol."""
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
    origin = LAYOUTS["origin"]
    hard, soft = feasibility(0.45, 100.0, origin, a_max=12.0), feasibility(0.45, 100.0, origin, a_max=4.0)
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

def test_the_default_layout_board_and_camera():
    """The default is the side base.  Its footprint reaches y = 0.53 m, so the board is shifted 10 cm away and clears it; the
    origin layout keeps the board centred.  The default camera sees the start of the push and the board to its far end."""
    import scene
    from scene import BASE, BOARD_Y, CAMERAS, DEFAULT_LAYOUT, LAYOUTS, Camera, Table, board_y, layout_of
    from physics import BOARD_X_MAX
    assert DEFAULT_LAYOUT == "side" and BASE == LAYOUTS["side"] == SIDE and layout_of(SIDE) == "side" and layout_of((0, 0, 0)) == "origin"
    assert BOARD_Y["side"] == board_y(SIDE) == -0.10 and board_y(LAYOUTS["origin"]) == 0.0 and board_y(None) == 0.0 and board_y((0.3, 0.5, 0.0)) == 0.0
    assert Table().y_centre is None and board_y(SIDE) + Table().width/2 == pytest.approx(0.50)      # the board's edge on the arm's side
    assert Camera().position == CAMERAS["full"] and CAMERAS["original"] == (0.75, 0.0, 0.8)
    half_x = Camera().position[2]*math.tan(Camera().fov_y/2)*Camera().width/Camera().height      # half the view along x on the table
    assert Camera().position[0] - half_x < 0.55 - 0.05 - 0.02 and Camera().position[0] + half_x > BOARD_X_MAX
    x, _, z = CAMERAS["original"]
    assert x + z*math.tan(Camera().fov_y/2)*Camera().width/Camera().height < 1.25                  # what the original lost

def test_link_clearance_at_the_default_base():
    """experiments/arm_clearance.py at the speed cap: every moving link stays well above the board and off the object, the
    base clears the shifted board's edge (and would overlap a centred one), and nothing comes near the camera."""
    from arm import limited_peak_speed
    from arm_clearance import clearance
    c = clearance(SIDE, limited_peak_speed(100.0, base=SIDE))
    moving = {k: v for k, v in c["links"].items() if k not in ("link 0", "paddle")}
    assert len(moving) == 7 and min(v["plane"] for v in moving.values()) > 0.04
    assert min(v["object"] for v in moving.values()) > 0.04 and min(v["camera"] for v in c["links"].values()) > 0.5
    assert abs(c["links"]["paddle"]["plane"] - 0.003) < 1e-3                # the paddle runs 3 mm above the board
    assert c["foot_overlap"] < -0.02                                        # clears the edge at y = +0.50 m
    assert clearance(SIDE, 0.5, board_centre=0.0)["foot_overlap"] > 0.05    # a centred 1.2 m board would be under the base

def test_cli_defaults_of_the_evaluator():
    """--v3 is the saved protocol; with no options the runner uses the side base with computed cap, practice speed and targets."""
    import subprocess, sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = lambda *args: subprocess.run([sys.executable, os.path.join(root, "evaluation", "run_arm_evaluation.py"), "--pair", "all", "--seeds", "7",
                                        "--check-only", *args], capture_output=True, text=True, check=True).stdout
    v3 = run("--v3")
    assert "'layout': 'origin'" in v3 and "peak cap 0.45 m/s, practice 0.45 m/s, targets [0.6, 0.62, 0.64]" in v3
    new = run()
    assert "'layout': 'side'" in new and "targets [0.6, 0.77, 0.94]" in new and "peak cap 1.0" in new

def test_height_gate_drops_paddle_face_pixels_but_keeps_the_object():
    """perception.segment's face_gate: paddle-face pixels that depth noise puts just inside the x window sit at every
    height; within a few millimetres of the face only pixels at the object's top height are kept."""
    from perception import segment
    rows, cols = np.mgrid[0:200, 0:300]
    P = np.dstack([cols*0.001, rows*0.001, np.zeros((200, 300))])            # a 1 mm grid on the table plane z = 0
    obj = (P[..., 0] >= 0.060) & (P[..., 0] < 0.160) & (P[..., 1] >= 0.050) & (P[..., 1] < 0.150)
    P[..., 2][obj] = 0.050                                                   # the object's top, 5 cm up, its back edge at x = 0.060
    leak = (P[..., 0] >= 0.057) & (P[..., 0] < 0.060) & (P[..., 1] >= 0.060) & (P[..., 1] < 0.140)
    P[..., 2][leak] = 0.100                                                  # paddle-face pixels just ahead of the face at x = 0.0565
    plane, window = (np.array([0.0, 0.0, 1.0]), 0.0), (0.0570, 0.6)
    plain, _ = segment(P, plane, x_window=window)
    gated, _ = segment(P, plane, x_window=window, face_gate=(0.0565, 0.004, 0.050, 0.006))
    inside = leak & (P[..., 0] > window[0])                                  # the leaked pixels the x window lets through
    assert plain[:, 0].min() < 0.0595 and len(plain) == obj.sum() + inside.sum() > obj.sum()    # they join the object's blob
    assert gated[:, 0].min() == pytest.approx(0.060) and len(gated) == obj.sum()     # gated: the object alone, back edge intact

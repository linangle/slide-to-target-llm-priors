"""The 20 Nov trial wrapper (evaluation/run_trial.py): the two ways of measuring a miss (D4), without Drake."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"evaluation"))
import arm
import run_trial as T

def test_the_model_release_point_is_the_planners():
    for mu, u, J in ((0.14, 0.9, 100.0), (0.26, 0.5, 500.0)):
        v_r = u - (mu*arm.G)**2/(2*J)
        assert math.isclose(T.release_model_x(u, mu, J) + v_r*v_r/(2*mu*arm.G), arm.rest_x(u, mu, J), abs_tol=1e-12)

def test_a_start_gap_error_moves_the_rest_measure_and_not_the_slide_measure():
    mu, u, J, target = 0.26, 0.6, 100.0, 0.77
    x_fk, x_rest = T.release_model_x(u, mu, J), 0.7712           # nominal gap: FK reports the model's release point
    nominal = T.miss_slide(x_rest, x_fk, target, u, mu, J)
    assert math.isclose(nominal, x_rest - target, abs_tol=1e-12)  # no gap error: the two measures agree
    shift = -0.002                                                # paddle starts 2 mm further back: release and rest move with it
    assert math.isclose(T.miss_slide(x_rest + shift, x_fk + shift, target, u, mu, J), nominal, abs_tol=1e-12)
    assert math.isclose((x_rest + shift) - target, nominal + shift, abs_tol=1e-12)

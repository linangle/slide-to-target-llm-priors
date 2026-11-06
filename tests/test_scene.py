"""scene.Table is the 48 in board (5 Nov decision 1): its extent, its strip tiling, and that the E0 runs fit on it."""
import numpy as np
from physics import BOARD_LENGTH, BOARD_X_MAX, BOARD_X_MIN, OBJECT_SIZE
from scene import Obj, Table, strip_targets

def test_table_is_the_48_in_board():
    t = Table()
    assert abs(BOARD_LENGTH - 48*0.0254) < 1e-12 and abs(t.length - BOARD_LENGTH) < 1e-12
    e = t.edges()
    assert e[0] == BOARD_X_MIN and abs(e[-1] - BOARD_X_MAX) < 1e-12 and len(e) == t.n_strips + 1 == 25
    assert np.allclose(np.diff(e), 2*0.0254)                    # 24 equal strips of 2 in (nominal 5 cm)
    assert len(strip_targets(0.3, 0.0, t, 0)) == t.n_strips
    # the default start (x0 = 0.55) and the longest E0 slide (E0-4: rest at x = 1.35 m) both sit on the board
    half = Obj().size[0]/2
    assert Obj().size == OBJECT_SIZE
    assert BOARD_X_MIN < 0.55 - half and 1.35 + half < BOARD_X_MAX
    # a table that is a whole number of strips tiles at exactly the strip length; a short one is a single strip
    assert np.allclose(np.diff(Table(x_min=0.0, x_max=1.0).edges()), 0.05)
    assert Table(x_min=0.0, x_max=0.02).edges().tolist() == [0.0, 0.02]

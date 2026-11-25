#!/usr/bin/env python3
"""
intro_figure.py -- the first-page figure: a strobe side view of one simulated push and slide.

One push-and-slide is simulated with the arm (push.push_slide with arm.ArmPusher, the contact settings in sim_config.yaml),
then drawn as a multiple-exposure photograph seen from the side (camera on -y, looking along +y: x right, z up):
  * the object is drawn at equal time steps from the start of the push until it is at rest, so the spacing between
    exposures shows its speed: growing while the paddle accelerates it, shrinking as friction brakes the free slide;
  * the iiwa is drawn at a few of the same instants.  Each link's outline is the side projection of the convex hull of its
    Drake visual mesh, posed by forward kinematics from the simulated joint angles, so the arm is the one in scene.py;
  * one exposure falls exactly on the release (t*, from the release model), and the last one is the rest pose;
  * the board, its friction strips, the start x_0, the release x_r, the push, the slide d and the target x* are annotated
    (no equation or exposure note in the image: those go in the caption).

Colour: the arm, board and labels are grey or ink; the object is blue and the target red (CVD-checked: protan dE 21).
Every exposure is opaque, pre-blended with the background and drawn over the older ones, so overlapping exposures never
stack up darker: the object's copies are a tint that darkens steadily towards the rest pose (the only solid one, on the
red target strip), and the arm's are lighter when older, solid at the release.
The default push is the acceptance push (0.8 m/s, J = 100 m/s^3, base at the origin) at mu = 0.15, chosen so that the
slide (21 cm) is about as long as the push.  The board runs off the right edge (its far end is 0.6 m past the rest point).
The figure is drawn at print size for one IEEE column (3.5 in, 7.5 pt labels): include it at width = column width.

    python3 figures/intro_figure.py                         # intro_figure.pdf and .png, one IEEE column wide (3.5 in) (~4 s)
    python3 figures/intro_figure.py --mu 0.15 --v 0.8 --J 100 --dt-strobe 0.08 --out intro_figure
    python3 figures/intro_figure.py --no-labels             # artwork only, to label in the report
"""
import argparse, math
import numpy as np
import matplotlib
from paths import FIGURES
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from scipy.spatial import ConvexHull
from pydrake.all import AddMultibodyPlantSceneGraph, Box, DiagramBuilder, Mesh, RigidTransform, Role

from arm import ArmPusher
from push import push_slide
from physics import BOARD_X_MIN, OBJECT_SIZE
from scene import add_arm

# colours: greys and ink, one blue for the object, one red for the target
BG = "#ffffff"
ARM_EDGE = "#3d434b"
ARM_FILL = {"grey": "#d6d9dd", "orange": "#b4b9c0", "band": "#a3a9b1", "paddle": "#6f757d"}   # the iiwa's orange covers in grey
OBJ, OBJ_EDGE = "#2f6db5", "#1b3f6b"
TARGET = "#d1352b"
BOARD_LIGHT, BOARD_DARK, BOARD_EDGE, MOUNT = "#f2f2f0", "#e5e5e2", "#9a9a96", "#d4d4d0"
INK, MUTED = "#262626", "#6b6b6b"

def blend(color, alpha, bg=BG):
    """The colour `alpha` of the way from the background to `color` (an opaque stand-in for a translucent exposure)."""
    c, b = np.array(matplotlib.colors.to_rgb(color)), np.array(matplotlib.colors.to_rgb(bg))
    return tuple(b + alpha*(c - b))

class ArmOutline:
    """Side-view outlines of the iiwa's visual geometry (convex hull of each mesh, projected on x-z) at given joint angles."""

    def __init__(self, base=(0.0, 0.0, 0.0)):
        builder = DiagramBuilder()
        self.plant, sg = AddMultibodyPlantSceneGraph(builder, 0.0)
        self.iiwa, _ = add_arm(self.plant, None, RigidTransform(np.asarray(base, float)), collision=False)
        self.plant.Finalize()
        self.ctx = self.plant.CreateDefaultContext()
        insp = sg.model_inspector()
        self.parts = []   # (body, X_BG, vertices in G, colour key)
        for gid in insp.GetAllGeometryIds(Role.kIllustration):
            name, shape = insp.GetName(gid), insp.GetShape(gid)
            if "kuka" in name:   # the logo plate
                continue
            body = self.plant.GetBodyFromFrameId(insp.GetFrameId(gid))
            if isinstance(shape, Mesh):
                hull = shape.GetConvexHull()   # in the geometry frame, scale applied
                verts = np.array([hull.vertex(i) for i in range(hull.num_vertices())])
            elif isinstance(shape, Box):
                verts = 0.5*np.array([[sx*shape.width(), sy*shape.depth(), sz*shape.height()]
                                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            else:
                continue
            key = "orange" if "orange" in name else "band" if "band" in name else "paddle" if "paddle" in name else "grey"
            self.parts.append((body, insp.GetPoseInFrame(gid), verts, key))

    def polygons(self, q):
        """[(xz polygon, colour key, depth y)] at joint angles q, far (large y) first."""
        self.plant.SetPositions(self.ctx, self.iiwa, q)
        out = []
        for body, X_BG, verts, key in self.parts:
            X_WG = self.plant.EvalBodyPoseInWorld(self.ctx, body) @ X_BG
            pw = (X_WG.rotation().matrix() @ verts.T).T + X_WG.translation()
            xz = pw[:, [0, 2]]
            out.append((xz[ConvexHull(xz).vertices], key, float(pw[:, 1].mean())))
        return sorted(out, key=lambda o: -o[2])

def strobe_times(t_start, t_sep, t_rest, dt):
    """Equal steps through the release instant t_sep, from the first step after the push starts to the last one before rest."""
    k0, k1 = math.ceil((t_start - t_sep)/dt), math.floor((t_rest - t_sep)/dt)
    return t_sep + dt*np.arange(k0, k1 + 1), -k0   # times and the index of the release exposure

def draw(r, out, dt_strobe=0.08, arm_at=None, labels=True, width_in=3.5):
    tr = r["traj"]
    t, x = tr["t"], tr["x"]
    sx, _, sz = OBJECT_SIZE
    z0 = float(np.median(tr["z"][-50:])) - sz/2        # object bottom at rest (the hydroelastic sink is sub-mm)
    t_start = float(t[np.argmax(x > x[0] + 1e-4)])     # the object starts to move
    times, k_rel = strobe_times(t_start, r["t_sep"], r["t_rest"], dt_strobe)
    times = np.append(times, t[-1])                    # the rest pose
    n = len(times)
    arm_k = sorted(set(arm_at if arm_at is not None else np.linspace(0, k_rel, 4).round().astype(int)))
    outline = ArmOutline()
    at = lambda tk: min(int(np.searchsorted(t, tk)), len(t) - 1)
    x_0, x_r, x_rest = x[at(times[0])], r["x_sep"], r["x_rest"]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7.5,    # IEEE column: 10 pt body, 8 pt captions
                         "pdf.fonttype": 42})                              # TrueType, not Type 3 (IEEE PDF eXpress)
    xlim, zlim = (-0.17, 1.10), (-0.20, 0.62)
    fig = plt.figure(figsize=(width_in, width_in*(zlim[1] - zlim[0])/(xlim[1] - xlim[0])))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("equal"); ax.axis("off")

    # the mount under the arm and the board with its friction strips (scene.Table: 2 in strips), running off both edges
    th = 0.05
    ax.add_patch(Rectangle((xlim[0] - 0.1, -th), BOARD_X_MIN - xlim[0] + 0.1, th, fc=MOUNT, ec=BOARD_EDGE, lw=0.6, zorder=1))
    w = 0.0508
    for i in range(int(math.ceil((xlim[1] - BOARD_X_MIN)/w)) + 1):
        ax.add_patch(Rectangle((BOARD_X_MIN + i*w, -th), w, th, fc=BOARD_DARK if i % 2 == 0 else BOARD_LIGHT, lw=0, zorder=1))
    ax.add_patch(Rectangle((BOARD_X_MIN, -th), xlim[1] + 0.5 - BOARD_X_MIN, th, fill=False, ec=BOARD_EDGE, lw=0.6, zorder=1))

    # the target: a red strip set into the board's top face, wider than the object so it shows on both sides of it
    pad = 0.025
    ax.add_patch(Rectangle((x_rest - sx/2 - pad, -0.011), sx + 2*pad, 0.011, fc=TARGET, lw=0, zorder=2))

    # exposures, oldest first, each opaque, pre-blended with the background and drawn over the older ones, so overlaps never
    # stack up and the shade only grows with time.  The arm: lighter when older, solid at the release.  The object: a tint
    # that darkens steadily towards the rest pose, the only solid one.
    for k, tk in enumerate(times):
        i = at(tk)
        if k in arm_k:
            a_arm = 1.0 if k == k_rel else 0.25 + 0.40*k/max(k_rel, 1)
            for poly, key, _ in outline.polygons(tr["q_arm"][i]):
                ax.add_patch(Polygon(poly, closed=True, fc=blend(ARM_FILL[key], a_arm), ec=blend(ARM_EDGE, a_arm), lw=0.5,
                                     joinstyle="round", zorder=3 + k*0.01))
        if k == n - 1:
            ax.add_patch(Rectangle((x[i] - sx/2, z0), sx, sz, fc=OBJ, ec=OBJ_EDGE, lw=0.8, zorder=4.5))
        else:
            a = 0.10 + 0.40*k/(n - 2)
            ax.add_patch(Rectangle((x[i] - sx/2, z0), sx, sz, fc=blend(OBJ, a), ec=blend(OBJ, a + 0.08), lw=0.4,
                                   zorder=3 + k*0.01 + 0.005))

    if not labels:
        fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
        return fig

    # dimension chain under the board: start, release, rest (the target's line in red)
    zd = -th - 0.065
    for xv, lab, c in ((x_0, r"$x_0$", INK), (x_r, r"$x_r$", INK), (x_rest, r"$x^*$", TARGET)):
        ax.plot([xv, xv], [-th - 0.006, zd - 0.018], color=c, lw=0.6, ls=(0, (2, 1.5)), zorder=2)
        ax.text(xv, zd - 0.024, lab, ha="center", va="top", color=c)
    for xa, xb, lab in ((x_0, x_r, "push"), (x_r, x_rest, r"slide $d$")):
        ax.add_patch(FancyArrowPatch((xa, zd), (xb, zd), arrowstyle="<|-|>", mutation_scale=6, lw=0.7, color=INK, zorder=5,
                                     shrinkA=0, shrinkB=0))
        ax.text((xa + xb)/2, zd + 0.009, lab, ha="center", va="bottom")

    # release: the label at the paddle and the release speed over the object's path
    i_rel = at(r["t_sep"])
    xa = x[i_rel] + 0.01
    ax.add_patch(FancyArrowPatch((xa, z0 + sz + 0.035), (xa + 0.10, z0 + sz + 0.035), arrowstyle="-|>", mutation_scale=6, lw=0.9,
                                 color=INK, zorder=6, shrinkA=0, shrinkB=0))
    ax.text(xa + 0.05, z0 + sz + 0.045, r"$v_r$", ha="center", va="bottom")
    ax.text(x_r - sx/2 + 0.012, z0 + sz + 0.10, "release", ha="left", va="bottom")

    # the target's marker over the rest pose
    ax.plot([x_rest], [z0 + sz + 0.032], marker="v", ms=5.5, color=TARGET, mec="none", zorder=7)
    ax.text(x_rest, z0 + sz + 0.055, "target", ha="center", va="bottom", color=TARGET)

    ax.text(BOARD_X_MIN + 0.015, -th/2, r"unknown friction $\mu$", ha="left", va="center", fontsize=6.5, color=MUTED, zorder=2)

    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    return fig

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mu", type=float, default=0.15)
    ap.add_argument("--v", type=float, default=0.8, help="peak paddle speed, m/s")
    ap.add_argument("--J", type=float, default=100.0, help="braking jerk, m/s^3")
    ap.add_argument("--dt-strobe", type=float, default=0.08, help="time between exposures, s")
    ap.add_argument("--arm-at", type=int, nargs="*", help="exposure indices that also show the arm (default: 4 up to the release)")
    ap.add_argument("--width", type=float, default=3.5, help="figure width, in (3.5: one IEEE column, drawn at print size)")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--out", default=str(FIGURES/"intro_figure"))
    a = ap.parse_args()
    from scene import LAYOUTS      # the figure is drawn for the frontal layout (mount under the arm, side view): keep the base at the origin
    r = push_slide(a.mu, a.v, a.J, pusher=ArmPusher(base=LAYOUTS["origin"]))
    print(f"mu {a.mu}  v_peak {a.v} m/s  J {a.J} m/s^3: release at x_r = {r['x_sep']:.3f} m with v_r = {r['v_sep']:.3f} m/s, "
          f"slide {r['d_slide']*100:.1f} cm (v_r^2/(2 mu g) = {r['d_pred']*100:.1f} cm), rest at x = {r['x_rest']:.3f} m")
    draw(r, a.out, a.dt_strobe, a.arm_at, not a.no_labels, a.width)
    print(f"wrote {a.out}.pdf, {a.out}.png")

if __name__ == "__main__":
    main()

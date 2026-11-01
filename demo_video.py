#!/usr/bin/env python3
"""
demo_video.py -- a demo video of the Drake scene, rendered offscreen with VTK and encoded with ffmpeg.

Four segments:
  1. the scene: the camera circles the iiwa, paddle, patchy table and object;
  2. a passive slide (scene.run_slide) in slow motion, with the predicted stopping point drawn on the table;
  3. the same slide for each object shape, side by side;
  4. landing perception (perception.py): the overhead depth image, the segmented object, its fitted outline.

The physics is scene.py's passive slide, without the arm.  Frames come from a second copy of the scene that
includes the arm, posed by IK with the paddle just behind the object.  The arm is not simulated (it has no
controller yet), so it stays still while the object slides.

Needs ffmpeg on the PATH (macOS: brew install ffmpeg; Ubuntu: apt install ffmpeg).

Usage:
    python3 demo_video.py                                   # demo.mp4: pair Y03, 1.4 m/s, 1920 x 1080
    python3 demo_video.py --pair M07 --v0 1.2 --out m07.mp4
    python3 demo_video.py --no-text                         # no captions, to add your own in an editor
    python3 demo_video.py --size 1280 720 --slowmo 0.5
"""
import argparse, math, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import ConvexHull
from pydrake.all import (AddMultibodyPlantSceneGraph, Box, CameraInfo, ClippingRange, ColorRenderCamera, DiagramBuilder,
                         EventStatus, GeometryInstance, InverseKinematics, LightParameter, MakeRenderEngineVtk,
                         PerceptionProperties, RenderCameraCore, RenderEngineVtkParams, Rgba, RigidTransform,
                         RollPitchYaw, RotationMatrix, Simulator, Solve)
import scene
from scene import (DEFAULT_PAIRS, PADDLE, Camera, Contact, Obj, Table, build_scene, camera_pose, load_pairs,
                   predicted_distance, run_slide)
from perception import Noise, add_noise, fit_plane, intrinsics, perceive, render_depth, to_points

SHAPES = ("box", "disc", "ring", "footed")
FOV_Y = math.radians(40)
RED = (215, 40, 40)

def look_at(eye, target):
    """X_WC for a camera at eye looking at target, image up = world +z (Drake cameras: +z forward, +y down)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = (target - eye)/np.linalg.norm(target - eye)
    r = np.cross(f, [0, 0, 1])
    r /= np.linalg.norm(r)
    return RigidTransform(RotationMatrix(np.c_[r, np.cross(f, r), f]), eye)

def orbit(target, radius, height, azimuth):
    target = np.asarray(target, float)
    return look_at(target + [radius*math.cos(azimuth), radius*math.sin(azimuth), height], target)

def ease(u):
    return 0.5 - 0.5*math.cos(math.pi*min(max(u, 0.0), 1.0))

def arm_pose(x0, obj, gap=0.004):
    """iiwa joint angles with the paddle upright, face along +x, `gap` behind the object at x0 and above the table."""
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
    _, paddle = scene.add_arm(plant, Contact())
    plant.Finalize()
    p = np.array([x0 - obj.size[0]/2 - gap - PADDLE[2]/2, 0, gap + PADDLE[1]/2])
    R = RotationMatrix(np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]]))   # paddle x, y, z along world y, z, x
    ik = InverseKinematics(plant, plant.CreateDefaultContext())
    ik.AddPositionConstraint(paddle.body_frame(), np.zeros(3), plant.world_frame(), p - 1e-4, p + 1e-4)
    ik.AddOrientationConstraint(plant.world_frame(), R, paddle.body_frame(), RotationMatrix(), 1e-3)
    q_nom = np.array([0, 0.8, 0, -1.8, 0, 0.8, math.pi/2])            # planar, elbow up
    ik.prog().AddQuadraticErrorCost(np.diag([10, 0.1, 10, 0.1, 10, 0.1, 1]), q_nom, ik.q())
    ik.prog().SetInitialGuess(ik.q(), q_nom)
    res = Solve(ik.prog())
    if not res.is_success():
        sys.exit(f"no arm pose puts the paddle behind an object at x0 = {x0}; try a larger --x0")
    return res.GetSolution(ik.q())

def slide(pair, obj, table, v0, seed, x0):
    """Passive slide without the arm (scene.run_slide).  Returns times from launch, object poses, x speeds and
    the predicted stopping distance."""
    builder = DiagramBuilder()
    sc = build_scene(builder, pair, obj, table, arm=False, seed=seed, x0=x0)
    sim = Simulator(builder.Build())
    ctx = sc.plant.GetMyContextFromRoot(sim.get_context())
    log = []
    def monitor(root):
        log.append((root.get_time(), sc.plant.EvalBodyPoseInWorld(ctx, sc.body),
                    sc.plant.EvalBodySpatialVelocityInWorld(ctx, sc.body).translational()[0]))
        return EventStatus.Succeeded()
    sim.set_monitor(monitor)
    X0, _ = run_slide(sim, sc, v0)
    i = next(i for i, (_, _, v) in enumerate(log) if v > v0/2)   # first step after the launch
    t, X, v = zip(*([(log[i - 1][0], X0, v0)] + log[i:]))
    return np.array(t) - t[0], list(X), np.array(v), predicted_distance(sc, obj, v0)

class Stage:
    """The scene with the arm, drawn by its own VTK renderer with shadows.  Nothing is simulated: the object is
    placed by hand on each frame.  x_stop = None leaves out the predicted-stop line."""
    def __init__(self, pair, obj, table, seed, x0, q_arm, x_stop=None):
        builder = DiagramBuilder()
        self.sc = sc = build_scene(builder, pair, obj, table, arm=True, seed=seed, x0=x0, camera=Camera())
        sg = sc.scene_graph
        sg.AddRenderer("demo", MakeRenderEngineVtk(RenderEngineVtkParams(
            default_clear_color=[0.92, 0.93, 0.95], cast_shadows=True, shadow_map_size=4096,
            lights=[LightParameter(type="directional", frame="world", direction=[0.35, 0.5, -1.0], intensity=1.1),
                    LightParameter(type="directional", frame="camera", direction=[0, 0, 1], intensity=0.4)])))
        if x_stop is not None:   # the predicted stop, a red line across the table; it would show in the depth camera too
            source = sg.RegisterSource("demo_video")
            line = sg.RegisterAnchoredGeometry(source, GeometryInstance(RigidTransform([x_stop, 0, 0.0005]),
                                                                        Box(0.004, table.width, 0.001), "predicted_stop"))
            props = PerceptionProperties()
            props.AddProperty("phong", "diffuse", Rgba(*(np.array(RED)/255), 1.0))
            sg.AssignRole(source, line, props)
        self.diagram = builder.Build()
        self.ctx = self.diagram.CreateDefaultContext()
        self.pctx = sc.plant.GetMyMutableContextFromRoot(self.ctx)
        sc.plant.SetPositions(self.pctx, sc.iiwa, q_arm)
        self.cameras = {}

    def info(self, size):
        if size not in self.cameras:
            info = CameraInfo(*size, FOV_Y)
            self.cameras[size] = (info, ColorRenderCamera(RenderCameraCore("demo", info, ClippingRange(0.05, 20),
                                                                           RigidTransform()), False))
        return self.cameras[size][0]

    def render(self, X_WO, X_WC, size):
        self.info(size)
        self.sc.plant.SetFreeBodyPose(self.pctx, self.sc.body, X_WO)
        sg = self.sc.scene_graph
        query = sg.get_query_output_port().Eval(sg.GetMyContextFromRoot(self.ctx))
        return query.RenderColorImage(self.cameras[size][1], sg.world_frame_id(), X_WC).data[:, :, :3].copy()

    def project(self, p_W, X_WC, size):
        info, p = self.info(size), X_WC.inverse() @ np.asarray(p_W, float)
        return info.focal_x()*p[0]/p[2] + info.center_x(), info.focal_y()*p[1]/p[2] + info.center_y()

    def depth(self, X_WO):
        self.sc.plant.SetFreeBodyPose(self.pctx, self.sc.body, X_WO)
        return render_depth(self.sc, self.ctx)

def font(px):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            pass
    return ImageFont.load_default(px)

class Captions:
    """Text boxes drawn onto frames; with on=False they draw nothing, for footage to caption elsewhere."""
    def __init__(self, height, on=True):
        self.on, self.u = on, height/1080
        self.fonts = {k: font(round(s*self.u)) for k, s in (("title", 64), ("body", 34), ("small", 28))}

    def box(self, img, lines, xy=(40, 40), anchor="tl", alpha=0.72):
        """lines: [(text, style)] or [(text, style, rgb)], in a dark box with its `anchor` corner at xy (px at 1080p)."""
        if not self.on:
            return img
        im = Image.fromarray(img).convert("RGBA")
        draw, pad = ImageDraw.Draw(im), round(18*self.u)
        lines = [l if len(l) == 3 else (*l, (255, 255, 255)) for l in lines]
        sizes = [draw.textbbox((0, 0), t, font=self.fonts[s]) for t, s, _ in lines]
        gap = round(10*self.u)
        w = max(b[2] for b in sizes) + 2*pad
        h = sum(self.fonts[s].size for _, s, _ in lines) + gap*(len(lines) - 1) + 2*pad
        x, y = xy[0]*self.u, xy[1]*self.u
        x, y = (x - w if "r" in anchor else x), (y - h if "b" in anchor else y)
        overlay = Image.new("RGBA", im.size)
        ImageDraw.Draw(overlay).rounded_rectangle([x, y, x + w, y + h], radius=round(12*self.u),
                                                  fill=(20, 22, 28, round(255*alpha)))
        im = Image.alpha_composite(im, overlay)
        draw, yy = ImageDraw.Draw(im), y + pad
        for text, style, rgb in lines:
            draw.text((x + pad, yy), text, font=self.fonts[style], fill=rgb)
            yy += self.fonts[style].size + gap
        return np.asarray(im.convert("RGB"))

    def label(self, img, text, xy, rgb=RED):
        """Bare text centred above pixel xy, with a light halo so it reads on the table."""
        if not self.on:
            return img
        im = Image.fromarray(img)
        draw, f = ImageDraw.Draw(im), self.fonts["small"]
        draw.text(xy, text, font=f, fill=rgb, anchor="md", stroke_width=round(3*self.u), stroke_fill=(245, 245, 245))
        return np.asarray(im)

    def legend(self, img, lo, hi, xy):
        """Strip-friction colour bar, using scene.build_scene's shading (darker = more friction)."""
        if not self.on:
            return img
        im = Image.fromarray(img)
        draw, f, u = ImageDraw.Draw(im), self.fonts["small"], self.u
        x, y, w, h = xy[0]*u, xy[1]*u, 360*u, 26*u
        for i in range(int(w)):
            s = 0.85 - 0.35*i/w
            draw.line([x + i, y, x + i, y + h], fill=tuple(int(255*c) for c in (s, s*0.9, s*0.8)))
        draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=max(1, round(2*u)))
        draw.text((x, y + h + 6*u), f"μ {lo:.2f}", font=f, fill=(255, 255, 255), anchor="la")
        draw.text((x + w, y + h + 6*u), f"{hi:.2f}", font=f, fill=(255, 255, 255), anchor="ra")
        return np.asarray(im)

class Video:
    """Raw RGB frames piped to ffmpeg, H.264 in yuv420p so it plays everywhere."""
    def __init__(self, path, size, fps):
        self.proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                      "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                      "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                      path], stdin=subprocess.PIPE)
        self.fps, self.last, self.frames = fps, None, 0

    def add(self, frame, seconds=None):
        for _ in range(1 if seconds is None else max(1, round(seconds*self.fps))):
            self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
            self.frames += 1
        self.last = frame

    def fade(self, frame, seconds=0.3):
        """Cross-fade from the last frame to this one (the caller then adds the segment as usual)."""
        n = round(seconds*self.fps)
        for k in range(1, n) if self.last is not None else ():
            self.add(((1 - k/n)*self.last + k/n*frame.astype(float)).astype(np.uint8))

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            sys.exit("ffmpeg failed")

def at(t, ts):
    """Index of the last sample at or before time t."""
    return min(max(int(np.searchsorted(ts, t, side="right")) - 1, 0), len(ts) - 1)

def depth_colors(depth, lo=0.70, hi=0.81):
    """Depth image as RGB: near is bright yellow, far is dark blue; invalid pixels black."""
    stops = np.array([[253, 231, 37], [94, 201, 98], [33, 145, 140], [59, 82, 139], [68, 1, 84]], float)
    u = np.clip((np.nan_to_num(depth, nan=hi) - lo)/(hi - lo), 0, 1)*(len(stops) - 1)
    rgb = np.stack([np.interp(u, np.arange(len(stops)), stops[:, c]) for c in range(3)], -1)
    rgb[~np.isfinite(depth) | (depth <= 0)] = 0
    return rgb.astype(np.uint8)

def blob_mask(depth, cam, rng, x_window, band=(0.005, 0.15)):
    """Pixel mask of the blob perception.segment keeps: largest connected region in the height band, in the window."""
    P = to_points(depth, cam)
    n, c = fit_plane(P[np.isfinite(P[..., 2])], rng)
    with np.errstate(invalid="ignore"):
        h = P @ n + c
        mask = (h > band[0]) & (h < band[1]) & (P[..., 0] > x_window[0]) & (P[..., 0] < x_window[1])
    labels, k = ndimage.label(mask)
    return labels == 1 + int(np.argmax(ndimage.sum(mask, labels, range(1, k + 1)))) if k else mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--pair", default="Y03")
    ap.add_argument("--v0", type=float, default=1.4, help="launch speed, m/s")
    ap.add_argument("--x0", type=float, default=0.65, help="object start; the paddle cannot reach behind much below 0.6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slowmo", type=float, default=0.25, help="playback speed of the slides")
    ap.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-text", action="store_true", help="leave out the captions")
    ap.add_argument("--out", default="demo.mp4")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found (macOS: brew install ffmpeg; Ubuntu: apt install ffmpeg)")
    W, H = (s//2*2 for s in a.size)
    size, half = (W, H), (W//2, H//2)
    pair, table = load_pairs(a.pairs)[a.pair], Table()
    cap = Captions(H, on=not a.no_text)

    print("simulating slides ...")
    runs = {}
    for shape in SHAPES:
        obj = Obj(shape=shape)
        runs[shape] = (obj, *slide(pair, obj, table, a.v0, a.seed, a.x0))
        print(f"  {shape}: d = {(runs[shape][2][-1].translation()[0] - a.x0)*100:.2f} cm, "
              f"predicted {runs[shape][4]*100:.2f} cm")
    q_arm = arm_pose(a.x0, Obj())
    stages = {s: Stage(pair, runs[s][0], table, a.seed, a.x0, q_arm, a.x0 + runs[s][4]) for s in SHAPES}
    video = Video(a.out, size, a.fps)
    mu_lo, mu_hi = stages["box"].sc.mu_strips.min(), stages["box"].sc.mu_strips.max()
    source = "placeholder value" if "PLACEHOLDER" in pair["sources"] else pair["sources"].split(",")[0]

    print("1/4 scene")
    obj, ts, Xs, vs, d_pred = runs["box"]
    n = round(4.5*a.fps)
    for k in range(n):
        az = math.radians(-20 - 58*ease(k/(n - 1)))
        img = stages["box"].render(Xs[0], orbit([0.5, 0, 0.18], 1.75, 0.8, az), size)
        img = cap.box(img, [("Slide-to-Target", "title"),
                            ("Drake scene: iiwa 14 with paddle, patchy table, hydroelastic object", "body"),
                            (f"pair {a.pair}: {pair['object_material']} on {pair['surface_material']}, "
                             f"μ = {float(pair['mu_k_nominal']):.3f} ({source})", "small", (200, 205, 215))])
        img = cap.box(img, [("table strips, darker = more friction", "small"), ("", "small"), ("", "small")],
                      xy=(40, 1040), anchor="bl")
        video.add(cap.legend(img, mu_lo, mu_hi, (58, 958)))

    print("2/4 passive slide")
    side = look_at([0.82, -1.5, 0.62], [0.58, 0, 0.14])
    stop_xy = stages["box"].project([a.x0 + d_pred, -table.width/2, 0], side, size)
    def slide_frame(i, lines):
        img = stages["box"].render(Xs[i], side, size)
        img = cap.label(img, "predicted stop", (stop_xy[0], stop_xy[1] - 10*cap.u))
        return cap.box(img, lines)
    head = [(f"passive slide, launched at v0 = {a.v0:.2f} m/s", "body"),
            (f"{a.slowmo:g}× speed · arm not simulated (no controller yet)", "small", (200, 205, 215))]
    first = slide_frame(0, head + [("t = 0.000 s    v = %.2f m/s    d = 0.0 cm" % a.v0, "small")])
    video.fade(first)
    video.add(first, 1.0)
    for k in range(int(ts[-1]/a.slowmo*a.fps) + 1):
        i = at(k/a.fps*a.slowmo, ts)
        d = Xs[i].translation()[0] - Xs[0].translation()[0]
        video.add(slide_frame(i, head + [(f"t = {ts[i]:.3f} s    v = {vs[i]:.2f} m/s    d = {d*100:.1f} cm", "small")]))
    d = Xs[-1].translation()[0] - Xs[0].translation()[0]
    yaw = math.degrees(RollPitchYaw(Xs[-1].rotation()).yaw_angle() - RollPitchYaw(Xs[0].rotation()).yaw_angle())
    yaw = round(yaw, 2) + 0.0                      # no "-0.00"
    video.add(slide_frame(len(Xs) - 1, head + [(f"stopped at d = {d*100:.1f} cm · predicted {d_pred*100:.1f} cm "
                                                f"({d/d_pred - 1:+.1%}) · yaw {yaw:+.2f}°", "small", (255, 220, 120))]), 2.5)

    print("3/4 shapes")
    close = look_at([0.86, -0.92, 0.42], [0.84, 0, 0.0])
    t_end = max(runs[s][1][-1] for s in SHAPES)
    def grid(t):
        img = np.zeros((H, W, 3), np.uint8)
        for j, s in enumerate(SHAPES):
            obj, ts_s, Xs_s, _, d_pred_s = runs[s]
            i = at(t, ts_s)
            d = Xs_s[i].translation()[0] - Xs_s[0].translation()[0]
            lines = [(s, "body")]
            if i == len(ts_s) - 1:
                lines.append((f"d = {d*100:.1f} cm · predicted {d_pred_s*100:.1f} ({d/d_pred_s - 1:+.1%})", "small",
                              (255, 220, 120)))
            panel = cap.box(stages[s].render(Xs_s[i], close, half), lines, xy=(24, 24))
            img[(j//2)*half[1]:(j//2 + 1)*half[1], (j % 2)*half[0]:(j % 2 + 1)*half[0]] = panel
        img[H//2 - 1:H//2 + 1], img[:, W//2 - 1:W//2 + 1] = 255, 255
        return img
    first = grid(0.0)
    video.fade(first)
    video.add(first, 0.8)
    for k in range(int(t_end/a.slowmo*a.fps) + 1):
        video.add(grid(k/a.fps*a.slowmo))
    video.add(grid(t_end), 2.5)

    print("4/4 perception")
    obj, ts, Xs, vs, d_pred = runs["box"]
    cam, rng = Camera(), np.random.default_rng(a.seed)
    depth = add_noise(Stage(pair, obj, table, a.seed, a.x0, q_arm).depth(Xs[-1]), cam, Noise(), rng)
    window = (a.x0 - obj.size[0]/2 + 0.02, np.inf)   # leave out the paddle and wrist, as perceive's x_window
    est = perceive(depth, cam, obj, rng, x_window=window)
    if est is None:
        sys.exit("perception found no object")
    mask = blob_mask(depth, cam, np.random.default_rng(a.seed), window)
    s = min(H/cam.height, 0.66*W/cam.width)       # depth image on the left; text and a close-up on the right
    dw, dh = round(cam.width*s), round(cam.height*s)
    ox = oy = (H - dh)//2
    info, X_CW = intrinsics(cam), camera_pose(cam).inverse()
    def pixel(p_W):
        """Depth-image position of a world point: perception.to_points inverted (pixel u's centre is at u + 0.5)."""
        p = X_CW @ np.asarray(p_W, float)
        return np.array([info.focal_x()*p[0]/p[2] + info.center_x(), info.focal_y()*p[1]/p[2] + info.center_y()])
    colors = depth_colors(depth)
    dim = (0.35*colors + 0.65*110).astype(np.uint8)
    dim[mask] = colors[mask]
    dim[mask & ~ndimage.binary_erosion(mask)] = 255
    R = np.array([[math.cos(est["yaw"]), -math.sin(est["yaw"])], [math.sin(est["yaw"]), math.cos(est["yaw"])]])
    corners = np.array([pixel([*((est["x"], est["y"]) + R @ c), z]) for z in (0.0, obj.size[2])
                        for c in obj.size[0]/2*np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])])
    outline_px = corners[ConvexHull(corners).vertices]   # fitted box's silhouette: top face plus visible side
    centre_px = pixel([est["x"], est["y"], obj.size[2]])
    zoom = 56                                       # close-up: depth pixels within this many of the object centre
    crop = (*np.round(centre_px - zoom).astype(int), *np.round(centre_px + zoom).astype(int))
    side = round(400*cap.u)
    zk = side/(2*zoom)                              # close-up magnification
    ix, iy = ox + dw + round(30*cap.u), H - oy - side
    lw = max(2, round(3*cap.u))
    def draw_fit(draw, f):
        """The fitted outline and centre, with f mapping depth-image positions into the frame."""
        pts = [tuple(f(p)) for p in outline_px]
        draw.line(pts + pts[:1], fill=RED, width=lw)
        (cx, cy), r = f(centre_px), 10*cap.u
        draw.line([cx - r, cy, cx + r, cy], fill=RED, width=lw)
        draw.line([cx, cy - r, cx, cy + r], fill=RED, width=lw)
    d_true = Xs[-1].translation()[0] - Xs[0].translation()[0]
    d_hat = est["x"] - Xs[0].translation()[0]
    steps = [(["1  depth image with stereo noise"], colors, False),
             (["2  table plane removed (RANSAC),", "    largest blob kept"], dim, False),
             (["3  outline fitted to the blob edge", "    (2-D ICP)"], dim, True)]
    for j, (_, picture, fitted) in enumerate(steps):
        im = Image.new("RGB", size, (24, 26, 32))
        im.paste(Image.fromarray(picture).resize((dw, dh), Image.NEAREST), (ox, oy))
        if fitted:
            im.paste(Image.fromarray(dim).crop(crop).resize((side, side), Image.NEAREST), (ix, iy))
            draw = ImageDraw.Draw(im)
            draw_fit(draw, lambda p: (ox + p[0]*s, oy + p[1]*s))
            draw_fit(draw, lambda p: (ix + (p[0] - crop[0])*zk, iy + (p[1] - crop[1])*zk))
            draw.rectangle([ox + crop[0]*s, oy + crop[1]*s, ox + crop[2]*s, oy + crop[3]*s], outline=(255, 255, 255))
            draw.rectangle([ix, iy, ix + side, iy + side], outline=(255, 255, 255), width=max(1, round(2*cap.u)))
        lines = [("landing perception", "body"), ("overhead depth camera", "small", (200, 205, 215)),
                 (f"{cam.width}×{cam.height}, {cam.position[2]:.1f} m above the table", "small", (200, 205, 215)),
                 ("", "small")]
        lines += [(t, "small", (255, 255, 255) if k <= j else (110, 115, 125)) for k, (texts, _, _) in enumerate(steps)
                  for t in texts]
        if fitted:
            lines += [("", "small"), (f"perceived d = {d_hat*100:.2f} cm", "small", (255, 220, 120)),
                      (f"true d = {d_true*100:.2f} cm", "small", (255, 220, 120)),
                      (f"error {(d_hat - d_true)*1e3:+.2f} mm", "small", (255, 220, 120))]
        img = cap.box(np.asarray(im), lines, xy=((ox + dw)/cap.u + 12, oy/cap.u), alpha=0.0)
        if j == 0:
            video.fade(img)
        video.add(img, 3.5 if fitted else 1.6)
    video.close()
    print(f"wrote {a.out}: {video.frames} frames, {video.frames/a.fps:.1f} s; perception error "
          f"{(d_hat - d_true)*1e3:+.2f} mm")

if __name__ == "__main__":
    main()

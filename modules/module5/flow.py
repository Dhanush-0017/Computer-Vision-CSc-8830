"""
flow.py -- the maths for Part A (optical flow and tracking).

Used by the web page (__init__.py) and by the two scripts:
    make_flow_videos.py    Part A: flow for every frame of a 30 s clip,
                           written out as a video, plus the evidence plots
    validate_tracking.py   Part A: the tracking equations run on two
                           consecutive frames and checked against where
                           the pixels actually went

What's here:
    read_frames        load a clip
    dense_flow         Farneback dense optical flow (OpenCV)
    camera_motion      how the (hand-held) camera moved: homography, RANSAC
    camera_flow        the flow that camera motion alone would cause
    textured           where there is enough texture to measure flow
    flow_to_color      flow -> colour picture (hue = direction, brightness = speed)
    draw_arrows        flow -> arrows drawn on the frame
    color_wheel        the legend for flow_to_color
    frame_stats        the numbers the "what can we infer" section is built on
    bilinear           bilinear interpolation, written out by hand (theory.md §3)
    gradients          Ix, Iy by central differences
    lk_track           Lucas-Kanade tracking, from scratch: pyramid + iterations,
                       uses bilinear() for every sub-pixel sample (theory.md §2)
    lk_workings        one point, one pyramid level, every number printed --
                       the "by hand" check
    ncc_locate         where a patch actually went, measured independently of LK
                       (normalised cross-correlation + sub-pixel peak)
"""
import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def read_frames(path, max_frames=None, step=1):
    """All frames of a clip as a list of BGR arrays, and its frame rate."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames, fps / step


def gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


# ----------------------------------------------------------------------------
# Dense optical flow
# ----------------------------------------------------------------------------
def dense_flow(prev_bgr, next_bgr):
    """Farneback dense flow. Returns an HxWx2 array: flow[y, x] = (u, v), the
    displacement in pixels from prev to next.

    Farneback fits a quadratic polynomial to each neighbourhood in both frames
    and solves for the shift that maps one onto the other. A 5-level pyramid
    (each level half the size) lets it follow cars moving 20+ px per frame.
    """
    return cv2.calcOpticalFlowFarneback(
        gray(prev_bgr), gray(next_bgr), None,
        pyr_scale=0.5, levels=5, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)


def camera_motion(prev_bgr, next_bgr):
    """How the CAMERA moved between two frames (my phone was hand-held).

    Corners spread over the whole frame are tracked, and a homography is
    fitted to them with RANSAC. Most of the frame is static background, so
    the fit follows the background; moving cars are outliers and are ignored.
    For a camera that only rotates (hand shake, panning while standing still)
    a homography describes the background motion exactly, whatever the depth.
    Returns (H, fraction of corners that agree with it).
    """
    g1, g2 = gray(prev_bgr), gray(next_bgr)
    p = cv2.goodFeaturesToTrack(g1, 600, 0.01, 12)
    if p is None or len(p) < 8:
        return np.eye(3), 0.0
    q, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p, None, winSize=(21, 21),
                                        maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return np.eye(3), 0.0
    H, inl = cv2.findHomography(p[ok], q[ok], cv2.RANSAC, 1.5)
    if H is None:
        return np.eye(3), 0.0
    return H, float(inl.mean())


def textured(bgr, grow=15):
    """Where the image has enough texture for flow to be measured at all.

    Flow comes from brightness gradients; on a flat patch (clear sky, plain
    road) there are none, so Farneback returns ~0 there whatever really moved
    (the aperture problem, theory.md A1). When the camera pans, 'measured 0'
    minus 'camera moved 7 px' would then look like an object moving. So only
    pixels within `grow` px of real texture (smallest structure-tensor
    eigenvalue above the frame's median) are allowed to count as moving."""
    lam = cv2.cornerMinEigenVal(gray(bgr), 5)
    t = (lam > np.median(lam)).astype(np.uint8)
    return cv2.dilate(t, np.ones((grow, grow), np.uint8)) > 0


def camera_flow(H, shape):
    """The flow field the camera motion H alone would produce."""
    h, w = shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.dstack([x, y]).reshape(-1, 1, 2)
    to = cv2.perspectiveTransform(pts, H).reshape(h, w, 2)
    return to - np.dstack([x, y])


def flow_to_color(flow, max_mag):
    """Standard flow colouring: hue = direction, brightness = speed.

    Speed is divided by a FIXED max_mag (not this frame's max) so that the
    same colour means the same speed in every frame of the video.
    """
    u, v = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)          # OpenCV hue is 0..180
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def color_wheel(size=90):
    """Legend for flow_to_color: direction -> hue, centre = no motion."""
    r = size // 2
    y, x = np.mgrid[-r:r, -r:r].astype(np.float32)
    fl = np.dstack([x, y]) / r
    img = flow_to_color(fl, 1.0)
    img[np.hypot(x, y) > r] = 255
    return img


def draw_arrows(bgr, flow, step=16, min_mag=0.5, scale=3.0,
                color=(0, 255, 0)):
    """Sparse arrows on a grid, drawn only where something moves.

    Arrows are drawn `scale` times longer than the real displacement so that
    1-2 px/frame is visible.
    """
    out = bgr.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = flow[y, x]
            if u * u + v * v < min_mag * min_mag:
                continue
            cv2.arrowedLine(out, (x, y),
                            (int(round(x + scale * u)), int(round(y + scale * v))),
                            color, 1, cv2.LINE_AA, tipLength=0.35)
    return out


def frame_stats(flow, thr, cam=None, tex=None):
    """Numbers used as evidence for what the flow tells us.

    flow: the measured flow. cam: the part of it caused by the camera moving
    (camera_flow), or None for a fixed camera. What is left over,
    flow - cam, is how things moved in the world, and a pixel counts as
    moving when that is faster than thr (px/frame) -- and, if tex is given
    (see textured), it is somewhere flow can actually be measured.
    """
    if cam is None:
        cam = np.zeros_like(flow)
    obj = flow - cam
    mag = np.hypot(obj[..., 0], obj[..., 1])
    moving = mag > thr
    if tex is not None:
        moving &= tex
    n = int(moving.sum())
    cmag = np.hypot(cam[..., 0], cam[..., 1])
    st = {
        'moving_fraction': n / moving.size,
        # camera motion: its median speed over the frame, and its direction
        'camera_speed': float(np.median(cmag)),
        'camera_dx': float(np.median(cam[..., 0])),
        'camera_dy': float(np.median(cam[..., 1])),
        # background after removing the camera motion: should be ~0
        'residual_background': float(np.median(mag[~moving])) if n < moving.size else 0.0,
        'moving_mean_speed': float(mag[moving].mean()) if n else 0.0,
        'mean_u': float(obj[..., 0][moving].mean()) if n else 0.0,
        'mean_v': float(obj[..., 1][moving].mean()) if n else 0.0,
    }
    # moving regions = connected blobs of moving pixels (roughly: vehicles)
    m8 = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((5, 5), np.uint8))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    k, _, s, _ = cv2.connectedComponentsWithStats(m8)
    st['moving_blobs'] = int((s[1:, cv2.CC_STAT_AREA] >= 400).sum()) if k > 1 else 0
    return st, moving


# ----------------------------------------------------------------------------
# Tracking, from scratch (theory.md sections 2 and 3)
# ----------------------------------------------------------------------------
def bilinear(img, x, y):
    """Value of img at non-integer (x, y), by bilinear interpolation.

        x0 = floor(x), a = x - x0      y0 = floor(y), b = y - y0
        I(x, y) = (1-a)(1-b) I[y0, x0]  +  a(1-b) I[y0, x0+1]
                + (1-a) b   I[y0+1, x0] +  a b    I[y0+1, x0+1]

    x, y may be arrays of any (matching) shape. Coordinates outside the image
    are clamped to the border.
    """
    img = img.astype(np.float64)
    h, w = img.shape
    x = np.clip(np.asarray(x, np.float64), 0, w - 1.000001)
    y = np.clip(np.asarray(y, np.float64), 0, h - 1.000001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    a = x - x0
    b = y - y0
    return ((1 - a) * (1 - b) * img[y0, x0] + a * (1 - b) * img[y0, x0 + 1]
            + (1 - a) * b * img[y0 + 1, x0] + a * b * img[y0 + 1, x0 + 1])


def gradients(img):
    """Ix, Iy by central differences: Ix = (I[x+1] - I[x-1]) / 2."""
    img = img.astype(np.float64)
    Ix = np.zeros_like(img)
    Iy = np.zeros_like(img)
    Ix[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    Iy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return Ix, Iy


def _lk_level(I1, I2, Ix, Iy, x, y, d0, half, iters, eps):
    """One pyramid level of iterative Lucas-Kanade for one point.

    Solves  G d = b  repeatedly, where over the window W around (x, y)
        G = sum [Ix^2   IxIy]      b = sum [Ix * e]
                [IxIy   Iy^2]              [Iy * e]
        e = I1(x, y) - I2(x + d)   (I2 sampled with bilinear())
    Returns (d, number of iterations, smallest eigenvalue of G).
    """
    oy, ox = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    px, py = x + ox, y + oy
    T = bilinear(I1, px, py)                       # template from frame 1
    gx = bilinear(Ix, px, py)
    gy = bilinear(Iy, px, py)
    G = np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                  [np.sum(gx * gy), np.sum(gy * gy)]])
    lam_min = float(np.linalg.eigvalsh(G)[0])
    if lam_min < 1e-6:
        return d0, 0, lam_min
    Ginv = np.linalg.inv(G)
    d = d0.copy()
    k = 0
    for k in range(1, iters + 1):
        e = T - bilinear(I2, px + d[0], py + d[1])
        b = np.array([np.sum(gx * e), np.sum(gy * e)])
        step = Ginv @ b
        d = d + step
        if np.hypot(*step) < eps:
            break
    return d, k, lam_min


def lk_track(img1, img2, pts, win=21, levels=4, iters=30, eps=0.01):
    """Pyramidal iterative Lucas-Kanade, from scratch.

    pts: Nx2 array of (x, y) in img1. Returns (new_pts Nx2, info list).
    Coarse-to-fine: solve at the smallest pyramid level, double the answer,
    use it as the starting guess one level up.
    """
    I1 = gray(img1).astype(np.float64)
    I2 = gray(img2).astype(np.float64)
    pyr1, pyr2 = [I1], [I2]
    for _ in range(levels - 1):
        pyr1.append(cv2.pyrDown(pyr1[-1]))
        pyr2.append(cv2.pyrDown(pyr2[-1]))
    grads = [gradients(p) for p in pyr1]
    half = win // 2
    out, info = [], []
    for (x, y) in np.asarray(pts, np.float64):
        d = np.zeros(2)
        for L in range(levels - 1, -1, -1):
            s = 2.0 ** L
            d, k, lam = _lk_level(pyr1[L], pyr2[L], *grads[L],
                                  x / s, y / s, d, half, iters, eps)
            if L > 0:
                d = d * 2.0
        out.append([x + d[0], y + d[1]])
        info.append({'iters_last_level': k, 'lambda_min': lam})
    return np.array(out), info


def lk_workings(img1, img2, pt, win=7, iters=10):
    """Every number of LK for one point at full resolution -- the worked
    example in the report. Returns a dict with the window, gradients, G, b
    and the iterates."""
    I1 = gray(img1).astype(np.float64)
    I2 = gray(img2).astype(np.float64)
    Ix, Iy = gradients(I1)
    x, y = float(pt[0]), float(pt[1])
    half = win // 2
    oy, ox = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    px, py = x + ox, y + oy
    T = bilinear(I1, px, py)
    gx = bilinear(Ix, px, py)
    gy = bilinear(Iy, px, py)
    G = np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                  [np.sum(gx * gy), np.sum(gy * gy)]])
    rows = []
    d = np.zeros(2)
    for k in range(1, iters + 1):
        e = T - bilinear(I2, px + d[0], py + d[1])
        b = np.array([np.sum(gx * e), np.sum(gy * e)])
        step = np.linalg.solve(G, b)
        rows.append({'k': k, 'b': b.copy(), 'step': step.copy(),
                     'd': (d + step).copy(), 'ssd': float(np.sum(e * e))})
        d = d + step
        if np.hypot(*step) < 0.01:
            break
    e = T - bilinear(I2, px + d[0], py + d[1])
    return {'x': x, 'y': y, 'win': win, 'I1': T, 'Ix': gx, 'Iy': gy,
            'It0': bilinear(I2, px, py) - T, 'G': G,
            'eig': np.linalg.eigvalsh(G), 'iters': rows, 'd': d,
            'ssd_final': float(np.sum(e * e))}


def ncc_locate(img1, img2, pt, patch=21, search=40):
    """Where did the patch around pt actually go? Measured WITHOUT any flow
    equation: slide the patch over a search window (+/-40 px) in img2, take the peak of
    normalised cross-correlation, refine to sub-pixel with a parabola through
    the peak and its neighbours. Returns ((x, y), peak score)."""
    I1 = gray(img1).astype(np.float32)
    I2 = gray(img2).astype(np.float32)
    x, y = int(round(pt[0])), int(round(pt[1]))
    h = patch // 2
    r = search
    H, W = I1.shape
    if not (h + r <= x < W - h - r and h + r <= y < H - h - r):
        return None, 0.0
    T = I1[y - h:y + h + 1, x - h:x + h + 1]
    S = I2[y - h - r:y + h + r + 1, x - h - r:x + h + r + 1]
    R = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(R)

    def sub(c_m, c_0, c_p):
        den = c_m - 2 * c_0 + c_p
        return 0.0 if abs(den) < 1e-9 else 0.5 * (c_m - c_p) / den

    dx = sub(R[my, mx - 1], R[my, mx], R[my, mx + 1]) if 0 < mx < R.shape[1] - 1 else 0.0
    dy = sub(R[my - 1, mx], R[my, mx], R[my + 1, mx]) if 0 < my < R.shape[0] - 1 else 0.0
    # R[0, 0] is a shift of (-r, -r), so the peak at (mx, my) is a shift of
    # (mx - r, my - r)
    return (pt[0] + (mx - r) + dx, pt[1] + (my - r) + dy), float(score)

"""
sfm.py -- Part B: structure from motion from four views of a flat object.

Usage (from this folder):
    python sfm.py --width-mm 170 --height-mm 230 --diag-mm 280

(the purple panel of the cover, measured with the iPhone Measure app)

The object is a spiral notebook ("NEVER STOP" cover) lying flat on a library
table, photographed from four positions with the same iPhone I calibrated in
Module 2 (main 1x camera). The cover is flat, so it is the planar object the
assignment suggests. Two kinds of point are reconstructed:
    - feature points on the cover's artwork (SIFT), matched across all four
      photos -- they give the correspondences, and
    - the 4 corners of the purple panel of the cover: the object's boundary.

Only image measurements and K are used to reconstruct. The panel's size,
measured with the iPhone Measure app, is used for:
    - one known length (the panel's width) to fix the scale, which SfM can
      never recover by itself, and
    - afterwards, as a check: the reconstructed height and diagonal should
      match the measurements, and the corners should come out at 90 degrees.

Steps (theory.md, Part B, has the maths):
    1. detect points, undistort to normalised coordinates  m = K^-1 x
    2. homography view 1 -> view i from the matched points  (my DLT)
    3. H = R + t n^T / d  ->  R_i, t_i, n                    (4 solutions each,
                                                              pick consistent)
    4. triangulate every point from all 4 views             (my linear DLT)
    5. refine: re-estimate each camera from the points, re-triangulate, repeat
    6. fix the scale with one known length, express in the object's plane
    7. boundary = polygon through the reconstructed panel corners

Writes to results/:
    sfm_points.csv, sfm_cameras.csv, sfm_summary.json,
    sfm_views.png, sfm_3d.png, sfm_boundary.png, workings_sfm.md
"""
import argparse
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SFM = os.path.join(HERE, 'data', 'sfm')
RESULTS = os.path.join(HERE, 'results')
CORNERS = ('TL', 'TR', 'BR', 'BL')


# ----------------------------------------------------------------------------
# 1. Points in each image
# ----------------------------------------------------------------------------
def load_camera():
    """Intrinsics for the full-resolution 3024x4032 photos (Module 2
    calibration) and the distortion coefficients."""
    with open(os.path.join(SFM, 'camera.json')) as f:
        c = json.load(f)
    return np.array(c['K_full_resolution']), np.array(c['dist']), c


def panel_outline(img):
    """Outline of the purple panel of the cover, in pixels.

    Purple = hue 110-150 (OpenCV scale), saturation > 70. The largest purple
    region, closed and taken as a convex hull, is the panel. Its bottom-right
    corner is covered by the black 'classmate PULSE' label, so the hull has
    one corner cut off; panel_corners deals with that.
    Works on a half-size copy (the kernel sizes are tuned for it) and returns
    full-resolution pixel coordinates."""
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    p = ((hsv[..., 1] > 70) & (hsv[..., 0] > 110) & (hsv[..., 0] < 150)).astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p)
    j = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    m = ((lab == j) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    hull = cv2.convexHull(c.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    full = lambda p: (p + 0.5) * 2 - 0.5      # half-size pixel -> full-size pixel
    return full(hull), full(c)


def panel_corners(img, K, dist):
    """The 4 corners of the panel, in NORMALISED coordinates.

    The outline is simplified to a 4- or 5-sided polygon (5 when the label
    cuts a corner: the shortest side is then dropped). Each of the 4 real
    sides gets a straight line fitted through the middle 76 % of its outline
    points -- after undistortion, so the sides really are straight -- and the
    corners are where adjacent lines meet. That also gives sharp corners even
    though the printed panel has rounded ones."""
    hull, cnt = panel_outline(img)
    for eps in np.linspace(0.005, 0.08, 60):
        ap = cv2.approxPolyDP(hull.astype(np.float32).reshape(-1, 1, 2),
                              eps * cv2.arcLength(hull.astype(np.float32), True),
                              True).reshape(-1, 2).astype(np.float64)
        if len(ap) in (4, 5):
            break
    cnt_n = normalise(cnt, K, dist)
    ap_n = normalise(ap, K, dist)
    sides = [(ap_n[k], ap_n[(k + 1) % len(ap_n)]) for k in range(len(ap_n))]
    if len(sides) == 5:
        cut = int(np.argmin([np.linalg.norm(b - a) for a, b in sides]))
        sides = sides[cut + 1:] + sides[:cut]
    lines = []
    for a, c in sides:
        d = c - a
        L = np.linalg.norm(d)
        t = ((cnt_n - a) @ d) / L ** 2
        off = np.abs(d[0] * (cnt_n[:, 1] - a[1]) - d[1] * (cnt_n[:, 0] - a[0])) / L
        q = cnt_n[(t > 0.12) & (t < 0.88) & (off < 0.015 * L + 0.004)]
        mu = q.mean(0)
        _, _, vt = np.linalg.svd(q - mu)
        nrm = vt[1]
        lines.append(np.array([nrm[0], nrm[1], -nrm @ mu]))
    out = []
    for k in range(4):
        p = np.cross(lines[k - 1], lines[k])
        out.append(p[:2] / p[2])
    return np.array(out)


def order_by_layout(c):
    """TL, TR, BR, BL by image position (used for view 1 only)."""
    s = c.sum(1)
    d = c[:, 0] - c[:, 1]
    return np.array([c[np.argmin(s)], c[np.argmax(d)], c[np.argmax(s)], c[np.argmin(d)]])


def match_features(imgs, K, dist, masks):
    """SIFT points on the cover, matched from view 1 to every other view.

    Matches pass Lowe's ratio test (0.8) and must agree with one homography
    (RANSAC, 4 px at full resolution) -- the cover is flat, so every correct
    match does. A
    point is kept only if it was found in all four views.
    Returns (list of Nx2 normalised arrays, one per view; the homographies in
    pixel coordinates view 1 -> view i)."""
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02)
    kd = []
    for im, mk in zip(imgs, masks):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(g, mk)
        kd.append((np.array([k.pt for k in kp]), des))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    keep = None
    pairs, Hpx = [], []
    for i in range(1, len(imgs)):
        mt = bf.knnMatch(kd[0][1], kd[i][1], k=2)
        good = {m.queryIdx: m.trainIdx for m, n in mt if m.distance < 0.8 * n.distance}
        q = np.array(list(good.keys()))
        t = np.array([good[k] for k in q])
        H, inl = cv2.findHomography(kd[0][0][q], kd[i][0][t], cv2.RANSAC, 4.0)
        ok = inl.ravel() == 1
        d = dict(zip(q[ok], t[ok]))
        pairs.append(d)
        Hpx.append(H)
        keep = set(d) if keep is None else keep & set(d)
    keep = sorted(keep)
    pts = [kd[0][0][keep]] + [kd[i][0][[pairs[i - 1][k] for k in keep]]
                              for i in range(1, len(imgs))]
    return [normalise(p, K, dist) for p in pts], Hpx


def label_corners(c_n, ref_px, H, K, dist):
    """Give view i's detected corners the same names as view 1's: map view
    1's corners into view i with the feature homography and take the nearest
    detected corner for each."""
    mapped = cv2.perspectiveTransform(ref_px.reshape(-1, 1, 2), H).reshape(-1, 2)
    mapped_n = normalise(mapped, K, dist)
    idx = [int(np.argmin(np.linalg.norm(c_n - p, axis=1))) for p in mapped_n]
    if len(set(idx)) != 4:
        raise RuntimeError('corner labelling is ambiguous')
    return c_n[idx]


def panel_mask(img, K, dist, shrink=0.03):
    """Mask of the panel (slightly shrunk) for the feature detector."""
    hull, _ = panel_outline(img)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [hull.astype(np.int32)], 255)
    k = int(shrink * np.sqrt(cv2.contourArea(hull.astype(np.float32)))) | 1
    return cv2.erode(m, np.ones((k, k), np.uint8))


def normalise(px, K, dist):
    return cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64),
                               K, dist).reshape(-1, 2)


def to_pixels(m, K):
    return (K @ np.c_[m, np.ones(len(m))].T).T[:, :2]


# ----------------------------------------------------------------------------
# 2. Homography by DLT (with Hartley normalisation)
# ----------------------------------------------------------------------------
def _hartley(p):
    mu = p.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(p - mu, axis=1))
    return np.array([[s, 0, -s * mu[0]], [0, s, -s * mu[1]], [0, 0, 1]])


def dlt_homography(p, q):
    """H with q ~ H p. Each correspondence gives two rows of A h = 0:
        [ -x -y -1   0  0  0   x'x  x'y  x' ]
        [  0  0  0  -x -y -1   y'x  y'y  y' ]
    h = right singular vector of A for the smallest singular value."""
    Tp, Tq = _hartley(p), _hartley(q)
    ph = (Tp @ np.c_[p, np.ones(len(p))].T).T
    qh = (Tq @ np.c_[q, np.ones(len(q))].T).T
    A = []
    for (x, y, _), (u, v, _) in zip(ph, qh):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, S, Vt = np.linalg.svd(np.array(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tq) @ Hn @ Tp
    return H / H[2, 2], S


# ----------------------------------------------------------------------------
# 3. Homography -> (R, t, n)
# ----------------------------------------------------------------------------
def decompose_by_hand(Hs):
    """The 4 solutions of Hs = R + t n^T (middle singular value already 1).

    theory.md B4: H^T H = V diag(s1^2, 1, s3^2) V^T,
        u = (sqrt(1-s3^2) v1 +/- sqrt(s1^2-1) v3) / sqrt(s1^2-s3^2)
        U = [v2, u, v2 x u],  W = [H v2, H u, H v2 x H u]
        R = W U^T,  n = v2 x u,  t = (H - R) n
    and each (t, n) can also be (-t, -n)."""
    _, S, Vt = np.linalg.svd(Hs.T @ Hs)
    s1, s3 = np.sqrt(S[0]), np.sqrt(S[2])
    v1, v2, v3 = Vt
    out = []
    for sg in (1, -1):
        u = (np.sqrt(max(1 - s3 ** 2, 0)) * v1
             + sg * np.sqrt(max(s1 ** 2 - 1, 0)) * v3) / np.sqrt(s1 ** 2 - s3 ** 2)
        U = np.c_[v2, u, np.cross(v2, u)]
        W = np.c_[Hs @ v2, Hs @ u, np.cross(Hs @ v2, Hs @ u)]
        R = W @ U.T
        n = np.cross(v2, u)
        t = (Hs - R) @ n
        out += [(R, t, n), (R, -t, -n)]
    return out


def decompose(H, m1):
    """Euclidean homography (normalised coords) -> motion and plane.

    H is scaled so that its middle singular value is 1; then
    H = R + t n^T (with t already divided by the plane distance d). The 4
    algebraic solutions come from decompose_by_hand; the ones that put the
    plane in front of camera 1 (n . m > 0 for every point seen in view 1)
    survive. Returns the list of surviving (R, t, n) and the scaled H."""
    Hs = H / np.linalg.svd(H, compute_uv=False)[1]
    if np.linalg.det(Hs) < 0:
        Hs = -Hs
    out = []
    for R, t, n in decompose_by_hand(Hs):
        if np.all(np.c_[m1, np.ones(len(m1))] @ n > 0):
            out.append((R, t, n))
    return out, Hs


def check_against_opencv(Hs, cands):
    """Largest difference between my surviving solutions and the matching
    ones from cv2.decomposeHomographyMat (a check, not used in the pipeline)."""
    _, Rs, ts, ns = cv2.decomposeHomographyMat(Hs, np.eye(3))
    worst = 0.0
    for R, t, n in cands:
        diffs = [max(np.abs(R - R2).max(), np.abs(t - t2.ravel()).max(),
                     np.abs(n - n2.ravel()).max()) for R2, t2, n2 in zip(Rs, ts, ns)]
        worst = max(worst, min(diffs))
    return worst


def choose_motion(cands):
    """Each pair (1,2), (1,3), (1,4) leaves 2 candidates. The true plane
    normal n (in camera-1 coordinates) must be the SAME for all three pairs,
    so pick the combination whose normals agree best."""
    combos = []
    for a in cands[0]:
        for b in cands[1]:
            for c in cands[2]:
                ns = [a[2], b[2], c[2]]
                spread = sum(np.degrees(np.arccos(np.clip(ns[i] @ ns[j], -1, 1)))
                             for i in range(3) for j in range(i + 1, 3))
                combos.append((spread, (a, b, c)))
    combos.sort(key=lambda x: x[0])
    choose_motion.runner_up = combos[1][0] if len(combos) > 1 else float('nan')
    return combos[0][1], combos[0][0]


# ----------------------------------------------------------------------------
# 4. Triangulation (linear, all views at once)
# ----------------------------------------------------------------------------
def triangulate(Ps, ms):
    """Ps: list of 3x4 [R|t]; ms: list of the point's normalised (x, y) in
    each view. x = (P1 X)/(P3 X) gives  x P3 X - P1 X = 0, likewise for y:
    two rows per view, X = last right singular vector."""
    A = []
    for P, (x, y) in zip(Ps, ms):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, S, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3], np.array(A), S


def reproj_px(Ps, X, ms, K):
    """RMS reprojection error in pixels over all views and points."""
    err = []
    for i, P in enumerate(Ps):
        Xc = (P[:, :3] @ X.T).T + P[:, 3]
        proj = Xc[:, :2] / Xc[:, 2:3]
        err.append(to_pixels(proj, K) - to_pixels(ms[i], K))
    e = np.concatenate(err)
    return float(np.sqrt(np.mean(np.sum(e ** 2, 1)))), e


# ----------------------------------------------------------------------------
# 5. Refinement: resection / intersection
# ----------------------------------------------------------------------------
def refine(Ps, X, ms, n_iter=20):
    """Alternate: re-estimate cameras 2-4 from the current 3D points
    (solvePnP, iterative least squares on reprojection error), then
    re-triangulate the points from the new cameras. Camera 1 stays [I|0].
    The overall scale is a free choice (gauge); keep the plane at distance 1
    from camera 1 so it doesn't drift."""
    Ps = [P.copy() for P in Ps]
    for _ in range(n_iter):
        for i in range(1, len(Ps)):
            rv = cv2.Rodrigues(Ps[i][:, :3])[0]
            tv = Ps[i][:, 3].reshape(3, 1).copy()
            _, rv, tv = cv2.solvePnP(X, ms[i], np.eye(3), None, rv, tv,
                                     useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
            Ps[i] = np.c_[cv2.Rodrigues(rv)[0], tv.ravel()]
        X = np.array([triangulate(Ps, [m[j] for m in ms])[0]
                      for j in range(len(X))])
        n, d = fit_plane(X)
        s = 1.0 / d
        X = X * s
        for i in range(1, len(Ps)):
            Ps[i][:, 3] *= s
    return Ps, X


def fit_plane(X):
    """Least-squares plane n.X = d through points X (n unit, facing camera 1)."""
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu)
    n = Vt[-1]
    if n @ mu < 0:
        n = -n
    return n, float(n @ mu)


# ----------------------------------------------------------------------------
# The whole pipeline
# ----------------------------------------------------------------------------
def run(width_mm, height_mm, diag_mm=None, files=None, verbose=True, write=True):
    K, dist, camjson = load_camera()
    files = files or sorted(glob.glob(os.path.join(SFM, '*.jpg')))
    names = [os.path.basename(f) for f in files]
    imgs = [cv2.imread(f) for f in files]
    masks = [panel_mask(im, K, dist) for im in imgs]
    feat_m, Hpx = match_features(imgs, K, dist, masks)
    NG = len(feat_m[0])
    corners = [panel_corners(im, K, dist) for im in imgs]
    corners[0] = order_by_layout(corners[0])
    ref_px = to_pixels(corners[0], K)
    # view 1's corners in distorted pixels, to map with the pixel homography
    ref_px_d = cv2.projectPoints(np.c_[corners[0], np.ones(4)], np.zeros(3),
                                 np.zeros(3), K, dist)[0].reshape(-1, 2)
    for i in range(1, len(imgs)):
        corners[i] = label_corners(corners[i], ref_px_d, Hpx[i - 1], K, dist)
    grid_m = feat_m
    scr_m = corners
    ms = [np.vstack([g, s]) for g, s in zip(grid_m, scr_m)]   # features + 4 corners

    # 2-3: motion from homographies of the feature points, view 1 -> view i
    Hs, cands, dlt_S, cv_diff = [], [], [], []
    for i in range(1, 4):
        H, S = dlt_homography(grid_m[0], grid_m[i])
        c, Hn = decompose(H, grid_m[0])
        cv_diff.append(check_against_opencv(Hn, c))
        Hs.append((H, Hn))
        dlt_S.append(S)
        cands.append(c)
    (m2, m3, m4), spread = choose_motion(cands)
    Ps0 = [np.c_[np.eye(3), np.zeros(3)]] + [np.c_[R, t] for R, t, _ in (m2, m3, m4)]

    # 4: linear triangulation of the feature points
    X0 = np.array([triangulate(Ps0, [m[j] for m in grid_m])[0] for j in range(NG)])
    rms0, _ = reproj_px(Ps0, X0, grid_m, K)
    # 5: refinement, feature points only
    Ps, Xg = refine(Ps0, X0, grid_m)
    rms1, E = reproj_px(Ps, Xg, grid_m, K)
    # the boundary: the 4 panel corners, triangulated from the final cameras
    Xs = np.array([triangulate(Ps, [m[j] for m in scr_m])[0] for j in range(4)])
    rms_b, Eb = reproj_px(Ps, Xs, scr_m, K)
    X = np.vstack([Xg, Xs])

    # 6: object frame on the plane + metric scale from ONE length
    n, d = fit_plane(Xg)
    planarity = Xg @ n - d
    e1 = Xs[1] - Xs[0]                       # along the top edge, TL -> TR
    e1 -= (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    if e2 @ (Xs[3] - Xs[0]) < 0:             # e2 runs down, TL -> BL
        e2 = -e2
    e3 = np.cross(e1, e2)                    # into the cover, away from the cameras
    known_units = np.linalg.norm(Xs[1] - Xs[0])
    scale = width_mm / known_units           # the one known length: the top edge
    O = Xs[0]
    Bm = np.vstack([e1, e2, e3])
    XY = ((X - O) @ Bm.T) * scale            # object coordinates, mm

    cams = []
    for i, P in enumerate(Ps):
        C = -P[:, :3].T @ P[:, 3]
        cams.append(((C - O) @ Bm.T) * scale)
    cams = np.array(cams)

    # ---- checks against the measurements (NOT used above, except the width) ----
    scr = XY[NG:, :2]
    w_top = np.linalg.norm(scr[1] - scr[0])
    w_bot = np.linalg.norm(scr[2] - scr[3])
    h_l = np.linalg.norm(scr[3] - scr[0])
    h_r = np.linalg.norm(scr[2] - scr[1])
    ang = []
    for k in range(4):
        a = scr[k - 1] - scr[k]
        b = scr[(k + 1) % 4] - scr[k]
        ang.append(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))
    inside = ((XY[:NG, 0] > -2) & (XY[:NG, 0] < width_mm + 2) &
              (XY[:NG, 1] > -2) & (XY[:NG, 1] < height_mm + 2)).mean()

    camrows = []
    for i, nm in enumerate(names):
        rv = cv2.Rodrigues(Ps[i][:, :3])[0].ravel()
        camrows.append({
            'view': i + 1, 'image': nm,
            'Cx_mm': cams[i, 0], 'Cy_mm': cams[i, 1], 'Cz_mm': cams[i, 2],
            'dist_to_object_mm': float(np.linalg.norm(cams[i] - XY[NG:].mean(0))),
            'rot_vs_cam1_deg': float(np.degrees(np.linalg.norm(rv))),
        })
    camdf = pd.DataFrame(camrows)
    # viewing angle of each camera to the cover (0 = straight down on it)
    view_angles = [float(np.degrees(np.arccos(abs((C - XY[NG:].mean(0))[2]) /
                                              np.linalg.norm(C - XY[NG:].mean(0)))))
                   for C in cams]
    summary = {
        'width_mm_measured': width_mm, 'height_mm_measured': height_mm,
        'n_feature_points': int(NG),
        'normal_spread_deg_sum': float(spread),
        'decomposition_vs_opencv_maxdiff': float(max(cv_diff)),
        'reproj_rms_px_linear': rms0, 'reproj_rms_px_refined': rms1,
        'reproj_max_px_refined': float(np.max(np.linalg.norm(E, axis=1))),
        'boundary_reproj_rms_px': rms_b,
        'boundary_reproj_max_px': float(np.max(np.linalg.norm(Eb, axis=1))),
        'boundary_out_of_plane_mm': [float(v) for v in (Xs @ n - d) * scale],
        'planarity_rms_mm': float(np.sqrt(np.mean(planarity ** 2)) * scale),
        'features_inside_boundary': float(inside),
        'width_top_mm': float(w_top), 'width_bottom_mm': float(w_bot),
        'height_left_mm': float(h_l), 'height_right_mm': float(h_r),
        'height_error_mm': float((h_l + h_r) / 2 - height_mm),
        'diagonal_mm': float(np.linalg.norm(scr[2] - scr[0])),
        'diagonal_mm_measured': diag_mm,
        'corner_angles_deg': [float(a) for a in ang],
        'aspect': float((w_top + w_bot) / (h_l + h_r)),
        'aspect_measured': float(width_mm / height_mm),
        'viewing_angles_deg': view_angles,
    }
    ctx = dict(K=K, dist=dist, names=names, imgs=imgs, ms=ms,
               Hs=Hs, cands=cands, chosen=(m2, m3, m4), dlt_S=dlt_S, Ps0=Ps0,
               X0=X0, Ps=Ps, X=X, XY=XY, cams=cams, scale=scale,
               known_units=known_units, width_mm=width_mm, height_mm=height_mm,
               NG=NG, n=n, d=d, e=(e1, e2, e3), O=O, summary=summary,
               camdf=camdf, spread=spread, rms0=rms0, rms1=rms1, rms_b=rms_b)
    if not write:
        return ctx
    os.makedirs(RESULTS, exist_ok=True)
    pts = pd.DataFrame({
        'id': np.arange(len(X)),
        'kind': ['feature'] * NG + ['corner_' + k for k in CORNERS],
        'X_cam1': X[:, 0], 'Y_cam1': X[:, 1], 'Z_cam1': X[:, 2],
        'x_obj_mm': XY[:, 0], 'y_obj_mm': XY[:, 1], 'z_obj_mm': XY[:, 2]})
    for i, nm in enumerate(names):
        px = to_pixels(ms[i], K)
        pts['u_' + nm[:-4]] = px[:, 0]
        pts['v_' + nm[:-4]] = px[:, 1]
    pts.to_csv(os.path.join(RESULTS, 'sfm_points.csv'), index=False, float_format='%.4f')
    camdf.to_csv(os.path.join(RESULTS, 'sfm_cameras.csv'), index=False, float_format='%.2f')
    with open(os.path.join(RESULTS, 'sfm_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    plots(ctx)
    write_workings(ctx)
    if verbose:
        print(json.dumps(summary, indent=2))
        print(camdf.round(1).to_string(index=False))
    return ctx


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plots(c):
    K, NG = c['K'], c['NG']
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for i, ax in enumerate(axes):
        im = cv2.undistort(c['imgs'][i], K, c['dist'])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        px = to_pixels(c['ms'][i], K)
        ax.plot(px[:NG, 0], px[:NG, 1], '.', color='yellow', ms=2)
        q = np.vstack([px[NG:], px[NG]])
        ax.plot(q[:, 0], q[:, 1], '-', color='red', lw=1.5)
        # reprojection of the refined 3D points
        P = c['Ps'][i]
        Xc = (P[:, :3] @ c['X'].T).T + P[:, 3]
        rp = to_pixels(Xc[:, :2] / Xc[:, 2:3], K)
        ax.plot(rp[NG:, 0], rp[NG:, 1], '+', color='lime', ms=8, mew=2)
        ax.set_title('view %d: %s' % (i + 1, c['names'][i]), fontsize=9)
        ax.axis('off')
    fig.suptitle('The four views (undistorted). yellow: %d feature points matched '
                 'in all four views, red: boundary (panel corners), green +: '
                 'reconstructed corners re-projected' % NG, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'sfm_views.png'), dpi=100)
    plt.close(fig)

    XY, cams = c['XY'], c['cams']
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XY[:NG, 0], XY[:NG, 1], XY[:NG, 2], s=6, c='k', label='feature points on the cover')
    q = np.vstack([XY[NG:], XY[NG]])
    ax.plot(q[:, 0], q[:, 1], q[:, 2], 'r-', lw=2, label='boundary (panel corners)')
    for i, C in enumerate(cams):
        R = c['Ps'][i][:, :3]
        Bm = np.vstack(c['e'])
        z = Bm @ R.T @ np.array([0, 0, 1.0])       # optical axis, object frame
        ax.quiver(*C, *(z * 150), color='tab:blue', lw=1.5)
        ax.scatter(*C, color='tab:blue', s=30)
        ax.text(*C, '  view %d' % (i + 1), fontsize=8)
    ax.scatter([], [], color='tab:blue', label='camera (SfM) + viewing direction')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    allp = np.vstack([XY, cams])
    ax.set_box_aspect(np.ptp(allp, 0))
    ax.view_init(elev=-150, azim=-60)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('Reconstruction, object frame (origin = top-left corner)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'sfm_3d.png'), dpi=110)
    plt.close(fig)

    s = c['summary']
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(XY[:NG, 0], XY[:NG, 1], 'k.', ms=2, alpha=0.5, label='reconstructed feature points')
    q = np.vstack([XY[NG:, :2], XY[NG, :2]])
    ax.fill(q[:, 0], q[:, 1], color='red', alpha=0.08)
    ax.plot(q[:, 0], q[:, 1], 'r-o', lw=2, ms=5, label='reconstructed boundary')
    for k, nm in enumerate(['TL', 'TR', 'BR', 'BL']):
        ax.annotate('%s\n(%.0f, %.0f)' % (nm, XY[NG + k, 0], XY[NG + k, 1]),
                    XY[NG + k, :2], fontsize=8, ha='center',
                    xytext=(0, 10 if k < 2 else -24), textcoords='offset points')
    ax.set_aspect('equal')
    ax.margins(0.1)
    ax.invert_yaxis()
    ax.set_xlabel('x on the object (mm)')
    ax.set_ylabel('y on the object (mm)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title('Estimated boundary of the object, in its own plane\n'
                 'width %.0f / %.0f mm (top / bottom), height %.0f / %.0f mm '
                 '(left / right), corners %s deg'
                 % (s['width_top_mm'], s['width_bottom_mm'],
                    s['height_left_mm'], s['height_right_mm'],
                    ', '.join('%.1f' % a for a in s['corner_angles_deg'])),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'sfm_boundary.png'), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# The worked calculation, with this run's numbers
# ----------------------------------------------------------------------------
def M(a, nd=4):
    a = np.atleast_2d(a)
    return (r'\begin{bmatrix}' + r' \\ '.join(
        ' & '.join(('%.' + str(nd) + 'f') % v for v in r) for r in a)
        + r'\end{bmatrix}')


def write_workings(c):
    K, NG = c['K'], c['NG']
    L = []
    L.append('## Part B — worked calculation (numbers from this run of `sfm.py`)\n')
    L.append('### Step 0 — camera parameters\n')
    L.append('Intrinsics from the Module 2 calibration (same phone, same main '
             'camera, full 3024×4032 resolution):\n\n$$K = %s$$\n' % M(K, 2))
    L.append('Distortion $(k_1, k_2, p_1, p_2, k_3)$ = (%s). Every detected '
             'pixel is undistorted and converted to normalised coordinates '
             '$\\mathbf{m} = K^{-1}\\mathbf{x}$ first, so from here on the '
             'cameras are ideal pinholes with $K = I$.\n'
             % ', '.join('%.4f' % v for v in c['dist'].ravel()))
    p = to_pixels(c['ms'][0][:1], K)[0]
    L.append('Example, feature point 0 in view 1: undistorted pixel '
             '$(%.2f, %.2f)$ → $\\mathbf{m} = ((%.2f-%.2f)/%.2f,\\ (%.2f-%.2f)/%.2f) '
             '= (%.5f, %.5f)$.\n'
             % (p[0], p[1], p[0], K[0, 2], K[0, 0], p[1], K[1, 2], K[1, 1],
                c['ms'][0][0, 0], c['ms'][0][0, 1]))

    L.append('### Step 1 — homography view 1 → view 2 (DLT)\n')
    H, Hn = c['Hs'][0]
    S = c['dlt_S'][0]
    L.append('%d feature correspondences give a %d×9 matrix $A$. Its two smallest '
             'singular values are %.2e and %.2e — the smallest is far below the '
             'next, so the points are related by a single homography (they lie on a '
             'plane). $\\mathbf{h}$ = the singular vector of the smallest:\n'
             % (NG, 2 * NG, S[-2], S[-1]))
    L.append('$$H_{12} = %s$$\n' % M(H, 5))
    sv = np.linalg.svd(H, compute_uv=False)
    L.append('Singular values of $H_{12}$: %s. Dividing by the middle one gives '
             'the Euclidean homography $\\hat H = R + \\mathbf{t}\\mathbf{n}^T/d$:\n'
             % ', '.join('%.5f' % v for v in sv))
    L.append('$$\\hat H_{12} = %s$$\n' % M(Hn, 5))

    L.append('### Step 2 — decomposition into $R, \\mathbf{t}, \\mathbf{n}$\n')
    L.append('From $\\hat H^T\\hat H$: $\\sigma_1 = %.5f$, $\\sigma_2 = %.5f$, '
             '$\\sigma_3 = %.5f$. The formulas in theory.md B4 give 4 solutions '
             '(checked against `cv2.decomposeHomographyMat`: largest difference '
             '%.1e over all three pairs).\n'
             % (*np.sqrt(np.linalg.svd(Hn.T @ Hn, compute_uv=False)),
                c['summary']['decomposition_vs_opencv_maxdiff']))
    L.append('$\\hat H$ has 4 algebraic decompositions. Two put the plane '
             'behind camera 1 ($\\mathbf{n}^T\\mathbf{m} < 0$ for some point) '
             'and are rejected. The two left for each pair:\n')
    L.append('| pair | candidate | $\\mathbf{n}$ (camera 1 frame) | $\\mathbf{t}/d$ |')
    L.append('|---|---|---|---|')
    for i, cand in enumerate(c['cands']):
        for k, (R, t, n) in enumerate(cand):
            L.append('| 1→%d | %d | (%.3f, %.3f, %.3f) | (%.3f, %.3f, %.3f) |'
                     % (i + 2, k + 1, *n, *t))
    L.append('')
    L.append('The plane is the same physical plane in all three pairs, so its '
             'normal in camera 1 must be the same, so the combination whose '
             'normals agree best is chosen: the three normals are %.1f° apart in '
             'total (sum of pairwise angles), against %.1f° for the next-best '
             'combination. Chosen:\n' % (c['spread'], choose_motion.runner_up))
    for i, (R, t, n) in enumerate(c['chosen']):
        L.append('$$R_{1%d} = %s,\\quad \\mathbf{t}_{1%d}/d = %s,\\quad '
                 '\\mathbf{n} = %s$$\n' % (i + 2, M(R, 4), i + 2, M(t.reshape(3, 1), 4),
                                         M(n.reshape(3, 1), 4)))

    L.append('### Step 3 — triangulation of one boundary point\n')
    j = NG            # panel top-left
    X, A, S = triangulate(c['Ps'], [m[j] for m in c['ms']])
    L.append('Panel corner TL. Normalised coordinates in the 4 views: %s. '
             'Camera matrices (after refinement) $P_1 = [I|\\mathbf{0}]$, $P_i = [R_{1i}|\\mathbf{t}_{1i}]$. '
             'Each view contributes $x\\,\\mathbf{p}_3^T - \\mathbf{p}_1^T$ and '
             '$y\\,\\mathbf{p}_3^T - \\mathbf{p}_2^T$:\n'
             % ', '.join('(%.4f, %.4f)' % tuple(m[j]) for m in c['ms']))
    L.append('$$A = %s$$\n' % M(A, 4))
    L.append('Singular values of $A$: %s. The last is $\\approx 0$, as it should be. '
             'Its singular vector, divided by its 4th entry: '
             '$\\mathbf{X} = (%.4f, %.4f, %.4f)$ in units of $d$ (the distance '
             'from camera 1 to the plane).\n'
             % (', '.join('%.2e' % v for v in S), X[0], X[1], X[2]))
    L.append('The %d feature points triangulated this way (from the homography '
             'cameras): RMS reprojection error **%.2f px**. After the refinement '
             '(re-estimate cameras 2–4 from the points, re-triangulate, 20 '
             'rounds): **%.2f px**. The 4 panel corners are then triangulated '
             'from the refined cameras: **%.2f px** RMS (they come from fitted '
             'edge lines, so they are located less precisely than the feature '
             'points).\n' % (NG, c['rms0'], c['rms1'], c['rms_b']))

    L.append('### Step 4 — scale and the object frame\n')
    L.append('SfM gives shape only up to scale. One length fixes it: the top '
             'edge of the panel (TL to TR) is %.1f mm (measured). In the '
             'reconstruction it is %.5f units long, so '
             '$s = %.1f / %.5f = %.2f$ mm per unit ($d = %.0f$ mm).\n'
             % (c['width_mm'], c['known_units'], c['width_mm'], c['known_units'],
                c['scale'], c['scale']))
    e1, e2, e3 = c['e']
    L.append('Plane fitted to the points: $\\mathbf{n} = (%.4f, %.4f, %.4f)$. '
             'Object axes: $\\mathbf{e}_1$ along the top edge (TL to TR), '
             '$\\mathbf{e}_2 = \\mathbf{n}\\times\\mathbf{e}_1$, origin at the '
             'TL corner. Object coordinates $= s\\,[\\mathbf{e}_1\\ \\mathbf{e}_2\\ '
             '\\mathbf{e}_3]^T(\\mathbf{X}-\\mathbf{X}_0)$.\n' % tuple(c['n']))
    s = c['summary']
    xy = c['XY'][NG:]
    L.append('Panel corners on the object (mm): TL (%.1f, %.1f), TR (%.1f, %.1f), '
             'BR (%.1f, %.1f), BL (%.1f, %.1f). Out of plane: all within %.2f mm.\n'
             % (*xy[0, :2], *xy[1, :2], *xy[2, :2], *xy[3, :2],
                np.abs(xy[:, 2]).max()))
    L.append('Width = |TR − TL| = %.1f mm (set by the scale), height = |BL − TL| '
             '= %.1f mm against %.1f mm measured; corner angles %s.\n'
             % (s['width_top_mm'], s['height_left_mm'], c['height_mm'],
                ', '.join('%.1f°' % a for a in s['corner_angles_deg'])))
    L.append('### Step 5 — camera positions\n')
    L.append('Camera centre $\\mathbf{C}_i = -R_{1i}^T\\mathbf{t}_{1i}$, converted to '
             'the object frame and mm:\n')
    L.append('| view | image | C (mm) | distance to object (mm) | rotation vs view 1 |')
    L.append('|---|---|---|---|---|')
    for _, r in c['camdf'].iterrows():
        L.append('| %d | %s | (%.0f, %.0f, %.0f) | %.0f | %.1f° |'
                 % (r['view'], r.image, r.Cx_mm, r.Cy_mm, r.Cz_mm,
                    r.dist_to_object_mm, r.rot_vs_cam1_deg))
    L.append('')
    with open(os.path.join(RESULTS, 'workings_sfm.md'), 'w') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--width-mm', type=float, required=True,
                    help='width of the purple panel, measured (the one known length)')
    ap.add_argument('--height-mm', type=float, required=True,
                    help='height of the purple panel, measured (used only as a check)')
    ap.add_argument('--diag-mm', type=float,
                    help='diagonal of the purple panel (used only as a check)')
    a = ap.parse_args()
    run(a.width_mm, a.height_mm, a.diag_mm)


if __name__ == '__main__':
    main()

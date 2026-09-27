"""
sfm.py -- Part B: structure from motion from four views of a flat object.

Usage (from this folder):
    python sfm.py                     # the 4 views in data/sfm/
    python sfm.py --square-mm 30      # the one known length (see below)

The object is my monitor screen showing the chessboard page I used for the
Module 2 calibration, photographed from four positions with the same
(calibrated) iPhone. The screen is flat, so it is the planar object the
assignment suggests. Two kinds of point are reconstructed:
    - the 54 inner chessboard corners (they give the correspondences), and
    - the 4 corners of the lit screen area: the object's boundary.

Only image measurements and K are used to reconstruct. The chessboard's real
geometry is NOT used, except for:
    - one known length (8 squares along the top row) to fix the scale, which
      SfM can never recover by itself, and
    - afterwards, as ground truth to check the result (squares should come out
      square, camera positions should match a solvePnP pose).

Steps (theory.md, Part B, has the maths):
    1. detect points, undistort to normalised coordinates  m = K^-1 x
    2. homography view 1 -> view i from the grid points     (my DLT)
    3. H = R + t n^T / d  ->  R_i, t_i, n                    (4 solutions each,
                                                              pick consistent)
    4. triangulate every point from all 4 views             (my linear DLT)
    5. refine: re-estimate each camera from the points, re-triangulate, repeat
    6. fix the scale with one known length, express in the object's plane
    7. boundary = polygon through the reconstructed screen corners

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
GRID = (9, 6)            # inner corners: 9 along a row, 6 rows


# ----------------------------------------------------------------------------
# 1. Points in each image
# ----------------------------------------------------------------------------
def load_camera():
    with open(os.path.join(SFM, 'camera.json')) as f:
        c = json.load(f)
    return np.array(c['K']), np.array(c['dist']), c


def detect_grid(img):
    """54 chessboard inner corners, sub-pixel, ordered row by row starting at
    the corner nearest the top-left of the image (same physical corner in
    every view, as long as the phone is held upright)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ok, c = cv2.findChessboardCorners(
        s, GRID, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        raise RuntimeError('chessboard not found')
    c = c * 2.0
    c = cv2.cornerSubPix(g, c, (7, 7), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    c = c.reshape(-1, 2)
    if c[0].sum() > c[-1].sum():          # detected from the other end
        c = c[::-1].copy()
    return c


def detect_screen(img, K, dist):
    """The 4 corners of the lit screen, ordered TL, TR, BR, BL.

    The lit area is found by Otsu threshold; its outline is undistorted, each
    of the 4 sides gets a straight line fitted through all its boundary
    pixels, and the corners are where adjacent lines meet. Returns the corners
    in NORMALISED coordinates and in (undistorted) pixel coordinates."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(g, (7, 7), 0).astype(np.float32)
    top = np.percentile(b, 99.5)
    found = None
    # The screen is the large bright quadrilateral. Its grey margins are much
    # darker than the white page in the middle, and the desk lamp adds glow
    # above it, so no single threshold suits every photo: try levels from low
    # to high and keep the first one whose biggest blob really is a clean
    # four-sided shape (fills >= 97% of its 4-corner polygon).
    for frac in np.arange(0.15, 0.75, 0.05):
        m = (b > frac * top).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        ap = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(ap) == 4 and cv2.contourArea(cnt) / cv2.contourArea(ap) > 0.97 \
                and cv2.contourArea(cnt) > 0.02 * b.size:
            found = (cnt.reshape(-1, 2).astype(np.float64), ap.reshape(-1, 2))
            break
    if found is None:
        raise RuntimeError('screen outline is not a quadrilateral')
    cnt, ap = found
    # undistorted, normalised outline
    un = cv2.undistortPoints(cnt.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    apn = cv2.undistortPoints(ap.astype(np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    # which contour points belong to which side: nearest side of the rough quad
    lines = []
    for k in range(4):
        a, c = apn[k], apn[(k + 1) % 4]
        dvec = c - a
        L = np.linalg.norm(dvec)
        t = ((un - a) @ dvec) / L ** 2
        dist_to = np.abs(dvec[0] * (un[:, 1] - a[1]) - dvec[1] * (un[:, 0] - a[0])) / L
        sel = (t > 0.1) & (t < 0.9) & (dist_to < 0.01)
        pts = un[sel]
        # total least squares line: through the mean, along the main direction
        mu = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mu)
        nrm = vt[1]
        lines.append(np.array([nrm[0], nrm[1], -nrm @ mu]))
    corners = []
    for k in range(4):
        p = np.cross(lines[k - 1], lines[k])
        corners.append(p[:2] / p[2])
    corners = np.array(corners)
    # order TL, TR, BR, BL
    s = corners.sum(1)
    d = corners[:, 0] - corners[:, 1]
    ordered = np.array([corners[np.argmin(s)], corners[np.argmax(d)],
                        corners[np.argmax(s)], corners[np.argmin(d)]])
    return ordered


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
    best = None
    for a in cands[0]:
        for b in cands[1]:
            for c in cands[2]:
                ns = [a[2], b[2], c[2]]
                spread = sum(np.degrees(np.arccos(np.clip(ns[i] @ ns[j], -1, 1)))
                             for i in range(3) for j in range(i + 1, 3))
                if best is None or spread < best[0]:
                    best = (spread, (a, b, c))
    return best[1], best[0]


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
def run(square_mm, verbose=True):
    K, dist, camjson = load_camera()
    files = sorted(glob.glob(os.path.join(SFM, '*.jpg')))
    names = [os.path.basename(f) for f in files]
    imgs = [cv2.imread(f) for f in files]
    grid_px = [detect_grid(im) for im in imgs]
    grid_m = [normalise(g, K, dist) for g in grid_px]
    scr_m = [detect_screen(im, K, dist) for im in imgs]
    ms = [np.vstack([g, s]) for g, s in zip(grid_m, scr_m)]   # 54 + 4 points
    NG = GRID[0] * GRID[1]

    # 2-3: motion from homographies of the grid points, view 1 -> view i
    Hs, cands, dlt_S, cv_diff = [], [], [], []
    for i in range(1, 4):
        H, S = dlt_homography(grid_m[0], grid_m[i])
        c, Hn = decompose(H, grid_m[0])
        cv_diff.append(check_against_opencv(Hn, c))
        Hs.append((H, Hn))
        dlt_S.append(S)
        cands.append(c)
    (m2, m3, m4), spread = choose_motion(cands)
    n_avg = np.mean([m2[2], m3[2], m4[2]], 0)
    n_avg /= np.linalg.norm(n_avg)
    Ps0 = [np.c_[np.eye(3), np.zeros(3)]] + [np.c_[R, t] for R, t, _ in (m2, m3, m4)]

    # 4: linear triangulation of the grid points
    X0 = np.array([triangulate(Ps0, [m[j] for m in grid_m])[0] for j in range(NG)])
    rms0, _ = reproj_px(Ps0, X0, grid_m, K)
    # 5: refinement, grid points only (their detections are the most precise)
    Ps, Xg = refine(Ps0, X0, grid_m)
    rms1, E = reproj_px(Ps, Xg, grid_m, K)
    # the boundary: the 4 screen corners, triangulated from the final cameras
    Xs = np.array([triangulate(Ps, [m[j] for m in scr_m])[0] for j in range(4)])
    rms_b, Eb = reproj_px(Ps, Xs, scr_m, K)
    X = np.vstack([Xg, Xs])

    # 6: object frame on the plane + metric scale from ONE length
    n, d = fit_plane(Xg)
    planarity = Xg @ n - d
    G = X[:NG].reshape(GRID[1], GRID[0], 3)
    e1 = G[0, -1] - G[0, 0]
    e1 -= (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    if e2 @ (G[-1, 0] - G[0, 0]) < 0:       # e2 runs down the rows, like the board
        e2 = -e2
    e3 = np.cross(e1, e2)                    # = board z (away from the cameras)
    known_units = np.linalg.norm(G[0, -1] - G[0, 0])
    scale = (GRID[0] - 1) * square_mm / known_units
    O = G[0, 0]
    Bm = np.vstack([e1, e2, e3])             # rows: object axes in camera-1 frame
    XY = ((X - O) @ Bm.T) * scale            # object coordinates, mm

    # camera centres in the object frame: C = -R^T t (camera-1 coords) -> object
    cams = []
    for i, P in enumerate(Ps):
        C = -P[:, :3].T @ P[:, 3]
        cams.append(((C - O) @ Bm.T) * scale)
    cams = np.array(cams)

    # ---- checks against ground truth (NOT used above) ----
    gxy = XY[:NG, :2].reshape(GRID[1], GRID[0], 2)
    sq_h = np.linalg.norm(np.diff(gxy, axis=1), axis=2).ravel()   # along rows
    sq_v = np.linalg.norm(np.diff(gxy, axis=0), axis=2).ravel()   # down columns
    objp = np.zeros((NG, 3))
    objp[:, :2] = np.mgrid[0:GRID[0], 0:GRID[1]].T.reshape(-1, 2) * square_mm
    grid_err = np.linalg.norm(XY[:NG, :2] - objp[:, :2], axis=1)
    pnp = []
    for i in range(4):
        _, rv, tv = cv2.solvePnP(objp, grid_m[i], np.eye(3), None)
        R = cv2.Rodrigues(rv)[0]
        pnp.append((-R.T @ tv).ravel())
    pnp = np.array(pnp)
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

    os.makedirs(RESULTS, exist_ok=True)
    # --- tables ---
    kinds = ['grid'] * NG + ['screen_' + k for k in ('TL', 'TR', 'BR', 'BL')]
    pts = pd.DataFrame({
        'id': np.arange(len(X)), 'kind': kinds,
        'X_cam1': X[:, 0], 'Y_cam1': X[:, 1], 'Z_cam1': X[:, 2],
        'x_obj_mm': XY[:, 0], 'y_obj_mm': XY[:, 1], 'z_obj_mm': XY[:, 2]})
    for i, nm in enumerate(names):
        px = to_pixels(ms[i], K)
        pts['u_' + nm[:-4]] = px[:, 0]
        pts['v_' + nm[:-4]] = px[:, 1]
    pts.to_csv(os.path.join(RESULTS, 'sfm_points.csv'), index=False, float_format='%.4f')

    camrows = []
    for i, nm in enumerate(names):
        R = Ps[i][:, :3]
        # rotation of camera i relative to camera 1, as one angle about one axis
        rv = cv2.Rodrigues(R)[0].ravel()
        camrows.append({
            'view': i + 1, 'image': nm,
            'Cx_mm': cams[i, 0], 'Cy_mm': cams[i, 1], 'Cz_mm': cams[i, 2],
            'dist_to_object_mm': float(np.linalg.norm(cams[i] - XY[:NG].mean(0))),
            'rot_vs_cam1_deg': float(np.degrees(np.linalg.norm(rv))),
            'pnp_Cx_mm': pnp[i, 0], 'pnp_Cy_mm': pnp[i, 1], 'pnp_Cz_mm': pnp[i, 2],
            'diff_vs_pnp_mm': float(np.linalg.norm(cams[i] - pnp[i])),
        })
    camdf = pd.DataFrame(camrows)
    camdf.to_csv(os.path.join(RESULTS, 'sfm_cameras.csv'), index=False, float_format='%.2f')

    summary = {
        'square_mm_assumed': square_mm,
        'normal_spread_deg_sum': float(spread),
        'decomposition_vs_opencv_maxdiff': float(max(cv_diff)),
        'reproj_rms_px_linear': rms0, 'reproj_rms_px_refined': rms1,
        'reproj_max_px_refined': float(np.max(np.linalg.norm(E, axis=1))),
        'boundary_reproj_rms_px': rms_b,
        'boundary_reproj_max_px': float(np.max(np.linalg.norm(Eb, axis=1))),
        'boundary_out_of_plane_mm': [float(v) for v in (Xs @ n - d) * scale],
        'planarity_rms_mm': float(np.sqrt(np.mean(planarity ** 2)) * scale),
        'square_side_along_rows_mm': [float(sq_h.mean()), float(sq_h.std())],
        'square_side_down_cols_mm': [float(sq_v.mean()), float(sq_v.std())],
        'grid_vs_true_rms_mm': float(np.sqrt(np.mean(grid_err ** 2))),
        'grid_vs_true_max_mm': float(grid_err.max()),
        'screen_width_top_mm': float(w_top), 'screen_width_bottom_mm': float(w_bot),
        'screen_height_left_mm': float(h_l), 'screen_height_right_mm': float(h_r),
        'screen_corner_angles_deg': [float(a) for a in ang],
        'screen_aspect': float((w_top + w_bot) / (h_l + h_r)),
        'screen_diagonal_in': float(np.hypot((w_top + w_bot) / 2, (h_l + h_r) / 2) / 25.4),
        'camera_vs_pnp_mm_mean': float(camdf.diff_vs_pnp_mm.mean()),
        'camera_vs_pnp_mm_max': float(camdf.diff_vs_pnp_mm.max()),
    }
    with open(os.path.join(RESULTS, 'sfm_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    ctx = dict(K=K, dist=dist, names=names, imgs=imgs, grid_px=grid_px, ms=ms,
               Hs=Hs, cands=cands, chosen=(m2, m3, m4), dlt_S=dlt_S, Ps0=Ps0,
               X0=X0, Ps=Ps, X=X, XY=XY, cams=cams, pnp=pnp, scale=scale,
               known_units=known_units, square_mm=square_mm, NG=NG, n=n, d=d,
               e=(e1, e2, e3), O=O, summary=summary, camdf=camdf, spread=spread,
               rms0=rms0, rms1=rms1, rms_b=rms_b)
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
        ax.plot(px[:NG, 0], px[:NG, 1], '.', color='yellow', ms=3)
        q = np.vstack([px[NG:], px[NG]])
        ax.plot(q[:, 0], q[:, 1], '-', color='red', lw=1.5)
        ax.plot(px[0, 0], px[0, 1], 'o', mfc='none', mec='cyan', ms=8)
        # reprojection of the refined 3D points
        P = c['Ps'][i]
        Xc = (P[:, :3] @ c['X'].T).T + P[:, 3]
        rp = to_pixels(Xc[:, :2] / Xc[:, 2:3], K)
        ax.plot(rp[:, 0], rp[:, 1], '+', color='lime', ms=4)
        ax.set_title('view %d: %s' % (i + 1, c['names'][i]), fontsize=9)
        ax.axis('off')
    fig.suptitle('The four views (undistorted). yellow: grid corners, red: screen '
                 'boundary, cyan: grid corner 0, green +: refined 3D points '
                 're-projected', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'sfm_views.png'), dpi=100)
    plt.close(fig)

    XY, cams, pnp = c['XY'], c['cams'], c['pnp']
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XY[:NG, 0], XY[:NG, 1], XY[:NG, 2], s=6, c='k', label='grid points')
    q = np.vstack([XY[NG:], XY[NG]])
    ax.plot(q[:, 0], q[:, 1], q[:, 2], 'r-', lw=2, label='screen boundary')
    for i, C in enumerate(cams):
        R = c['Ps'][i][:, :3]
        Bm = np.vstack(c['e'])
        z = Bm @ R.T @ np.array([0, 0, 1.0])       # optical axis, object frame
        ax.quiver(*C, *(z * 300), color='tab:blue', lw=1.5)
        ax.scatter(*C, color='tab:blue', s=30)
        ax.scatter(*pnp[i], color='tab:orange', marker='x', s=40)
        ax.text(*C, '  view %d' % (i + 1), fontsize=8)
    ax.scatter([], [], color='tab:blue', label='camera (SfM) + viewing direction')
    ax.scatter([], [], color='tab:orange', marker='x', label='camera (solvePnP, check)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    allp = np.vstack([XY, cams])
    ax.set_box_aspect(np.ptp(allp, 0))
    ax.view_init(elev=-150, azim=-60)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('Reconstruction, object frame (origin = grid corner 0)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'sfm_3d.png'), dpi=110)
    plt.close(fig)

    s = c['summary']
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(XY[:NG, 0], XY[:NG, 1], 'k.', ms=5, label='reconstructed grid corners')
    q = np.vstack([XY[NG:, :2], XY[NG, :2]])
    ax.fill(q[:, 0], q[:, 1], color='red', alpha=0.08)
    ax.plot(q[:, 0], q[:, 1], 'r-o', lw=2, ms=5, label='reconstructed screen boundary')
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
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('Estimated boundary of the object, in its own plane\n'
                 'width %.0f / %.0f mm (top / bottom), height %.0f / %.0f mm '
                 '(left / right), corners %s deg'
                 % (s['screen_width_top_mm'], s['screen_width_bottom_mm'],
                    s['screen_height_left_mm'], s['screen_height_right_mm'],
                    ', '.join('%.1f' % a for a in s['screen_corner_angles_deg'])),
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
    L.append('Intrinsics from the Module 2 calibration, scaled to the 1512×2016 '
             'images used here:\n\n$$K = %s$$\n' % M(K, 2))
    L.append('Distortion $(k_1, k_2, p_1, p_2, k_3)$ = (%s). Every detected '
             'pixel is undistorted and converted to normalised coordinates '
             '$\\mathbf{m} = K^{-1}\\mathbf{x}$ first, so from here on the '
             'cameras are ideal pinholes with $K = I$.\n'
             % ', '.join('%.4f' % v for v in c['dist'].ravel()))
    p = to_pixels(c['ms'][0][:1], K)[0]
    L.append('Example, grid corner 0 in view 1: undistorted pixel '
             '$(%.2f, %.2f)$ → $\\mathbf{m} = ((%.2f-%.2f)/%.2f,\\ (%.2f-%.2f)/%.2f) '
             '= (%.5f, %.5f)$.\n'
             % (p[0], p[1], p[0], K[0, 2], K[0, 0], p[1], K[1, 2], K[1, 1],
                c['ms'][0][0, 0], c['ms'][0][0, 1]))

    L.append('### Step 1 — homography view 1 → view 2 (DLT)\n')
    H, Hn = c['Hs'][0]
    S = c['dlt_S'][0]
    L.append('54 grid correspondences give a 108×9 matrix $A$. Its two smallest '
             'singular values are %.2e and %.2e — one clearly near zero, so the '
             'points really are related by a single homography (they lie on a '
             'plane). $\\mathbf{h}$ = the singular vector of the smallest:\n'
             % (S[-2], S[-1]))
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
             'normal in camera 1 must be the same. Only one combination agrees '
             '(sum of pairwise angles between the chosen normals: %.2f°); the '
             'other candidates disagree by tens of degrees. Chosen:\n'
             % c['spread'])
    for i, (R, t, n) in enumerate(c['chosen']):
        L.append('$$R_{1%d} = %s,\\quad \\mathbf{t}_{1%d}/d = %s,\\quad '
                 '\\mathbf{n} = %s$$\n' % (i + 2, M(R, 4), i + 2, M(t.reshape(3, 1), 4),
                                         M(n.reshape(3, 1), 4)))

    L.append('### Step 3 — triangulation of one boundary point\n')
    j = NG            # screen top-left
    X, A, S = triangulate(c['Ps'], [m[j] for m in c['ms']])
    L.append('Screen corner TL. Normalised coordinates in the 4 views: %s. '
             'Camera matrices (after refinement) $P_1 = [I|\\mathbf{0}]$, $P_i = [R_{1i}|\\mathbf{t}_{1i}]$. '
             'Each view contributes $x\\,\\mathbf{p}_3^T - \\mathbf{p}_1^T$ and '
             '$y\\,\\mathbf{p}_3^T - \\mathbf{p}_2^T$:\n'
             % ', '.join('(%.4f, %.4f)' % tuple(m[j]) for m in c['ms']))
    L.append('$$A = %s$$\n' % M(A, 4))
    L.append('Singular values of $A$: %s. The last is ≈ 0, as it should be. '
             'Its singular vector, divided by its 4th entry: '
             '$\\mathbf{X} = (%.4f, %.4f, %.4f)$ in units of $d$ (the distance '
             'from camera 1 to the plane).\n'
             % (', '.join('%.2e' % v for v in S), X[0], X[1], X[2]))
    L.append('The 54 grid points triangulated this way (from the homography '
             'cameras): RMS reprojection error **%.2f px**. After the refinement '
             '(re-estimate cameras 2–4 from the points, re-triangulate, 20 '
             'rounds): **%.2f px**. The 4 screen corners are then triangulated '
             'from the refined cameras: **%.2f px** RMS (their edges are soft '
             'glowing edges, so they are located less precisely than chessboard '
             'corners).\n' % (c['rms0'], c['rms1'], c['rms_b']))

    L.append('### Step 4 — scale and the object frame\n')
    L.append('SfM gives shape only up to scale. One length fixes it: grid '
             'corners 0 and 8 are 8 squares apart, 8 × %.1f mm = %.1f mm. In the '
             'reconstruction they are %.5f units apart, so '
             '$s = %.1f / %.5f = %.2f$ mm per unit ($d = %.0f$ mm).\n'
             % (c['square_mm'], 8 * c['square_mm'], c['known_units'],
                8 * c['square_mm'], c['known_units'], c['scale'], c['scale']))
    e1, e2, e3 = c['e']
    L.append('Plane fitted to the points: $\\mathbf{n} = (%.4f, %.4f, %.4f)$. '
             'Object axes: $\\mathbf{e}_1$ along the top row of the grid, '
             '$\\mathbf{e}_2 = \\mathbf{n}\\times\\mathbf{e}_1$, origin at grid '
             'corner 0. Object coordinates $= s\\,[\\mathbf{e}_1\\ \\mathbf{e}_2\\ '
             '\\mathbf{e}_3]^T(\\mathbf{X}-\\mathbf{X}_0)$.\n' % tuple(c['n']))
    s = c['summary']
    xy = c['XY'][NG:]
    L.append('Screen corners on the object (mm): TL (%.1f, %.1f), TR (%.1f, %.1f), '
             'BR (%.1f, %.1f), BL (%.1f, %.1f). Out of plane: all within %.2f mm.\n'
             % (*xy[0, :2], *xy[1, :2], *xy[2, :2], *xy[3, :2],
                np.abs(xy[:, 2]).max()))
    L.append('Width = |TR − TL| = %.1f mm, height = |BL − TL| = %.1f mm, aspect '
             '%.3f.\n' % (s['screen_width_top_mm'], s['screen_height_left_mm'],
                          s['screen_aspect']))
    L.append('### Step 5 — camera positions\n')
    L.append('Camera centre $\\mathbf{C}_i = -R_{1i}^T\\mathbf{t}_{1i}$, converted to '
             'the object frame and mm:\n')
    L.append('| view | image | C (mm) | distance to object (mm) | rotation vs view 1 |')
    L.append('|---|---|---|---|---|')
    for _, r in c['camdf'].iterrows():
        L.append('| %d | %s | (%.0f, %.0f, %.0f) | %.0f | %.1f° |'
                 % (r.view, r.image, r.Cx_mm, r.Cy_mm, r.Cz_mm,
                    r.dist_to_object_mm, r.rot_vs_cam1_deg))
    L.append('')
    with open(os.path.join(RESULTS, 'workings_sfm.md'), 'w') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--square-mm', type=float, default=30.0,
                    help='size of one chessboard square on the screen, mm '
                         '(the one known length; default = the value used in '
                         'the Module 2 calibration)')
    a = ap.parse_args()
    run(a.square_mm)


if __name__ == '__main__':
    main()

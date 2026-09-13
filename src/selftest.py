"""
================================================================================
selftest.py  --  Pipeline smoke test (run this BEFORE taking real photos)
================================================================================
Renders synthetic chessboard images, runs the full calibration + measurement
pipeline on them, and checks the recovered numbers against the known truth.

If this prints ALL CHECKS PASSED, your OpenCV install works with this code and
you can safely go shoot real photos.

Run:
    python selftest.py
================================================================================
"""
import sys
import numpy as np
import cv2

print(f'Python  : {sys.version.split()[0]}')
print(f'OpenCV  : {cv2.__version__}')
print(f'NumPy   : {np.__version__}\n')

COLS, ROWS = 9, 6          # inner corners
SQ_MM = 22.0               # square size
W, H = 1280, 960           # synthetic image size

# ---- ground-truth camera we will try to recover ----------------------------
K_true = np.array([[1100.0, 0, W / 2 - 8],
                   [0, 1100.0, H / 2 + 5],
                   [0, 0, 1]])
dist_true = np.zeros(5)

# board corners in board frame (Z=0 plane), plus full squares for rendering
objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
objp *= SQ_MM


def render_board(rvec, tvec):
    """Draw a chessboard as seen by K_true from the given pose."""
    img = np.full((H, W), 255, np.uint8)
    # render each full square (board has COLS+1 x ROWS+1 squares)
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                continue
            quad = np.array([
                [(c - 1) * SQ_MM, (r - 1) * SQ_MM, 0],
                [c * SQ_MM, (r - 1) * SQ_MM, 0],
                [c * SQ_MM, r * SQ_MM, 0],
                [(c - 1) * SQ_MM, r * SQ_MM, 0]], np.float32)
            proj, _ = cv2.projectPoints(quad, rvec, tvec, K_true, dist_true)
            cv2.fillConvexPoly(img, proj.reshape(-1, 2).astype(np.int32), 0,
                               lineType=cv2.LINE_AA)
    return img


# ---- TEST 1: corner detection ----------------------------------------------
print('[1/4] Rendering synthetic boards and detecting corners...')
poses = [
    (np.array([0.0, 0.0, 0.0]),   np.array([-90.0, -60.0, 600.0])),
    (np.array([0.30, 0.0, 0.0]),  np.array([-90.0, -60.0, 620.0])),
    (np.array([-0.30, 0.0, 0.0]), np.array([-90.0, -60.0, 640.0])),
    (np.array([0.0, 0.35, 0.0]),  np.array([-95.0, -60.0, 610.0])),
    (np.array([0.0, -0.35, 0.0]), np.array([-85.0, -60.0, 630.0])),
    (np.array([0.22, 0.22, 0.05]), np.array([-95.0, -55.0, 650.0])),
    (np.array([-0.22, 0.22, -0.05]), np.array([-85.0, -65.0, 590.0])),
    (np.array([0.15, -0.28, 0.10]), np.array([-92.0, -58.0, 670.0])),
    (np.array([-0.18, -0.20, 0.0]), np.array([-88.0, -62.0, 605.0])),
    (np.array([0.0, 0.0, 0.25]),  np.array([-90.0, -60.0, 660.0])),
    (np.array([0.28, 0.12, 0.0]), np.array([-93.0, -57.0, 615.0])),
    (np.array([-0.25, -0.12, 0.08]), np.array([-87.0, -63.0, 635.0])),
]

objpoints, imgpoints = [], []
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i, (rv, tv) in enumerate(poses):
    img = render_board(rv, tv)
    ok, corners = cv2.findChessboardCorners(img, (COLS, ROWS), None)
    if ok:
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), crit)
        objpoints.append(objp)
        imgpoints.append(corners)

print(f'      detected in {len(objpoints)}/{len(poses)} synthetic images')
assert len(objpoints) >= 8, 'FAIL: corner detection not working on this OpenCV build'
print('      OK — findChessboardCorners + cornerSubPix work\n')

# ---- TEST 2: calibration ----------------------------------------------------
print('[2/4] Running cv2.calibrateCamera...')
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (W, H), None, None)
print(f'      RMS reprojection error : {rms:.4f} px')
print(f'      fx recovered {K[0,0]:8.2f}   (true {K_true[0,0]:.2f})')
print(f'      fy recovered {K[1,1]:8.2f}   (true {K_true[1,1]:.2f})')
print(f'      cx recovered {K[0,2]:8.2f}   (true {K_true[0,2]:.2f})')
print(f'      cy recovered {K[1,2]:8.2f}   (true {K_true[1,2]:.2f})')

assert rms < 1.0, f'FAIL: reprojection error too high ({rms})'
assert abs(K[0, 0] - K_true[0, 0]) / K_true[0, 0] < 0.05, 'FAIL: fx off by >5%'
assert abs(K[1, 1] - K_true[1, 1]) / K_true[1, 1] < 0.05, 'FAIL: fy off by >5%'
print('      OK — calibration recovers the true intrinsics\n')

# ---- TEST 3: save / load ----------------------------------------------------
print('[3/4] Testing save + load of camera_params.npz...')
np.savez('/tmp/_selftest_params.npz', K=K, dist=dist, img_size=np.array([W, H]))
d = np.load('/tmp/_selftest_params.npz')
assert np.allclose(d['K'], K), 'FAIL: save/load mismatch'
print('      OK — np.savez round-trip works\n')

# ---- TEST 4: the measurement math ------------------------------------------
print('[4/4] Testing the perspective-projection measurement...')


def pixel_to_camera(u, v, Z, K):
    return np.array([(u - K[0, 2]) * Z / K[0, 0],
                     (v - K[1, 2]) * Z / K[1, 1]])


def measure(p1, p2, Z, K, dist=None):
    pts = np.array([p1, p2], np.float32).reshape(-1, 1, 2)
    if dist is not None:
        pts = cv2.undistortPoints(pts, K, dist, P=K)
    (u1, v1), (u2, v2) = pts.reshape(-1, 2)
    return float(np.linalg.norm(pixel_to_camera(u2, v2, Z, K) -
                                pixel_to_camera(u1, v1, Z, K)))


# put a known 300 mm bar at Z = 2500 mm, fronto-parallel, and project it
Z = 2500.0
bar = np.array([[-150.0, 0, Z], [150.0, 0, Z]], np.float32)   # 300 mm wide
proj, _ = cv2.projectPoints(bar, np.zeros(3), np.zeros(3), K_true, dist_true)
p1, p2 = proj.reshape(-1, 2)

est = measure(p1, p2, Z, K_true, dist_true)
err = abs(est - 300.0)
print(f'      true length 300.00 mm  ->  measured {est:.2f} mm   (error {err:.3f} mm)')
assert err < 1.0, f'FAIL: measurement math wrong (error {err} mm)'

# and at an angle, to confirm it is not just an x-axis special case
bar2 = np.array([[-100.0, -80.0, Z], [120.0, 90.0, Z]], np.float32)
true2 = float(np.linalg.norm(bar2[1][:2] - bar2[0][:2]))
proj2, _ = cv2.projectPoints(bar2, np.zeros(3), np.zeros(3), K_true, dist_true)
q1, q2 = proj2.reshape(-1, 2)
est2 = measure(q1, q2, Z, K_true, dist_true)
print(f'      true length {true2:.2f} mm  ->  measured {est2:.2f} mm   '
      f'(error {abs(est2-true2):.3f} mm)')
assert abs(est2 - true2) < 1.0, 'FAIL: diagonal measurement wrong'
print('      OK — perspective projection measurement is correct\n')

print('=' * 60)
print('ALL CHECKS PASSED — your OpenCV build works with this code.')
print('You can safely go take real photos now.')
print('=' * 60)

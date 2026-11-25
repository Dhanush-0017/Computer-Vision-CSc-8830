"""
stereo.py -- the maths for Module 7, Problem 1.

Measure an object from TWO photos taken from two spots 5 cm apart
(calibrated stereo). The steps are:

  1. Load the camera matrix K from the Module 2 chessboard calibration.
  2. Match features between the two photos and find how the phone moved
     between them (rotation R and direction t) from the essential matrix.
     t only has a direction, so I multiply it by the measured slide
     (the baseline B = 5 cm) to get real units.
  3. For each point I click in the left photo, find the same point in the
     right photo by searching along its epipolar line.
  4. Triangulate each pair into a 3D point (X, Y, Z) in millimetres.
     Z is the distance from the camera.
  5. The size of the object is just the 3D distance between those points.

Not run on its own: the web page (__init__.py) and measure_stereo.py both
import it. To run Problem 1 from the command line:
    python measure_stereo.py
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, 'calibration', 'camera_params.npz')


# ----------------------------------------------------------------------------
# Step 1 - the camera
# ----------------------------------------------------------------------------
def load_camera(image_width, image_height):
    """Return K and the distortion values for a photo of this size.

    The calibration was done on full size 3024 x 4032 photos. If the photo is
    smaller (for example a half size copy), fx, fy, cx, cy all shrink by the
    same amount. The distortion values don't change with size.
    """
    data = np.load(CALIB_PATH)
    K = data['K'].copy()
    dist = data['dist'].copy()
    calib_width = int(data['img_size'][0])

    scale = image_width / calib_width
    K[0, 0] = K[0, 0] * scale     # fx
    K[1, 1] = K[1, 1] * scale     # fy
    K[0, 2] = K[0, 2] * scale     # cx
    K[1, 2] = K[1, 2] * scale     # cy
    return K, dist


def undistort_points(points, K, dist):
    """Remove lens distortion from pixel points (keeps them in pixels)."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    pts = cv2.undistortPoints(pts, K, dist, P=K)
    return pts.reshape(-1, 2)


# ----------------------------------------------------------------------------
# Step 2 - how did the phone move between the two photos?
# ----------------------------------------------------------------------------
def find_camera_motion(left_gray, right_gray, K, dist, baseline_mm):
    """Find R and t (in mm) of the right camera relative to the left camera.

    Returns R, t, and the matched points (so the page can show them).
    """
    sift = cv2.SIFT_create(5000)
    kp_left, desc_left = sift.detectAndCompute(left_gray, None)
    kp_right, desc_right = sift.detectAndCompute(right_gray, None)

    # Lowe's ratio test: keep a match only if it is clearly better than the
    # second best one
    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(desc_left, desc_right, k=2)
    good = []
    for best, second in pairs:
        if best.distance < 0.75 * second.distance:
            good.append(best)

    pts_left = []
    pts_right = []
    for m in good:
        pts_left.append(kp_left[m.queryIdx].pt)
        pts_right.append(kp_right[m.trainIdx].pt)
    pts_left = undistort_points(pts_left, K, dist)
    pts_right = undistort_points(pts_right, K, dist)

    # Essential matrix with RANSAC throws away the wrong matches
    E, mask = cv2.findEssentialMat(pts_left, pts_right, K,
                                   method=cv2.RANSAC, prob=0.999,
                                   threshold=1.0)
    count, R, t, mask = cv2.recoverPose(E, pts_left, pts_right, K, mask=mask)

    # recoverPose gives t with length 1. The real length is the slide.
    t = t * baseline_mm

    keep = mask.ravel() > 0
    return R, t, pts_left[keep], pts_right[keep]


def rotation_in_degrees(R):
    """How much the phone turned between the photos, as 3 angles."""
    rvec, _ = cv2.Rodrigues(R)
    return np.degrees(rvec.ravel())


# ----------------------------------------------------------------------------
# Step 3 - find the clicked point in the right photo
# ----------------------------------------------------------------------------
def fundamental_from_pose(K, R, t):
    """F = K^-T [t]x R K^-1. Maps a left pixel to a line in the right photo."""
    tx = np.array([[0, -t[2, 0], t[1, 0]],
                   [t[2, 0], 0, -t[0, 0]],
                   [-t[1, 0], t[0, 0], 0]])
    E = tx @ R
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F


def find_match(left_gray, right_gray, point, F, patch=25,
               max_shift=600):
    """Look for the clicked left point in the right photo.

    The match has to lie on the epipolar line l = F x, so I only slide a small
    patch along that line and keep the spot that looks most alike
    (normalised cross correlation).
    """
    x, y = point
    half = patch // 2
    h, w = left_gray.shape

    if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
        return None, 0.0
    template = left_gray[int(y) - half:int(y) + half + 1,
                         int(x) - half:int(x) + half + 1]

    a, b, c = F @ np.array([x, y, 1.0])     # line: a*u + b*v + c = 0

    best_score = -1.0
    best_point = None
    for u in range(int(x) - max_shift, int(x) + max_shift + 1):
        if abs(b) < 1e-9:
            continue
        v = -(a * u + c) / b
        if u - half < 0 or v - half < 0 or u + half >= w or v + half >= h:
            continue
        window = right_gray[int(v) - half:int(v) + half + 1,
                            u - half:u + half + 1]
        score = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)[0, 0]
        if score > best_score:
            best_score = score
            best_point = (float(u), float(v))
    return best_point, float(best_score)


# ----------------------------------------------------------------------------
# Step 4 - triangulate
# ----------------------------------------------------------------------------
def triangulate(point_left, point_right, K, dist, R, t):
    """Turn one left/right pixel pair into a 3D point in mm (left camera)."""
    pl = undistort_points([point_left], K, dist)
    pr = undistort_points([point_right], K, dist)

    P_left = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = K @ np.hstack([R, t])

    X = cv2.triangulatePoints(P_left, P_right, pl.T, pr.T)
    X = X[:3, 0] / X[3, 0]
    return X


# ----------------------------------------------------------------------------
# Step 5 - sizes
# ----------------------------------------------------------------------------
def distance_3d(A, B):
    return float(np.linalg.norm(np.array(A) - np.array(B)))


def rectangle_size(P):
    """P = 4 corners in order: top-left, top-right, bottom-right, bottom-left.

    Width is the average of the top and bottom edges, height is the average
    of the left and right edges.
    """
    top = distance_3d(P[0], P[1])
    right = distance_3d(P[1], P[2])
    bottom = distance_3d(P[2], P[3])
    left = distance_3d(P[3], P[0])
    width = (top + bottom) / 2
    height = (left + right) / 2
    return width, height, [top, right, bottom, left]


def circle_diameter(P):
    """P = 4 points on the rim: left, right, top, bottom.

    Diameter = average of the left-right and top-bottom distances.
    """
    across = distance_3d(P[0], P[1])
    down = distance_3d(P[2], P[3])
    return (across + down) / 2, across, down


def polygon_edges(P):
    """P = corners in order around the shape. Returns every edge length,
    including the one that closes the shape (last corner -> first)."""
    edges = []
    for i in range(len(P)):
        A = P[i]
        B = P[(i + 1) % len(P)]
        edges.append(distance_3d(A, B))
    return edges


# ----------------------------------------------------------------------------
# Checks on the matrices (for the report)
# ----------------------------------------------------------------------------
def matrix_checks(K, R, t):
    """Build E and F from R and t, and check the properties they must have.

    E = [t]x R      -> rank 2, two equal singular values, the third is 0
    F = K^-T E K^-1 -> rank 2 (det F = 0)
    R               -> det(R) = +1 and R R^T = I
    """
    t_unit = t.ravel() / np.linalg.norm(t)
    tx = np.array([[0, -t_unit[2], t_unit[1]],
                   [t_unit[2], 0, -t_unit[0]],
                   [-t_unit[1], t_unit[0], 0]])
    E = tx @ R

    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    F = F / F[2, 2]                       # usual way to write F: last entry 1

    E_singular = np.linalg.svd(E, compute_uv=False)
    F_singular = np.linalg.svd(F, compute_uv=False)
    F_rank = int(np.sum(F_singular > F_singular[0] * 1e-6))

    angle = np.degrees(np.arccos((np.trace(R) - 1) / 2))

    return {
        'E': E,
        'F': F,
        'E_singular_values': E_singular,
        'F_rank': F_rank,
        'det_R': float(np.linalg.det(R)),
        'R_orthogonal_error': float(np.linalg.norm(R @ R.T - np.eye(3))),
        'rotation_angle_deg': float(angle),
    }

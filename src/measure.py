"""
measure.py -- Step 2, turning pixels into millimetres.

Same idea as the Step 2 tab, just as a script. Given the calibration from
Step 1 and a distance Z (measured with a tape measure from the camera to the
object), it back-projects two clicked pixel points to real 3D coordinates
and returns the distance between them.

    u = fx*X/Z + cx   ->   X = (u - cx)*Z/fx
    v = fy*Y/Z + cy   ->   Y = (v - cy)*Z/fy
    D = sqrt((X2-X1)^2 + (Y2-Y1)^2)

Only works cleanly if both points are roughly the same distance from the
camera -- i.e. the face you're measuring is facing the camera, not tilted.
Keep Z in the same units you used for --square in calibrate.py (mm here).

Run with known pixel coords:
    python measure.py --calib ../calibration/camera_params.npz \
                      --Z 2500 --p1 400 300 --p2 950 300

Run and click the two points on the image instead:
    python measure.py --calib ../calibration/camera_params.npz \
                      --Z 2500 --image ../data/obj.jpg
"""
import argparse
import numpy as np
import cv2


def load_calib(path):
    d = np.load(path)
    return d['K'], d['dist']


def pixel_to_camera(u, v, Z, K):
    """Back-project a pixel to a 3D point at depth Z (camera frame)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * Z / fx, (v - cy) * Z / fy])


def measure(p1, p2, Z, K, dist=None):
    """Real-world distance between two pixel points at depth Z."""
    pts = np.array([p1, p2], np.float32).reshape(-1, 1, 2)
    if dist is not None:
        # remove lens distortion, keep result in pixel coordinates (P=K)
        pts = cv2.undistortPoints(pts, K, dist, P=K)
    (u1, v1), (u2, v2) = pts.reshape(-1, 2)
    P1 = pixel_to_camera(u1, v1, Z, K)
    P2 = pixel_to_camera(u2, v2, Z, K)
    return float(np.linalg.norm(P2 - P1))


def click_two_points(image_path):
    """Open an image and let the user click two points."""
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit('Cannot read image')
    pts, disp = [], img.copy()
    win = 'click 2 points (ESC when done)'

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            if len(pts) == 2:
                cv2.line(disp, pts[0], pts[1], (0, 255, 0), 2)
            cv2.imshow(win, disp)

    cv2.imshow(win, disp)
    cv2.setMouseCallback(win, cb)
    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    if len(pts) < 2:
        raise SystemExit('Need two points')
    return pts[0], pts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True)
    ap.add_argument('--Z', type=float, required=True, help='camera-object distance (mm)')
    ap.add_argument('--p1', type=float, nargs=2)
    ap.add_argument('--p2', type=float, nargs=2)
    ap.add_argument('--image')
    args = ap.parse_args()

    K, dist = load_calib(args.calib)
    if args.image:
        p1, p2 = click_two_points(args.image)
    else:
        if not (args.p1 and args.p2):
            raise SystemExit('Provide --p1 and --p2, or --image')
        p1, p2 = args.p1, args.p2

    D = measure(p1, p2, args.Z, K, dist)
    print(f'p1={p1}, p2={p2}, Z={args.Z}')
    print(f'Real-world distance: {D:.2f} (same units as Z)')


if __name__ == '__main__':
    main()

"""
================================================================================
common.py  --  Shared helpers used across all assignment modules
================================================================================
Calibration load/save and the perspective-projection measurement math live here
so every module page can import them without duplication.
================================================================================
"""
import os
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(ROOT, 'calibration', 'camera_params.npz')


def load_calib():
    """Return (K, dist) from the saved calibration, or (None, None)."""
    if os.path.exists(CALIB_PATH):
        d = np.load(CALIB_PATH)
        return d['K'], d['dist']
    return None, None


def save_calib(K, dist, img_size):
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    np.savez(CALIB_PATH, K=K, dist=dist, img_size=np.array(img_size))


def pixel_to_camera(u, v, Z, K):
    """Back-project a pixel to a 3D point at depth Z.

    From the pinhole equations  u = fx*X/Z + cx,  v = fy*Y/Z + cy:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * Z / fx, (v - cy) * Z / fy])


def measure(p1, p2, Z, K, dist=None):
    """Real-world distance between two pixel points lying at depth Z."""
    pts = np.array([p1, p2], np.float32).reshape(-1, 1, 2)
    if dist is not None:
        pts = cv2.undistortPoints(pts, K, dist, P=K)
    (u1, v1), (u2, v2) = pts.reshape(-1, 2)
    return float(np.linalg.norm(pixel_to_camera(u2, v2, Z, K) -
                                pixel_to_camera(u1, v1, Z, K)))


def decode_upload(uploaded):
    """Streamlit UploadedFile -> BGR numpy image."""
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

"""
================================================================================
calibrate.py  --  Camera Calibration (Step 1)
================================================================================
Purpose : Estimate the intrinsic matrix K and lens-distortion coefficients of a
          smartphone camera using OpenCV's chessboard calibration.

How to run:
    python calibrate.py --images ../data/calibration_images \
                        --rows 6 --cols 9 --square 25.0 \
                        --out ../calibration/camera_params.npz

Arguments:
    --images : folder of chessboard photos taken with YOUR phone
    --rows   : number of INNER corners along the short side of the board
    --cols   : number of INNER corners along the long side
               (a printed 10x7 board of squares has 9x6 inner corners)
    --square : printed square size in millimetres (keep units consistent
               with Z later; this does NOT change the intrinsics)
    --out    : where the calibration is saved (.npz)

Tips:
    - Print a chessboard (search "OpenCV chessboard 9x6"), tape it flat.
    - Take 15-25 photos from varied angles/distances, board filling
      different parts of the frame each time.
    - Aim for a reprojection error < ~0.5 px.
================================================================================
"""
import argparse
import glob
import os
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--rows', type=int, default=6, help='inner corners (short side)')
    ap.add_argument('--cols', type=int, default=9, help='inner corners (long side)')
    ap.add_argument('--square', type=float, default=25.0, help='square size in mm')
    ap.add_argument('--out', default='../calibration/camera_params.npz')
    args = ap.parse_args()

    pattern = (args.cols, args.rows)

    # 3D coordinates of the board corners in the board's own frame:
    # (0,0,0), (1,0,0), (2,0,0)... then scaled by the real square size.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square

    objpoints, imgpoints = [], []
    files = sorted(glob.glob(os.path.join(args.images, '*')))
    if not files:
        raise SystemExit(f'No images found in {args.images}')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = None
    used = 0

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (width, height)
        ok, corners = cv2.findChessboardCorners(gray, pattern, None)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            used += 1
            print(f'[OK]   {os.path.basename(f)}')
        else:
            print(f'[SKIP] {os.path.basename(f)} (corners not found)')

    if used < 5:
        raise SystemExit(f'Only {used} usable images - need >= 10 for a good result.')

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    # Mean reprojection error (a second sanity check next to the RMS above)
    total_err = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        # OpenCV 5 returns projectPoints with a different array shape than
        # OpenCV 4, so reshape both to (N,2) before comparing.
        a = np.asarray(imgpoints[i], np.float64).reshape(-1, 2)
        b = np.asarray(proj, np.float64).reshape(-1, 2)
        total_err += float(np.linalg.norm(a - b) / len(b))
    mean_err = total_err / len(objpoints)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, K=K, dist=dist, img_size=np.array(img_size))

    print('\n=== Calibration complete ===')
    print(f'Images used       : {used}/{len(files)}')
    print(f'RMS reproj error  : {rms:.4f} px')
    print(f'Mean reproj error : {mean_err:.4f} px')
    print(f'fx, fy            : {K[0,0]:.2f}, {K[1,1]:.2f}')
    print(f'cx, cy            : {K[0,2]:.2f}, {K[1,2]:.2f}')
    print(f'distortion        : {dist.ravel()}')
    print(f'Saved to          : {args.out}')


if __name__ == '__main__':
    main()

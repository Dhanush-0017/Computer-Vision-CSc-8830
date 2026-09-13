"""
================================================================================
modules/module2.py  --  Module 2: Object Dimension Measurement
================================================================================
Camera calibration + perspective projection to measure real-world 2D object
dimensions, validated over 20 measurements, plus the two-view theory.

Rendered inside app.py — not run directly.
================================================================================
"""
import os
import numpy as np
import pandas as pd
import cv2
import streamlit as st

# --- metadata read by app.py to build the navigation ------------------------
NUMBER = 2
TITLE = 'Object Dimension Measurement'
SUBTITLE = ('Camera calibration and perspective projection to recover real-world '
            '2D dimensions from a single image, validated over 20 experiments.')
STATUS = 'complete'

from common import load_calib, save_calib, measure, decode_upload, CALIB_PATH

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as img_coords
    HAS_CLICK = True
except Exception:
    HAS_CLICK = False


def render():
    st.title("Module 2 — Object Dimension Measurement")
    st.caption("Camera calibration and perspective projection to recover "
               "real-world 2D dimensions from a single image.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ['Step 1 — Calibration', 'Step 2 — Measure', 'Step 3 — Validation', 'Theory'])


    # ----------------------------------------------------------------------------
    # STEP 1 : CALIBRATION
    # ----------------------------------------------------------------------------
    with tab1:
        st.header('Step 1 — Camera Calibration')
        st.markdown(
            'Upload chessboard photos taken with your smartphone. OpenCV locates the '
            'inner corners in each image and solves for the intrinsic matrix **K** '
            '(focal lengths `fx, fy` and principal point `cx, cy`) plus the lens '
            'distortion coefficients.')

        c1, c2, c3 = st.columns(3)
        cols = c1.number_input('Inner corners (long side)', 2, 20, 9)
        rows = c2.number_input('Inner corners (short side)', 2, 20, 6)
        square = c3.number_input('Square size (mm)', 1.0, 100.0, 22.0)

        files = st.file_uploader('Chessboard images', type=['jpg', 'jpeg', 'png'],
                                 accept_multiple_files=True)

        if st.button('Run calibration', type='primary') and files:
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            objp *= square

            objpoints, imgpoints, size = [], [], None
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            prog = st.progress(0.0)
            log = []

            for i, f in enumerate(files):
                arr = np.frombuffer(f.getvalue(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ok, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
                if ok:
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    log.append(f'OK   — {f.name}')
                else:
                    log.append(f'skip — {f.name} (corners not found)')
                prog.progress((i + 1) / len(files))

            if len(objpoints) < 5:
                st.error(f'Only {len(objpoints)} usable images. Need at least 10 '
                         'for a reliable calibration. Check the corner counts above.')
            else:
                rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, size, None, None)

                err = 0.0
                for i in range(len(objpoints)):
                    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
                    # OpenCV 5 returns projectPoints with a different array shape
                    # than OpenCV 4, so reshape both to (N,2) before comparing.
                    a = np.asarray(imgpoints[i], np.float64).reshape(-1, 2)
                    b = np.asarray(proj, np.float64).reshape(-1, 2)
                    err += float(np.linalg.norm(a - b) / len(b))
                err /= len(objpoints)

                os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
                np.savez(CALIB_PATH, K=K, dist=dist, img_size=np.array(size))

                st.success(f'Calibration complete — {len(objpoints)}/{len(files)} images used.')
                m1, m2, m3 = st.columns(3)
                m1.metric('RMS reprojection error', f'{rms:.4f} px')
                m2.metric('Mean reprojection error', f'{err:.4f} px')
                m3.metric('Resolution', f'{size[0]}x{size[1]}')

                st.subheader('Intrinsic matrix K')
                st.dataframe(pd.DataFrame(K, columns=[' ', '  ', '   '],
                                          index=[' ', '  ', '   ']).style.format('{:.3f}'))
                st.write(f'**fx** = {K[0,0]:.2f}  **fy** = {K[1,1]:.2f}  '
                         f'**cx** = {K[0,2]:.2f}  **cy** = {K[1,2]:.2f}')
                st.subheader('Distortion coefficients')
                st.code(np.array2string(dist.ravel(), precision=6))
                with st.expander('Per-image log'):
                    st.text('\n'.join(log))

        K, dist = load_calib()
        if K is not None:
            st.info(f'Saved calibration in use — fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, '
                    f'cx={K[0,2]:.1f}, cy={K[1,2]:.1f}')


    # ----------------------------------------------------------------------------
    # STEP 2 : MEASUREMENT
    # ----------------------------------------------------------------------------
    with tab2:
        st.header('Step 2 — Measure a Real-World Dimension')
        K, dist = load_calib()

        if K is None:
            st.warning('Run Step 1 first — no calibration found.')
        else:
            st.latex(r'X=\frac{(u-c_x)\,Z}{f_x}\qquad Y=\frac{(v-c_y)\,Z}{f_y}'
                     r'\qquad D=\sqrt{(X_2-X_1)^2+(Y_2-Y_1)^2}')
            st.caption('Assumes the measured face is roughly perpendicular to the '
                       'optical axis (fronto-parallel) at depth Z.')

            Z = st.number_input('Camera-to-object distance Z (mm)',
                                100.0, 20000.0, 2500.0, step=50.0)
            up = st.file_uploader('Object photo', type=['jpg', 'jpeg', 'png'],
                                  key='measure_img')

            if up:
                arr = np.frombuffer(up.getvalue(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                disp_w = 900
                scale = disp_w / w
                small = cv2.resize(img, (disp_w, int(h * scale)))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                if 'pts' not in st.session_state:
                    st.session_state.pts = []

                if HAS_CLICK:
                    st.write('**Click the two endpoints of the dimension.**')
                    from PIL import Image
                    val = img_coords(Image.fromarray(rgb), key='clicker')
                    if val:
                        p = (val['x'] / scale, val['y'] / scale)   # back to full-res px
                        if len(st.session_state.pts) >= 2:
                            st.session_state.pts = []
                        st.session_state.pts.append(p)
                    if st.button('Reset points'):
                        st.session_state.pts = []
                    st.write(f'Points selected: {len(st.session_state.pts)}/2')
                else:
                    st.image(rgb, caption='Enter pixel coordinates below '
                                          '(install streamlit-image-coordinates to click)')
                    c1, c2, c3, c4 = st.columns(4)
                    u1 = c1.number_input('u1', 0.0, float(w), 0.0)
                    v1 = c2.number_input('v1', 0.0, float(h), 0.0)
                    u2 = c3.number_input('u2', 0.0, float(w), 0.0)
                    v2 = c4.number_input('v2', 0.0, float(h), 0.0)
                    st.session_state.pts = [(u1, v1), (u2, v2)]

                if len(st.session_state.pts) == 2:
                    p1, p2 = st.session_state.pts
                    D = measure(p1, p2, Z, K, dist)

                    ann = small.copy()
                    a = (int(p1[0] * scale), int(p1[1] * scale))
                    b = (int(p2[0] * scale), int(p2[1] * scale))
                    cv2.line(ann, a, b, (0, 255, 0), 3)
                    cv2.circle(ann, a, 7, (0, 0, 255), -1)
                    cv2.circle(ann, b, 7, (0, 0, 255), -1)
                    st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))

                    st.metric('Measured real-world length',
                              f'{D:.1f} mm   ({D/10:.2f} cm)')
                    px = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    st.caption(f'Pixel length: {px:.1f} px   |   '
                               f'p1=({p1[0]:.0f}, {p1[1]:.0f})   '
                               f'p2=({p2[0]:.0f}, {p2[1]:.0f})   |   Z={Z:.0f} mm')
                    st.caption('Copy these into your validation CSV: '
                               f'`{Z:.0f},{p1[0]:.0f},{p1[1]:.0f},{p2[0]:.0f},{p2[1]:.0f}`')


    # ----------------------------------------------------------------------------
    # STEP 3 : VALIDATION
    # ----------------------------------------------------------------------------
    with tab3:
        st.header('Step 3 — Validation over 20 Measurements')
        K, dist = load_calib()

        if K is None:
            st.warning('Run Step 1 first — no calibration found.')
        else:
            st.markdown('Upload your `measurements.csv`, or edit the table directly. '
                        'Columns: `id, Z_mm, u1, v1, u2, v2, gt_mm` where `gt_mm` is '
                        'the tape-measured ground truth.')

            csv = st.file_uploader('measurements.csv', type=['csv'], key='valcsv')
            default = pd.DataFrame({
                'id': range(1, 21), 'Z_mm': [2500.0] * 20,
                'u1': [0.0] * 20, 'v1': [0.0] * 20,
                'u2': [0.0] * 20, 'v2': [0.0] * 20, 'gt_mm': [0.0] * 20})
            df = pd.read_csv(csv) if csv else default

            edited = st.data_editor(df, num_rows='dynamic', use_container_width=True)

            if st.button('Compute error statistics', type='primary'):
                d = edited[edited.gt_mm > 0].copy()
                if len(d) == 0:
                    st.error('No rows with a ground-truth value yet.')
                else:
                    est = [measure((r.u1, r.v1), (r.u2, r.v2), r.Z_mm, K, dist)
                           for _, r in d.iterrows()]
                    d['est_mm'] = np.round(est, 2)
                    d['error_mm'] = np.round(d.est_mm - d.gt_mm, 2)
                    d['error_pct'] = np.round(100 * (d.est_mm - d.gt_mm) / d.gt_mm, 2)

                    e = d.error_mm.values
                    p = d.error_pct.values
                    st.dataframe(d, use_container_width=True)

                    st.subheader('Error statistics')
                    a, b, c, dd = st.columns(4)
                    a.metric('MAE', f'{np.mean(np.abs(e)):.2f} mm')
                    b.metric('RMSE', f'{np.sqrt(np.mean(e**2)):.2f} mm')
                    c.metric('MAPE', f'{np.mean(np.abs(p)):.2f} %')
                    dd.metric('Max abs error', f'{np.max(np.abs(e)):.2f} mm')
                    e1, e2, e3 = st.columns(3)
                    e1.metric('Mean signed error', f'{np.mean(e):.2f} mm')
                    e2.metric('Std of error', f'{np.std(e, ddof=1):.2f} mm' if len(e) > 1 else 'n/a')
                    e3.metric('N', f'{len(d)}')

                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    f1, ax = plt.subplots(1, 2, figsize=(11, 4.5))
                    ax[0].scatter(d.gt_mm, d.est_mm)
                    lim = [min(d.gt_mm.min(), d.est_mm.min()),
                           max(d.gt_mm.max(), d.est_mm.max())]
                    ax[0].plot(lim, lim, 'r--', label='ideal (y=x)')
                    ax[0].set_xlabel('Ground truth (mm)'); ax[0].set_ylabel('Estimated (mm)')
                    ax[0].set_title('Measured vs Ground Truth'); ax[0].legend(); ax[0].grid(True)
                    ax[1].scatter(d.Z_mm, d.error_mm)
                    ax[1].axhline(0, color='r', ls='--')
                    ax[1].set_xlabel('Distance Z (mm)'); ax[1].set_ylabel('Error (mm)')
                    ax[1].set_title('Error vs Distance'); ax[1].grid(True)
                    plt.tight_layout()
                    st.pyplot(f1)

                    st.download_button('Download validation_results.csv',
                                       d.to_csv(index=False).encode(),
                                       'validation_results.csv', 'text/csv')


    # ----------------------------------------------------------------------------
    # THEORY
    # ----------------------------------------------------------------------------
    with tab4:
        st.header('Theory — Two-View Geometry')
        st.markdown('**Problem.** Camera 1 is static; camera 2 sits at a distance and '
                    'oblique orientation from it. Point **P=(X,Y,Z)** is in both FoVs. '
                    'Relate its image coordinates in the two cameras.')

        st.subheader('Assumptions')
        st.markdown("""
    1. Pinhole model, distortion removed by calibration.
    2. Both cameras calibrated — `K1`, `K2` known (equal if the same phone is used).
    3. Rigid relative pose `(R, t)`: the oblique orientation is `R`, the offset is `t`.
    4. Camera 1's frame is taken as the world frame (free choice, no loss of generality).
    5. P has positive depth in both views.
    """)

        st.subheader('1. Projection in each camera')
        st.latex(r'\lambda_1\tilde{p}_1 = K_1 X_1 \qquad '
                 r'\lambda_2\tilde{p}_2 = K_2 (R X_1 + t)')
        st.latex(r'\hat{x}_1 = K_1^{-1}\tilde{p}_1 \propto X_1 \qquad '
                 r'\hat{x}_2 = K_2^{-1}\tilde{p}_2 \propto X_2')

        st.subheader('2. Coplanarity → epipolar constraint')
        st.markdown('`X2`, `t` and `R·X1` are coplanar, so their scalar triple product vanishes:')
        st.latex(r'X_2 \cdot (t \times R X_1) = 0 \;\Longrightarrow\; X_2^{T} [t]_\times R X_1 = 0')
        st.latex(r'E = [t]_\times R \quad\text{(essential matrix)}')
        st.markdown('Substituting the normalized coordinates gives the relationship in pixels:')
        st.latex(r'\boxed{\;\tilde{p}_2^{T} F \tilde{p}_1 = 0\;,\qquad '
                 r'F = K_2^{-T}[t]_\times R\,K_1^{-1}\;}')
        st.latex(r'[t]_\times=\begin{bmatrix}0&-t_z&t_y\\ t_z&0&-t_x\\ -t_y&t_x&0\end{bmatrix}')

        st.markdown('The image of P in camera 2 must lie on the **epipolar line** '
                    r'$l_2 = F\tilde{p}_1$, and in camera 1 on $l_1 = F^{T}\tilde{p}_2$. '
                    'It is a *line*, not a point, because one image point fixes only the '
                    '**direction** of the ray to P, not its depth — every point along that '
                    'ray projects to the same pixel in camera 1 but to different pixels in camera 2.')

        st.subheader('3. If the depth is known')
        st.markdown('With `Z` (hence `λ₁`) known, P is fully determined and maps to a single point:')
        st.latex(r'X_1 = \lambda_1 K_1^{-1}\tilde{p}_1, \qquad '
                 r'\lambda_2 \tilde{p}_2 = K_2(R X_1 + t)')

        st.subheader('4. Parameters and how they are obtained')
        st.table(pd.DataFrame({
            'Symbol': ['K1, K2', 'R', 't', 'E = [t]×R', 'F = K2⁻ᵀEK1⁻¹', 'P = (X,Y,Z)'],
            'Meaning': ['intrinsic matrices', 'relative rotation', 'relative translation',
                        'essential matrix', 'fundamental matrix', 'the 3D scene point'],
            'How determined': [
                'Step-1 chessboard calibration',
                'measured mounting angle, or from cv2.recoverPose',
                'ruler-measured baseline, or recoverPose (up to scale)',
                'from R,t or cv2.findEssentialMat',
                'from E, or the 8-point algorithm',
                'cv2.triangulatePoints from both views']}))

        st.markdown("""
    **If `R` and `t` are unknown:** match ≥8 correspondences (ORB/SIFT) →
    `cv2.findFundamentalMat` → `E = K2ᵀ F K1` → `cv2.recoverPose(E, ...)` gives `R`
    and a **unit** `t`. Absolute scale is unrecoverable from images alone
    (monocular scale ambiguity); one known physical length fixes it.
    """)

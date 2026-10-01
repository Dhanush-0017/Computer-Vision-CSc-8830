"""
modules/module5 -- Modules 5 & 6 (Assignment 6): Optical Flow & Structure from Motion.

The page in the web app. app.py imports this; it doesn't run on its own.

Where each part of the assignment lives:
  A. optical flow as a video, what can be inferred, evidence   -> tab 1
     tracking equations validated on two consecutive frames     -> tab 2
     derivations (tracking equations, bilinear interpolation)   -> tab 4
  B. structure from motion, 4 views of a flat object, boundary,
     camera positions and parameters, worked maths              -> tab 3

The flow videos, evidence plots and the SfM figures are made by the scripts
(make_flow_videos.py, validate_tracking.py, sfm.py) and read from results/.
The page also re-computes the smaller things live, so they can be changed.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from . import flow as F

# --- metadata read by app.py to build the navigation ------------------------
NUMBER = 5
TITLE = 'Optical Flow & Structure from Motion'
SUBTITLE = ('Modules 5 & 6: dense optical flow on two videos, Lucas-Kanade '
            'tracking derived and checked against real pixel positions, and '
            'structure from motion of a flat object from four views.')
STATUS = 'complete'

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RES = os.path.join(HERE, 'results')


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def _clips():
    with open(os.path.join(DATA, 'clips.json')) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _json(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _csv(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def _text(path):
    with open(path) as f:
        return f.read()


@st.cache_resource(show_spinner='Loading video...')
def _frames(name):
    return F.read_frames(os.path.join(DATA, 'videos', _clips()[name]['file']))


def _compass(deg):
    """Image-coordinate angle (0 = right, 90 = down) -> words."""
    names = ['right', 'down-right', 'down', 'down-left', 'left', 'up-left',
             'up', 'up-right']
    return names[int(((deg % 360) + 22.5) // 45) % 8]


def _pick_clip(key):
    clips = _clips()
    return st.radio('Video', list(clips), key=key, horizontal=True,
                    format_func=lambda k: clips[k]['title'])


# ----------------------------------------------------------------------------
# Tab 1: optical flow
# ----------------------------------------------------------------------------
def _pan_segments(df, thr=1.0):
    """Stretches where the camera turns faster than thr px/frame:
    [(start s, end s, mean horizontal camera flow, peak speed)]."""
    pan = (df.camera_speed > thr).values
    out, start = [], None
    for k, p in enumerate(list(pan) + [False]):
        if p and start is None:
            start = k
        elif not p and start is not None:
            seg = df.iloc[start:k]
            if seg.t.iloc[-1] - seg.t.iloc[0] >= 1.0:
                out.append((seg.t.iloc[0], seg.t.iloc[-1],
                            seg.camera_dx.mean(), seg.camera_speed.max()))
            start = None
    return out


def _tab_flow():
    st.subheader('A. Optical flow of two 30-second videos')
    name = _pick_clip('m5_flow_clip')
    clip = _clips()[name]
    st.caption('%s  \nSource: %s' % (clip['about'], clip['source']))
    vid = os.path.join(RES, 'flow_%s.mp4' % name)
    if os.path.exists(vid):
        st.markdown(
            '**The flow as a video.** Right: the optical flow, colour = '
            'direction (legend top right), brightness = speed. Left: the clip, '
            'with moving objects outlined in red and their own motion as '
            'arrows. I held the phone in my hand, so the camera moves too; '
            'its motion is measured each frame (a homography fitted with '
            'RANSAC to corners across the frame) and subtracted before '
            'deciding what is moving.')
        st.video(vid)
    else:
        st.warning('Run `python make_flow_videos.py` to make the flow video.')

    with st.expander('Look at any single frame pair (computed live)'):
        frames, fps = _frames(name)
        i = st.slider('frame', 0, len(frames) - 2, len(frames) // 2,
                      key='m5_frame_' + name)
        fl = F.dense_flow(frames[i], frames[i + 1])
        Hc, _ = F.camera_motion(frames[i], frames[i + 1])
        cam = F.camera_flow(Hc, fl.shape)
        s, moving = F.frame_stats(fl, clip['moving_thr'], cam,
                                  F.textured(frames[i]))
        obj = (fl - cam) * moving[..., None]
        c1, c2 = st.columns(2)
        c1.image(_rgb(F.draw_arrows(frames[i], obj, 20, 1.0, 2, (0, 255, 255))),
                 caption='frame %d, t = %.2f s: object motion (camera motion '
                 'removed)' % (i, i / fps), width='stretch')
        c2.image(_rgb(F.flow_to_color(fl, clip['max_mag'])),
                 caption='optical flow %d → %d' % (i, i + 1), width='stretch')
        a, b, c, d = st.columns(4)
        a.metric('camera motion', '%.2f px/frame' % s['camera_speed'])
        b.metric('moving pixels', '%.1f %%' % (100 * s['moving_fraction']))
        c.metric('speed of moving pixels', '%.1f px/frame' % s['moving_mean_speed'])
        d.metric('moving regions', s['moving_blobs'])

    st.markdown('#### What can be inferred from optical flow — with evidence')
    ev = os.path.join(RES, 'evidence_%s.png' % name)
    if os.path.exists(ev):
        st.image(ev, width='stretch')
    sm = _json('summary_%s.json' % name)
    df = _csv('stats_%s.csv' % name)
    if sm is None or df is None:
        return
    db = sm['depth_bins']
    lines = [
        '1. **What is moving — segmentation with no model of a car.** After '
        'removing the camera motion, pixels faster than %.1f px/frame outline '
        'the moving vehicles (plot a, red). Something is moving in %.0f %% of '
        'the frames, up to %d separate moving regions at once.'
        % (clip['moving_thr'], 100 * sm['frames_with_motion'], sm['max_blobs']),
    ]
    segs = _pan_segments(df)
    if not segs:
        lines.append(
            '2. **How the camera moved.** The background flow says the hand-held '
            'phone moved only %.2f px/frame (median; %.2f at the 95th '
            'percentile) — hand shake, not a pan. After subtracting it the '
            'background is left at %.2f px/frame (plot b, orange vs green).'
            % (sm['camera_speed_median'], sm['camera_speed_p95'],
               sm['residual_background_median']))
    else:
        desc = []
        for t0, t1, dx, peak in segs:
            # scene content moving right in the image = camera turning left
            desc.append('%.0f–%.0f s turning **%s** (picture slides %s, up to '
                        '%.1f px/frame)' % (t0, t1, 'left' if dx > 0 else 'right',
                                            'right' if dx > 0 else 'left', peak))
        still = df[df.camera_speed <= 1.0]
        lines.append(
            '2. **How the camera moved.** The flow of the background gives the '
            'camera\'s own motion frame by frame: held still at first (%.2f '
            'px/frame, hand shake), then %s (plot b, orange). Subtracting it '
            'leaves the background at %.2f px/frame, so the cars can still be '
            'picked out while the camera turns.'
            % (still.camera_speed.median(), '; '.join(desc),
               sm['residual_background_median']))
    lr = sm['left_vs_right'][0]
    lines += [
        '3. **Direction of travel.** Plot c is the direction histogram of '
        'all moving pixels. %s'
        % ('Traffic goes both ways: %.0f %% of the moving pixels move left and '
           '%.0f %% right — the two directions of the road.' % (100 * lr, 100 * (1 - lr))
           if 0.25 < lr < 0.75 else
           'Nearly all of it is **%s** (%.0f %%).'
           % ('left' if lr > 0.5 else 'right', 100 * max(lr, 1 - lr))),
        '4. **Distance.** The same road speed gives a larger image speed '
        'nearer the camera ($u \\propto 1/Z$). Plot d puts each moving region '
        'at the y where it meets the road (lower = nearer): vehicles meeting '
        'the road around y ≈ %d px move %.1f px/frame (median), those around '
        'y ≈ %d px, the nearest lane, %.1f px/frame.'
        % (db[0][0], db[0][1], db[-1][0], db[-1][1]),
        '5. **How fast.** Moving pixels travel on average %.1f px/frame '
        '(%.0f px/s at %.0f fps), up to %.0f px/frame for the nearest cars.'
        % (sm['moving_speed_px'], sm['moving_speed_px'] * sm['fps'], sm['fps'],
           sm['moving_speed_max']),
        '6. **Where flow cannot be measured.** On flat patches (sky, plain '
        'road) there is no gradient, so flow reads ~0 whatever moved — the '
        'aperture problem. While panning that would look like motion against '
        'the camera, so only pixels near texture are allowed to count as '
        'moving (`flow.textured`).',
    ]
    st.markdown('\n'.join(lines))


# ----------------------------------------------------------------------------
# Tab 2: tracking validation
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner='Tracking with my Lucas-Kanade...')
def _track_live(name, i):
    frames, _ = _frames(name)
    f1, f2 = frames[i], frames[i + 1]
    fl = F.dense_flow(f1, f2)
    Hc, _ = F.camera_motion(f1, f2)
    obj = fl - F.camera_flow(Hc, fl.shape)
    g = F.gray(f1)
    mag = np.hypot(obj[..., 0], obj[..., 1])
    m = cv2.erode(((mag > _clips()[name]['moving_thr']) * 255).astype(np.uint8),
                  np.ones((7, 7), np.uint8))
    m[:40], m[-40:], m[:, :40], m[:, -40:] = 0, 0, 0, 0
    p = cv2.goodFeaturesToTrack(g, 20, 0.01, 12, mask=m, blockSize=7)
    if p is None:
        return None
    P = np.round(p.reshape(-1, 2)).astype(np.float64)
    pred, _ = F.lk_track(f1, f2, P)
    rows = []
    for (x, y), (px, py) in zip(P, pred):
        act, sc = F.ncc_locate(f1, f2, (x, y))
        if act is None:
            continue
        rows.append({'x': int(x), 'y': int(y), 'predicted x': px,
                     'predicted y': py, 'actual x': act[0], 'actual y': act[1],
                     'moved (px)': np.hypot(act[0] - x, act[1] - y),
                     'error (px)': np.hypot(px - act[0], py - act[1]),
                     'NCC peak': sc})
    return pd.DataFrame(rows)


def _tab_tracking():
    st.subheader('A. Tracking between two consecutive frames — theory vs actual')
    st.markdown(
        '**Predicted:** my Lucas–Kanade (`flow.lk_track`) solves '
        '$G\\,\\mathbf d = \\mathbf b$ from the derivation (tab 4), iterating '
        'with bilinear interpolation, on a 4-level pyramid (cars move up to '
        '~25 px per frame).  \n'
        '**Actual:** where the pixel really went, measured *without* any flow '
        'equation — normalised cross-correlation of the 21×21 patch around '
        'the point, peak refined to sub-pixel.')
    name = _pick_clip('m5_trk_clip')
    s = _json('tracking_summary_%s.json' % name)
    df = _csv('tracking_%s.csv' % name)
    if s is None or df is None:
        st.warning('Run `python validate_tracking.py` first.')
        return
    st.markdown('**Frames %d → %d** (a frame with a typical amount of motion).'
                % (s['frame'], s['frame'] + 1))
    st.image(os.path.join(RES, 'tracking_%s.png' % name), width='stretch')
    a, b, c, d = st.columns(4)
    a.metric('reliable points', '%d of %d (median shift %.0f px)'
             % (s['reliable'], s['points'], s['median_moved_px']))
    b.metric('median |predicted − actual|', '%.2f px' % s['median_err_vs_actual'])
    c.metric('90th percentile', '%.2f px' % s['p90_err_vs_actual'])
    d.metric('median vs OpenCV LK', '%.3f px' % s['median_err_vs_opencv'])
    st.caption('"Reliable" = the correlation peak is ≥ 0.9, i.e. the patch '
               'still looks like itself in frame 2 (motion blur and wheels '
               'turning make some car patches change). %d of the %d reliable '
               'points agree within 0.5 px. The few that disagree by several '
               'pixels sit on a car\'s outline, where the window holds both '
               'the moving car and the still road: two motions in one window, '
               'which the "flow is constant in the window" assumption of LK '
               'does not allow.' % (s['within_half_px'], s['reliable']))
    show = df.rename(columns={'pred_x': 'predicted x', 'pred_y': 'predicted y',
                              'actual_x': 'actual x', 'actual_y': 'actual y',
                              'err_vs_actual_px': 'error (px)',
                              'moved_px': 'moved (px)', 'ncc_peak': 'NCC peak'})
    st.dataframe(show[['x', 'y', 'kind', 'predicted x', 'predicted y',
                       'actual x', 'actual y', 'moved (px)', 'error (px)',
                       'NCC peak', 'reliable']].round(2),
                 hide_index=True, width='stretch')
    wk = os.path.join(RES, 'workings_%s.md' % name)
    if os.path.exists(wk):
        with st.expander('One point worked through by hand — every number'):
            st.markdown(_text(wk))

    with st.expander('Try another frame pair (computed live)'):
        frames, _ = _frames(name)
        i = st.slider('first frame', 0, len(frames) - 2, s['frame'],
                      key='m5_trk_f_' + name)
        r = _track_live(name, i)
        if r is None or r.empty:
            st.info('Nothing moving enough in this frame.')
        else:
            ok = r['NCC peak'] >= 0.9
            st.metric('median error over %d reliable points' % ok.sum(),
                      '%.2f px' % r[ok]['error (px)'].median())
            st.dataframe(r.round(2), hide_index=True, width='stretch')


# ----------------------------------------------------------------------------
# Tab 3: structure from motion
# ----------------------------------------------------------------------------
def _tab_sfm():
    st.subheader('B. Structure from motion — four views of a flat object')
    st.markdown(
        'The object is my **monitor screen**, which is flat, showing the '
        'chessboard page from Module 2. It was photographed from four positions with '
        'the calibrated iPhone. The 54 chessboard corners give the '
        'correspondences between views. The **4 corners of the lit screen are '
        'the boundary** to estimate. Only the image points and $K$ are used to '
        'reconstruct; the board\'s real size is used for one length (the scale) '
        'and afterwards to check the result.')
    s = _json('sfm_summary.json')
    cams = _csv('sfm_cameras.csv')
    if s is None:
        st.warning('Run `python sfm.py` first.')
        return
    st.image(os.path.join(RES, 'sfm_views.png'), width='stretch')

    st.markdown('#### Camera parameters')
    cj = json.load(open(os.path.join(DATA, 'sfm', 'camera.json')))
    K = np.array(cj['K'])
    c1, c2 = st.columns([1, 1])
    c1.markdown('Intrinsics $K$ (Module 2 calibration, 1512×2016 images):')
    c1.latex(r'K=\begin{bmatrix}%.1f&0&%.1f\\0&%.1f&%.1f\\0&0&1\end{bmatrix}'
             % (K[0, 0], K[0, 2], K[1, 1], K[1, 2]))
    c2.markdown('Distortion $(k_1, k_2, p_1, p_2, k_3)$:')
    c2.code(', '.join('%.4f' % v for v in cj['dist']))
    c2.caption(cj['note'])

    st.markdown('#### Camera positions (recovered by SfM)')
    show = cams.copy()
    show['C (mm)'] = show.apply(lambda r: '(%.0f, %.0f, %.0f)'
                                % (r.Cx_mm, r.Cy_mm, r.Cz_mm), axis=1)
    show['solvePnP check (mm)'] = show.apply(
        lambda r: '(%.0f, %.0f, %.0f)' % (r.pnp_Cx_mm, r.pnp_Cy_mm, r.pnp_Cz_mm), axis=1)
    st.dataframe(show[['view', 'image', 'C (mm)', 'dist_to_object_mm',
                       'rot_vs_cam1_deg', 'solvePnP check (mm)', 'diff_vs_pnp_mm']]
                 .rename(columns={'dist_to_object_mm': 'distance to object (mm)',
                                  'rot_vs_cam1_deg': 'rotation vs view 1 (°)',
                                  'diff_vs_pnp_mm': 'difference (mm)'}).round(1),
                 hide_index=True, width='stretch')
    st.caption('Object frame: origin at chessboard corner 0, x along the top '
               'row, y down the rows, z into the screen, so cameras have z < 0. '
               'The check column is where `cv2.solvePnP` puts each camera using '
               'the board\'s true geometry. SfM never uses that.')

    c1, c2 = st.columns(2)
    c1.image(os.path.join(RES, 'sfm_3d.png'), width='stretch')
    c2.image(os.path.join(RES, 'sfm_boundary.png'), width='stretch')

    st.markdown('#### How good is it')
    a, b, c, d = st.columns(4)
    a.metric('reprojection (grid)', '%.2f px' % s['reproj_rms_px_refined'])
    b.metric('reprojection (boundary)', '%.2f px' % s['boundary_reproj_rms_px'])
    c.metric('flatness (RMS off plane)', '%.2f mm' % s['planarity_rms_mm'])
    d.metric('cameras vs solvePnP', '%.1f mm mean' % s['camera_vs_pnp_mm_mean'])
    a, b, c, d = st.columns(4)
    a.metric('square side, along rows', '%.2f mm' % s['square_side_along_rows_mm'][0])
    b.metric('square side, down columns', '%.2f mm' % s['square_side_down_cols_mm'][0])
    c.metric('grid vs true board', '%.2f mm RMS' % s['grid_vs_true_rms_mm'])
    d.metric('boundary corner angles', ', '.join(
        '%.1f°' % v for v in s['screen_corner_angles_deg']))
    st.markdown(
        'Estimated boundary: **%.0f × %.0f mm** (width × height, mean of the '
        'opposite sides), aspect ratio **%.3f**. The four corners are %.1f mm '
        'or less off the plane of the grid points. The scale comes from one '
        'assumed length: a chessboard square is %.0f mm on the screen, the value '
        'used in the Module 2 calibration. Every length scales with that one '
        'number; the angles, flatness and aspect ratio do not depend on it.'
        % ((s['screen_width_top_mm'] + s['screen_width_bottom_mm']) / 2,
           (s['screen_height_left_mm'] + s['screen_height_right_mm']) / 2,
           s['screen_aspect'], max(abs(v) for v in s['boundary_out_of_plane_mm']),
           s['square_mm_assumed']))

    with st.expander('The mathematical workings, with this run\'s numbers'):
        wk = os.path.join(RES, 'workings_sfm.md')
        if os.path.exists(wk):
            st.markdown(_text(wk))
    with st.expander('The reconstructed points (CSV)'):
        st.dataframe(_csv('sfm_points.csv').round(3), hide_index=True)


def render():
    st.title('Modules 5 & 6 — ' + TITLE)
    st.caption(SUBTITLE)
    t1, t2, t3, t4 = st.tabs(['1 · Optical flow', '2 · Tracking check',
                              '3 · Structure from motion', '4 · Theory'])
    with t1:
        _tab_flow()
    with t2:
        _tab_tracking()
    with t3:
        _tab_sfm()
    with t4:
        st.markdown(_text(os.path.join(HERE, 'theory.md')))

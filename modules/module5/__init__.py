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
def _tab_flow():
    st.subheader('A. Optical flow of two 30-second videos')
    name = _pick_clip('m5_flow_clip')
    clip = _clips()[name]
    st.caption('%s  \nSource: %s — [%s](%s), %s'
               % (clip['about'], clip['source'], clip['url'], clip['url'],
                  clip['license']))
    vid = os.path.join(RES, 'flow_%s.mp4' % name)
    if os.path.exists(vid):
        st.markdown('**The flow as a video.** Left: the clip with flow arrows '
                    '(drawn 3× longer). Right: the flow field, colour = '
                    'direction (legend top right), brightness = speed.')
        st.video(vid)
    else:
        st.warning('Run `python make_flow_videos.py` to make the flow video.')

    with st.expander('Look at any single frame pair (computed live)'):
        frames, fps = _frames(name)
        i = st.slider('frame', 0, len(frames) - 2, len(frames) // 2,
                      key='m5_frame_' + name)
        fl = F.dense_flow(frames[i], frames[i + 1])
        c1, c2 = st.columns(2)
        c1.image(_rgb(F.draw_arrows(frames[i], fl, 16, 0.5, 3)),
                 caption='frame %d, t = %.2f s' % (i, i / fps), width='stretch')
        c2.image(_rgb(F.flow_to_color(fl, clip['max_mag'])),
                 caption='flow %d → %d' % (i, i + 1), width='stretch')
        s, _ = F.frame_stats(fl, clip['moving_thr'])
        a, b, c, d = st.columns(4)
        a.metric('moving pixels', '%.1f %%' % (100 * s['moving_fraction']))
        b.metric('speed of moving pixels', '%.2f px/frame' % s['moving_mean_speed'])
        c.metric('background speed', '%.3f px/frame' % s['background_median'])
        d.metric('moving regions', s['moving_blobs'])

    st.markdown('#### What can be inferred from optical flow — with evidence')
    ev = os.path.join(RES, 'evidence_%s.png' % name)
    if os.path.exists(ev):
        st.image(ev, width='stretch')
    sm = _json('summary_%s.json' % name)
    df = _csv('stats_%s.csv' % name)
    if sm is None or df is None:
        return
    rs = [v for v in sm['row_speed'] if v > 0]
    # a lighting change: the background's median flow jumps far above its
    # normal level (0.2 px/frame is > 10x the normal median in both clips)
    flick = df[df.background_median > 0.2]
    lines = [
        '1. **Which pixels move — segmentation with no model.** Thresholding '
        'the flow speed at %.1f px/frame outlines the moving people (plot a, '
        'red). Up to %d separate moving regions appear in one frame.'
        % (clip['moving_thr'], sm['max_blobs']),
        '2. **The camera is fixed.** The median speed of the background is '
        '%.3f px/frame over the whole clip (plot b, orange). If the camera '
        'panned or shook, every pixel would have flow.' % sm['background_median_px'],
        '3. **Direction of travel.** Plot c is the direction histogram of all '
        'moving pixels. The most common direction is %.0f° in image '
        'coordinates, i.e. **%s** in the picture.'
        % (sm['dominant_dirs_deg'][0], _compass(sm['dominant_dirs_deg'][0])),
        '4. **Speed vs depth.** The same walking speed gives a larger image '
        'speed nearer the camera ($u \\propto 1/Z$). The mean speed of '
        'moving pixels rises from %.1f px/frame at the top of the image (far) to '
        '%.1f px/frame near the bottom (near) (plot d).' % (rs[0], rs[-1]),
        '5. **How fast.** Moving pixels travel on average %.2f px/frame = '
        '%.0f px/s at %.0f fps.'
        % (sm['moving_speed_px'], sm['moving_speed_px'] * sm['fps'], sm['fps']),
    ]
    if len(flick):
        # group neighbouring frames into events
        ev, cur = [], [flick.t.iloc[0], flick.t.iloc[0]]
        for t in flick.t.iloc[1:]:
            if t - cur[1] <= 0.2:
                cur[1] = t
            else:
                ev.append(cur)
                cur = [t, t]
        ev.append(cur)
        when = ' and '.join(('%.1f s' % a) if b - a < 0.05 else ('%.1f–%.1f s' % (a, b))
                            for a, b in ev)
        lines.append(
            '6. **The camera moved briefly.** At t = %s the whole '
            'background suddenly has flow (median up to %.2f px/frame, against '
            '%.3f normally; plot b, orange spikes). The background flow in '
            'those frames flips up and down from one frame to the next, and the '
            'average brightness changes by about 1 grey level, so it is a short '
            'camera shake, not a lighting change. Flow picks up camera motion '
            'even when it is well under a pixel.'
            % (when, flick.background_median.max(), sm['background_median_px']))
    st.markdown('\n'.join(lines))


# ----------------------------------------------------------------------------
# Tab 2: tracking validation
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner='Tracking with my Lucas-Kanade...')
def _track_live(name, i):
    frames, _ = _frames(name)
    f1, f2 = frames[i], frames[i + 1]
    fl = F.dense_flow(f1, f2)
    g = F.gray(f1)
    mag = np.hypot(fl[..., 0], fl[..., 1])
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
        'with bilinear interpolation, on a 3-level pyramid.  \n'
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
    a.metric('reliable points', '%d of %d' % (s['reliable'], s['points']))
    b.metric('median |predicted − actual|', '%.2f px' % s['median_err_vs_actual'])
    c.metric('90th percentile', '%.2f px' % s['p90_err_vs_actual'])
    d.metric('median vs OpenCV LK', '%.3f px' % s['median_err_vs_opencv'])
    st.caption('"Reliable" = the correlation peak is ≥ 0.9, i.e. the patch '
               'still looks like itself in frame 2. Points below that (an arm '
               'swinging, a person passing in front) are listed but not '
               'counted. There the "actual" position itself is uncertain, and '
               'brightness constancy does not hold either.')
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

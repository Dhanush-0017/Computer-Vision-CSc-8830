"""
validate_tracking.py -- Part A: check the tracking equations on real frames.

Usage (from this folder):
    python validate_tracking.py                  # both clips
    python validate_tracking.py --clip pedestrians --frame 95

For each clip it takes two CONSECUTIVE frames (by default a frame with a
typical amount of motion, see typical_frame) and:

  1. picks corner points (Shi-Tomasi) on the moving people, plus a few on the
     static background as a control;
  2. predicts where each point goes with MY Lucas-Kanade (flow.lk_track):
     the equations derived in theory.md, with bilinear interpolation for every
     sub-pixel sample;
  3. measures where each point ACTUALLY went, without using any flow
     equation: normalised cross-correlation of the patch around it, searched
     over +/-24 px in the second frame, peak refined to sub-pixel;
  4. also runs OpenCV's pyramidal LK (cv2.calcOpticalFlowPyrLK) as a second
     reference;
  5. writes one point's full calculation (window, gradients, G, b, every
     iteration) -- the "by hand" check.

Writes to results/:
    tracking_<clip>.csv     per point: start, predicted, actual, errors
    tracking_<clip>.png     frame 2 with predicted (+) vs actual (o), with zooms
    workings_<clip>.md      the one-point worked calculation
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flow as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RESULTS = os.path.join(HERE, 'results')


def typical_frame(name):
    """A frame with a typical amount of motion -- not the fastest, not the
    slowest: among frames with at least the median number of moving regions
    (and no lighting flicker), the one whose speed is closest to the median.
    Uses results/stats_<clip>.csv from make_flow_videos.py."""
    d = pd.read_csv(os.path.join(RESULTS, 'stats_%s.csv' % name))
    d = d[(d.background_median < 0.05) & (d.moving_blobs >= d.moving_blobs.median())]
    med = d.moving_mean_speed.median()
    return int(d.loc[(d.moving_mean_speed - med).abs().idxmin(), 'frame'])


def pick_points(f1, fl, thr, n_moving=30, n_static=8, border=40):
    """Integer corner locations: some on moving things, some on background."""
    g = F.gray(f1)
    mag = np.hypot(fl[..., 0], fl[..., 1])
    H, W = g.shape
    inner = np.zeros_like(g)
    inner[border:H - border, border:W - border] = 255
    mov = ((mag > thr) * 255).astype(np.uint8)
    mov = cv2.erode(mov, np.ones((7, 7), np.uint8)) & inner
    still = ((mag < 0.1) * 255).astype(np.uint8)
    still = cv2.erode(still, np.ones((15, 15), np.uint8)) & inner
    out = []
    for mask, n, kind in ((mov, n_moving, 'moving'), (still, n_static, 'static')):
        p = cv2.goodFeaturesToTrack(g, n, 0.01, 12, mask=mask, blockSize=7)
        if p is not None:
            for (x, y) in p.reshape(-1, 2):
                out.append((int(round(x)), int(round(y)), kind))
    return out


def fmt_mat(M, nd=1):
    rows = [' & '.join(('%.' + str(nd) + 'f') % v for v in r) for r in np.atleast_2d(M)]
    return r'\begin{bmatrix}' + r' \\ '.join(rows) + r'\end{bmatrix}'


def workings_md(name, w, actual, i):
    L = []
    L.append('### Worked example — %s, frames %d → %d, point (x, y) = (%d, %d)\n'
             % (name, i, i + 1, w['x'], w['y']))
    L.append('Window %d×%d centred on the point, full resolution, no pyramid. '
             'Rows are y (top to bottom), columns are x (left to right).\n'
             % (w['win'], w['win']))
    L.append('$I_1$ (frame %d grey levels):\n\n$$%s$$\n' % (i, fmt_mat(w['I1'], 0)))
    L.append('$I_x = \\frac{I_1(x+1,y) - I_1(x-1,y)}{2}$:\n\n$$%s$$\n'
             % fmt_mat(w['Ix'], 1))
    L.append('$I_y = \\frac{I_1(x,y+1) - I_1(x,y-1)}{2}$:\n\n$$%s$$\n'
             % fmt_mat(w['Iy'], 1))
    L.append('$I_t = I_2(x,y) - I_1(x,y)$ at $d = 0$:\n\n$$%s$$\n'
             % fmt_mat(w['It0'], 0))
    G = w['G']
    L.append('Structure tensor (sums over the window):\n\n'
             '$$G = \\begin{bmatrix}\\sum I_x^2 & \\sum I_xI_y\\\\ \\sum I_xI_y & '
             '\\sum I_y^2\\end{bmatrix} = %s,\\qquad '
             '\\lambda_{min} = %.1f,\\ \\lambda_{max} = %.1f$$\n'
             % (fmt_mat(G, 1), w['eig'][0], w['eig'][1]))
    L.append('Both eigenvalues are well above zero, so $G$ is invertible and '
             'the point is trackable (not an edge or a flat patch — the '
             'aperture problem does not apply here).\n')
    L.append('Iterations: $e = I_1(\\mathbf{x}) - I_2(\\mathbf{x} + \\mathbf{d})$ '
             '(with $I_2$ sampled by bilinear interpolation), '
             '$\\mathbf{b} = \\sum [I_x e,\\ I_y e]^T$, '
             '$\\Delta\\mathbf{d} = G^{-1}\\mathbf{b}$, '
             '$\\mathbf{d} \\leftarrow \\mathbf{d} + \\Delta\\mathbf{d}$.\n')
    L.append('| k | b | Δd | d after step | SSD before step |')
    L.append('|---|---|---|---|---|')
    for r in w['iters']:
        L.append('| %d | (%.1f, %.1f) | (%.3f, %.3f) | (%.3f, %.3f) | %.0f |'
                 % (r['k'], r['b'][0], r['b'][1], r['step'][0], r['step'][1],
                    r['d'][0], r['d'][1], r['ssd']))
    L.append('')
    px, py = w['x'] + w['d'][0], w['y'] + w['d'][1]
    L.append('Final SSD over the window: %.0f (was %.0f at d = 0).\n'
             % (w['ssd_final'], w['iters'][0]['ssd']))
    L.append('**Predicted** position in frame %d: (%.2f, %.2f) + (%.3f, %.3f) = '
             '**(%.2f, %.2f)**.  \n' % (i + 1, w['x'], w['y'], w['d'][0],
                                        w['d'][1], px, py))
    L.append('**Actual** position (patch correlation, independent of LK): '
             '**(%.2f, %.2f)**. Difference: %.2f px.\n'
             % (actual[0], actual[1], np.hypot(px - actual[0], py - actual[1])))
    return '\n'.join(L)


def run(name, clip, frame=None):
    frames, fps = F.read_frames(os.path.join(DATA, 'videos', clip['file']))
    if frame is None:
        frame = typical_frame(name)
    f1, f2 = frames[frame], frames[frame + 1]
    fl = F.dense_flow(f1, f2)
    pts = pick_points(f1, fl, clip['moving_thr'])
    P = np.array([[x, y] for x, y, _ in pts], np.float64)

    mine, info = F.lk_track(f1, f2, P, win=21, levels=3)
    cvp, stt, _ = cv2.calcOpticalFlowPyrLK(
        F.gray(f1), F.gray(f2), P.astype(np.float32).reshape(-1, 1, 2), None,
        winSize=(21, 21), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    cvp = cvp.reshape(-1, 2)

    rows = []
    for k, (x, y, kind) in enumerate(pts):
        act, score = F.ncc_locate(f1, f2, (x, y), patch=21, search=24)
        if act is None:
            continue
        rows.append({
            'x': x, 'y': y, 'kind': kind,
            'pred_x': mine[k, 0], 'pred_y': mine[k, 1],
            'actual_x': act[0], 'actual_y': act[1], 'ncc_peak': score,
            'opencv_x': cvp[k, 0], 'opencv_y': cvp[k, 1],
            'moved_px': np.hypot(act[0] - x, act[1] - y),
            'err_vs_actual_px': np.hypot(mine[k, 0] - act[0], mine[k, 1] - act[1]),
            'err_vs_opencv_px': np.hypot(mine[k, 0] - cvp[k, 0], mine[k, 1] - cvp[k, 1]),
            'lambda_min': info[k]['lambda_min'],
        })
    df = pd.DataFrame(rows)
    # a correlation peak below 0.9 means the patch changed too much to be a
    # trustworthy "actual" (occlusion, a limb swinging) -- flagged, not hidden
    df['reliable'] = df.ncc_peak >= 0.9
    df.to_csv(os.path.join(RESULTS, 'tracking_%s.csv' % name), index=False,
              float_format='%.3f')

    # worked example: a moving point with a small (1-3 px) shift, so one level
    # at full resolution is enough and every number fits on the page
    wk = None
    cand = df[(df.kind == 'moving') & df.reliable].copy()
    cand = cand.iloc[np.argsort(np.abs(cand.moved_px.values - 2.0))]
    for _, r in cand.iterrows():
        w = F.lk_workings(f1, f2, (r.x, r.y), win=7)
        a = (r.actual_x, r.actual_y)
        if np.hypot(r.x + w['d'][0] - a[0], r.y + w['d'][1] - a[1]) < 0.5:
            wk = (w, a)
            break
    if wk:
        with open(os.path.join(RESULTS, 'workings_%s.md' % name), 'w') as f:
            f.write(workings_md(name, wk[0], wk[1], frame))

    plot(name, clip, f1, f2, df, frame, wk)
    rel = df[df.reliable]
    s = {
        'frame': frame, 'points': len(df), 'reliable': int(len(rel)),
        'moving_reliable': int((rel.kind == 'moving').sum()),
        'median_err_vs_actual': float(rel.err_vs_actual_px.median()),
        'mean_err_vs_actual': float(rel.err_vs_actual_px.mean()),
        'p90_err_vs_actual': float(rel.err_vs_actual_px.quantile(0.9)),
        'median_err_vs_opencv': float(rel.err_vs_opencv_px.median()),
        'median_moved_px': float(rel[rel.kind == 'moving'].moved_px.median()),
        'max_moved_px': float(rel.moved_px.max()),
        'static_max_pred_shift': float(np.hypot(
            df[df.kind == 'static'].pred_x - df[df.kind == 'static'].x,
            df[df.kind == 'static'].pred_y - df[df.kind == 'static'].y).max()),
    }
    with open(os.path.join(RESULTS, 'tracking_summary_%s.json' % name), 'w') as f:
        json.dump(s, f, indent=2)
    print(name, json.dumps(s))


def plot(name, clip, f1, f2, df, frame, wk):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axes[0]
    ax.imshow(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
    for _, r in df.iterrows():
        c = 'yellow' if r.reliable else '0.6'
        ax.plot([r.x, r.actual_x], [r.y, r.actual_y], '-', color='w', lw=0.8)
        ax.plot(r.actual_x, r.actual_y, 'o', mfc='none', mec='red', ms=7)
        ax.plot(r.pred_x, r.pred_y, '+', color=c, ms=7, mew=1.5)
    ax.set_title('frame %d → %d. white line: start → actual.  red o: actual '
                 '(correlation)   yellow +: predicted by my LK' % (frame, frame + 1),
                 fontsize=9)
    ax.axis('off')

    ax = axes[1]
    rel = df[df.reliable]
    ax.scatter(rel.moved_px, rel.err_vs_actual_px, s=18, label='vs actual (correlation)')
    ax.scatter(rel.moved_px, rel.err_vs_opencv_px, s=18, marker='x',
               label='vs OpenCV pyramidal LK')
    ax.set_xlabel('how far the point moved (px)')
    ax.set_ylabel('|predicted − reference| (px)')
    ax.set_title('error of my LK prediction, %d reliable points' % len(rel),
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(clip['title'], fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'tracking_%s.png' % name), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--clip')
    ap.add_argument('--frame', type=int, help='first of the two frames')
    a = ap.parse_args()
    with open(os.path.join(DATA, 'clips.json')) as f:
        clips = json.load(f)
    for name, clip in clips.items():
        if a.clip and name != a.clip:
            continue
        run(name, clip, a.frame)


if __name__ == '__main__':
    main()

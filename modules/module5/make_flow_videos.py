"""
make_flow_videos.py -- Part A: optical flow for each 30 s clip, as a video,
plus the evidence for what the flow tells us.

Usage (from this folder):
    python make_flow_videos.py                  # both clips in data/clips.json
    python make_flow_videos.py --clip intersection

Both clips were filmed hand-held on my iPhone, so the camera moves too. For
every pair of consecutive frames the script computes:
    - the dense optical flow (Farneback), and
    - the camera's own motion (flow.camera_motion: a homography fitted with
      RANSAC to corners across the frame);
the difference between the two is how things moved in the world.

Writes, into results/:
    flow_<clip>.mp4       left: the video, moving objects outlined in red,
                          arrows = their motion with the camera motion removed;
                          right: the raw optical flow in colour (hue =
                          direction, brightness = speed, legend top right).
                          Same length and frame rate as the clip.
    stats_<clip>.csv      one row per frame pair: camera motion, fraction of
                          moving pixels, their speed and direction, number of
                          moving regions
    evidence_<clip>.png   the four plots used in the report to back up
                          "what can be inferred from optical flow"
    sample_<clip>.png     one frame, original | flow colour, for the PDF
    vehicles_<clip>.csv   every moving region in every frame: where it
                          touches the road (bottom y) and its speed
    summary_<clip>.json   the headline numbers

Needs ffmpeg on the PATH to make the mp4 playable in a browser (H.264).
Without it the video is still written, as MPEG-4 part 2.
"""
import argparse
import json
import os
import shutil
import subprocess

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
PANEL_W = 480
N_ANG = 36          # direction histogram bins (10 degrees each)


def load_clips():
    with open(os.path.join(DATA, 'clips.json')) as f:
        return json.load(f)


def _panel(img, w=PANEL_W):
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _label(img, text, y=18):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(img, (4, y - h - 4), (12 + w, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)


def outline(frame, moving, color=(0, 0, 255)):
    out = frame.copy()
    cnts, _ = cv2.findContours(moving.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 400]
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def compose(frame, fl, obj, moving, max_mag, t, st):
    """One output frame: objects + their motion | colour-coded raw flow."""
    left = outline(F.draw_arrows(frame, obj * moving[..., None], step=20,
                                 min_mag=1.0, scale=2,
                                 color=(0, 255, 255)), moving)
    left = _panel(left)
    right = _panel(F.flow_to_color(fl, max_mag))
    right[30:100, -78:-8] = F.color_wheel(70)
    _label(left, 't = %.1f s   red: moving objects   arrows: their motion (x2)' % t)
    _label(left, 'camera motion: %.2f px/frame' % st['camera_speed'], 36)
    _label(right, 'optical flow: hue = direction, brightness = speed '
           '(max %.0f px/frame)' % max_mag)
    return np.hstack([left, right])


def to_h264(src, dst):
    if shutil.which('ffmpeg') is None:
        os.replace(src, dst)
        return
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', src,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26',
                    '-movflags', '+faststart', dst], check=True)
    os.remove(src)


def process(name, clip):
    cap = cv2.VideoCapture(os.path.join(DATA, 'videos', clip['file']))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    thr, max_mag = clip['moving_thr'], clip['max_mag']
    os.makedirs(RESULTS, exist_ok=True)
    tmp = os.path.join(RESULTS, '_tmp_%s.mp4' % name)
    vw = None

    ok, prev = cap.read()
    H = prev.shape[0]
    ang_hist = np.zeros(N_ANG)
    rows = []
    blobs = []
    best = (-1, None)
    i = 0
    while True:
        ok, nxt = cap.read()
        if not ok:
            break
        fl = F.dense_flow(prev, nxt)
        Hc, inl = F.camera_motion(prev, nxt)
        cam = F.camera_flow(Hc, fl.shape)
        obj = fl - cam
        st, moving = F.frame_stats(fl, thr, cam, F.textured(prev))
        st.update(frame=i, t=i / fps, camera_inliers=inl)
        rows.append(st)

        mag = np.hypot(obj[..., 0], obj[..., 1])
        ang = np.degrees(np.arctan2(obj[..., 1], obj[..., 0])) % 360
        ang_hist += np.bincount((ang[moving] // (360 / N_ANG)).astype(int),
                                minlength=N_ANG)[:N_ANG]
        # each moving vehicle: where it touches the road (bottom of its
        # region -- lower in the image = nearer the camera) and its speed
        k, lab, cs, _ = cv2.connectedComponentsWithStats(moving.astype(np.uint8))
        for j in range(1, k):
            if cs[j, cv2.CC_STAT_AREA] < 3000:
                continue
            bottom = cs[j, cv2.CC_STAT_TOP] + cs[j, cv2.CC_STAT_HEIGHT]
            blobs.append((i, bottom, float(np.median(mag[lab == j]))))

        # frame shown in the report: the most object motion (for the panning
        # clip, only frames where the camera is actually turning)
        score = st['moving_fraction'] * st['moving_mean_speed']
        if clip.get('sample_while_panning') and st['camera_speed'] < 2:
            score = -1
        if score > best[0]:
            best = (score, (i, prev.copy(), fl, obj, moving))

        out = compose(prev, fl, obj, moving, max_mag, i / fps, st)
        if vw is None:
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                 (out.shape[1], out.shape[0]))
        vw.write(out)
        prev = nxt
        i += 1
    vw.release()
    cap.release()
    to_h264(tmp, os.path.join(RESULTS, 'flow_%s.mp4' % name))
    n_frames = i + 1
    print('%s: %d frames at %.0f fps = %.1f s' % (name, n_frames, fps, n_frames / fps))

    df = pd.DataFrame(rows)[['frame', 't', 'camera_speed', 'camera_dx', 'camera_dy',
                             'camera_inliers', 'residual_background',
                             'moving_fraction', 'moving_mean_speed', 'mean_u',
                             'mean_v', 'moving_blobs']]
    df.to_csv(os.path.join(RESULTS, 'stats_%s.csv' % name), index=False,
              float_format='%.4f')

    i, frame, fl, obj, moving = best[1]
    sample = np.hstack([_panel(outline(frame, moving)),
                        _panel(F.flow_to_color(fl, max_mag))])
    cv2.imwrite(os.path.join(RESULTS, 'sample_%s.png' % name), sample)
    bl = pd.DataFrame(blobs, columns=['frame', 'bottom_y', 'speed'])
    bl.to_csv(os.path.join(RESULTS, 'vehicles_%s.csv' % name), index=False,
              float_format='%.3f')
    evidence_plot(name, clip, frame, obj, moving, i / fps, df, ang_hist,
                  bl, fps, n_frames)
    summary = {
        'frames': n_frames, 'fps': fps, 'sample_frame': int(i),
        'camera_speed_median': float(df.camera_speed.median()),
        'camera_speed_p95': float(df.camera_speed.quantile(0.95)),
        'camera_speed_max': float(df.camera_speed.max()),
        'camera_inliers_mean': float(df.camera_inliers.mean()),
        'residual_background_median': float(df.residual_background.median()),
        'moving_speed_px': float(df.moving_mean_speed[df.moving_blobs > 0].mean()),
        'moving_speed_max': float(df.moving_mean_speed.max()),
        'max_blobs': int(df.moving_blobs.max()),
        'frames_with_motion': float((df.moving_blobs > 0).mean()),
        'depth_bins': [[round(y), round(v, 2), n] for y, v, n in depth_bins(bl)],
        'dominant_dirs_deg': (np.argsort(ang_hist)[::-1][:3] * 360 / N_ANG
                              + 5).tolist(),
        'left_vs_right': [float(ang_hist[(np.arange(N_ANG) * 10 + 5 > 90) &
                                         (np.arange(N_ANG) * 10 + 5 < 270)].sum()
                                / max(ang_hist.sum(), 1))],
    }
    with open(os.path.join(RESULTS, 'summary_%s.json' % name), 'w') as f:
        json.dump(summary, f, indent=2)
    print('  ', json.dumps(summary))


def depth_bins(bl, H=540, step=30):
    """Median vehicle speed for each 30 px band of 'where it touches the
    road' (only bands with enough samples)."""
    out = []
    for y0 in range(0, H, step):
        s = bl[(bl.bottom_y >= y0) & (bl.bottom_y < y0 + step)]
        if len(s) >= 15:
            out.append((y0 + step / 2, float(s.speed.median()), len(s)))
    return out


def evidence_plot(name, clip, frame, obj, moving, t, df, ang_hist, bl, fps,
                  n_frames):
    fig = plt.figure(figsize=(12, 8.2))

    ax = fig.add_subplot(2, 2, 1)
    over = outline(F.draw_arrows(frame, obj * moving[..., None], step=16,
                                 min_mag=clip['moving_thr'],
                                 scale=2, color=(0, 255, 255)), moving)
    ax.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
    ax.set_title('(a) t = %.1f s: moving objects (red outline, speed > %.1f px/frame\n'
                 'after removing camera motion) and their flow (arrows, x2)'
                 % (t, clip['moving_thr']), fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(df.t, df.camera_speed, lw=1, color='tab:orange',
            label='camera motion (median over frame)')
    ax.plot(df.t, df.moving_mean_speed.where(df.moving_blobs > 0), lw=1,
            color='tab:blue', label='speed of moving objects')
    ax.plot(df.t, df.residual_background, lw=1, color='tab:green',
            label='background after removing camera motion')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('px / frame')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('(b) camera motion vs object motion over time', fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 3, projection='polar')
    th = np.radians(np.arange(N_ANG) * 360 / N_ANG + 5)
    # image y points down; negate so "up the image" is up on the plot
    ax.bar(-th, ang_hist / max(ang_hist.sum(), 1), width=np.radians(360 / N_ANG),
           color='tab:blue', alpha=0.8)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['right', 'up', 'left', 'down'])
    ax.set_yticklabels([])
    ax.set_title('(c) direction of object motion, all moving pixels, whole clip',
                 fontsize=9)

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(bl.bottom_y, bl.speed, s=4, alpha=0.25, color='tab:blue',
               label='one moving region in one frame')
    db = depth_bins(bl, frame.shape[0])
    if db:
        ax.plot([d[0] for d in db], [d[1] for d in db], 'o-', color='tab:red',
                label='median per 30 px band')
    ax.set_xlabel('y where the vehicle meets the road (px)   '
                  'farther  ->  nearer the camera')
    ax.set_ylabel('speed in the image (px / frame)')
    ax.set_title('(d) image speed vs distance from the camera', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('%s  (%d frames, %.0f fps)' % (clip['title'], n_frames, fps),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'evidence_%s.png' % name), dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--clip', help='one clip name from data/clips.json')
    a = ap.parse_args()
    for name, clip in load_clips().items():
        if a.clip and name != a.clip:
            continue
        process(name, clip)


if __name__ == '__main__':
    main()

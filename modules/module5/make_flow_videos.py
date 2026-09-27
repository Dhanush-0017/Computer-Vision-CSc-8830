"""
make_flow_videos.py -- Part A: optical flow for each 30 s clip, as a video,
plus the evidence for what the flow tells us.

Usage (from this folder):
    python make_flow_videos.py                 # both clips in data/clips.json
    python make_flow_videos.py --clip pedestrians

For every pair of consecutive frames it computes dense (Farneback) flow and
writes, into results/:
    flow_<clip>.mp4       left: the video with flow arrows, right: flow colour
                          (hue = direction, brightness = speed; legend in the
                          corner). Same length and frame rate as the clip.
    stats_<clip>.csv      one row per frame pair: moving fraction, speed of
                          the moving pixels, background speed, mean direction,
                          number of moving regions
    evidence_<clip>.png   the four plots used in the report to back up
                          "what can be inferred from optical flow"
    sample_<clip>.png     one frame, original | flow colour, for the PDF

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
N_ROWS = 12         # image-row bands for the speed-vs-row plot


def load_clips():
    with open(os.path.join(DATA, 'clips.json')) as f:
        return json.load(f)


def _panel(img, w=PANEL_W):
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _label(img, text):
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def compose(frame, fl, max_mag, t):
    """One output frame: arrows | colour-coded flow with legend."""
    left = _panel(F.draw_arrows(frame, fl, step=16, min_mag=0.5, scale=3))
    right = _panel(F.flow_to_color(fl, max_mag))
    wheel = F.color_wheel(70)
    right[30:100, -78:-8] = wheel
    _label(left, 't = %.1f s   arrows = flow, drawn 3x longer' % t)
    _label(right, 'hue = direction, brightness = speed (max %.0f px/frame)'
           % max_mag)
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
    frames, fps = F.read_frames(os.path.join(DATA, 'videos', clip['file']))
    thr, max_mag = clip['moving_thr'], clip['max_mag']
    print('%s: %d frames at %.0f fps = %.1f s' % (name, len(frames), fps,
                                                 len(frames) / fps))
    os.makedirs(RESULTS, exist_ok=True)
    tmp = os.path.join(RESULTS, '_tmp_%s.mp4' % name)
    first = compose(frames[0], np.zeros(frames[0].shape[:2] + (2,), np.float32),
                    max_mag, 0)
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                         (first.shape[1], first.shape[0]))

    H = frames[0].shape[0]
    ang_hist = np.zeros(N_ANG)
    row_sum = np.zeros(N_ROWS)
    row_cnt = np.zeros(N_ROWS)
    rows = []
    best = (-1, None)
    for i in range(len(frames) - 1):
        fl = F.dense_flow(frames[i], frames[i + 1])
        st, moving = F.frame_stats(fl, thr)
        st['frame'] = i
        st['t'] = i / fps
        rows.append(st)

        mag = np.hypot(fl[..., 0], fl[..., 1])
        ang = np.degrees(np.arctan2(fl[..., 1], fl[..., 0])) % 360
        ang_hist += np.bincount((ang[moving] // (360 / N_ANG)).astype(int),
                                minlength=N_ANG)[:N_ANG]
        band = (np.arange(H) * N_ROWS // H)
        ys = np.nonzero(moving)[0]
        np.add.at(row_sum, band[ys], mag[moving])
        np.add.at(row_cnt, band[ys], 1)

        # frame shown in the report: the most total motion, skipping frames
        # where the whole background "moves" (lighting flicker)
        score = st['moving_fraction'] * st['moving_mean_speed']
        if st['background_median'] < 0.05 and score > best[0]:
            best = (score, (i, fl, moving))
        vw.write(compose(frames[i], fl, max_mag, i / fps))
    vw.release()
    to_h264(tmp, os.path.join(RESULTS, 'flow_%s.mp4' % name))

    df = pd.DataFrame(rows)[['frame', 't', 'moving_fraction', 'moving_mean_speed',
                             'background_median', 'mean_u', 'mean_v',
                             'moving_blobs']]
    df.to_csv(os.path.join(RESULTS, 'stats_%s.csv' % name), index=False,
              float_format='%.4f')

    i, fl, moving = best[1]
    sample = np.hstack([_panel(frames[i]), _panel(F.flow_to_color(fl, max_mag))])
    cv2.imwrite(os.path.join(RESULTS, 'sample_%s.png' % name), sample)
    evidence_plot(name, clip, frames[i], fl, moving, i / fps, df, ang_hist,
                  row_sum, row_cnt, fps)
    summary = {
        'frames': len(frames), 'fps': fps, 'sample_frame': int(i),
        'background_median_px': float(df.background_median.median()),
        'moving_speed_px': float(df.moving_mean_speed[df.moving_blobs > 0].mean()),
        'max_blobs': int(df.moving_blobs.max()),
        'row_speed': (row_sum / np.maximum(row_cnt, 1)).round(3).tolist(),
        'dominant_dirs_deg': (np.argsort(ang_hist)[::-1][:3] * 360 / N_ANG
                              + 5).tolist(),
    }
    with open(os.path.join(RESULTS, 'summary_%s.json' % name), 'w') as f:
        json.dump(summary, f, indent=2)
    print('  ', json.dumps(summary))


def evidence_plot(name, clip, frame, fl, moving, t, df, ang_hist, row_sum,
                  row_cnt, fps):
    fig = plt.figure(figsize=(12, 8.2))

    ax = fig.add_subplot(2, 2, 1)
    over = F.draw_arrows(frame, fl, step=12, min_mag=clip['moving_thr'],
                         scale=3, color=(0, 255, 255))
    cnts, _ = cv2.findContours(moving.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 60]
    cv2.drawContours(over, cnts, -1, (0, 0, 255), 2)
    ax.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
    ax.set_title('(a) t = %.1f s: moving pixels (red outline, |flow| > %.1f px)\n'
                 'and flow arrows (x3)' % (t, clip['moving_thr']), fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(df.t, df.moving_mean_speed, lw=1, label='mean speed of moving pixels')
    ax.plot(df.t, df.background_median, lw=1, label='median speed of background')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('px / frame')
    ax2 = ax.twinx()
    ax2.plot(df.t, df.moving_blobs, lw=0.8, color='0.6', label='moving regions')
    ax2.set_ylabel('number of moving regions', color='0.4')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('(b) speed over time; background stays at ~0 '
                 '-> the camera is not moving', fontsize=9)

    ax = fig.add_subplot(2, 2, 3, projection='polar')
    th = np.radians(np.arange(N_ANG) * 360 / N_ANG + 5)
    # image y points down; negate so "up the image" is up on the plot
    ax.bar(-th, ang_hist / ang_hist.sum(), width=np.radians(360 / N_ANG),
           color='tab:blue', alpha=0.8)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['right', 'up', 'left', 'down'])
    ax.set_yticklabels([])
    ax.set_title('(c) direction of motion, all moving pixels, whole clip',
                 fontsize=9)

    ax = fig.add_subplot(2, 2, 4)
    H = frame.shape[0]
    centers = (np.arange(N_ROWS) + 0.5) * H / N_ROWS
    speed = row_sum / np.maximum(row_cnt, 1)
    ok = row_cnt > 500
    ax.plot(centers[ok], speed[ok], 'o-')
    ax.set_xlabel('image row y (px)   top of image -> bottom (nearer camera)')
    ax.set_ylabel('mean speed of moving pixels (px / frame)')
    ax.set_title('(d) speed vs position in the image: nearer = faster in pixels',
                 fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('%s  (%d frames, %.0f fps)' % (clip['title'], len(df) + 1, fps),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'evidence_%s.png' % name), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--clip', help='one clip name from data/clips.json')
    a = ap.parse_args()
    clips = load_clips()
    for name, clip in clips.items():
        if a.clip and name != a.clip:
            continue
        process(name, clip)


if __name__ == '__main__':
    main()

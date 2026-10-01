# Modules 5 & 6 — Optical Flow & Structure from Motion

CSc 8830, Computer Vision. Dhanush Nagarajan. (Assignment 6, which combines
Modules 5 and 6.)

## What the assignment asked for

Take 2 videos, at least 30 s each with motion. For each:

- **A.** Compute optical flow and show it as a video. Explain what can be
  inferred from it, with evidence. Derive the motion tracking equations from
  first principles and set up tracking for two frames. Derive bilinear
  interpolation. Take two consecutive frames from each video and check the
  theoretical tracking result against the actual pixel locations.
- **B.** Structure from motion from four views of an object (flat, for
  simplicity). Reconstruct its points so its boundary can be estimated, show
  the maths, and give the images, camera positions and camera parameters.

## What's in this folder

| File | What it does |
|---|---|
| `__init__.py` | The page in the web app (4 tabs) |
| `flow.py` | Part A maths: Farneback flow, flow colouring, camera motion (homography + RANSAC), **my Lucas–Kanade** (pyramid + iterations), **bilinear interpolation**, correlation-based "actual" position |
| `make_flow_videos.py` | Part A: flow video, camera-motion removal and evidence plots for each clip |
| `validate_tracking.py` | Part A: predicted vs actual positions on two consecutive frames, and one point worked through by hand |
| `sfm.py` | Part B: DLT homography → R, t, n (by hand) → triangulation → refinement → scale → boundary |
| `theory.md` | All the derivations (A: flow constraint, LK tracking, bilinear; B: plane homography, decomposition, triangulation) |
| `data/videos/` | The two 30 s clips (my own recordings) |
| `data/clips.json` | Where the clips come from, licence, thresholds |
| `data/sfm/` | The 4 photos used for SfM, and `camera.json` (K and distortion) |
| `results/` | Everything the scripts write (videos, plots, CSVs, worked calculations) |

## Running it

Web app, from the repo root: `streamlit run app.py`, then Module 5 in the sidebar.

Scripts, from this folder:

```
cd modules/module5
python make_flow_videos.py     # A: results/flow_*.mp4, evidence_*.png, stats_*.csv  (~2.5 min)
python validate_tracking.py    # A: results/tracking_*.csv/.png, workings_*.md
python sfm.py                  # B: results/sfm_*.csv/.png/.json, workings_sfm.md
python sfm.py --square-mm 28.5 # B, if a square on the screen measures differently
```

`make_flow_videos.py` uses `ffmpeg` (if installed) to make the videos play in
a browser.

## Part A — results

Both videos are my own, recorded hand-held on my iPhone at a road intersection
(`data/videos/`, cut from `IMG_1893.MOV` and `IMG_1895.MOV`, which stay on my
machine and are git-ignored). **Video 1** (0–30 s): cars crossing both ways,
phone held as still as I could. **Video 2** (5–35 s): held still at first,
then I turn left and later right across the scene. Both were tone-mapped from
HDR to SDR and scaled to 960×540 at 30 fps.

Because the phone was hand-held, every frame's **camera motion** is measured
too (`flow.camera_motion`: a homography fitted with RANSAC to corners across
the frame) and subtracted before deciding what is moving.

What the flow shows, over each whole clip (`evidence_*.png`):

| | Video 1 | Video 2 |
|---|---|---|
| camera motion | 0.10 px/frame median (hand shake) | 0.15 still, then turning left (10–20 s) and right (25–30 s), up to 7.4 px/frame |
| background after removing it | 0.07 px/frame | 0.10 px/frame |
| moving cars: speed | 10.8 px/frame average, up to 28 | 11.3 px/frame average, up to 26 |
| direction | both ways (48 % left, 52 % right) | 91 % right |
| image speed, far → near lane | 10 → 24 px/frame | 6 → 20 px/frame |

Tracking check, two consecutive frames with a typical amount of motion
(`validate_tracking.py`):

| | Video 1 (frames 583→584) | Video 2 (frames 289→290) |
|---|---|---|
| points (reliable) | 38 (29) | 36 (26) |
| how far the cars moved (median) | 10.8 px | 21.3 px |
| median \|predicted − actual\| | **0.08 px** | **0.09 px** |
| reliable points within 0.5 px | 26 of 29 | 22 of 26 |
| median difference from OpenCV's LK | 0.003 px | 0.006 px |

The few points that disagree by several pixels sit on a car's outline, where
the window holds both the moving car and the still road (two motions in one
window, which LK's assumption doesn't allow).

## Part B — results

The object is my monitor screen, which is flat, showing the Module 2
chessboard. It was photographed from 4 positions (IMG_1553, 1554, 1560, 1565
from the Module 2 photo set) with the same calibrated iPhone. The chessboard
corners give the correspondences. The screen's 4 corners are the boundary.

| | |
|---|---|
| reprojection error, 54 grid points × 4 views | **0.12 px** RMS |
| reprojection error, boundary corners | 1.99 px RMS |
| square side recovered (true 30 mm) | 30.00 mm along rows, 29.96 mm down columns |
| all 54 grid points vs the true board | 0.31 mm RMS |
| camera positions vs `solvePnP` | 9.1 mm mean, 23.4 mm worst, at 1.2–1.8 m |
| estimated boundary | **643 × 370 mm**, corners 89.8°, 90.0°, 90.3°, 90.0° |

The scale comes from one length: a square on the screen is taken as 30 mm,
the value from the Module 2 calibration. If it measures differently with a
ruler, re-run with `--square-mm`. All lengths scale with it; the angles,
flatness and aspect ratio do not change.

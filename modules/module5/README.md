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
| `flow.py` | Part A maths: Farneback flow, flow colouring, **my Lucas–Kanade** (pyramid + iterations), **bilinear interpolation**, correlation-based "actual" position |
| `make_flow_videos.py` | Part A: flow video + evidence plots for each clip |
| `validate_tracking.py` | Part A: predicted vs actual positions on two consecutive frames, and one point worked through by hand |
| `sfm.py` | Part B: DLT homography → R, t, n (by hand) → triangulation → refinement → scale → boundary |
| `theory.md` | All the derivations (A: flow constraint, LK tracking, bilinear; B: plane homography, decomposition, triangulation) |
| `data/videos/` | The two 30 s clips |
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

Videos: **Video 1** is OpenCV's sample `vtest.avi`, the first 30 s: a fixed
camera over a campus path, 10 fps. **Video 2** is Intel's `store-aisle-detection.mp4`
(CC BY 4.0), 5–35 s: a fixed camera looking down a shop aisle, 30 fps.

What the flow shows, measured over each whole clip (`evidence_*.png`):

| | Video 1 | Video 2 |
|---|---|---|
| background median speed (camera fixed?) | 0.016 px/frame | 0.002 px/frame |
| mean speed of moving pixels | 4.7 px/frame | 2.0 px/frame |
| main direction of motion | left | up-left (walking away up the aisle) |
| speed, top of image → bottom (far → near) | 1.5 → 6.6 px/frame | 1.0 → 3.0 px/frame |
| camera shake detected | no | yes, at 5.2 s and 14.1–14.4 s |

Tracking check, two consecutive frames with a typical amount of motion
(`validate_tracking.py`):

| | Video 1 (frames 138→139) | Video 2 (frames 800→801) |
|---|---|---|
| points (reliable) | 38 (23) | 38 (38) |
| median \|predicted − actual\| | **0.10 px** | **0.08 px** |
| 90th percentile | 0.18 px | 0.38 px |
| median difference from OpenCV's LK | 0.007 px | 0.008 px |

"Reliable" means the patch still correlates at ≥ 0.9 in the next frame. Video 1
is only 10 fps, so legs and arms change shape between frames. Those points are
listed in the CSV but not counted.

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
| all 54 grid points vs the true board | 0.29 mm RMS |
| camera positions vs `solvePnP` | 5.7 mm mean, 9.3 mm worst, at 1.2–1.8 m |
| estimated boundary | **643 × 370 mm**, corners 89.8°, 90.0°, 90.3°, 90.0° |

The scale comes from one length: a square on the screen is taken as 30 mm,
the value from the Module 2 calibration. If it measures differently with a
ruler, re-run with `--square-mm`. All lengths scale with it; the angles,
flatness and aspect ratio do not change.

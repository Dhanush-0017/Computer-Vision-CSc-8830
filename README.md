# Module 2 — Object Dimension Measurement via Perspective Projection

CSc 8830 (Computer Vision), Module 2. The idea is to take a single photo from
my phone and get a real-world measurement (in mm) out of it, using camera
calibration plus the pinhole projection equations run backwards.

**Repo:** https://github.com/Dhanush-0017/Computer-Vision-CSc-8830

## How it's laid out

The actual deliverable is the Streamlit web app (`src/app.py`), which hosts
this module (and will host future ones) behind one URL, with a tab for each
step. I also kept the three pieces as plain command-line scripts
(`calibrate.py`, `measure.py`, `validate.py`) since that's what I actually
tested with first before wiring them into the UI — they share the same math
through `common.py`, so the two versions can't drift apart.

```
src/
  app.py           # the web app shell, auto-loads modules/moduleN.py
  common.py        # calibration load/save + the pinhole math, shared by everything
  calibrate.py     # Step 1 as a script
  measure.py       # Step 2 as a script
  validate.py      # Step 3 as a script
  selftest.py       # sanity check with synthetic images, run before real photos
  modules/
    module2.py     # this assignment, rendered inside app.py
data/
  calibration_images/   # chessboard photos
  measurement_images/   # photos used for the Step 3 validation
  measurements.csv      # the 20 rows for Step 3
calibration/
  camera_params.npz      # K + distortion, written by Step 1
docs/
  theory.md              # the two-camera derivation, for the PDF
```

## Running it

```
pip install -r requirements.txt
cd src
streamlit run app.py
```
Opens at `http://localhost:8501`. Module 2 is in the sidebar.

## Step 1 — Calibration

Chessboard photos go in `data/calibration_images/`. In the app that's just an
upload box; from the terminal it's:
```
python calibrate.py --images ../data/calibration_images --rows 6 --cols 9 --square 22.0
```
`--rows`/`--cols` are inner corners, not squares. I aimed for reprojection
error under 0.5 px — took a few tries, mostly because photographing a
chessboard off a monitor instead of a printed one introduces moire that
throws off corner detection, and a couple of shots had the board too small in
frame to detect reliably.

## Step 2 — Measurement

Photograph the object from ≥2 m, roughly facing the camera, and note the
distance Z with a tape measure. Then click the two endpoints (in the app) or
pass pixel coordinates directly:
```
python measure.py --calib ../calibration/camera_params.npz --Z 2500 --image ../data/obj.jpg
```
This only holds up if both points are at basically the same depth — if the
object is tilted relative to the camera, the single-Z assumption breaks down
and the estimate gets worse the more it's tilted.

## Step 3 — Validation

`data/measurements.csv` holds 20 rows of `Z_mm, u1, v1, u2, v2, gt_mm`. Run:
```
python validate.py --calib ../calibration/camera_params.npz --csv ../data/measurements.csv --plot
```
Prints MAE, RMSE, MAPE, and signed error stats, plus a measured-vs-ground-truth
plot. My run came out to about 1.6 mm MAE / 2.4% MAPE — width measurements
were consistently more accurate than height/diagonal ones, which I think comes
down to the object sitting well off the image center combined with the lens
distortion not correcting both axes equally at that position (more on this in
the PDF).

## Theory

`docs/theory.md` has the two-camera derivation — relating a point's image
coordinates across two views, ending at the epipolar/fundamental-matrix
relationship.

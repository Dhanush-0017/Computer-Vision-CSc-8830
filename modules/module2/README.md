# Module 2 — Object Dimension Measurement

CSc 8830, Computer Vision. Dhanush Nagarajan.

Take one photo with my phone and get a real-world measurement in millimetres out
of it, using camera calibration plus the pinhole projection equations run
backwards.

## What the assignment asked for

1. Calibrate a camera using OpenCV, with a smartphone camera.
2. Write a script that finds the real-world 2D size of an object using
   perspective projection.
3. Validate it over 20 measurements taken more than 2 metres away, and report
   error statistics.
4. Theory: derive how a 3D point's image coordinates relate between two cameras
   that are at different positions and angles.

## What's in this folder

| File | What it does |
|---|---|
| `__init__.py` | The Module 2 page in the web app — four tabs, one per part of the assignment |
| `common.py` | The maths. Calibration load/save and the pinhole projection, shared by the page and the scripts |
| `calibrate.py` | Step 1 as a command line script |
| `measure.py` | Step 2 as a command line script |
| `validate.py` | Step 3 as a command line script |
| `selftest.py` | Runs the pipeline on generated images with a known camera, to check the code before using real photos |
| `theory.md` | The two-camera derivation, for the PDF |
| `data/calibration_images/` | 18 chessboard photos |
| `data/measurement_images/` | The 4 photos used for validation |
| `data/measurements.csv` | The 20 validation rows |
| `calibration/camera_params.npz` | K and the distortion coefficients, written by step 1 |
| `validation_plot.png`, `validation_results.csv` | Output from `validate.py --plot` |

## Running it

The web app is the main thing — from the repo root:

```
streamlit run app.py
```

Module 2 is in the sidebar. The scripts below assume `cd modules/module2` first.

### Step 1 — calibration

```
python calibrate.py --images data/calibration_images --rows 6 --cols 9 --square 30.0
```

`--rows` and `--cols` are inner corners, not squares. A printed board of 10 by 7
squares has 9 by 6 inner corners.

I was aiming for reprojection error under 0.5 px. It took a few tries — my first
set of photos was of a chessboard on a monitor rather than a printed one, and
photographing a screen creates a moire pattern that stops the corner detection
working. A couple of other shots had the board too small in the frame.

**Result: 18 out of 18 images used, 0.3920 px RMS reprojection error.**
fx = 3023.59, fy = 3028.34, cx = 1532.54, cy = 1999.80

### Step 2 — measurement

```
python measure.py --calib calibration/camera_params.npz --Z 2500 --image data/obj.jpg
```

Give it the distance Z to the object, click the two endpoints, and it converts
the pixel distance into millimetres:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
D = sqrt((X2-X1)^2 + (Y2-Y1)^2)
```

This only works properly if both points are about the same distance from the
camera, so the face being measured has to be roughly facing the camera. If it's
tilted, one Z can't describe both points and the estimate drifts.

### Step 3 — validation

```
python validate.py --calib calibration/camera_params.npz --csv data/measurements.csv --plot
```

I used a MARTA Breeze card, which is a standard ID-1 size, 85.60 by 53.98 mm, so
I have exact ground truth without measuring the card myself. It's taped to a wall
and photographed from 4 distances. From each photo I took 5 measurements — the
width, both height edges and both diagonals — which gives 20 data points.

I didn't have a tape measure for the distance, so I used the card's known width to
solve for Z instead, then used that Z to predict the other four measurements per
photo. Those four weren't used to work out Z, so checking them against ground
truth is still a fair test. The distances came out between 2.11 and 2.54 m, so the
over-2-metre requirement is met.

**Result: MAE 1.64 mm, RMSE 1.93 mm, MAPE 2.37%, over 20 measurements.**

Width measurements came out more accurate than height and diagonal ones, and
always in the same direction. I checked it wasn't a bug in my corner detection by
rewriting that part and getting the same pattern. I think it's because the card
sits well off to one side of the frame rather than near the centre, and the
distortion correction isn't equally good on both axes out there — probably made
worse by the phone not being perfectly square to the wall.

## Theory

`theory.md` has the derivation: two cameras looking at the same 3D point, ending
at the epipolar constraint and the fundamental matrix.

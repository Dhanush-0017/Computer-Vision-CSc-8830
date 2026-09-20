# CSc 8830 — Computer Vision · Assignment Portfolio

Dhanush Nagarajan, Georgia State University.

Every assignment for this course lives in this one repo and is reachable from a
single Streamlit web app, which is what the course asks for. `app.py` is a
shell that scans `modules/` and builds its own navigation from whatever
module folders it finds, so each new assignment drops in as one self-contained
folder without touching the app. Each module keeps its own data, its own
theory doc, and plain command-line scripts alongside its UI page, sharing math
through a small module-local library so the two versions can't drift apart.

**Repo:** https://github.com/Dhanush-0017/Computer-Vision-CSc-8830

## Modules

| # | Topic | Folder |
|---|---|---|
| 2 | Camera calibration & object dimension measurement | `modules/module2/` |
| 3 | Image blurring & the convolution theorem | `modules/module3/` |

## Running the site

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`; pick a module from the sidebar.

## Layout

Each module is a self-contained folder: its page, its own math/helper files,
its own data, and its own theory doc all live together. `app.py` at the root
is the only shared file — it never needs to change when a module is added.

```
app.py                       # the shell, auto-discovers modules/moduleN/
requirements.txt
modules/
  _template.py                # copy this to scaffold a new module
  module2/
    __init__.py                # the page (app.py imports this as modules.module2)
    common.py                  # calibration I/O + pinhole math
    calibrate.py                # Step 1, as a script
    measure.py                  # Step 2, as a script
    validate.py                  # Step 3, as a script
    selftest.py                   # sanity check on synthetic images
    theory.md                     # the two-camera / epipolar derivation
    data/
      calibration_images/          # chessboard photos
      measurement_images/          # Step 3 validation photos
      measurements.csv             # the 20 validation rows
    calibration/
      camera_params.npz            # K + distortion, written by Step 1
  module3/
    __init__.py                 # the page
    filtering.py                  # kernels, spatial + FFT convolution, metrics
    blur.py                       # blurring, as a script
    verify_convolution.py          # self-test, equivalence table, timing sweep
    theory.md                      # the convolution theorem, proved and worked by hand
    data/
      module3_samples/             # sample image
files/
  SUBMISSION.md, SUBMISSION_M3.md  # PDF sources (not tracked -- personal working files)
  VIDEO_M3.md, EXPLAIN.md          # recording scripts (not tracked)
```

---

## Module 2 — Object Dimension Measurement via Perspective Projection

The idea is to take a single photo from my phone and get a real-world
measurement (in mm) out of it, using camera calibration plus the pinhole
projection equations run backwards.

The actual deliverable is the Streamlit web app (`app.py`, Module 2's page at
`modules/module2/__init__.py`). I also kept the three pieces as plain
command-line scripts (`calibrate.py`, `measure.py`, `validate.py`) since
that's what I actually tested with first before wiring them into the UI —
they share the same math through `common.py`, so the two versions can't drift
apart.

### Running it

```
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`. Module 2 is in the sidebar. The scripts
below assume you `cd modules/module2` first.

### Step 1 — Calibration

Chessboard photos go in `data/calibration_images/`. In the app that's just an
upload box; from the terminal it's:
```
cd modules/module2
python calibrate.py --images data/calibration_images --rows 6 --cols 9 --square 30.0
```
`--rows`/`--cols` are inner corners, not squares. I aimed for reprojection
error under 0.5 px — took a few tries, mostly because photographing a
chessboard off a monitor instead of a printed one introduces moire that
throws off corner detection, and a couple of shots had the board too small in
frame to detect reliably.

### Step 2 — Measurement

Photograph the object from ≥2 m, roughly facing the camera, and note the
distance Z with a tape measure. Then click the two endpoints (in the app) or
pass pixel coordinates directly:
```
python measure.py --calib calibration/camera_params.npz --Z 2500 --image data/obj.jpg
```
This only holds up if both points are at basically the same depth — if the
object is tilted relative to the camera, the single-Z assumption breaks down
and the estimate gets worse the more it's tilted.

### Step 3 — Validation

`data/measurements.csv` holds 20 rows of `Z_mm, u1, v1, u2, v2, gt_mm`. Run:
```
python validate.py --calib calibration/camera_params.npz --csv data/measurements.csv --plot
```
Prints MAE, RMSE, MAPE, and signed error stats, plus a measured-vs-ground-truth
plot. My run came out to about 1.6 mm MAE / 2.4% MAPE — width measurements
were consistently more accurate than height/diagonal ones, which I think comes
down to the object sitting well off the image center combined with the lens
distortion not correcting both axes equally at that position (more on this in
the PDF).

### Theory

`modules/module2/theory.md` has the two-camera derivation — relating a
point's image coordinates across two views, ending at the
epipolar/fundamental-matrix relationship.

---

## Module 3 — Image Blurring & the Convolution Theorem

Blur an image two ways — by sliding a kernel over it, and by multiplying its
spectrum by the kernel's spectrum — and show that the results are the same
image, not merely similar ones.

### What's implemented

Box (mean) and Gaussian kernels, both normalised to sum to 1, applied by an
explicit 2D convolution written out in `filtering.py`. No `cv2.blur`,
`cv2.GaussianBlur` or `cv2.filter2D` produces any result in this module —
OpenCV appears once, in `reference_opencv()`, purely as a cross-check that my
convolution is right. Four boundary rules are selectable (`wrap`, `zero`,
`reflect`, `replicate`), and the Gaussian can also run separably as two 1D
passes.

The Fourier side has both variants: `convolve_fft_circular()` (the plain DFT
product, which is circular convolution) and `convolve_fft_linear()`
(zero-padded to N+k−1, which gives linear convolution).

### Running it

```
cd modules/module3

# blur something
python blur.py --image data/module3_samples/sample_photo.jpg \
               --filter gaussian --sigma 3 --out blurred.png

# blur it both ways and print how far apart they are
python blur.py --image data/module3_samples/sample_photo.jpg \
               --filter gaussian --sigma 3 --compare

# the evidence: 21 assertions, the equivalence table, the timing sweep
python verify_convolution.py --all
```

`--all` writes `results_equivalence.csv`, `results_timing.csv` and
`timing_plot.png` into `modules/module3/`. Everything in the PDF is
reproducible from it.

### What came out

Across box and Gaussian kernels from 3×3 to 31×31 on a 384×512 image, the
spatial and Fourier routes agreed to a maximum of **8.8e-13** grey levels on a
0–255 scale — PSNR above 300 dB. That's round-off in double precision, not
disagreement.

The thing that took the longest to get right was the kernel centring. The DFT
treats index 0 as the origin, but a blur kernel's origin is its middle tap, so
the kernel has to be rolled by −(k//2) before transforming. Miss that and the
result is correct but *shifted* diagonally by (k−1)/2 pixels — which is easy to
stare past, because a shifted blur still looks like a blur.

The other subtlety is genuinely interesting rather than a bug. The DFT assumes
the image is periodic, so it computes circular convolution; a spatial filter that
mirrors or zero-fills at the border is computing something different there. Match
the conventions and the two agree everywhere. Mismatch them on purpose and the
difference is 70 grey levels at the border and **5.7e-13 across the entire
interior** — confined to a strip of exactly (k−1)/2 pixels, which is the distance
the kernel overhangs the edge. That's a sharper demonstration of the theorem than
the matched case, so the app shows both.

On cost: the FFT route's time is flat in kernel size (0.0023–0.0025 s across the
whole sweep) while naive spatial convolution grows like k², with the crossover at
**k = 7** on a 384×384 image. Separable filtering stays competitive to 31×31,
which is why it, not the FFT, is what production code usually reaches for at
ordinary blur radii.

### Theory

`modules/module3/theory.md` has the convolution theorem proved in both the
continuous and discrete cases, the circular-vs-linear discussion, and a
hand-worked 8-sample example with the full DFT tables — the same one tab 5 of
the app recomputes live.

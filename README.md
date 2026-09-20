# CSc 8830 — Computer Vision · Assignment Portfolio

Dhanush Nagarajan, Georgia State University.

Every assignment for this course lives in this one repo and is reachable from a
single Streamlit web app, which is what the course asks for. `src/app.py` is a
shell that scans `src/modules/` and builds its own navigation from whatever
`moduleN.py` files it finds, so each new assignment drops in as one file without
touching the app. Each module also keeps plain command-line scripts alongside the
UI, sharing the same math through a common library, so the two versions can't
drift apart.

**Repo:** https://github.com/Dhanush-0017/Computer-Vision-CSc-8830

## Modules

| # | Topic | Page | Implementation | Theory |
|---|---|---|---|---|
| 2 | Camera calibration & object dimension measurement | `src/modules/module2.py` | `common.py`, `calibrate.py`, `measure.py`, `validate.py` | `docs/theory.md` |
| 3 | Image blurring & the convolution theorem | `src/modules/module3.py` | `filtering.py`, `blur.py`, `verify_convolution.py` | `docs/theory_module3.md` |

## Running the site

```
pip install -r requirements.txt
cd src
streamlit run app.py
```

Opens at `http://localhost:8501`; pick a module from the sidebar.

## Layout

```
src/
  app.py                  # the shell, auto-discovers modules/moduleN.py
  modules/
    module2.py            # Module 2 page
    module3.py            # Module 3 page
  common.py               # Module 2: calibration I/O + pinhole math
  calibrate.py            # Module 2, step 1, as a script
  measure.py              # Module 2, step 2, as a script
  validate.py             # Module 2, step 3, as a script
  selftest.py             # Module 2 sanity check on synthetic images
  filtering.py            # Module 3: kernels, spatial + FFT convolution, metrics
  blur.py                 # Module 3 blurring, as a script
  verify_convolution.py   # Module 3 self-test, equivalence experiment, timing sweep
data/
  calibration_images/     # Module 2 chessboard photos
  measurement_images/     # Module 2 validation photos
  measurements.csv        # Module 2, the 20 validation rows
  module3_samples/        # Module 3 sample image
calibration/
  camera_params.npz       # K + distortion, written by Module 2 step 1
docs/
  theory.md               # Module 2: the two-camera / epipolar derivation
  theory_module3.md       # Module 3: the convolution theorem, proved and worked by hand
files/
  SUBMISSION.md           # Module 2 PDF source
  SUBMISSION_M3.md        # Module 3 PDF source
  VIDEO_M3.md             # Module 3 recording script
```

---

## Module 2 — Object Dimension Measurement via Perspective Projection

CSc 8830 (Computer Vision), Module 2. The idea is to take a single photo from
my phone and get a real-world measurement (in mm) out of it, using camera
calibration plus the pinhole projection equations run backwards.

### How it's laid out

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

### Running it

```
pip install -r requirements.txt
cd src
streamlit run app.py
```
Opens at `http://localhost:8501`. Module 2 is in the sidebar.

### Step 1 — Calibration

Chessboard photos go in `data/calibration_images/`. In the app that's just an
upload box; from the terminal it's:
```
python calibrate.py --images ../data/calibration_images --rows 6 --cols 9 --square 30.0
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
python measure.py --calib ../calibration/camera_params.npz --Z 2500 --image ../data/obj.jpg
```
This only holds up if both points are at basically the same depth — if the
object is tilted relative to the camera, the single-Z assumption breaks down
and the estimate gets worse the more it's tilted.

### Step 3 — Validation

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

### Theory

`docs/theory.md` has the two-camera derivation — relating a point's image
coordinates across two views, ending at the epipolar/fundamental-matrix
relationship.

---

## Module 3 — Image Blurring & the Convolution Theorem

Blur an image two ways — by sliding a kernel over it, and by multiplying its
spectrum by the kernel's spectrum — and show that the results are the same image,
not merely similar ones.

### What's implemented

Box (mean) and Gaussian kernels, both normalised to sum to 1, applied by an
explicit 2D convolution written out in `filtering.py`. No `cv2.blur`,
`cv2.GaussianBlur` or `cv2.filter2D` produces any result in this module —
OpenCV appears once, in `reference_opencv()`, purely as a cross-check that my
convolution is right. Four boundary rules are selectable (`wrap`, `zero`,
`reflect`, `replicate`), and the Gaussian can also run separably as two 1D
passes.

The Fourier side has both variants: `convolve_fft_circular()` (the plain DFT
product, which is circular convolution) and `convolve_fft_linear()` (zero-padded
to N+k−1, which gives linear convolution).

### Running it

```
cd src

# blur something
python blur.py --image ../data/module3_samples/sample_photo.jpg \
               --filter gaussian --sigma 3 --out blurred.png

# blur it both ways and print how far apart they are
python blur.py --image ../data/module3_samples/sample_photo.jpg \
               --filter gaussian --sigma 3 --compare

# the evidence: 21 assertions, the equivalence table, the timing sweep
python verify_convolution.py --all
```

`--all` writes `results_equivalence.csv`, `results_timing.csv` and
`timing_plot.png` into `src/`. Everything in the PDF is reproducible from it.

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

On cost: the FFT route's time is flat in kernel size (0.0026–0.0027 s across the
whole sweep) while naive spatial convolution grows like k², with the crossover at
**k = 7** on a 384×384 image. Separable filtering stays competitive to 31×31,
which is why it, not the FFT, is what production code usually reaches for at
ordinary blur radii.

### Theory

`docs/theory_module3.md` has the convolution theorem proved in both the
continuous and discrete cases, the circular-vs-linear discussion, and a
hand-worked 8-sample example with the full DFT tables — the same one tab 5 of the
app recomputes live.


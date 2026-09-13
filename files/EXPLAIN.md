# Explaining the Code — Narration Cheat Sheet

Read this once before recording. You do not need to memorise it; keep it on a
second screen. Aim for 3-5 minutes of video total.

---

## The one-sentence summary
"I calibrate my phone camera to find its focal length, then use the pinhole
projection equations in reverse to convert pixel distances into real-world
millimetres, and I validate that over 20 measurements."

---

## 1. `calibrate.py` / Step 1 tab — WHY calibration

**What it does:** finds the camera's intrinsic matrix K and distortion coefficients.

**Say this:**
"A camera projects the 3D world onto a 2D sensor. To reverse that projection I
need to know the camera's internal parameters — mainly the focal length in
pixels. Calibration recovers those by photographing an object whose real
geometry I already know."

**Why a chessboard:**
"The chessboard gives me free ground truth. I tell OpenCV the squares are 22 mm,
so it knows the true 3D position of every corner without me measuring any of
them. And the corners are high-contrast X junctions, so `cornerSubPix` can
locate them to a fraction of a pixel."

**Why many photos at angles:**
"Each photo gives a set of 2D-to-3D correspondences. Shooting from varied angles
makes the system well-conditioned — if every shot were head-on, focal length and
distance would be mathematically indistinguishable."

**Key lines:**
- `objp` = the known 3D corner positions on the board (Z=0 plane)
- `findChessboardCorners` = locate them in the image
- `cornerSubPix` = refine to sub-pixel precision
- `calibrateCamera` = least-squares solve for K + distortion

**The output K:**
```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
```
"fx, fy are focal length in pixels. cx, cy are the principal point — where the
optical axis hits the sensor, roughly the image centre."

**Reprojection error:** "I take the recovered parameters, re-project the known
3D corners back into the image, and measure how far they land from the detected
corners. Under half a pixel means the calibration is good."

---

## 2. `measure.py` / Step 2 tab — THE ACTUAL MEASUREMENT

**The pinhole equations (forward — world to image):**
```
u = fx · X/Z + cx
v = fy · Y/Z + cy
```

**Say this:** "This is just similar triangles. An object's image size shrinks in
proportion to its distance. If I know two of the three — image size, real size,
distance — I can solve for the third."

**Rearranged (backward — image to world), which is what the code does:**
```
X = (u - cx) · Z / fx
Y = (v - cy) · Z / fy
D = sqrt( (X2-X1)² + (Y2-Y1)² )
```

**Say this:** "I measure Z with a tape measure, so it's known. Then each clicked
pixel back-projects to a real 3D point at that depth, and the distance between
the two points is the object's real dimension."

**Why subtract cx, cy first:** "Pixel coordinates are measured from the image
corner, but the projection maths is defined from the optical axis. Subtracting
the principal point moves the origin to the right place."

**Why `undistortPoints`:** "The lens bends straight lines slightly, most visibly
near the edges. This removes that distortion before I do the geometry, using
the coefficients from calibration."

**The key assumption — say this unprompted, it shows you understand the limits:**
"This assumes the measured face is fronto-parallel — perpendicular to the
optical axis — so both endpoints are at the same depth Z. If the object is
slanted, the two points are at different depths and a single Z is no longer
valid."

---

## 3. `validate.py` / Step 3 tab — THE EXPERIMENT

**Say this:** "For each of 20 measurements I record the tape-measured distance Z,
the two pixel points, and the tape-measured true length. The script estimates
each length and compares against ground truth."

**The statistics:**
- **MAE** — mean absolute error, average magnitude of error in mm
- **RMSE** — root mean square error, penalises large errors more heavily
- **MAPE** — mean absolute percentage error, scale-independent
- **Mean signed error** — if clearly non-zero, there's a systematic bias
  (usually a slightly wrong Z), not just random noise
- **Std of error** — the spread / repeatability

**Error vs distance plot:** "Error grows with distance, because at larger Z each
pixel subtends more millimetres — so the same one-pixel click error translates
into a bigger physical error."

**Main error sources — naming these earns credit:**
1. Tape-measuring Z (biggest one; a 1% error in Z gives a 1% error in size)
2. Clicking the endpoints by hand (±2-3 px)
3. Object not perfectly fronto-parallel
4. Residual calibration error

---

## 4. Theory tab — TWO CAMERAS

**The question:** relate the image of point P in camera 1 to its image in camera 2.

**The chain, in order:**
1. Camera 1's frame is the world frame (free choice).
2. A point maps between frames by `X2 = R·X1 + t` — R is the oblique rotation,
   t the offset.
3. Those three vectors `X2`, `t`, `R·X1` are **coplanar**, so the scalar triple
   product is zero: `X2ᵀ [t]× R X1 = 0`
4. `E = [t]× R` is the **essential matrix**.
5. Convert to pixels with the intrinsics: `F = K2⁻ᵀ [t]× R K1⁻¹`
6. Result: **`p̃2ᵀ F p̃1 = 0`**

**The insight that matters most — say this:**
"A point in image 1 maps to a *line* in image 2, not a point. That's because one
pixel only fixes the direction of the ray to P, not its depth. Every point along
that ray looks identical in camera 1 but projects to different places in camera
2 — and that set of places is the epipolar line."

**If asked how to get R and t:** "Match at least 8 correspondences, run the
8-point algorithm for F, convert to E with the intrinsics, then decompose E via
SVD with `recoverPose`. That gives R and t, but t only up to scale — one known
physical length fixes the absolute scale."

---

## 5. The app architecture (if asked)

"`app.py` scans the `modules/` folder and auto-discovers any file named
moduleN.py, reads its metadata, and builds the navigation. So adding a future
assignment means dropping in one file — the shell never changes. Shared camera
helpers live in `common.py` so later modules can reuse this calibration."

---

## Questions you might get, with short answers

**"Why not just use a reference object of known size in the frame?"**
"That works and needs no calibration, but it requires a known object in every
shot. Calibrating once lets me measure anything, as long as I know Z."

**"What if you don't know Z?"**
"Then a single image is fundamentally ambiguous — a small near object and a
large far one produce identical images. You'd need a second view, a depth
sensor, or a known reference in the scene."

**"Why is fx different from fy?"**
"They'd be identical for perfectly square pixels. Small differences reflect
non-square sensor pixels or the calibration's numerical tolerance."

**"What limits your accuracy most?"**
"Measuring Z by hand. The relationship is linear, so a 1% error in Z gives a 1%
error in every dimension at that distance."

---

# Tying it to the lecture slides (use HIS notation in your video)

## Slide "Pixel coordinates" — what fx really is
His K is factored into two parts:
```
K = [ mx        ] [ f      px ]   =   [ αx      βx ]
    [     my    ] [    f   py ]       [     αy  βy ]
    [         1 ] [        1  ]       [         1  ]
     pixels/m        metres              pixels
```
So **fx = αx = mx · f** — focal length in metres times pixels-per-metre.

**Say this:** "The focal length OpenCV reports isn't in millimetres, it's in
pixels. It's the physical focal length multiplied by the sensor's pixel density.
That's why I never need to know my phone's sensor size — the pixel units absorb
it, and it's exactly what I need since I measure in pixels."

**And:** "fx and fy differ slightly because mx and my differ — pixels aren't
perfectly square. cx, cy are his px, py: the principal point."

## Slide "Camera calibration: x = K[R t]X"
That's the full projection. Our Step 2 is the **special case where the object
plane is fronto-parallel**: R = I and the plane sits at constant Z, which
collapses `x = K[R|t]X` down to `u = fx·X/Z + cx`. Worth saying — it shows you
know where your simplification comes from.

## Slides "Direct linear calibration" — why we don't use it
His listed disadvantages are exactly why `cv2.calibrateCamera` is the right call:
- doesn't tell you the camera parameters directly
- **doesn't model radial distortion**
- can't impose constraints
- doesn't minimise the right error function

**Say this:** "The direct linear method solves for the projection matrix P in one
least-squares step, but it can't model radial distortion and it minimises an
algebraic error rather than geometric reprojection error. As his slides say,
non-linear methods are preferred — `calibrateCamera` uses a DLT initialisation
refined by Levenberg-Marquardt, minimising true reprojection error in pixels."

That's exactly the error my app reports as the reprojection error.

## Slide "Alternative: multi-plane calibration" — THIS IS OUR METHOD
This slide is literally what we do. His stated advantages:
- only requires a plane  ✔ our printed chessboard
- don't have to know positions/orientations  ✔ we just wave the board around
- good code available — he names **OpenCV** and **Zhang's** method  ✔

**Say this:** "This is Zhang's multi-plane calibration, which is what
`cv2.calibrateCamera` implements. I only need a planar target, and I don't have
to know the pose of the board in any shot — the algorithm recovers a homography
per view and solves for the intrinsics from those."

## Slide "degenerate solutions for coplanar points"
His note that coplanar points give degenerate solutions is **exactly why you
must shoot from many different angles.**

**Say this:** "A single planar view is degenerate — you can't separate focal
length from distance from one flat board head-on. Each additional orientation
adds independent constraints, which is why I took 20 photos at varied angles
rather than 20 similar ones."

That single sentence answers the most likely question he'll ask.

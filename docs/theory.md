# Theory: Relating a 3D Point's Image Coordinates Across Two Cameras

**Problem.** Camera 1 is static; camera 2 is placed at a certain distance and
at an oblique orientation from camera 1. A 3D point **P = (X, Y, Z)** lies in
the field of view of both cameras. Derive the mathematical relationship between
the image coordinates of P in camera 1 and camera 2.

---

## Assumptions (and justification)

1. **Pinhole camera model**, lens distortion removed via calibration. This lets
   projection be written as a single linear map in homogeneous coordinates.
2. **Both cameras are calibrated**, so the intrinsic matrices `K1`, `K2` are
   known. If the same phone is used for both views, `K1 = K2 = K`.
3. **Rigid relative pose (R, t)** between the two cameras, constant during
   capture. "Oblique orientation" is captured entirely by the rotation `R`;
   "a certain distance" by the translation `t`.
4. **Camera 1's frame is chosen as the world frame.** This is a free choice and
   removes one redundant transform without loss of generality.
5. **P has positive depth in both cameras and is within both FoVs**, as stated.

Notation: `p̃ = (u, v, 1)^T` is a homogeneous pixel; `X1 = (X, Y, Z)^T` is P in
camera 1's frame; `[t]×` is the 3×3 skew-symmetric matrix of `t`.

---

## 1. Projection in each camera

With camera 1's frame as the world frame, its projection matrix is
`P1 = K1 [ I | 0 ]`, so

```
λ1 · p̃1 = K1 · X1                      (camera 1)
```

Camera 2 is obtained by rotating and translating camera 1's frame. A point maps
from camera 1's frame to camera 2's frame by `X2 = R·X1 + t`, giving
`P2 = K2 [ R | t ]`:

```
λ2 · p̃2 = K2 · (R·X1 + t)              (camera 2)
```

Define **normalized (calibration-free) coordinates** by pre-multiplying with the
inverse intrinsics:

```
x̂1 = K1⁻¹ · p̃1  ∝  X1
x̂2 = K2⁻¹ · p̃2  ∝  R·X1 + t = X2
```

---

## 2. Epipolar (coplanarity) constraint

The three vectors `X2`, `t`, and `R·X1` satisfy `X2 = R·X1 + t`, so they are
**coplanar**. Coplanarity is equivalent to a zero scalar triple product:

```
X2 · ( t × R·X1 ) = 0
```

Writing the cross product as a matrix (`t × v = [t]× · v`):

```
X2ᵀ · [t]× · R · X1 = 0
        └─────┬─────┘
              E   (Essential matrix,  E = [t]× R)
```

so **X2ᵀ · E · X1 = 0**. Substituting `X1 ∝ K1⁻¹ p̃1` and `X2 ∝ K2⁻¹ p̃2`:

```
p̃2ᵀ · ( K2⁻ᵀ · [t]× · R · K1⁻¹ ) · p̃1 = 0
        └────────────┬────────────┘
                     F   (Fundamental matrix)
```

### Result

```
p̃2ᵀ · F · p̃1 = 0 ,      F = K2⁻ᵀ [t]× R K1⁻¹
```

This is the exact relationship between the two image points: the image of P in
camera 2 must lie on the **epipolar line** `l2 = F · p̃1`, and its image in
camera 1 lies on `l1 = Fᵀ · p̃2`.

where

```
        ⎡  0   -t_z   t_y ⎤
[t]× =  ⎢ t_z    0   -t_x ⎥
        ⎣-t_y   t_x    0  ⎦
```

**Why a line, not a point?** A single image point in camera 1 fixes only the
*direction* of the ray to P, not its depth. All points along that ray project to
one point in camera 1 but to a whole line in camera 2 — the epipolar line.

---

## 3. Point-to-point relationship when depth is known

If the depth `Z` (equivalently the scale `λ1`) is known, P is fully determined
and its image in camera 2 is a single point:

```
X1 = λ1 · K1⁻¹ · p̃1
λ2 · p̃2 = K2 · ( R · X1 + t )
```

Dividing the right-hand side by its third component recovers `(u2, v2)`
explicitly. Eliminating the unknown depth from this pair of equations reproduces
exactly the epipolar line of Section 2.

---

## 4. Static parameters, variables, and how to obtain them

| Symbol | Meaning | How determined |
|--------|---------|----------------|
| `K1, K2` | intrinsic matrices | Step-1 calibration (`cv2.calibrateCamera`); equal if same phone |
| `R` | relative rotation (3×3) | measured mounting angles, or estimated from correspondences |
| `t` | relative translation / baseline | measured with a ruler, or estimated up to scale |
| `[t]×` | skew matrix of `t` | built directly from `t` |
| `E = [t]× R` | essential matrix | from `R, t`, or from calibrated correspondences |
| `F = K2⁻ᵀ E K1⁻¹` | fundamental matrix | from `E`, or the 8-point algorithm |
| `P = (X, Y, Z)` | the 3D scene point | triangulated from both views |

**If R and t are unknown**, estimate them from image data:

1. Detect and match **≥ 8 point correspondences** between the two images
   (e.g. ORB/SIFT features).
2. `F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)` — the 8-point
   algorithm.
3. `E = K2ᵀ · F · K1` (or `cv2.findEssentialMat` directly with K).
4. `cv2.recoverPose(E, pts1, pts2, K)` decomposes `E` via SVD into `R` and a
   **unit** translation `t`. Absolute scale of `t` is not recoverable from
   images alone (monocular scale ambiguity) — resolve it with one known
   physical length or baseline.
5. With `K, R, t` known, recover P by triangulation
   (`cv2.triangulatePoints`).

**Scale ambiguity note.** Two-view reconstruction from a single moving/second
camera is only determined *up to a global scale*. A single known real-world
measurement fixes that scale; everything else then follows.

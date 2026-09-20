# Theory: relating a 3D point's image across two cameras

**The setup.** Camera 1 stays put. Camera 2 is somewhere else — some distance
away and pointed at a different angle ("oblique orientation" in the prompt).
There's a 3D point P = (X, Y, Z) that both cameras can see. I want the
mathematical relationship between where P shows up in image 1 and where it
shows up in image 2.

I worked this the same way it's usually done in two-view geometry (this is
essentially deriving the epipolar constraint from scratch), so a lot of this
will look familiar from the multi-view lecture slides — I've tried to connect
it back to that notation where it's relevant.

## Assumptions I'm making, and why

1. **Pinhole camera, distortion already corrected.** Both cameras have been
   calibrated the way I did in Step 1, so I can treat projection as a clean
   linear map and not worry about lens distortion messing up the algebra.
2. **Both cameras are calibrated** — I know `K1` and `K2`. If it's literally
   the same phone taking both shots, `K1 = K2`.
3. **The relative pose between the cameras, `(R, t)`, is fixed** during the
   time both photos are taken. `R` is the "oblique orientation" from the
   prompt, `t` is the offset/baseline.
4. **I put camera 1's frame as the world frame.** No physical meaning to
   this, it's just a free choice that removes one transform from the algebra
   without losing anything.
5. **P is in front of both cameras** (positive depth), which is already
   stated in the problem.

Notation: `p̃ = (u, v, 1)ᵀ` is a pixel in homogeneous coordinates. `X1` is P
written in camera 1's frame. `[t]×` is the skew-symmetric matrix built from
`t`, defined further down.

## 1. Writing down the projection for each camera

Since camera 1's frame *is* the world frame, its projection is the plain
pinhole equation:

```
λ1 · p̃1 = K1 · X1
```

Camera 2 is just camera 1's frame rotated and shifted. A point moves between
the two frames as `X2 = R·X1 + t`, so camera 2's projection is:

```
λ2 · p̃2 = K2 · (R·X1 + t)
```

It's convenient to divide out the intrinsics and work with "normalized"
coordinates — i.e. undo the calibration and just look at the ray direction:

```
x̂1 = K1⁻¹ p̃1  ∝  X1
x̂2 = K2⁻¹ p̃2  ∝  X2 = R·X1 + t
```

## 2. Getting to the epipolar constraint

Here's the key geometric fact: the three vectors `X2`, `t`, and `R·X1` all lie
in the same plane, because `X2 - t = R·X1` — that's literally the equation
from step 1 rearranged. Three coplanar vectors means their scalar triple
product is zero:

```
X2 · (t × R·X1) = 0
```

Cross product with a fixed vector is just a linear map, so `t × v = [t]× v`
where

```
        ⎡  0   -t_z   t_y ⎤
[t]× =  ⎢ t_z    0   -t_x ⎥
        ⎣-t_y   t_x    0  ⎦
```

Substituting that in:

```
X2ᵀ · [t]× · R · X1 = 0
```

Call `E = [t]× R` — this is the **essential matrix**. So `X2ᵀ E X1 = 0`. Now
swap in the normalized-coordinate expressions from step 1 (`X1 ∝ K1⁻¹p̃1`,
`X2 ∝ K2⁻¹p̃2`):

```
p̃2ᵀ (K2⁻ᵀ [t]× R K1⁻¹) p̃1 = 0
```

The thing in parentheses is the **fundamental matrix**:

```
F = K2⁻ᵀ [t]× R K1⁻¹
```

**So the answer to "how do the two image points relate" is:**

```
p̃2ᵀ · F · p̃1 = 0
```

That's the whole relationship. Given `p̃1`, this says `p̃2` has to lie on the
line `l2 = F·p̃1` — the epipolar line. Same thing the other way with
`l1 = Fᵀ·p̃2`.

**Why a line and not a single point?** One pixel in camera 1 only tells you
the *direction* of the ray from the camera through P — not how far along
that ray P actually is. Every possible depth gives a different 3D point, all
of which project to the exact same pixel in camera 1, but to different
pixels in camera 2 as you slide along that ray. The set of all those
possible camera-2 pixels traces out the epipolar line. So one image alone is
ambiguous about depth — you need the second view (or a known depth, see
below) to pin P down.

## 3. If I actually know the depth

Everything above holds even without knowing Z. But if the depth *is* known
(say from a rangefinder, or because I fixed Z the way I did in Step 2), then
`λ1` is known and P is fully pinned down — not just a ray:

```
X1 = λ1 K1⁻¹ p̃1
λ2 p̃2 = K2 (R·X1 + t)
```

Dividing the right side by its own third component gives the exact pixel
`(u2, v2)` in camera 2 — a single point, not a line. If you eliminate the
now-known depth from these two equations, you get back exactly the epipolar
line from part 2, which makes sense: knowing depth is what's collapsing that
line down to one point.

## 4. Where each parameter actually comes from

| Symbol | What it is | How I'd get it |
|---|---|---|
| K1, K2 | intrinsics | Step 1 calibration for each camera (same matrix if it's the same phone) |
| R | relative rotation | measured mounting angle, or estimated from image correspondences |
| t | relative translation | measured baseline with a ruler, or estimated (up to scale) |
| E = [t]×R | essential matrix | built from R, t once you have them |
| F = K2⁻ᵀEK1⁻¹ | fundamental matrix | from E and the intrinsics, or straight from image correspondences via the 8-point algorithm |
| P = (X,Y,Z) | the actual 3D point | triangulated once you have both views |

**If R and t aren't known ahead of time**, they can be recovered from the
images themselves:
1. Find at least 8 matching points between the two photos (ORB or SIFT
   features, matched between images).
2. `cv2.findFundamentalMat(...)` — the 8-point algorithm — gets F directly
   from the correspondences.
3. `E = K2ᵀ F K1` (or just call `cv2.findEssentialMat` with K baked in).
4. `cv2.recoverPose(E, pts1, pts2, K)` decomposes E via SVD into R and a
   **unit-length** t.
5. That unit t is the catch — from images alone, translation is only
   recoverable up to an unknown scale (this is the classic monocular scale
   ambiguity — a scene could be twice as big and twice as far away and
   produce identical images). One known real-world length anywhere in the
   scene fixes that scale, and then `cv2.triangulatePoints` gives the actual
   P.

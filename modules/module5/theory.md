# Modules 5 & 6 — Theory

## Part A — Optical flow and tracking

### A1. The brightness constancy assumption

Let $I(x, y, t)$ be the image brightness at pixel $(x, y)$ at time $t$. A point
on a moving object is at $(x, y)$ at time $t$ and at $(x + \delta x, y + \delta y)$
at time $t + \delta t$. The one assumption everything rests on is that **the point
looks the same in both frames**:

$$
I(x + \delta x,\; y + \delta y,\; t + \delta t) = I(x, y, t)
$$

Expand the left side as a first-order Taylor series about $(x, y, t)$:

$$
I(x, y, t) + \frac{\partial I}{\partial x}\delta x + \frac{\partial I}{\partial y}\delta y
+ \frac{\partial I}{\partial t}\delta t + \text{(higher order)} = I(x, y, t)
$$

Cancel $I(x, y, t)$, drop the higher-order terms, divide by $\delta t$ and let
$\delta t \to 0$. With $u = dx/dt$, $v = dy/dt$ (the optical flow) and the
shorthand $I_x = \partial I/\partial x$ etc.:

$$
\boxed{\,I_x u + I_y v + I_t = 0\,}
\qquad\text{or}\qquad \nabla I \cdot \mathbf{u} = -I_t
$$

This is the **optical flow constraint equation**. Between two video frames
$\delta t = 1$ frame, so $(u, v)$ is in pixels per frame.

**The aperture problem.** It is one equation in two unknowns. It only fixes the
component of motion along the gradient, $u_\perp = -I_t / |\nabla I|$; motion
along an edge is invisible. A second assumption is needed to get $(u, v)$. Lucas
and Kanade [1] assume the flow is constant over a small window; Horn and Schunck
[2] assume it varies smoothly over the whole image; Farneback [3] (used for the
dense flow videos) fits a quadratic polynomial to each neighbourhood and solves
for the shift between the two polynomials.

### A2. What optical flow tells us

For a static camera and a scene point at depth $Z$ moving with velocity
$(\dot X, \dot Y, \dot Z)$, perspective projection $x = fX/Z$ gives

$$
u = \frac{f\dot X - x\dot Z}{Z}, \qquad v = \frac{f\dot Y - y\dot Z}{Z}
$$

so:

1. **Where things move** — the flow is zero on the static background and
   non-zero on moving objects. Thresholding $|\mathbf u|$ segments the moving
   objects without any model of what they are.
2. **Which way they move** — the direction of $\mathbf u$ is the direction of
   motion in the image.
3. **How fast, relative to depth** — the same real speed gives image speed
   $\propto 1/Z$. Nearer objects move faster in pixels (motion parallax).
4. **Coming or going** — the $\dot Z$ term makes an approaching object expand
   (flow points outward from a point, positive divergence) and a receding object
   contract toward a point.
5. **Whether the camera moves** — camera motion puts flow on *every* pixel,
   including the static background. A background with zero flow means a fixed camera.
6. **Where the assumption breaks** — brightness constancy fails when lighting
   changes, so a lighting flicker shows up as flow on everything at once.

The evidence for each one, in the two videos, is in the report and on the
web page.

### A3. Tracking a point between two frames — setting up the problem

Given frame 1 $I_1$, frame 2 $I_2$ and a point $\mathbf x = (x, y)$ in frame 1, find
the displacement $\mathbf d = (d_x, d_y)$ so that the window $W$ around $\mathbf x$ in
frame 1 matches the window around $\mathbf x + \mathbf d$ in frame 2. Write
it as a least-squares problem:

$$
E(\mathbf d) = \sum_{\mathbf p \in W} \big[\, I_2(\mathbf p + \mathbf d) - I_1(\mathbf p) \,\big]^2
$$

**Linearise.** For small $\mathbf d$, $I_2(\mathbf p + \mathbf d) \approx I_2(\mathbf p) + \nabla I(\mathbf p)^T\mathbf d$.
With $I_t(\mathbf p) = I_2(\mathbf p) - I_1(\mathbf p)$:

$$
E(\mathbf d) \approx \sum_{W} \big[\, \nabla I^T \mathbf d + I_t \,\big]^2
$$

This is the constraint equation from A1, one per pixel of the window, solved in
the least-squares sense.

**Minimise.** Set $\partial E / \partial \mathbf d = 0$:

$$
\sum_W 2\,\nabla I\,\big(\nabla I^T \mathbf d + I_t\big) = 0
\;\;\Longrightarrow\;\;
\underbrace{\begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}}_{G}
\mathbf d
= \underbrace{-\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}}_{\mathbf b}
$$

$$
\boxed{\,\mathbf d = G^{-1}\mathbf b\,}
$$

These are the **Lucas–Kanade tracking equations** [1]. Equivalently, stacking the
$n$ pixels of the window, $A\mathbf d = -\mathbf i_t$ with $A = [I_x\ I_y]$ ($n \times 2$),
and $\mathbf d = (A^TA)^{-1}A^T(-\mathbf i_t)$, the normal equations.

**When it can be solved.** $G$ is the structure tensor of the window. Its
eigenvalues $\lambda_1 \ge \lambda_2$ say what the window looks like:

| $\lambda_1$, $\lambda_2$ | window | tracking |
|---|---|---|
| both small | flat | impossible, $G$ singular |
| $\lambda_1 \gg \lambda_2 \approx 0$ | edge | aperture problem, only the normal component |
| both large | corner / texture | well-posed |

So points are chosen where $\lambda_{\min}$ is large. That is the Shi–Tomasi
criterion [4], which `cv2.goodFeaturesToTrack` uses.

**Iterate (Newton–Raphson).** The linearisation holds only for small $\mathbf d$.
Repeat, re-sampling frame 2 at the current estimate each time:

$$
e_k(\mathbf p) = I_1(\mathbf p) - I_2(\mathbf p + \mathbf d_k), \qquad
\mathbf b_k = \sum_W \begin{bmatrix} I_x e_k \\ I_y e_k \end{bmatrix}, \qquad
\mathbf d_{k+1} = \mathbf d_k + G^{-1}\mathbf b_k
$$

until $|\Delta\mathbf d| < 0.01$ px. $G$ is built from frame 1's gradients, so it
doesn't change and is inverted once. $\mathbf p + \mathbf d_k$ is not a whole
pixel, so $I_2$ has to be interpolated. That is what A4 is for.

**Large motions (pyramid).** The Taylor step only converges if $|\mathbf d|$ is
within about the window's half-width. For faster motion, build image pyramids
(halve the resolution $L$ times), solve at the coarsest level where the motion
is $2^{L}$ times smaller, then double $\mathbf d$ and use it as the starting
guess one level up [5]. `flow.lk_track` does this with 3 levels and a 21×21 window.

**Gradients.** Central differences:
$I_x(x, y) = \tfrac12\,[I(x{+}1, y) - I(x{-}1, y)]$,
$I_y(x, y) = \tfrac12\,[I(x, y{+}1) - I(x, y{-}1)]$.

### A4. Bilinear interpolation

We need $I$ at a non-integer point $(x, y)$. Let $x_0 = \lfloor x \rfloor$,
$y_0 = \lfloor y \rfloor$, $a = x - x_0$, $b = y - y_0$ ($0 \le a, b < 1$). The
four surrounding pixels are

$$
f_{00} = I(x_0, y_0),\quad f_{10} = I(x_0{+}1, y_0),\quad
f_{01} = I(x_0, y_0{+}1),\quad f_{11} = I(x_0{+}1, y_0{+}1)
$$

**Step 1: interpolate linearly along x, on the two rows.** The straight line
through $(0, f_{00})$ and $(1, f_{10})$ evaluated at $a$:

$$
f_{a0} = f_{00} + a\,(f_{10} - f_{00}) = (1-a) f_{00} + a f_{10}
$$
$$
f_{a1} = (1-a) f_{01} + a f_{11}
$$

**Step 2: interpolate linearly along y between those two results:**

$$
I(x, y) \approx (1-b)\, f_{a0} + b\, f_{a1}
$$

**Expand:**

$$
\boxed{\,I(x, y) \approx (1-a)(1-b)\,f_{00} + a(1-b)\,f_{10} + (1-a)\,b\,f_{01} + a\,b\,f_{11}\,}
$$

Doing y first and then x gives the same result, so the order does not matter.
In matrix form:

$$
I(x, y) \approx \begin{bmatrix} 1-b & b \end{bmatrix}
\begin{bmatrix} f_{00} & f_{10} \\ f_{01} & f_{11} \end{bmatrix}
\begin{bmatrix} 1-a \\ a \end{bmatrix}
$$

**Properties.**

- The four weights are each $\ge 0$ and add up to 1, so the result is a weighted
  average of the four pixels and stays in their range.
- It is exact at the pixels: with $a = b = 0$ it returns $f_{00}$.
- It is continuous across cells but its derivative is not.
- It is the same as fitting $f(a, b) = c_0 + c_1 a + c_2 b + c_3 ab$ through the
  four corners. That surface is linear along any horizontal or vertical line,
  but the $ab$ term makes it a saddle (hyperbolic paraboloid), not a plane.

Area interpretation: each corner's weight is the area of the rectangle
*opposite* it in the unit cell. The nearer the point is to a corner, the larger
that corner's opposite rectangle, so the more that corner counts.

**Small example.** $f_{00}=10$, $f_{10}=20$, $f_{01}=30$, $f_{11}=60$, point at
$a = 0.25$, $b = 0.5$:
$f_{a0} = 0.75\cdot10 + 0.25\cdot20 = 12.5$,
$f_{a1} = 0.75\cdot30 + 0.25\cdot60 = 37.5$,
$I = 0.5\cdot12.5 + 0.5\cdot37.5 = 25.0$.
The expanded formula gives the same number:
$0.375\cdot10 + 0.125\cdot20 + 0.375\cdot30 + 0.125\cdot60 = 3.75 + 2.5 + 11.25 + 7.5 = 25.0$.

`flow.bilinear` is this formula. It is used for every sub-pixel sample in the
tracker: the template window in frame 1 (the points are integers, but the
coarser pyramid levels are not), its gradients, and $I_2(\mathbf p + \mathbf d_k)$ at every iteration.

### A5. Validating the tracking result against actual pixel locations

For two consecutive frames of each video (`validate_tracking.py`):

1. **Predicted:** the equations above give $\mathbf x + \mathbf d$ for each
   Shi–Tomasi corner.
2. **Actual:** where the point really went is measured *without* any flow
   equation. The 21×21 patch around $\mathbf x$ is slid over a ±24 px search area in
   frame 2, the peak of normalised cross-correlation is taken, and the peak is
   refined to sub-pixel with a parabola through it and its neighbours:
   $\delta = \tfrac12 (c_{-1} - c_{+1}) / (c_{-1} - 2c_0 + c_{+1})$.
3. The two are compared, and OpenCV's pyramidal LK is run as a second reference.

One point per video is also worked out in full (window values, $I_x$, $I_y$,
$I_t$, $G$, each iteration) in the report.

---

## Part B — Structure from motion with a planar object

### B1. Cameras and normalised coordinates

A pinhole camera maps a 3D point $\mathbf X$ (in world coordinates) to pixel
$\mathbf x$ by

$$
\lambda\,\tilde{\mathbf x} = K\,[R \mid \mathbf t]\,\tilde{\mathbf X},\qquad
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix}
$$

$K$ is known from the Module 2 calibration. After undistorting, every pixel is
converted to **normalised coordinates** $\tilde{\mathbf m} = K^{-1}\tilde{\mathbf x}$,
so each camera becomes $\lambda\tilde{\mathbf m} = [R\mid\mathbf t]\tilde{\mathbf X}$.
Take camera 1 as the world frame, $P_1 = [I \mid \mathbf 0]$. The unknowns are
$P_i = [R_i \mid \mathbf t_i]$ for $i = 2, 3, 4$ and the 3D points: that is
*structure* (the points) *from motion* (the camera poses).

### B2. Why a plane gives a homography

Let the object's points lie on a plane $\mathbf n^T\mathbf X = d$ in camera-1
coordinates, where $\mathbf n$ is the unit normal and $d$ is the distance from
camera 1. Then $\mathbf n^T\mathbf X / d = 1$ for every point on it, and

$$
\mathbf X_i = R_i\mathbf X + \mathbf t_i = R_i\mathbf X + \mathbf t_i \frac{\mathbf n^T\mathbf X}{d}
= \Big(R_i + \frac{\mathbf t_i\mathbf n^T}{d}\Big)\mathbf X
$$

Since $\mathbf X = \lambda_1 \tilde{\mathbf m}_1$ and $\mathbf X_i = \lambda_i\tilde{\mathbf m}_i$:

$$
\tilde{\mathbf m}_i \sim H_{1i}\,\tilde{\mathbf m}_1, \qquad
\boxed{\,H_{1i} = R_i + \frac{\mathbf t_i \mathbf n^T}{d}\,}
$$

Every point on the plane moves between the two views by the same 3×3 matrix, and
that matrix contains the motion $(R_i, \mathbf t_i)$ and the plane $(\mathbf n, d)$.

### B3. Estimating $H$ — DLT

Write $\mathbf m_i = (x', y')$ and $\mathbf m_1 = (x, y)$. From
$\tilde{\mathbf m}_i \times H\tilde{\mathbf m}_1 = \mathbf 0$, each correspondence gives two equations
linear in the 9 entries $\mathbf h$ of $H$:

$$
\begin{bmatrix}
-x & -y & -1 & 0 & 0 & 0 & x'x & x'y & x' \\
0 & 0 & 0 & -x & -y & -1 & y'x & y'y & y'
\end{bmatrix}\mathbf h = \mathbf 0
$$

With 54 points, $A$ is 108×9. $\mathbf h$ is the right singular vector of $A$ for
its smallest singular value, which minimises $\|A\mathbf h\|$ subject to
$\|\mathbf h\| = 1$. Before building $A$, each point set is shifted to zero mean and
scaled to mean distance $\sqrt 2$ (Hartley normalisation [6]); $H$ is then
de-normalised. Without this the columns of $A$ differ by orders of
magnitude and the answer is badly conditioned.

### B4. From $H$ to $R$, $\mathbf t$, $\mathbf n$

DLT returns $H$ only up to scale. The true $R + \mathbf t\mathbf n^T/d$ has its
**middle singular value equal to 1**. The reason: $\mathbf n^T\mathbf w = 0$ for any
$\mathbf w$ perpendicular to both $\mathbf n$ and $R^T\mathbf t$, so
$H\mathbf w = R\mathbf w$, which has the same length as $\mathbf w$. So the
estimate is divided by its middle singular value.

Then write $H^TH = V\,\mathrm{diag}(\sigma_1^2, 1, \sigma_3^2)\,V^T$ with columns
$\mathbf v_1, \mathbf v_2, \mathbf v_3$. The vector $\mathbf v_2$ keeps its length under $H$, and
two more unit vectors keep their length as well:

$$
\mathbf u_{1,2} = \frac{\sqrt{1-\sigma_3^2}\;\mathbf v_1 \pm \sqrt{\sigma_1^2-1}\;\mathbf v_3}{\sqrt{\sigma_1^2-\sigma_3^2}}
$$

All vectors in the plane spanned by $\mathbf v_2$ and $\mathbf u_1$ (or $\mathbf u_2$)
keep their length, so on that plane $H$ acts as a pure rotation. So, for each
choice [7, 8, and Ma et al., *An Invitation to 3-D Vision*, §5.3]:

$$
U = [\mathbf v_2,\ \mathbf u,\ \mathbf v_2\times\mathbf u],\quad
W = [H\mathbf v_2,\ H\mathbf u,\ H\mathbf v_2\times H\mathbf u],\quad
R = WU^T,\quad
\mathbf n = \mathbf v_2\times\mathbf u,\quad
\frac{\mathbf t}{d} = (H - R)\,\mathbf n
$$

It gives **four** solutions: $\mathbf u_1$ or $\mathbf u_2$, and for each the pair $(\mathbf t/d, \mathbf n)$ or $(-\mathbf t/d, -\mathbf n)$.

**Choosing the right one.**

1. *Visibility:* the plane is in front of camera 1, so every observed point has
   positive depth: $\lambda = d/(\mathbf n^T\tilde{\mathbf m}_1) > 0$, i.e.
   $\mathbf n^T\tilde{\mathbf m}_1 > 0$ for all points. This removes two.
2. *Four views:* there are three pairs (1→2, 1→3, 1→4), each with 2 survivors,
   but only one plane. Its normal $\mathbf n$ in camera 1 must be the same in all
   three, so the combination with the most consistent normals is chosen.
   The wrong candidates disagree by tens of degrees. This is where having four
   views helps. With two views the choice would be ambiguous.

All three $\mathbf t_i$ come out divided by the same $d$, so the cameras share
one scale. The reconstruction is in units of $d$.

### B5. Triangulation

For one 3D point $\tilde{\mathbf X}$ seen at $(x_i, y_i)$ in view $i$ with
$P_i$ rows $\mathbf p_{i1}^T, \mathbf p_{i2}^T, \mathbf p_{i3}^T$:

$$
x_i = \frac{\mathbf p_{i1}^T\tilde{\mathbf X}}{\mathbf p_{i3}^T\tilde{\mathbf X}}
\;\Rightarrow\; (x_i\mathbf p_{i3}^T - \mathbf p_{i1}^T)\tilde{\mathbf X} = 0,
\qquad
(y_i\mathbf p_{i3}^T - \mathbf p_{i2}^T)\tilde{\mathbf X} = 0
$$

Four views give 8 equations in the 4 homogeneous unknowns, $A\tilde{\mathbf X} = \mathbf 0$.
$\tilde{\mathbf X}$ is the singular vector for the smallest singular value, divided by its
4th entry. This is linear (DLT) triangulation [6].

### B6. Refinement

The linear steps minimise algebraic quantities, not image distances. So the
result is refined by alternating two steps:

- **resection:** re-estimate each camera $P_2, P_3, P_4$ from the current 3D points
  by minimising reprojection error (`cv2.solvePnP`, Levenberg–Marquardt),
- **intersection:** re-triangulate every point from the new cameras,

repeated 20 times. $P_1$ stays fixed. Scale is a free choice (the gauge), so after
each round everything is rescaled to keep the plane at distance 1 from camera 1.
This is a simple form of bundle adjustment. The quality measure is the RMS
**reprojection error**: the pixel distance between each detected point and
where its 3D point projects.

### B7. Scale, the object frame, and the boundary

SfM cannot recover absolute size: scaling the scene and the camera translations
by the same factor gives identical images. One known length fixes it. Here that
is the distance between two grid corners 8 squares apart. The factor is
$s = L_{\text{true}} / \lVert\mathbf X_a - \mathbf X_b\rVert$.

To draw the object flat, fit the plane to the points (SVD of the centred points;
the normal is the singular vector with the smallest singular value) and set up
axes on it: $\mathbf e_1$ along the grid's top row, $\mathbf e_2 = \mathbf n\times\mathbf e_1$,
$\mathbf e_3 = \mathbf e_1 \times \mathbf e_2$, origin at grid corner 0:

$$
\mathbf X_{\text{obj}} = s\,[\mathbf e_1\ \mathbf e_2\ \mathbf e_3]^T(\mathbf X - \mathbf X_0)
$$

The **boundary** is the polygon through the four reconstructed screen corners.
Its side lengths, corner angles and flatness are measured directly from the 3D points.
Camera centres are $\mathbf C_i = -R_i^T\mathbf t_i$, put into the same frame.

---

## References

1. B. D. Lucas and T. Kanade, "An iterative image registration technique with
   an application to stereo vision," *Proc. IJCAI*, 1981, pp. 674–679.
2. B. K. P. Horn and B. G. Schunck, "Determining optical flow," *Artificial
   Intelligence*, vol. 17, pp. 185–203, 1981.
3. G. Farnebäck, "Two-frame motion estimation based on polynomial expansion,"
   *Proc. Scandinavian Conf. on Image Analysis*, LNCS 2749, 2003, pp. 363–370.
4. J. Shi and C. Tomasi, "Good features to track," *Proc. IEEE CVPR*, 1994,
   pp. 593–600.
5. J.-Y. Bouguet, "Pyramidal implementation of the affine Lucas Kanade feature
   tracker — description of the algorithm," Intel Corporation, 2001.
6. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*,
   2nd ed., Cambridge University Press, 2004. Ch. 4 (DLT, normalisation),
   Ch. 12 (triangulation), Ch. 13 (plane-induced homography), Ch. 18 (bundle adjustment).
7. O. Faugeras and F. Lustman, "Motion and structure from motion in a piecewise
   planar environment," *Int. J. Pattern Recognition and Artificial
   Intelligence*, vol. 2, no. 3, pp. 485–508, 1988.
8. E. Malis and M. Vargas, "Deeper understanding of the homography decomposition
   for vision-based control," INRIA Research Report RR-6303, 2007.
9. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.
10. Video 1: OpenCV sample data `samples/data/vtest.avi`,
    https://github.com/opencv/opencv (first 30 s).
11. Video 2: Intel IoT DevKit, `store-aisle-detection.mp4`,
    https://github.com/intel-iot-devkit/sample-videos, CC BY 4.0 (5–35 s).

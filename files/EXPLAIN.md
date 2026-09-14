# Video script — Module 2 walkthrough (~10 min)

Read this basically as-is while you screen record. Each section has a rough
time so the whole thing lands around 10 minutes — don't stress about hitting
it exactly, just don't linger too long on any one tab.

Have the app open already, on the Home page, before you hit record.

---

## 0. Intro / project overview — ~1 min

**Say:**
"Hi, this is my Module 2 submission for CSc 8830, Computer Vision. This
module is about measuring the real-world size of an object from a single
smartphone photo. The overall idea has three steps: first I calibrate my
phone's camera to figure out its internal parameters, then I use those
parameters to convert pixel measurements into real-world millimeters using
the pinhole projection equations, and then I validate that whole pipeline by
running it on 20 real measurements and checking the error against a tape
measure. There's also a theory section where I derive the relationship
between how a 3D point shows up in two different cameras, which is the more
general version of the same projection math.

The whole thing runs as one web app built with Streamlit, and everything's
on GitHub — I'll show the tabs one at a time."

Click into Module 2 from the sidebar now.

---

## 1. Step 1 — Calibration — ~2 min

**Say:**
"A camera takes a 3D scene and flattens it onto a 2D sensor. To go the other
way — recover real-world size from a 2D photo — I first need to know the
camera's internal parameters, mainly its focal length in pixels and where
the optical axis actually hits the sensor. That's what calibration gives me.

I do this with a chessboard pattern, because it gives me free ground truth —
I tell OpenCV the real square size in millimeters, so it knows exactly where
every corner should be in 3D without me measuring anything by hand. The
corners are also high-contrast X-junctions, so OpenCV can locate them to
within a fraction of a pixel using cornerSubPix."

*[Point at the uploaded images / already-run result]*

"I took 18 photos of the board from different angles and distances — that
matters, because if every photo were the same head-on angle, the math can't
tell focal length apart from distance. Different angles give independent
constraints."

*[Point at the RMS reprojection error]*

"This number is the reprojection error — I take the recovered camera
parameters, project the known 3D corners back into each image, and measure
how far off they land from where OpenCV actually detected them. Mine came
out to about 0.39 pixels, which is under the half-pixel target I was aiming
for."

*[Point at K matrix]*

"This is the intrinsic matrix K — fx and fy are the focal length in pixels,
and cx, cy are the principal point, roughly the image center. These numbers
are what Step 2 actually uses."

---

## 2. Step 2 — Measurement — ~2 min

**Say:**
"This is the actual measurement step. The idea is just similar triangles —
an object's size in the image shrinks in proportion to how far away it is.
If I know two of the three things — image size in pixels, real size, and
distance — I can solve for the third. Here I know the calibration and I
measure the distance Z with a tape measure, so I can solve for the real
size."

*[Show the equations on screen]*

"The forward direction is u = fx·X/Z + cx. I need it backwards, so I
rearrange to X = (u - cx)·Z/fx, same idea for Y. Each pixel I click
back-projects to an actual 3D point at that known depth Z, and then the
distance between my two clicked points is the real-world measurement."

*[Upload a photo, enter Z, click two points]*

"I upload the object photo, enter the distance I measured with a tape
measure, and click the two endpoints of whatever I'm measuring. The app
back-projects both points and reports the distance in millimeters."

**Say this part even if not asked — it shows I understand the limitation:**
"This assumes the face I'm measuring is roughly perpendicular to the optical
axis, so both points are at basically the same depth. If the object is
tilted, the two points are actually at slightly different real depths, and
using one Z for both introduces error — that's actually something I ran
into in my validation, which I'll get to."

---

## 3. Step 3 — Validation — ~2.5 min

**Say:**
"For this step the assignment wants 20 measurements validated against ground
truth. I used a MARTA Breeze card, which is the same standard size as a
credit card — 85.6 by 53.98 millimeters — taped to a wall and photographed
from four different distances, all past the 2-meter minimum. From those 4
photos I pulled 5 independent measurements each — the width, both height
edges, and both diagonals — to get to 20 total data points without needing
20 separate photo sessions."

*[Show the measurements.csv / uploaded table]*

"Each row has the distance Z, the two pixel points, and the true
ground-truth length. Hitting compute runs all 20 through the same
measurement function as Step 2, and compares against ground truth."

*[Point at the stats]*

"My mean absolute error came out to about 1.6 millimeters, with a mean
percentage error around 2.4%. One thing I noticed digging into this: the
width measurements were consistently more accurate than the height and
diagonal ones — off by under 3% versus sometimes 6-9%. I actually spent time
checking whether this was a bug in my corner detection, but it held up even
after I completely changed the detection method, so it's not that. My best
explanation is that the card sits well off to one side of the image center,
and the lens distortion correction — plus the fact that the phone probably
wasn't held perfectly perpendicular to the wall — doesn't correct both axes
equally at that off-center position. So the error isn't random noise, it's a
systematic, explainable bias, which I think is a more useful finding than
just reporting a single clean error number."

*[Point at the plots]*

"These two plots show measured versus ground truth — ideally everything
sits on the red diagonal line — and error against distance, which shows the
same pixel-level detection error turns into a bigger physical error the
farther away the object is, since each pixel covers more real-world
millimeters at range."

---

## 4. Theory tab — ~2 min

**Say:**
"This section is the more general math problem — instead of one camera at a
known depth, this is: given two different cameras looking at the same
point, how do their two images relate to each other?

Camera 1 I set as the world frame, just to remove one transform from the
algebra — that's a free choice, no physical meaning. Camera 2 sits at some
rotation R and offset t from camera 1. A point maps between the two frames
as X2 = R·X1 + t.

The key geometric step: the vectors X2, t, and R·X1 all lie in the same
plane — you can see that directly from that equation rearranged — and three
coplanar vectors means their scalar triple product is zero. Writing the
cross product as a matrix gives the essential matrix E = [t]×R, and after
putting the camera intrinsics back in, you get the fundamental matrix F,
with the final relationship: p̃2 transpose times F times p̃1 equals zero."

**The one insight that matters most — say this clearly:**
"A single point in image 1 maps to a *line* in image 2, not a point. That's
because one pixel only tells you the direction of the ray toward that 3D
point, not how far along the ray it is. Every point along that ray looks
identical in camera 1, but lands on different pixels in camera 2 — and that
whole set of possible pixels is exactly the epipolar line. So one photo
alone can't fully pin down a 3D point without knowing depth, which is
exactly the problem Step 2 solved by measuring Z directly instead."

*[If time — mention how R, t are found if unknown]*

"If R and t aren't known in advance, you can recover them from image
correspondences — match at least 8 points between the two views, run the
8-point algorithm to get F, convert to E using the intrinsics, and decompose
E with recoverPose to get R and t. The catch is that t only comes out up to
an unknown scale — from images alone you can't tell a small close object
from a big far one — so you need one known real-world length somewhere in
the scene to fix the actual scale."

---

## 5. Wrap-up — ~15 sec

**Say:**
"That covers all three steps plus the theory. Code and everything I used is
on GitHub, link's in the description / PDF. Thanks for watching."

Stop recording.

---

## If you get asked a question live

**"Why not just put a known-size object in every shot instead of
calibrating?"**
"That also works and skips calibration entirely, but it needs a
known-size reference in every single photo. Calibrating once lets me
measure anything afterward, as long as I know the distance."

**"What if you don't know Z at all?"**
"Then a single image is fundamentally ambiguous — a small close object and
a large far one can produce the exact same picture. You'd need a second
camera view, a depth sensor, or a known reference object in frame."

**"Why are fx and fy slightly different?"**
"They'd be identical for perfectly square sensor pixels. The small
difference just reflects real sensor geometry and normal calibration
numerical tolerance."

**"What's the biggest source of error in your validation?"**
"Based on what I found, it's not random clicking error — it's the object
being off-center in the frame combined with imperfect distortion correction
and the camera not being perfectly perpendicular to the object. That shows
up as a repeatable, directional bias rather than noise."

---

## Connecting it to the lecture slides (mention this if your professor uses this notation)

**On focal length in pixels vs. physical units:**
"The focal length OpenCV reports isn't in millimeters — it's in pixels. It's
the physical focal length multiplied by how many pixels fit per unit length
on the sensor. That's exactly why I never had to know my phone's actual
sensor size — the pixel units already absorb that, and pixels are what I'm
measuring in anyway."

**On why full projection is x = K[R|t]X, but Step 2 looks simpler:**
"Step 2 is really the special case of that full projection equation where
R is identity and the object sits at a constant depth Z — which collapses
the general equation down to the simpler u = fx·X/Z + cx form I'm actually
using."

**On why we don't just solve for the projection matrix directly (direct
linear calibration):**
"That approach can't model radial lens distortion and it minimizes an
algebraic error instead of actual reprojection error in pixels. OpenCV's
calibrateCamera starts from something similar but then refines it with
Levenberg-Marquardt to directly minimize reprojection error — which is
exactly the number I reported in Step 1."

**On why many chessboard angles matter (degenerate coplanar case):**
"A single flat view, especially straight-on, is mathematically degenerate —
you can't separate focal length from distance using one flat plane at one
angle. Every additional distinct angle adds an independent constraint, which
is why I shot from a bunch of different angles instead of just several
similar ones."

# Video script — one take, Q&A style

Read the **Q** silently (or skip it, it's just there so you know what
you're answering), then say the **A** out loud while pointing at whatever's
in the `[Point at: ...]` bracket. Go in this exact order. Don't rush —
better to be a bit long and clear than fast and mumbled, since you're only
recording once.

---

## PART 0 — Before you hit record

Have these ready in browser tabs:
1. Your deployed app URL (the Streamlit Cloud one)
2. Your GitHub repo page
3. This script, on a second monitor or phone if possible

Start recording, then begin.

---

## PART 1 — What is this assignment, and why

**Q: What is this assignment asking you to do?**

**A:** "This is Module 2 for CSc 8830, Computer Vision. The assignment has
four parts: first, calibrate a camera using OpenCV — that means figure out
its internal parameters. Second, write a script that uses those parameters
to find the real-world 2D size of an object from a single photo, using
perspective projection. Third, validate that measurement approach over 20
different measurements at a distance greater than 2 meters, and report
error statistics. And fourth, a theory question — derive, mathematically,
how the same 3D point looks in two different cameras at different positions.

Everything has to run as one web application accessible through a link, the
code has to be in a GitHub repo, and I need to record this video as proof it
actually works."

**[Point at: GitHub repo page, e.g. the URL bar]**

"This is my repo — [read the URL out loud]. Everything I show in this video
is in here."

**[Point at: the deployed app URL]**

"And this is the live app — [read URL] — this isn't running on my laptop,
it's actually deployed, so anyone with the link can open it."

---

## PART 2 — app.py and how the site is put together

**Q: What is app.py, and why does the site look the way it does?**

**A:** "Before I go into the actual assignment, quickly — this `app.py` file
is the shell for the whole course. Instead of writing a separate site for
every module, this one file scans a folder called `modules/`, and any file
in there named `moduleN.py` automatically shows up in the sidebar. So Module
2 here is just `modules/module2.py` — that's the file that actually has all
the assignment code, and app.py never needed to change when I added it."

**[Point at: the sidebar showing Module 2, 3, 4]**

"Modules 3 and 4 are future assignments — they're just placeholders right
now, marked 'scheduled', that's expected and not part of what I'm submitting
today. Module 2 is the one that's complete."

Click into Module 2 now.

---

## PART 3 — Step 1: Calibration, and why a chessboard

**Q: What does calibration actually do, and why do we need it before we can
measure anything?**

**A:** "A camera takes the 3D world and flattens it onto a 2D sensor. To go
backwards — to take a 2D photo and recover a real-world size — I first need
to know exactly how that camera projects things. Specifically I need its
focal length in pixels, and where its optical center actually falls on the
sensor. That's what calibration finds. Those numbers together are called the
camera's intrinsic matrix, K."

**Q: Why a chessboard specifically, and not just any object?**

**A:** "Because a chessboard gives me free, perfect ground truth. I tell
OpenCV the real square size in millimeters — 22mm here — and from that it
already knows exactly where every single corner should be in 3D space,
without me measuring anything by hand. And the corners are sharp black/white
X-junctions, which OpenCV can locate to a fraction of a pixel using a
function called cornerSubPix. That precision is what makes the whole
calibration accurate."

**Q: Why take many photos from different angles instead of just one?**

**A:** "Because one flat photo, especially straight-on, is what's called
degenerate — mathematically, the algorithm can't tell focal length apart
from distance from a single flat view. Every different angle I shoot from
adds an independent constraint, so with enough varied angles the numbers
converge on the actual right answer."

**[Point at: the uploaded chessboard images]**

"I ended up using 18 photos here. I actually started with 16, but 6 of them
failed corner detection — some were photographed off a monitor screen
instead of a printed board, which causes a moire interference pattern that
confuses the detector, and a couple had the board too small and far away in
the frame. I swapped those out for 8 better ones and got all 18 to pass."

**[Point at: Run calibration button, then the results]**

"Hitting Run Calibration runs OpenCV's calibrateCamera function on all the
detected corners."

**[Point at: RMS reprojection error]**

"This number, reprojection error, is the real check of quality — I take the
calibration's recovered parameters, project the known 3D chessboard corners
back into each photo, and measure how far off they land from where the
corners were actually detected. Mine came out to 0.39 pixels, which is
solidly under the half-pixel target that's generally considered good."

**[Point at: the K matrix / fx, fy, cx, cy values]**

"fx and fy here are the focal length in pixels — they're not in millimeters,
they're in pixels, because that's a mix of the physical focal length and how
densely packed the sensor's pixels are, and pixels are exactly what I need
since I'm about to measure everything in pixel coordinates. cx and cy are the
principal point, roughly the image center."

---

## PART 4 — Step 2: turning pixels into real millimeters

**Q: How does Step 2 actually use those calibration numbers to measure
something?**

**A:** "This is just similar triangles, really. An object's size in the
photo shrinks proportionally to how far away it is. The forward version of
that is u = fx times X over Z, plus cx — that's how a real 3D point becomes
a pixel. I need it backwards: I already know u and v, the pixel I clicked,
and I know Z because I measured the real distance with a tape measure. So I
just rearrange the equation to solve for X and Y — the actual real-world
position — and the distance between two such points is the real-world
measurement."

**[Point at: the equations shown in the Step 2 tab]**

"This caption right here is important — it says this assumes the face I'm
measuring is roughly perpendicular to the camera, so both of my clicked
points are at about the same depth Z. If the object were tilted, that
assumption breaks and the measurement gets less accurate — that's actually
something I ran directly into in my validation, which I'll get to."

**[Upload a demo photo, enter Z, click two points]**

"So here I upload a photo, type in the distance I measured, click the two
endpoints of whatever I'm measuring, and it reports the real-world length in
millimeters."

---

## PART 5 — Step 3: why these measurement images, why a card

**Q: What are these measurement images, and why did you choose to measure a
card?**

**A:** "The assignment wants 20 different measurements validated against
ground truth, at a distance over 2 meters. I taped a MARTA Breeze card to a
wall — you can see it in these photos — and photographed it from four
different distances, all past 2 meters. I picked this card specifically
because it's a standardized size — it's the same ISO dimensions as a normal
credit card, 85.6 by 53.98 millimeters — so I have exact, trustworthy ground
truth without needing to physically tape-measure the card itself."

**[Point at: the 4 measurement_images photos]**

"There's also a notebook visible in these same photos, on the wall — I did
not use that one for measurement, only the card, because the notebook
doesn't have a standardized, verifiable real-world size the way the card
does. Ground truth has to be something I can actually trust."

**Q: The assignment wants 20 measurements — you only took 4 photos. How did
you get to 20?**

**A:** "From each of those 4 photos, I measured 5 different things off the
same card: the width, both height edges — since the card isn't perfectly
axis aligned in every shot, the left and right edges come out as slightly
different pixel lengths — and both diagonals. That gives 5 independent
measurements per photo, times 4 photos, equals 20 data points, without
needing 20 separate photo sessions."

**Q: How did you know the distance Z for these photos if you didn't tape-measure it?**

**A:** "I didn't have a tape measurement for these specific 4 shots, so
instead I used the card's own known width to solve for Z directly — since I
already know the card is 85.6mm wide, and I can measure its width in pixels,
I can invert the same projection equation to back out the distance. Then I
used that same distance to predict the *other* 4 measurements — the height
edges and diagonals — which weren't used to calculate Z, so comparing those
against ground truth is still a fair, independent check, not circular."

**[Point at: the measurements.csv table / Step 3 tab]**

"Each row has the distance, the two pixel points, and the true length. This
also confirms the actual distances came out to between 2.1 and 2.5 meters —
so the over-2-meter requirement is satisfied."

**[Click Compute error statistics, point at results]**

**Q: What did you actually find, and does it match what the assignment
asked for?**

**A:** "My mean absolute error came out to about 1.6 millimeters, with a
mean percentage error of roughly 2.4%. The assignment just asks to report
error statistics — it doesn't set a required accuracy threshold — so
technically this satisfies the requirement regardless of the exact number.
But what's actually interesting is I noticed the width measurements were
consistently more accurate, under 3% error, while the height and diagonal
measurements were off by anywhere from 2% to 9%, and always in the same
direction — always underestimating. I checked whether this was a bug in how
I was detecting the card's corners, and it held up even after I completely
rewrote the detection method, so it's not a bug. My conclusion is that it's
a real geometric effect — the card sits well off to one side of the image
rather than centered, and the lens distortion correction doesn't correct
both axes equally at that off-center position, especially combined with the
phone probably not being held perfectly perpendicular to the wall. So this
isn't random noise, it's a systematic, explainable bias, which I think is a
more useful finding than just a single clean error number."

**[Point at: the two plots]**

"This plot shows measured versus ground truth — everything should sit on
this diagonal line if it were perfect. And this one shows error against
distance, which shows the same one- or two-pixel detection error turns into
a bigger real-world error the farther away the object is, since each pixel
covers more real millimeters at range."

---

## PART 6 — Theory tab

**Q: What is the theory question actually asking?**

**A:** "Instead of one camera at a known distance, this is the more general
problem: two different cameras, at different positions and angles, both
looking at the same 3D point. How do the two images relate to each other?"

**[Point at: the assumptions listed]**

"I set camera 1's position as the world frame — that's just a free choice
that simplifies the algebra, it doesn't lose any generality. Camera 2 sits
at some rotation R and offset t relative to camera 1."

**[Point at: the coplanarity / essential matrix section]**

"The key step is noticing that a point's position in camera 2, the offset t,
and the rotated point from camera 1 are all coplanar — you can see that
directly from the equation X2 = R times X1 plus t, just rearranged. Three
coplanar vectors means their scalar triple product is zero, and writing that
cross product as a matrix gives what's called the essential matrix, E.
Bringing the camera intrinsics back in turns that into the fundamental
matrix, F, and the final relationship is: p2-transpose times F times p1
equals zero."

**Q: What's the single most important thing this result tells you?**

**A:** "That one point in image 1 maps to a *line* in image 2, not a single
point. That's because one pixel only tells you the direction of the ray
toward that point — not how far along the ray it actually is. Every possible
depth along that ray looks identical in camera 1, but lands on a different
pixel in camera 2 — and that whole set of possible pixels is exactly this
epipolar line. So a single photo alone is fundamentally ambiguous about
depth — which is exactly why, back in Step 2, I had to directly measure Z
with a tape measure instead of getting it from the image."

**[If time allows, point at the R/t-unknown section]**

"If R and t aren't known ahead of time, they can be recovered from image
correspondences instead — match at least 8 points between the two photos,
run the 8-point algorithm to get F, convert to E using the intrinsics, and
decompose E to get R and t. The one catch is t only comes out up to an
unknown scale — from images alone, a small close object and a big far one
look identical — so you need one known real-world length somewhere in the
scene to fix the actual scale."

---

## PART 7 — Wrap-up

**A:** "That covers all three steps and the theory question. The repo link
is [read it again], and the live app is at [read the URL again]. Thanks for
watching."

Stop recording.

---

## Quick answers if you get asked something live

**"Why calibrate instead of just using a known-size reference object in
every photo?"**
"A reference object works too and skips calibration, but it needs to be in
every single shot. Calibrating once lets me measure anything afterward, as
long as I know the distance."

**"What if you don't know the distance Z at all?"**
"Then a single photo is ambiguous — a small close object and a large far
one can look identical. You'd need a second camera view, a depth sensor, or
a known reference object in the frame."

**"Why are fx and fy slightly different numbers?"**
"They'd be identical for perfectly square sensor pixels. The small
difference is normal sensor geometry and calibration tolerance."

**"What's the biggest source of error in your validation?"**
"Based on what I found, it's not random clicking noise — it's the card
being off-center in the frame combined with imperfect distortion correction,
which shows up as a repeatable, directional bias rather than noise."

# Module 7 - Stereo Vision

CSc 8830, Computer Vision. Dhanush Nagarajan.

In Module 2 I measured objects from one photo, but I had to measure the
distance to the object with a tape first. In this module the distance comes
from the photos themselves. I take two photos from two spots a little apart,
the code finds how far away each point is, and then it uses that to get the
size.

## What the assignment asked for

1. Redo the Assignment 1 size measurement with calibrated stereo. Find the
   distance Z first, then the size:
   - rectangle: width and height
   - circle: diameter
   - any other shape: every edge length
2. Derive the same thing for uncalibrated stereo by hand. This one is
   handwritten and in the PDF, so there's no code for it.
3. Find the dimensions of a room from video.

## Problem 1 - Object size with calibrated stereo

### Taking the photos

I used the same phone I calibrated in Module 2. The phone stands on the
table, I take a photo, slide it 5 cm to the right along a ruler and take a
second photo. That 5 cm is the baseline B. The objects lean against the wall
about 30-40 cm away.

| Object | Shape | Measured with a ruler |
|---|---|---|
| Notebook | rectangle | 18.5 x 23.5 cm |
| Cream tube (front outline) | other shape, 4 edges | top 6 cm, sides 17 cm, bottom (cap) 3.2 cm |
| Lid | circle | 13.5 cm across |

### How the code works (`stereo.py`)

1. Load K from the Module 2 calibration (`calibration/camera_params.npz`).
   My photos here are half size, so I halve fx, fy, cx and cy.
2. Match SIFT features between the two photos. From the matches OpenCV gives
   the essential matrix, and from that the rotation R and the direction t of
   the move. t only has a direction, so I multiply it by B = 50 mm. Doing it
   this way also handles the phone turning a little while I slide it.
3. I click the object in the first photo. For each click the code looks for
   the same spot in the second photo, but only along the epipolar line,
   because that's the only place the match can be.
4. Triangulate each pair into a 3D point (X, Y, Z) in mm. Z is the distance
   from the camera. The simple version is Z = f * B / d, where d is how far
   the point moved between the photos.
5. The size is just the 3D distance between the points.

### Results

| Object | Measurement | Stereo | Ruler | Error |
|---|---|---|---|---|
| Notebook | width | 21.6 cm | 18.5 cm | +17.0% |
| Notebook | height | 28.8 cm | 23.5 cm | +22.5% |
| Tube | top | 4.6 cm | 6.0 cm | -23.5% |
| Tube | side | 12.9 cm | 17.0 cm | -24.3% |
| Tube | bottom | 2.2 cm | 3.2 cm | -32.4% |
| Tube | side | 12.9 cm | 17.0 cm | -24.1% |
| Lid | diameter | 22.0 cm | 13.5 cm | +63.0% |

The shapes come out right. For example the tube's side is 2.8 times its top
in my stereo numbers and also in my ruler numbers. What's off is the overall
scale.

I think the reason is the slide. In stereo every size is multiplied by B, so
if the phone really moved 4 cm and the code thinks 5 cm, everything comes out
25% too big. I slid the phone by hand, and working backwards from the errors
the real slides were about 4.3 cm (notebook), 6.6 cm (tube) and 3.1 cm
(lid). With a 5 cm baseline, being off by just 5 mm is already a 10% error,
so a longer slide or a proper stereo rig would help a lot.

The lid was the hardest one. It's plain blue on a plain wall, so only 42
feature matches were left, which makes the camera move estimate less
reliable.

### Checking the matrices

For each pair I also build F, E and R and check that they have the
properties they should.

| Check | Should be | Notebook | Tube | Lid |
|---|---|---|---|---|
| rank(F) | 2 | 2 | 2 | 2 |
| singular values of E | (s, s, 0) | (1, 1, ~0) | (1, 1, ~0) | (1, 1, ~0) |
| det(R) | +1 | 1.000000 | 1.000000 | 1.000000 |
| \|R R^T - I\| | 0 | 1e-15 | 2e-15 | 6e-16 |
| how much the phone turned | small | 4.73 deg | 3.71 deg | 1.54 deg |

All three pass. Here are the notebook's matrices (F scaled so its last entry
is 1):

```
F = [  1.70e-08   5.09e-07  -8.30e-04 ]
    [  8.63e-07  -2.14e-08  -2.58e-02 ]
    [ -5.58e-04   2.48e-02   1.00e+00 ]

E = [ -0.0010  -0.0306   0.0122 ]        R = [  0.9966  -0.0003   0.0824 ]
    [ -0.0519   0.0013   0.9986 ]            [  0.0004   1.0000  -0.0013 ]
    [ -0.0126  -0.9995   0.0003 ]            [ -0.0824   0.0013   0.9966 ]

t = (-49.97, 0.61, 1.53) mm
```

The web page shows these for every pair too, and `python measure_stereo.py`
prints the checks.

## Problem 2 - Uncalibrated stereo

Handwritten derivation, in the PDF.

## Problem 3 - Room size from video (`room.py`)

I measured my room from my videos using the straight lines of the walls, the
camera's focal length and one size I know (the door).

### How it works

1. Focal length from `turning_video.mp4`. When the camera only turns, two
   frames are connected by a homography H = K R K^-1. Only the right focal
   length makes K^-1 H K an actual rotation (R R^T = I), so the code tries f
   from 300 to 3000 px and keeps the best one. I needed this because video
   mode crops the sensor, so the Module 2 photo calibration doesn't fit the
   videos.
2. On a frame of each wall I marked some lines that are vertical on the wall
   (door edges, the room corner) and some that are horizontal (door tops, the
   floor line, the top of the wall). The vertical ones meet at one vanishing
   point and the horizontal ones at another. With K, each vanishing point
   gives a direction in 3D (d = K^-1 v), and crossing the two gives the
   wall's normal.
3. Every marked point is put on the plane n . X = 1. The distances between
   those points are right except for one unknown scale.
4. The scale comes from the door. I used 2032 mm (80 inches), which is the
   standard US door height. I didn't measure my door. This gives the room
   height and the width of the door wall.
5. The desk wall doesn't have a door, so I use the room height from step 4 as
   its scale, since all the walls are the same height.

### Result

| Measurement | Result |
|---|---|
| Room height (floor to top of wall) | 2.42 m |
| Door wall width | 4.21 m |
| Desk wall width | 3.44 m |

So my room is about 4.2 m x 3.4 m and 2.4 m high.

I didn't tape-measure the room, but a few things say the numbers make sense:

- The vertical and horizontal directions on each wall come out 89.5 and 89.7
  degrees apart, so the focal length and my lines agree (it should be 90).
- The door comes out 85 cm wide, and US doors are normally 76-91 cm.
- 2.42 m is very close to a normal 8 ft (2.44 m) ceiling.

The door height is the biggest guess here. Everything scales with it, so if
the door is really 2000 mm, every length is 1.6% smaller.

## Files

| File | What it does |
|---|---|
| `__init__.py` | The Module 7 page in the web app |
| `stereo.py` | Problem 1 maths |
| `measure_stereo.py` | Runs Problem 1 on the three objects from the command line |
| `room.py` | Problem 3 maths (focal length, vanishing points, room size) |
| `data/` | The photo pairs and `examples.json` (ruler sizes and my clicked points) |
| `data/room/` | The three room videos (made smaller) and `room.json` (my marked lines and points) |
| `calibration/camera_params.npz` | K and lens distortion from Module 2 |

## Running it

Web app, from the repo root:

```
streamlit run app.py
```

Then pick Module 7 in the sidebar. For Problem 1, choose an object and my
saved points are already clicked. You can clear them and click your own, or
upload your own two photos.

Command line, from `modules/module7`:

```
python measure_stereo.py
python measure_stereo.py --baseline 50
python room.py
```

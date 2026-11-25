"""
room.py -- the maths for Module 7, Problem 3: room size from video.

Each wall is measured from one video frame using the room's straight lines:

  1. Focal length f from the turning video. When a camera only turns, two
     frames are linked by a homography H = K R K^-1. Only the right f makes
     K^-1 H K a real rotation, so I try many values of f and keep the best.
  2. In a frame facing a wall, find two vanishing points:
       - where the wall's vertical lines meet (door edges, room corner)
       - where the wall's horizontal lines meet (door tops, floor line)
     With K, each vanishing point gives a 3D direction (d = K^-1 v). The wall's
     normal is the cross product of the two directions.
  3. Every clicked pixel on the wall is put onto the plane n . X = 1. Lengths
     between those points are correct up to one unknown scale.
  4. The scale comes from one known length: the door height (2032 mm). That
     gives the room height and the width of the door wall.
  5. The desk wall has no door, so its scale comes from the room height found
     in step 4 (the walls are all the same height).

Run from modules/module7:
    python room.py      prints the focal length, the room size and the checks
The web page (__init__.py, Problem 3 tab) uses the same functions.
"""
import os
import json

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOM_DIR = os.path.join(HERE, 'data', 'room')


def load_room_data():
    with open(os.path.join(ROOM_DIR, 'room.json')) as f:
        return json.load(f)


def read_frames(video_path, every=1):
    """Read a video and keep every n-th frame (grey)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(frame)
        i = i + 1
    cap.release()
    return frames


def first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    return frame


# ----------------------------------------------------------------------------
# Step 1 - focal length from a turning video
# ----------------------------------------------------------------------------
def homography_between(gray_a, gray_b):
    sift = cv2.SIFT_create(4000)
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    pts_a = []
    pts_b = []
    for best, second in pairs:
        if best.distance < 0.75 * second.distance:
            pts_a.append(kp_a[best.queryIdx].pt)
            pts_b.append(kp_b[best.trainIdx].pt)
    H, mask = cv2.findHomography(np.float32(pts_a), np.float32(pts_b),
                                 cv2.RANSAC, 2.0)
    # scale H so det(H) = 1, like a rotation
    H = H / np.cbrt(np.linalg.det(H))
    return H


def focal_from_turning_video(video_path, every=20):
    """Try f from 300 to 3000 px and keep the one where K^-1 H K is closest
    to a rotation matrix (R R^T = I) for all frame pairs."""
    frames = read_frames(video_path, every)
    grays = []
    for fr in frames:
        grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    h, w = grays[0].shape

    homographies = []
    for i in range(len(grays) - 1):
        homographies.append(homography_between(grays[i], grays[i + 1]))

    best_f = None
    best_error = None
    for f in range(300, 3000, 2):
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
        K_inv = np.linalg.inv(K)
        error = 0.0
        for H in homographies:
            R = K_inv @ H @ K
            error = error + np.linalg.norm(R @ R.T - np.eye(3))
        if best_error is None or error < best_error:
            best_error = error
            best_f = f
    return float(best_f), w, len(homographies)


# ----------------------------------------------------------------------------
# Step 2 - vanishing points and the wall plane
# ----------------------------------------------------------------------------
def vanishing_point(lines):
    """Least squares meeting point of several image lines.

    Each line through (x1,y1) and (x2,y2) is l = p1 x p2. The vanishing point
    v satisfies l . v = 0 for every line, so v is the last column of V in the
    SVD of the stacked lines.
    """
    rows = []
    for x1, y1, x2, y2 in lines:
        l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        l = l / np.linalg.norm(l[:2])
        rows.append(l)
    _, _, Vt = np.linalg.svd(np.array(rows))
    return Vt[-1]


def wall_geometry(K, vertical_lines, horizontal_lines):
    """Return the vertical direction, horizontal direction and normal of the
    wall (all unit vectors in camera coordinates), and the angle between the
    two directions (should be about 90 degrees)."""
    K_inv = np.linalg.inv(K)
    d_vert = K_inv @ vanishing_point(vertical_lines)
    d_vert = d_vert / np.linalg.norm(d_vert)
    d_horiz = K_inv @ vanishing_point(horizontal_lines)
    d_horiz = d_horiz / np.linalg.norm(d_horiz)
    normal = np.cross(d_horiz, d_vert)
    normal = normal / np.linalg.norm(normal)
    angle = np.degrees(np.arccos(abs(float(d_vert @ d_horiz))))
    return d_vert, d_horiz, normal, angle


def point_on_wall(K, normal, pixel):
    """Put a pixel onto the wall plane n . X = 1 (the true size is fixed later)."""
    ray = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    return ray / float(normal @ ray)


def scale_data(data, frame_width):
    """The clicks were made on 1920-wide frames. Scale them to this frame."""
    s = frame_width / data['original_width']
    out = {}
    for wall in ['door_wall', 'desk_wall']:
        w = data[wall]
        out[wall] = {
            'vertical_lines': [[v * s for v in l] for l in w['vertical_lines']],
            'horizontal_lines': [[v * s for v in l] for l in w['horizontal_lines']],
            'points': {k: [p[0] * s, p[1] * s] for k, p in w['points'].items()},
        }
    return out


# ----------------------------------------------------------------------------
# Steps 3-5 - the room size
# ----------------------------------------------------------------------------
def measure_room(f, frame_width, frame_height, data, door_height_mm):
    walls = scale_data(data, frame_width)
    K = np.array([[f, 0, frame_width / 2],
                  [0, f, frame_height / 2],
                  [0, 0, 1]], dtype=float)

    # --- door wall: scale from the door ------------------------------------
    door = walls['door_wall']
    d_vert, d_horiz, normal, angle_a = wall_geometry(
        K, door['vertical_lines'], door['horizontal_lines'])
    P = {}
    for name, pixel in door['points'].items():
        P[name] = point_on_wall(K, normal, pixel)

    def up_down(a, b):          # length along the vertical direction
        return abs(float((P[a] - P[b]) @ d_vert))

    def across(a, b):           # length along the wall's horizontal direction
        return abs(float((P[a] - P[b]) @ d_horiz))

    door_height_units = (up_down('door_top_left', 'door_bottom_left') +
                         up_down('door_top_right', 'door_bottom_right')) / 2
    scale_a = door_height_mm / door_height_units

    door_width = (across('door_top_left', 'door_top_right') +
                  across('door_bottom_left', 'door_bottom_right')) / 2 * scale_a
    # room height: from the top of the wall (at the corners) down to the
    # floor (the bottom of the door)
    room_height = (up_down('corner_left_top', 'door_bottom_left') +
                   up_down('corner_right_top', 'door_bottom_right')) / 2 * scale_a
    door_wall_width = across('corner_left_top', 'corner_right_top') * scale_a

    # --- desk wall: scale from the room height ------------------------------
    desk = walls['desk_wall']
    d_vert_b, d_horiz_b, normal_b, angle_b = wall_geometry(
        K, desk['vertical_lines'], desk['horizontal_lines'])
    Q = {}
    for name, pixel in desk['points'].items():
        Q[name] = point_on_wall(K, normal_b, pixel)
    desk_height_units = abs(float((Q['corner_left_top'] - Q['floor']) @ d_vert_b))
    scale_b = room_height / desk_height_units
    desk_wall_width = abs(float((Q['corner_left_top'] - Q['corner_right_top'])
                                @ d_horiz_b)) * scale_b

    return {
        'room_height_mm': room_height,
        'door_wall_width_mm': door_wall_width,
        'desk_wall_width_mm': desk_wall_width,
        'door_width_mm': door_width,
        'angle_door_wall_deg': angle_a,
        'angle_desk_wall_deg': angle_b,
    }


def draw_annotations(frame, wall, scale):
    """Draw the lines and points used for one wall on its frame."""
    out = frame.copy()
    for l in wall['vertical_lines']:
        p1 = (int(l[0] * scale), int(l[1] * scale))
        p2 = (int(l[2] * scale), int(l[3] * scale))
        cv2.line(out, p1, p2, (255, 120, 0), 3)          # blue: vertical
    for l in wall['horizontal_lines']:
        p1 = (int(l[0] * scale), int(l[1] * scale))
        p2 = (int(l[2] * scale), int(l[3] * scale))
        cv2.line(out, p1, p2, (0, 0, 255), 3)            # red: horizontal
    for name, p in wall['points'].items():
        c = (int(p[0] * scale), int(p[1] * scale))
        cv2.circle(out, c, 8, (0, 200, 0), -1)
        cv2.putText(out, name.replace('_', ' '), (c[0] + 10, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 2)
    return out


if __name__ == '__main__':
    # command line: print the room size
    data = load_room_data()
    turning = os.path.join(ROOM_DIR, data['turning_video'])
    f, width, pairs = focal_from_turning_video(turning)
    print('focal length from the turning video: %.0f px (frame width %d, %d frame pairs)'
          % (f, width, pairs))
    frame = first_frame(os.path.join(ROOM_DIR, data['door_wall']['video']))
    h, w = frame.shape[:2]
    r = measure_room(f, w, h, data, data['door_height_mm'])
    print('angle between vertical and horizontal (door wall): %.1f deg'
          % r['angle_door_wall_deg'])
    print('angle between vertical and horizontal (desk wall): %.1f deg'
          % r['angle_desk_wall_deg'])
    print('room height     : %.2f m' % (r['room_height_mm'] / 1000))
    print('door wall width : %.2f m' % (r['door_wall_width_mm'] / 1000))
    print('desk wall width : %.2f m' % (r['desk_wall_width_mm'] / 1000))
    print('door width check: %.0f mm (US doors are usually 762-914 mm)'
          % r['door_width_mm'])

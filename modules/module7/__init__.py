"""
modules/module7 -- Assignment 7: Stereo vision.

The page in the web app. app.py imports this; it doesn't run on its own.

Where each part of the assignment lives:
  Problem 1. object size with CALIBRATED stereo        -> tab 1 (stereo.py has the maths)
  Problem 2. the same with UNCALIBRATED stereo         -> handwritten, in the PDF
  Problem 3. room dimensions from video                 -> tab 2 (room.py has the maths)
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st


from . import stereo
from . import room

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as img_coords
except ImportError:
    img_coords = None

# --- metadata read by app.py to build the navigation ------------------------
NUMBER = 7
TITLE = 'Stereo Vision'
SUBTITLE = ('Measuring an object from two photos taken 5 cm apart '
            '(calibrated stereo), and room dimensions from video.')
STATUS = 'complete'

# the main numbers, shown in a box at the top of the page
RESULTS = [
    ('Room size', '4.21 x 3.44 m', '2.42 m high, from my room videos'),
    ('Notebook (stereo)', '21.6 x 28.8 cm', 'ruler 18.5 x 23.5 cm'),
    ('Matrix checks', 'all pass', 'rank(F) = 2, det(R) = 1, E = (1, 1, 0)'),
]
THUMBNAIL = 'results/room_walls.jpg'      # picture for the home page card

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'photos')   # Problem 1 photo pairs

# how many points to click for each shape
CLICK_HELP = {
    'rectangle': (4, 'Click the 4 corners in order: top-left, top-right, '
                     'bottom-right, bottom-left.'),
    'circle': (4, 'Click 4 points on the edge: left, right, top, bottom.'),
    'polygon': (None, 'Click every corner in order around the shape '
                      '(at least 3). Edges are measured between neighbours, '
                      'and the last corner joins back to the first.'),
}
SHAPE_NAMES = {'rectangle': 'Rectangle', 'circle': 'Circle',
               'polygon': 'Any other shape'}
DISPLAY_WIDTH = 520


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_examples():
    with open(os.path.join(DATA, 'examples.json')) as f:
        return json.load(f)


def decode_upload(uploaded):
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@st.cache_data(show_spinner='Matching features between the two photos...')
def camera_motion(left_bytes, right_bytes, baseline_mm):
    """Cached, so moving a click doesn't redo the feature matching."""
    left = cv2.imdecode(np.frombuffer(left_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    right = cv2.imdecode(np.frombuffer(right_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = left.shape
    K, dist = stereo.load_camera(w, h)
    R, t, pl, pr = stereo.find_camera_motion(left, right, K, dist, baseline_mm)
    return K, dist, R, t, len(pl)


def draw_points(img, points, color):
    out = img.copy()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        cv2.circle(out, (x, y), 14, color, -1)
        cv2.circle(out, (x, y), 14, (255, 255, 255), 3)
        cv2.putText(out, str(i + 1), (x + 18, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
    return out


# ----------------------------------------------------------------------------
# Problem 1
# ----------------------------------------------------------------------------
def problem1():
    st.header('Problem 1: Object size with calibrated stereo')
    st.markdown(
        'Two photos of the same object, taken with the phone slid sideways by '
        'a known distance (the **baseline B**). The camera matrix **K** comes '
        'from the Module 2 chessboard calibration. From the two photos the '
        'code finds the distance **Z** to every clicked point, turns each one '
        'into a 3D point, and measures the object from those 3D points.')
    st.latex(r'Z = \frac{f \, B}{d} \qquad X = \frac{(u - c_x)\,Z}{f_x} '
             r'\qquad Y = \frac{(v - c_y)\,Z}{f_y}')
    st.caption('d is the disparity: how far the point shifts between the two '
               'photos. In the code the phone is also allowed to turn a '
               'little, so it uses full triangulation, which is the same idea.')

    examples = load_examples()
    source = st.radio('Photos', list(examples.keys()) + ['Upload my own pair'],
                      horizontal=True, key='m7_source')

    if source == 'Upload my own pair':
        c1, c2 = st.columns(2)
        up_left = c1.file_uploader('First photo (left position)',
                                   type=['jpg', 'jpeg', 'png'], key='m7_upl')
        up_right = c2.file_uploader('Second photo (after sliding right)',
                                    type=['jpg', 'jpeg', 'png'], key='m7_upr')
        shape = st.selectbox('Shape', list(SHAPE_NAMES.keys()),
                             format_func=lambda s: SHAPE_NAMES[s], key='m7_shape')
        baseline = st.number_input('Slide distance B (mm)', value=50.0,
                                   min_value=1.0, key='m7_b_up')
        if up_left is None or up_right is None:
            st.info('Upload both photos to start.')
            return
        left_bytes = up_left.getvalue()
        right_bytes = up_right.getvalue()
        truth = None
        default_clicks = []
    else:
        ex = examples[source]
        with open(os.path.join(DATA, ex['left']), 'rb') as f:
            left_bytes = f.read()
        with open(os.path.join(DATA, ex['right']), 'rb') as f:
            right_bytes = f.read()
        shape = ex['shape']
        baseline = st.number_input('Slide distance B (mm)',
                                   value=float(ex['baseline_mm']),
                                   min_value=1.0, key='m7_b_' + source)
        truth = ex['true_mm']
        default_clicks = ex['clicks']

    left = cv2.imdecode(np.frombuffer(left_bytes, np.uint8), cv2.IMREAD_COLOR)
    right = cv2.imdecode(np.frombuffer(right_bytes, np.uint8), cv2.IMREAD_COLOR)
    if left.shape != right.shape:
        st.error('The two photos must be the same size.')
        return
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # --- step 2: camera motion ---------------------------------------------
    K, dist, R, t, n_matches = camera_motion(left_bytes, right_bytes, baseline)
    angles = stereo.rotation_in_degrees(R)
    a, b, c = st.columns(3)
    a.metric('Matched features', n_matches)
    b.metric('Phone turned (left/right)', '%.1f°' % angles[1])
    c.metric('Slide (from t)', '%.0f mm' % np.linalg.norm(t))
    checks = stereo.matrix_checks(K, R, t)
    with st.expander('Matrices K, F, E, R, t and their checks'):
        c1, c2 = st.columns(2)
        c1.markdown('**K** (camera matrix, scaled to this photo size)')
        c1.dataframe(pd.DataFrame(K).round(2), hide_index=True)
        c2.markdown('**F** (fundamental matrix, last entry set to 1)')
        F_text = []
        for row in checks['F']:
            F_text.append(['%.3e' % v for v in row])
        c2.dataframe(pd.DataFrame(F_text), hide_index=True)
        c1.markdown('**E** (essential matrix, E = [t]x R)')
        c1.dataframe(pd.DataFrame(checks['E']).round(4), hide_index=True)
        c2.markdown('**R** (how the phone turned)')
        c2.dataframe(pd.DataFrame(R).round(4), hide_index=True)
        st.write('t (mm) =', np.round(t.ravel(), 2))
        sv = checks['E_singular_values']
        st.dataframe(pd.DataFrame([
            {'check': 'rank(F)', 'should be': '2',
             'result': str(checks['F_rank'])},
            {'check': 'singular values of E', 'should be': '(s, s, 0)',
             'result': '(%.3f, %.3f, %.1e)' % (sv[0], sv[1], sv[2])},
            {'check': 'det(R)', 'should be': '+1',
             'result': '%.6f' % checks['det_R']},
            {'check': '|R R^T - I|', 'should be': '0',
             'result': '%.1e' % checks['R_orthogonal_error']},
            {'check': 'rotation angle', 'should be': 'small',
             'result': '%.2f deg' % checks['rotation_angle_deg']},
        ]), hide_index=True)

    # --- step 3: clicking ----------------------------------------------------
    needed, help_text = CLICK_HELP[shape]
    st.subheader('Click the object in the first photo')
    st.caption(help_text)

    state_key = 'm7_pts_' + source
    if state_key not in st.session_state:
        st.session_state[state_key] = [tuple(p) for p in default_clicks]
    points = st.session_state[state_key]

    b1, b2 = st.columns(2)
    if b1.button('Clear points', key='m7_clear'):
        st.session_state[state_key] = []
        st.rerun()
    if default_clicks and b2.button('Use my saved points', key='m7_reset'):
        st.session_state[state_key] = [tuple(p) for p in default_clicks]
        st.rerun()

    h, w = left.shape[:2]
    scale = DISPLAY_WIDTH / w
    shown_left = draw_points(left, points, (0, 0, 255))
    shown_left = cv2.resize(shown_left, (DISPLAY_WIDTH, int(h * scale)))

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('**First photo** (click here)')
        if img_coords is None:
            st.image(rgb(shown_left))
            st.warning('streamlit-image-coordinates is not installed, '
                       'so clicking is off. The saved points are used.')
        else:
            # a new key after every click, so the component forgets the old click
            if 'm7_click_count' not in st.session_state:
                st.session_state['m7_click_count'] = 0
            click_key = 'm7_click_' + source + '_' + str(st.session_state['m7_click_count'])
            value = img_coords(rgb(shown_left), key=click_key)
            if value is not None:
                st.session_state['m7_click_count'] += 1
                new_point = (value['x'] / scale, value['y'] / scale)
                if needed is not None and len(points) >= needed:
                    points = []
                points.append(new_point)
                st.session_state[state_key] = points
                st.rerun()

    # --- step 3b + 4: match in the right photo and triangulate --------------
    F = stereo.fundamental_from_pose(K, R, t)
    matches = []
    scores = []
    points_3d = []
    for p in points:
        m, s = stereo.find_match(left_gray, right_gray, p, F)
        if m is None:
            st.warning('Point too close to the photo edge, click again.')
            st.session_state[state_key] = []
            return
        matches.append(m)
        scores.append(s)
        points_3d.append(stereo.triangulate(p, m, K, dist, R, t))

    with col_right:
        st.markdown('**Second photo** (found automatically)')
        shown_right = draw_points(right, matches, (0, 160, 0))
        shown_right = cv2.resize(shown_right, (DISPLAY_WIDTH, int(h * scale)))
        st.image(rgb(shown_right))

    if len(points) == 0:
        return

    rows = []
    for i in range(len(points)):
        rows.append({
            'point': i + 1,
            'left (u, v)': '(%.0f, %.0f)' % points[i],
            'right (u, v)': '(%.0f, %.0f)' % matches[i],
            'match score': round(scores[i], 2),
            'X (cm)': round(points_3d[i][0] / 10, 1),
            'Y (cm)': round(points_3d[i][1] / 10, 1),
            'Z distance (cm)': round(points_3d[i][2] / 10, 1),
        })
    st.markdown('**3D points** (left camera at the origin, Z = distance from camera)')
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    # --- step 5: sizes -------------------------------------------------------
    if needed is not None and len(points) < needed:
        st.info('Click %d more point(s).' % (needed - len(points)))
        return
    if needed is None and len(points) < 3:
        st.info('Click at least 3 corners.')
        return

    st.subheader('Result')
    result_rows = []
    if shape == 'rectangle':
        width_mm, height_mm, edges = stereo.rectangle_size(points_3d)
        result_rows.append(['Width', width_mm, truth['width'] if truth else None])
        result_rows.append(['Height', height_mm, truth['height'] if truth else None])
    elif shape == 'circle':
        diameter, across, down = stereo.circle_diameter(points_3d)
        result_rows.append(['Diameter', diameter, truth['diameter'] if truth else None])
    else:
        edges = stereo.polygon_edges(points_3d)
        for i in range(len(edges)):
            true_value = None
            if truth and i < len(truth['edges']):
                true_value = truth['edges'][i]
            label = 'Edge %d (point %d to %d)' % (i + 1, i + 1, (i + 1) % len(edges) + 1)
            result_rows.append([label, edges[i], true_value])

    table = []
    for name, estimate, true_value in result_rows:
        row = {'measurement': name, 'stereo (cm)': round(estimate / 10, 2)}
        if true_value is not None:
            row['ruler (cm)'] = true_value / 10
            row['error (%)'] = round(100 * (estimate - true_value) / true_value, 1)
        table.append(row)
    st.dataframe(pd.DataFrame(table), hide_index=True)


# ----------------------------------------------------------------------------
# Problem 3
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner='Finding the focal length from the turning video...')
def focal_length(video_path):
    return room.focal_from_turning_video(video_path)


def problem3():
    st.header('Problem 3: Room size from video')
    st.markdown(
        'Each wall is measured from a frame of my room videos using the '
        'room\'s straight lines (vanishing points), the focal length found '
        'from the video, and the door as the known size.')

    data = room.load_room_data()
    room_dir = room.ROOM_DIR

    # --- step 1: focal length -----------------------------------------------
    st.subheader('Step 1: Focal length from the turning video')
    st.markdown('When the camera only turns, two frames are linked by '
                'H = K R K^-1. The code tries many focal lengths and keeps the '
                'one that makes K^-1 H K closest to a real rotation.')
    turning_path = os.path.join(room_dir, data['turning_video'])
    c1, c2 = st.columns([2, 1])
    c1.video(turning_path)
    f, video_width, n_pairs = focal_length(turning_path)
    c2.metric('Focal length f', '%.0f px' % f)
    c2.caption('at %d px wide, from %d frame pairs' % (video_width, n_pairs))

    # --- step 2: the two walls ----------------------------------------------
    st.subheader('Step 2: Lines on each wall')
    st.caption('Blue lines are vertical on the wall, red lines are horizontal '
               'on the wall. Their vanishing points give the wall\'s direction '
               'in 3D. Green dots are the points that get measured.')
    door_frame = room.first_frame(os.path.join(room_dir, data['door_wall']['video']))
    desk_frame = room.first_frame(os.path.join(room_dir, data['desk_wall']['video']))
    h, w = door_frame.shape[:2]
    scale = w / data['original_width']

    c1, c2 = st.columns(2)
    c1.markdown('**Door wall**')
    c1.image(rgb(room.draw_annotations(door_frame, data['door_wall'], scale)))
    c2.markdown('**Desk wall**')
    c2.image(rgb(room.draw_annotations(desk_frame, data['desk_wall'], scale)))

    # --- step 3: scale and result -------------------------------------------
    st.subheader('Step 3: Scale from the door, and the result')
    door_cm = st.number_input('Door height (cm)',
                              value=data['door_height_mm'] / 10,
                              min_value=100.0, max_value=300.0, step=0.5,
                              key='m7_door')
    st.caption(data['door_height_note'])
    r = room.measure_room(f, w, h, data, door_cm * 10)

    table = pd.DataFrame([
        {'measurement': 'Room height (floor to top of wall)',
         'result (m)': round(r['room_height_mm'] / 1000, 2)},
        {'measurement': 'Door wall width',
         'result (m)': round(r['door_wall_width_mm'] / 1000, 2)},
        {'measurement': 'Desk wall width',
         'result (m)': round(r['desk_wall_width_mm'] / 1000, 2)},
    ])
    st.dataframe(table, hide_index=True)
    st.markdown('**Room: %.2f m x %.2f m, %.2f m high**'
                % (r['door_wall_width_mm'] / 1000, r['desk_wall_width_mm'] / 1000,
                   r['room_height_mm'] / 1000))

    st.markdown('**Checks**')
    st.write('- Angle between the wall\'s vertical and horizontal directions: '
             '%.1f° (door wall) and %.1f° (desk wall). They should be 90°, so '
             'the focal length and the lines agree.'
             % (r['angle_door_wall_deg'], r['angle_desk_wall_deg']))
    st.write('- The door comes out %.0f cm wide. Standard US interior doors '
             'are 76-91 cm wide, so the scale looks right.'
             % (r['door_width_mm'] / 10))


def render():
    tab1, tab3 = st.tabs(['Problem 1: Object size', 'Problem 3: Room size'])
    with tab1:
        problem1()
    with tab3:
        problem3()
    st.caption('Problem 2 (uncalibrated stereo) is a hand-written derivation '
               'and is in the PDF report.')

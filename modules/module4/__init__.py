"""
modules/module4 -- Module 4: Human Boundary Detection (RGB & Thermal).

The Module 4 page. app.py imports this; it doesn't run on its own.

Where each part of the assignment lives:
  Q1 "boundary of a human, RGB camera, no ML, compare with SAM2"  -> tab 1
  Q2 "boundary of a human, thermal camera, no ML, compare with SAM2"
                                                                -> tab 2
  Q3 "edges and regions via Fourier analysis, with derivations"  -> tab 3

SAM2 never runs in here. Its masks were made once by sam2_reference.py and
are read from data/sam2_masks/, so the app needs no torch and no GPU.
"""
import json
import os

import cv2
import numpy as np
import streamlit as st

from . import segment as S

# --- metadata read by app.py to build the navigation ------------------------
NUMBER = 4
TITLE = 'Human Boundary Detection'
SUBTITLE = ('Classical (non-ML) outlines of people in RGB and thermal images, '
            'compared with SAM2, plus edges and regions via Fourier analysis.')
STATUS = 'complete'

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

GREEN = (0, 220, 0)
MAGENTA = (255, 0, 255)


@st.cache_data(show_spinner=False)
def _prompts():
    with open(os.path.join(DATA, 'prompts.json')) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _read(path, flag=cv2.IMREAD_COLOR):
    return cv2.imread(path, flag)


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr


def _pick(kind, key):
    """Image chooser plus the box. Returns (image_bgr, box, sam2_mask)."""
    P = _prompts()[kind]
    name = st.selectbox('Image', list(P), key=key + '_name')
    img = _read(os.path.join(DATA, kind, name))
    sam = _read(os.path.join(DATA, 'sam2_masks', kind, name),
                cv2.IMREAD_GRAYSCALE)
    sam = (sam > 0).astype(np.uint8)
    default = P[name]
    h, w = img.shape[:2]
    st.markdown('**Box around the person** — the only input. SAM2 was given '
                'this same box.')
    c1, c2 = st.columns(2)
    k = key + '_' + name     # new image -> sliders reset to its box
    x0, x1 = c1.slider('x range', 0, w, (default[0], default[2]), key=k + '_x')
    y0, y1 = c2.slider('y range', 0, h, (default[1], default[3]), key=k + '_y')
    box = S.clip_box((x0, y0, x1, y1), img.shape)
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        st.warning('The box is too small.')
        return None
    return img, box, sam


@st.cache_data(show_spinner=False, max_entries=32)
def _run_rgb(img, box):
    return S.grabcut_person(img, box)


@st.cache_data(show_spinner=False, max_entries=32)
def _run_thermal(img, box):
    return S.thermal_person(img, box)


def _show_result(img, box, mask, sam, zoom=False):
    """My result next to SAM2's, the scores, and where they disagree."""
    if zoom:        # thermal people are small in a 640x512 street scene
        zx0, zy0, zx1, zy1 = S.grow_box(box, 0.35, img.shape)
        view = lambda a: a[zy0:zy1, zx0:zx1]
    else:
        view = lambda a: a
    c1, c2, c3 = st.columns(3)
    c1.image(_rgb(view(S.draw_result(img, mask, box, color=GREEN))),
             caption='Classical result (green = boundary)', width='stretch')
    c2.image(_rgb(view(S.draw_result(img, sam, box, color=MAGENTA))),
             caption='SAM2, same box', width='stretch')
    c3.image(_rgb(view(S.disagreement_image(mask, sam))),
             caption='green: both · red: classical only · blue: SAM2 only',
             width='stretch')
    r = S.compare_masks(mask, sam)
    a, b = st.columns(2)
    a.metric('IoU with SAM2', '%.3f' % r['iou'])
    b.metric('Dice with SAM2', '%.3f' % r['dice'])


def _tab_rgb():
    st.subheader('Q1 — Boundary of a person in an RGB image, no ML')
    st.markdown(
        '**GrabCut**, started from a box. It fits two colour models '
        '(person / background) to *this* image\'s pixels and finds the '
        'labelling with a graph min-cut, then re-fits and cuts again. Nothing '
        'is trained beforehand. Before the cut, a thin strip down the middle '
        'of the box is fixed as person, and pixels whose colour mostly '
        'occurs outside the box start as background. The boundary is traced '
        'with `cv2.findContours`, keeping every boundary pixel.')
    got = _pick('rgb', 'm4rgb')
    if got is None:
        return
    img, box, sam = got
    with st.spinner('Segmenting...'):
        mask = _run_rgb(img, box)
    _show_result(img, box, mask, sam)


def _tab_thermal():
    st.subheader('Q2 — Boundary of a person in a thermal image, no ML')
    st.markdown(
        'In a thermal image brightness is roughly temperature, and a person '
        'is warmer than most of a night scene, so a **threshold** finds them. '
        'The road near the camera is warm too, so the background is '
        'estimated with a wide median filter and subtracted first. Then a '
        'hysteresis threshold (Otsu\'s level, plus a lower level for pixels '
        'connected to it), clean-up, and the contour.')
    got = _pick('thermal', 'm4th')
    if got is None:
        return
    img, box, sam = got
    mask = _run_thermal(img, box)
    _show_result(img, box, mask, sam, zoom=True)


@st.cache_data(show_spinner=False)
def _theory_text():
    with open(os.path.join(HERE, 'theory.md')) as f:
        return f.read()


def render():
    st.title('Module 4 — ' + TITLE)
    st.caption(SUBTITLE)
    st.markdown('**No deep learning or machine learning is used for the '
                'classical results.** SAM2 is only the reference to compare '
                'against; its masks were computed offline and are loaded '
                'from files.')
    t1, t2, t3 = st.tabs(['1 · RGB', '2 · Thermal', '3 · Theory'])
    with t1:
        _tab_rgb()
    with t2:
        _tab_thermal()
    with t3:
        st.markdown(_theory_text())

"""
ui.py -- the look of the site.

Everything visual that is shared between pages lives here: the CSS, the
sidebar, the home page banner and cards, and the header at the top of every
module page. app.py calls these; the module pages only draw their own content.
The main colour is a dark blue (#0039A6).
"""
import base64
import html
import os

import cv2
import streamlit as st


# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea, select {
    font-family: 'Inter', sans-serif;
}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; padding-left: 3rem; padding-right: 3rem; max-width: 1600px;}
h1, h2, h3 {letter-spacing: -0.3px; color: #0f1f3d;}
h2 {font-weight: 700;}
h3 {font-weight: 650;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #00205f 0%, #0039A6 100%);
    border-right: none;
}
[data-testid="stSidebar"] * {color: #e8eefc;}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {height: 1rem; padding: 0;}
.sb-brand {padding: 4px 6px 18px 6px; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom: 14px;}
.sb-brand .logo {
    display: inline-flex; align-items: center; justify-content: center;
    width: 38px; height: 38px; border-radius: 10px; background: white;
    color: #0039A6 !important; font-weight: 800; font-size: 0.95rem; margin-bottom: 10px;
}
.sb-brand .name {font-size: 1.25rem; font-weight: 800; color: white !important; line-height: 1.2;}
.sb-brand .sub {font-size: 0.85rem; color: #b9cbf2 !important; margin-top: 2px;}

/* the radio list becomes a menu */
[data-testid="stSidebar"] [role="radiogroup"] {gap: 4px;}
[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100%; padding: 10px 12px; border-radius: 10px; margin: 0;
    transition: background .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(255,255,255,0.10);}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {display: none !important;}
[data-testid="stSidebar"] [role="radiogroup"] > label, [data-testid="stSidebar"] [role="radiogroup"] > div {width: 100%;}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size: 0.92rem; font-weight: 500; color: #dbe5fb;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked), [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {background: white;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p, [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {color: #0039A6 !important; font-weight: 700;}

.sb-foot {margin-top: 26px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.15);
          font-size: 0.85rem; line-height: 1.6; color: #b9cbf2 !important;}
.sb-foot a {color: white !important; font-weight: 600; text-decoration: none;}
.sb-foot a:hover {text-decoration: underline;}

/* ---------- home banner ---------- */
.banner {
    background: linear-gradient(120deg, #00266e 0%, #0039A6 55%, #2a6fe0 100%);
    color: white; padding: 36px 40px; border-radius: 16px; margin-bottom: 22px;
    box-shadow: 0 10px 28px rgba(0,57,166,0.18);
}
.banner .kicker {text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.78rem;
                 font-weight: 700; color: #b9cdf6;}
.banner h1 {color: white; margin: 6px 0 6px 0; padding: 0; font-size: 2.3rem; font-weight: 800;}
.banner p {color: #dbe6fb; margin: 0; font-size: 1.05rem; max-width: 760px; line-height: 1.55;}
.banner a {color: white; font-weight: 700;}

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px; border-color: #e3e9f4;
    box-shadow: 0 2px 8px rgba(15,31,61,0.05); background: white;
}
.card-img {width: 100%; aspect-ratio: 16/9; object-fit: cover; border-radius: 10px; display: block;}
.card-number {color: #0039A6; font-weight: 700; font-size: 0.8rem; margin-top: 12px;
              text-transform: uppercase; letter-spacing: 0.6px;}
.card-title {font-size: 1.12rem; font-weight: 750; color: #0f1f3d; margin: 4px 0 4px 0; line-height: 1.3; min-height: 2.6em;}
.card-text {color: #4b5563; font-size: 0.9rem; line-height: 1.5; min-height: 6.2em;}
.card-result {margin: 10px 0 8px 0; padding: 8px 12px; border-radius: 8px; background: #eef3fc;
              display: flex; justify-content: space-between; gap: 8px; font-size: 0.85rem;}
.card-result .rl {color: #4b5563;}
.card-result .rv {color: #0039A6; font-weight: 800; white-space: nowrap;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;}
.pill-complete {background: #e3f4ea; color: #1e7a3c;}
.pill-in-progress {background: #fff3dc; color: #a35d00;}
.pill-scheduled {background: #eef0f3; color: #59606b;}
.pill-error {background: #fde8e8; color: #b42318;}

/* ---------- module page header ---------- */
.page-head {
    background: linear-gradient(120deg, #00266e 0%, #0039A6 60%, #2a6fe0 100%);
    border-radius: 16px; padding: 28px 34px; margin-bottom: 20px; color: white;
    box-shadow: 0 10px 28px rgba(0,57,166,0.15);
}
.page-head .num {text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.78rem;
                 font-weight: 700; color: #b9cdf6;}
.page-head h1 {color: white; font-size: 2rem; font-weight: 800; margin: 4px 0 6px 0; padding: 0;}
.page-head p {color: #dbe6fb; margin: 0; font-size: 1.02rem; max-width: 820px; line-height: 1.5;}
.page-head .results {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 12px; margin-top: 20px;}
.page-head .rcard {background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.20);
                   border-radius: 12px; padding: 12px 16px;}
.page-head .rl {font-size: 0.75rem; font-weight: 700; color: #b9cdf6; text-transform: uppercase; letter-spacing: 0.5px;}
.page-head .rv {font-size: 1.35rem; font-weight: 800; color: white; margin: 2px 0;}
.page-head .rn {font-size: 0.8rem; color: #c8d7f7;}

/* ---------- widgets ---------- */
[data-testid="stMetric"] {background: #f4f7fd; border: 1px solid #e3e9f4; border-radius: 12px; padding: 12px 16px;}
[data-testid="stMetricValue"] {color: #0039A6; font-weight: 750;}
.stButton > button {border-radius: 10px; font-weight: 600;}
.stTabs [role="tab"] p {font-weight: 600; font-size: 0.95rem;}
[data-testid="stImage"] img {border-radius: 10px;}

@media (max-width: 700px) {
    .banner, .page-head {padding: 24px 20px;}
    .banner h1 {font-size: 1.7rem;}
    .page-head h1 {font-size: 1.5rem;}
}
/* ================= cleaner, larger type and a warm page ================= */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select, label {
    font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {background: #f7f4ee;}
.block-container {
    background: white; border-radius: 18px; margin-top: 28px; margin-bottom: 28px;
    box-shadow: 0 4px 18px rgba(15,31,61,0.06);
    padding: 2.4rem 2.6rem !important;
    max-width: min(1500px, calc(100% - 56px)) !important;
}
.stMarkdown p, .stMarkdown li, [data-testid="stText"] {font-size: 1.05rem; line-height: 1.6; color: #1f2937;}
[data-testid="stCaptionContainer"] p {font-size: 0.92rem;}
[data-testid="stWidgetLabel"] p {font-size: 1rem !important; font-weight: 700 !important; color: #00266e !important;}
h1, h2, h3, [data-testid="stHeading"] * {color: #0039A6 !important; font-weight: 700 !important; letter-spacing: 0 !important;}
.banner h1, .page-head h1 {color: white !important;}
.banner p, .page-head p {color: #dbe6fb !important;}
.banner a {color: white !important;}
[data-testid="stHeading"] h2 {font-size: 1.75rem;}
[data-testid="stHeading"] h3 {font-size: 1.3rem;}

/* tabs as big boxed buttons */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {gap: 12px; border-bottom: none !important; padding-bottom: 6px;}
.stTabs [role="tab"] {
    flex: 1; justify-content: center; height: auto !important;
    padding: 14px 18px !important; border: 2px solid #0039A6 !important; border-radius: 12px !important;
    background: white !important;
}
.stTabs [role="tab"] p {color: #0039A6 !important; font-weight: 700 !important; font-size: 1.02rem !important;}
.stTabs [role="tab"][aria-selected="true"] {background: #0039A6 !important;}
.stTabs [role="tab"][aria-selected="true"] p {color: white !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none !important;}

/* buttons */
.stButton > button {
    background: #0039A6; color: white; border: none; font-weight: 700;
    padding: 0.55rem 1.2rem; border-radius: 10px;
}
.stButton > button p {font-weight: 700; color: white;}
.stButton > button:hover {background: #002d85; color: white;}
.stButton > button[kind="secondary"] {background: #e8eefb; color: #0039A6;}
.stButton > button[kind="secondary"] p {color: #0039A6;}

/* inputs */
[data-baseweb="input"], [data-baseweb="select"] > div {border-radius: 10px !important;}
[data-testid="stExpander"] details {border-radius: 12px; border-color: #d8e1f2;}
[data-testid="stExpander"] summary p {font-weight: 700; color: #0039A6;}
[data-testid="stMetricLabel"] p {font-weight: 600; color: #374151;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def pill(status):
    labels = {'complete': 'Complete', 'in progress': 'In progress',
              'scheduled': 'Scheduled', 'error': 'Error'}
    return ('<span class="pill pill-' + status.replace(' ', '-') + '">' +
            labels.get(status, status.title()) + '</span>')


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def sidebar_brand():
    st.markdown(
        '<div class="sb-brand"><div class="logo">CV</div>'
        '<div class="name">Computer Vision</div></div>',
        unsafe_allow_html=True)


def sidebar_footer(student, school, repo_url):
    st.markdown(
        '<div class="sb-foot">' + esc(student) + '<br>' + esc(school) +
        '<br><a href="' + repo_url + '" target="_blank">Source code on GitHub</a></div>',
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Home page
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def card_picture(path):
    """Cut a picture to 16:9, shrink it and return it as a data URL,
    so all the cards line up."""
    img = cv2.imread(path)
    if img is None:
        return ''
    h, w = img.shape[:2]
    target = 16 / 9
    if w / h > target:                      # too wide: cut the sides
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    else:                                   # too tall: cut top and bottom
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def banner(repo_url):
    st.markdown(
        '<div class="banner">'
        '<h1>Computer Vision</h1>'
        '<p>Camera calibration, image filtering, segmentation, optical flow, '
        '3D reconstruction and stereo vision, each with an interactive demo.</p>'
        '<p style="margin-top:14px"><a href="' + repo_url + '" target="_blank">'
        'View the source code on GitHub &rarr;</a></p></div>',
        unsafe_allow_html=True)


def card(m):
    picture = ''
    if m['thumbnail'] and os.path.exists(m['thumbnail']):
        picture = '<img class="card-img" src="' + card_picture(m['thumbnail']) + '" alt="">'
    result = ''
    if m['results']:
        label, value = m['results'][0][0], m['results'][0][1]
        result = ('<div class="card-result"><span class="rl">' + esc(label) +
                  '</span><span class="rv">' + esc(value) + '</span></div>')
    st.markdown(
        picture +
        '<div class="card-number">' + esc(m['label']) + '</div>'
        '<div class="card-title">' + esc(m['title']) + '</div>'
        '<div class="card-text">' + esc(m['subtitle']) + '</div>' +
        result + pill(m['status']),
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Module pages
# ----------------------------------------------------------------------------
def page_header(m):
    results_html = ''
    if m['results']:
        cells = ''
        for label, value, note in m['results']:
            cells += ('<div class="rcard"><div class="rl">' + esc(label) + '</div>'
                      '<div class="rv">' + esc(value) + '</div>'
                      '<div class="rn">' + esc(note) + '</div></div>')
        results_html = '<div class="results">' + cells + '</div>'
    st.markdown(
        '<div class="page-head"><div class="num">' + esc(m['label']) + '</div>'
        '<h1>' + esc(m['title']) + '</h1>'
        '<p>' + esc(m['subtitle']) + '</p>' + results_html + '</div>',
        unsafe_allow_html=True)


# kept so older pages that still call it don't break
def results_at_a_glance(results):
    return None

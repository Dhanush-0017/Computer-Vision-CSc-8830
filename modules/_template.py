"""
_template.py -- copy this to start a new module.

Underscore prefix keeps it out of the sidebar (app.py only picks up entries
under modules/ whose name starts with "module"). Each module is its own
self-contained folder: to make module 7, create modules/module7/, put this
file in there as __init__.py, set NUMBER/TITLE/SUBTITLE/STATUS, write
render(), and put any of that module's own helper files/data alongside it in
the same folder. Reload the browser -- app.py doesn't need any changes, it'll
just show up.

modules/module2/common.py has load_calib, save_calib, measure, decode_upload
if a future module also needs a camera -- copy the pattern, don't import
across module folders.
"""
import numpy as np
import pandas as pd
import cv2
import streamlit as st

# --- metadata read by app.py to build the navigation ------------------------
NUMBER = 0
TITLE = 'Template'
SUBTITLE = 'One-line description shown on the home page.'
STATUS = 'scheduled'        # 'complete' | 'in progress' | 'scheduled'


def render():
    """Draw this module's page. Called by app.py when selected."""
    st.title('Module ' + str(NUMBER) + ' — ' + TITLE)
    st.caption(SUBTITLE)

    tab1, tab2 = st.tabs(['Implementation', 'Theory'])

    with tab1:
        st.header('Implementation')
        st.write('Upload inputs, run the algorithm, show results.')

        up = st.file_uploader('Input image', type=['jpg', 'jpeg', 'png'])
        if up:
            arr = np.frombuffer(up.getvalue(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    with tab2:
        st.header('Theory')
        st.markdown('Derivation goes here. Use st.latex for equations:')
        st.latex(r'f(x,y) * h(x,y) \Longleftrightarrow F(u,v) \cdot H(u,v)')

"""
_template.py -- copy this to start a new module.

Underscore prefix keeps it out of the sidebar (app.py only picks up files
literally named moduleN.py). To make module 7: copy this to module7.py, set
NUMBER/TITLE/SUBTITLE/STATUS, write render(), reload the browser -- app.py
doesn't need any changes, it'll just show up.

common.py already has load_calib, save_calib, measure, decode_upload if the
new module also needs a camera.
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

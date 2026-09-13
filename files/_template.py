"""
================================================================================
modules/_template.py  --  Starting point for a new assignment module
================================================================================
The leading underscore keeps this file OUT of the navigation — app.py only
picks up files named moduleN.py.

TO CREATE MODULE 7:
    cp modules/_template.py modules/module7.py
    - set NUMBER = 7
    - set TITLE / SUBTITLE / STATUS
    - write render()
    Reload the browser. It appears in the sidebar automatically; app.py needs
    no changes at all.

Shared helpers available:
    from common import load_calib, save_calib, measure, decode_upload
================================================================================
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

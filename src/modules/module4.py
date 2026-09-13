"""
================================================================================
modules/module4.py  --  Module 4: Human Boundary Detection (RGB & Thermal)
================================================================================
PLACEHOLDER — to be implemented.

Assignment:
  1. Find exact boundaries of a human in an RGB image using classical OpenCV
     only (NO deep learning / machine learning). Compare against SAM2.
  2. Same for a thermal image. Compare against SAM2.
  3. Theory: derive how edge detection and region segmentation can be achieved
     through Fourier (frequency) domain analysis.
================================================================================
"""
import streamlit as st

# --- metadata read by app.py to build the navigation ------------------------
NUMBER = 4
TITLE = 'Human Boundary Detection'
SUBTITLE = ('Classical (non-ML) segmentation of humans in RGB and thermal '
            'imagery, compared against SAM2.')
STATUS = 'scheduled'


def render():
    st.title('Module 4 — Human Boundary Detection (RGB & Thermal)')
    st.info('Coming soon — this module is scheduled for a later submission.')

    st.subheader('Planned scope')
    st.markdown("""
**Part 1 — RGB, classical methods only (no ML)**
- Background subtraction / GrabCut initialised from a coarse box.
- Colour-space work (HSV, YCrCb skin ranges), morphological cleanup,
  contour extraction and largest-contour selection.

**Part 2 — Thermal**
- Humans appear as high-intensity regions, so thresholding (Otsu / adaptive)
  is far more effective here than in RGB.
- Morphological opening/closing, then contour extraction.

**Comparison with SAM2**
- Run SAM2 separately, then score the classical masks against it using
  IoU / Dice, plus boundary F-score. SAM2 is used only as a reference — the
  submitted method stays fully classical, as the assignment requires.

**Part 3 — Theory**
""")
    st.markdown('Edge detection as high-pass filtering in the frequency domain; '
                'the derivative theorem of the Fourier transform:')
    st.latex(r'\mathcal{F}\!\left\{\frac{\partial f}{\partial x}\right\} = j2\pi u\,F(u,v)')
    st.markdown('The `j2πu` factor amplifies high frequencies — which is exactly '
                'why differentiation detects edges, and why a high-pass filter '
                'and an edge detector are the same operation viewed from two '
                'domains.')

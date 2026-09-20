"""
This file only exists because of Streamlit Cloud.

The app used to live at src/app.py, and that path got baked into the deployment
when I first set it up. Streamlit Cloud doesn't let you change the main file
path afterwards, and redeploying would give me a new URL - which I've already
put in my Module 2 PDF.

So the real app is at the repo root now (app.py), and this just runs it.
Nothing else imports this file.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, 'app.py'), run_name='__main__')

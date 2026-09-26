# CSc 8830 — Computer Vision

Assignment portfolio. Dhanush Nagarajan, Georgia State University.

Every assignment for this course lives in this one repo and is reachable from a
single web app, which is what the course asks for. `app.py` is the shell: it
scans `modules/`, finds every module folder, and builds its own navigation. So a
new assignment drops in as one folder and `app.py` never changes.

Each module is self-contained — its page, its own scripts, its own data, its own
theory doc, and **its own README**.

## Modules

| # | Topic | Folder | Details |
|---|---|---|---|
| 2 | Camera calibration & object dimension measurement | `modules/module2/` | [README](modules/module2/README.md) |
| 3 | Image blurring & the convolution theorem | `modules/module3/` | [README](modules/module3/README.md) |
| 4 | Human boundary detection (RGB & thermal, vs SAM2) + Fourier theory | `modules/module4/` | [README](modules/module4/README.md) |

There's no Module 1 — Module 2 was the first assignment with code to submit.

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Pick a module from the sidebar.

Each module's own README has the command line instructions for that module's
scripts.

## Layout

```
app.py                      # the shell, finds modules/moduleN/ by itself
requirements.txt
modules/
  _template.py               # copy this to start a new module
  module2/
    README.md                 # what this module is, how to run it, results
    __init__.py               # the page
    common.py                 # the maths
    calibrate.py, measure.py, validate.py, selftest.py
    theory.md
    data/, calibration/
  module3/
    README.md
    __init__.py               # the page
    filtering.py              # the maths
    blur.py, verify_convolution.py
    theory.md
    data/
  module4/
    README.md
    __init__.py               # the page
    segment.py                # RGB + thermal methods, IoU/Dice
    find_person.py, compare.py, sam2_reference.py
    theory.md
    data/, figures/
files/                        # my working files for the PDFs and videos, not tracked
```

## How a module plugs in

`app.py` imports each `modules/moduleN/` folder and reads four things off it —
`NUMBER`, `TITLE`, `SUBTITLE`, `STATUS` — plus a `render()` function that draws
the page. That's the whole contract. Nothing is hardcoded in `app.py`, so adding
Module 4 later means creating the folder and nothing else.

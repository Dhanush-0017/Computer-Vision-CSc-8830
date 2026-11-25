# Computer Vision

Camera calibration, image filtering, segmentation, optical flow, 3D
reconstruction and stereo vision. Every module runs from one Streamlit web app.

## Modules

| # | Topic | Key result | Folder |
|---|---|---|---|
| 2 | Camera calibration and measuring objects from one photo | 1.64 mm mean error at 2+ m | [`modules/module2`](modules/module2/README.md) |
| 3 | Image blurring and the convolution theorem | spatial and Fourier blur match to 1.1e-12 | [`modules/module3`](modules/module3/README.md) |
| 4 | Human boundary detection in RGB and thermal images, compared with SAM2 | IoU 0.943 (RGB), 0.790 (thermal) | [`modules/module4`](modules/module4/README.md) |
| 5 & 6 | Optical flow, Lucas-Kanade tracking and structure from motion | 0.08 px tracking error, 0.81 px reprojection error | [`modules/module5`](modules/module5/README.md) |
| 7 | Stereo vision: object size from two photos, room size from video | room measured at 4.21 x 3.44 x 2.42 m | [`modules/module7`](modules/module7/README.md) |

## Getting started

Requires Python 3.10+.

```
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The home page has a card for every
module with its headline result. Each module page has its own link
(`?page=module7`) and starts with a header showing the key results.

Each module also has command line scripts; see that module's README.

## Architecture

`app.py` is a small shell that discovers modules at run time. It scans
`modules/`, loads every `moduleN/` package and builds the navigation and home
page from a few attributes each module exposes:

| Attribute | Purpose |
|---|---|
| `NUMBER`, `TITLE`, `SUBTITLE`, `STATUS` | Shown in the sidebar and on the home card |
| `THUMBNAIL` | Optional picture for the home card |
| `LABEL` | Optional display name, e.g. "Modules 5 & 6" |
| `RESULTS` | Key numbers shown on the home card and in the page header |
| `render()` | Draws the module's page |

Adding a module means adding a folder; `app.py` doesn't change. Each module is
loaded from its file path, so a failing module shows an error on its own page
without affecting the rest of the site.

## Repository layout

```
.
├── app.py                 web app shell (module discovery, home page, navigation)
├── ui.py                  the site's look: CSS, sidebar, home cards, page headers
├── requirements.txt
├── .streamlit/
│   └── config.toml        theme
├── src/
│   └── app.py             entry point used by the Streamlit Cloud deployment
└── modules/
    ├── _template.py       starting point for a new module
    └── moduleN/
        ├── README.md      approach, results, usage
        ├── __init__.py    the module's web page
        ├── *.py           algorithms and command line scripts
        ├── theory.md      derivations (where the module has them)
        ├── data/          inputs: photos, videos, calibration
        └── results/       outputs: plots, tables, figures
```

## Tech stack

Python, OpenCV, NumPy, pandas, Matplotlib, Streamlit.

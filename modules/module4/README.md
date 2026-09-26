# Module 4 — Human Boundary Detection (RGB & Thermal)

CSc 8830, Computer Vision. Dhanush Nagarajan.

## What the assignment asked for

1. Find the exact boundary of a human in an RGB image. OpenCV only, no
   deep learning or machine learning. Compare with SAM2.
2. Same for an image from a thermal camera. Compare with SAM2.
3. Theory: explain, with derivations, how edge detection and region
   segmentation can be done with Fourier (frequency) domain analysis.

## What's in this folder

| File | What it does |
|---|---|
| `__init__.py` | The Module 4 page in the web app |
| `segment.py` | Q1 and Q2: the RGB method, the thermal method, IoU/Dice |
| `find_person.py` | Outline a person in one image from the command line |
| `compare.py` | Compares both results with SAM2, writes the table and figures |
| `sam2_reference.py` | Runs SAM2 **once, offline**, and saves its masks. Not used by the app |
| `theory.md` | Q3: the derivations |
| `data/rgb/`, `data/thermal/` | One RGB image and one thermal image |
| `data/sam2_masks/` | SAM2's masks, made by `sam2_reference.py` |
| `data/prompts.json` | The box around the person in each image. My method and SAM2 get the same one |
| `data/SOURCES.md` | Where the images come from |
| `results_comparison.csv`, `figures/` | The comparison with SAM2 |

## How it works

Both methods start from a **box around the person**, which is the same
prompt SAM2 gets. The boundary is traced with
`cv2.findContours(..., CHAIN_APPROX_NONE)`, which keeps every boundary pixel.

**Q1, RGB: GrabCut** (`segment.grabcut_person`). GrabCut is a graph cut on
colour. It fits two colour models (person and background) to the pixels of
*this* image, then finds the best labelling with a min-cut, re-fits and cuts
again. Nothing is trained ahead of time, so it isn't machine learning. Before
the cut, a thin strip down the middle of the box is fixed as person, and
pixels whose colour mostly occurs outside the box start as background.

**Q2, thermal: threshold** (`segment.thermal_person`). In a thermal image,
brightness is roughly temperature, and a person is warmer than most of a
night scene. The road near the camera is warm too, so first the background
is estimated with a wide median filter and subtracted. Then a hysteresis
threshold: Otsu's level, plus a lower level for pixels connected to it.
Last, clean-up and fill holes.

**Q3: theory.** `theory.md` derives:

- the derivative theorem, and why it makes edge detection high-pass filtering
- why noise makes it band-pass (derivative of Gaussian, LoG)
- high-pass filters
- low-pass and band-pass energy for finding regions
- a small example worked by hand

## Running it

Web app, from the repo root:

```
streamlit run app.py
```

Command line, from this folder:

```
cd modules/module4
python find_person.py --sample FudanPed00013.png      # Q1
python find_person.py --sample FLIR_07583.png         # Q2
python compare.py        # comparison with SAM2: results_comparison.csv, figures/
```

`find_person.py` writes `out/<name>_mask.png`, `_overlay.png` and
`_boundary.txt` (the boundary as x, y points).

To remake the SAM2 masks (optional, needs torch):

```
pip install ultralytics torch torchvision
python sam2_reference.py
```

## Comparison with SAM2

| Image | Method | IoU | Dice |
|---|---|---|---|
| RGB — FudanPed00013 | GrabCut | 0.943 | 0.971 |
| Thermal — FLIR_07583 | Threshold | 0.790 | 0.883 |

- **RGB:** the background is plain, so GrabCut and SAM2 nearly agree. The
  small differences are along the edges: the feet and the edge of the
  backpack.
- **Thermal:** the threshold picks up warm things touching her, like the
  lamp-post crossbar at her head and a warm patch beside her arm. It misses
  her feet, which are cooler than the rest of her. SAM2 gets a cleaner
  outline because it knows the shape of a person.

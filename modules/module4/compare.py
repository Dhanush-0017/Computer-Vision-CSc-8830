"""
compare.py -- compares the classical results with SAM2 for Module 4.

    python compare.py

Runs GrabCut on the RGB image and the threshold method on the thermal image,
compares each with SAM2's mask (IoU and Dice), and writes
results_comparison.csv and figures/*.png (used in the PDF).

Needs data/sam2_masks/ -- made once by sam2_reference.py.
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import segment as S                                          # noqa: E402

DATA = os.path.join(HERE, 'data')
FIG = os.path.join(HERE, 'figures')

# (label, kind, function) -- the method used for each question
METHODS = [
    ('GrabCut', 'rgb', lambda im, b: S.grabcut_person(im, b)),
    ('Thermal threshold', 'thermal', lambda im, b: S.thermal_person(im, b)),
]


def load(kind, name):
    img = cv2.imread(os.path.join(DATA, kind, name))
    sam = cv2.imread(os.path.join(DATA, 'sam2_masks', kind, name), 0)
    return img, (sam > 0).astype(np.uint8)


def _label(img, text):
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _row(tiles, pad=6):
    h = max(t.shape[0] for t in tiles)
    return np.hstack([np.pad(t, ((0, h - t.shape[0]), (0, pad), (0, 0)),
                             constant_values=255) for t in tiles])


def main():
    P = json.load(open(os.path.join(DATA, 'prompts.json')))
    os.makedirs(FIG, exist_ok=True)
    rows = []
    for label, kind, fn in METHODS:
        for name, box in P[kind].items():
            img, sam = load(kind, name)
            box = tuple(box)
            m = fn(img, box)
            r = S.compare_masks(m, sam)
            rows.append(dict(kind=kind, image=name, method=label,
                             iou=round(r['iou'], 4), dice=round(r['dice'], 4)))
            print('%-8s %-20s IoU %.3f  Dice %.3f' % (kind, name, r['iou'], r['dice']))

            # figure: classical | SAM2 | where they differ
            x0, y0, x1, y1 = S.grow_box(box, 0.15, img.shape)
            c = lambda a: a[y0:y1, x0:x1]
            fig = _row([
                _label(c(S.draw_result(img, m, box, color=(0, 220, 0))), 'classical'),
                _label(c(S.draw_result(img, sam, box, color=(255, 0, 255))), 'SAM2'),
                _label(c(S.disagreement_image(m, sam)), 'IoU %.3f' % r['iou']),
            ])
            out = os.path.join(FIG, 'compare_%s.png' % kind)
            cv2.imwrite(out, fig)
            print('  wrote', out)

    with open(os.path.join(HERE, 'results_comparison.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['kind', 'image', 'method', 'iou', 'dice'])
        w.writeheader()
        w.writerows(rows)
    print('wrote results_comparison.csv')

    # the thermal method step by step
    name, box = list(P['thermal'].items())[0]
    img, sam = load('thermal', name)
    box = tuple(box)
    m, st = S.thermal_person(img, box, return_stages=True)
    x0, y0, x1, y1 = S.grow_box(box, 0.9, img.shape)
    c = lambda a: a[y0:y1, x0:x1]
    diff = cv2.normalize(st['difference'], None, 0, 255, cv2.NORM_MINMAX)
    fig = _row([
        _label(c(st['blurred']), '1 blurred'),
        _label(c(st['background']), '2 background'),
        _label(c(diff), '3 difference'),
        _label(c(st['raw'] * 255), '4 threshold'),
        _label(c(S.draw_result(img, m, box)), '5 result'),
    ])
    out = os.path.join(FIG, 'thermal_stages.png')
    cv2.imwrite(out, fig)
    print('wrote', out)


if __name__ == '__main__':
    main()

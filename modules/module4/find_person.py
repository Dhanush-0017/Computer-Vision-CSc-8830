"""
find_person.py -- outline a person in one image from the command line.

Classical methods only (see segment.py). Give it an image and a box around
the person; it writes the mask, an overlay with the traced boundary, and the
boundary itself as a list of (x, y) points.

Run from this folder:

    # RGB (Q1): GrabCut
    python find_person.py --image data/rgb/FudanPed00013.png \\
                          --box 381 179 562 490 --kind rgb

    # thermal (Q2): hysteresis threshold after background subtraction
    python find_person.py --image data/thermal/FLIR_07583.png \\
                          --box 232 210 322 468 --kind thermal

    # same thing, box taken from data/prompts.json
    python find_person.py --sample FLIR_07583.png

    # compare it with SAM2's mask (from sam2_reference.py)
    python find_person.py --sample FLIR_07583.png \\
                          --ref data/sam2_masks/thermal/FLIR_07583.png

Outputs go next to --out (default: out/<image name>_*.png / .txt).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import segment as S                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Outline a person, no ML.')
    ap.add_argument('--image')
    ap.add_argument('--box', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--kind', choices=['rgb', 'thermal'])
    ap.add_argument('--sample', help='file name from data/rgb or data/thermal')
    ap.add_argument('--ref', help='SAM2 mask to compare against')
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    a = ap.parse_args()

    if a.sample:
        P = json.load(open(os.path.join(HERE, 'data', 'prompts.json')))
        for kind in ('rgb', 'thermal'):
            if a.sample in P[kind]:
                a.kind = a.kind or kind
                a.image = os.path.join(HERE, 'data', kind, a.sample)
                a.box = a.box or P[kind][a.sample]
        if not a.image:
            ap.error('%s is not in data/prompts.json' % a.sample)
    if not (a.image and a.box and a.kind):
        ap.error('need --image, --box and --kind (or --sample)')

    img = cv2.imread(a.image)
    if img is None:
        ap.error('could not read ' + a.image)
    box = S.clip_box(a.box, img.shape)
    method = 'GrabCut' if a.kind == 'rgb' else 'thermal threshold'

    t = time.time()
    if a.kind == 'rgb':
        mask = S.grabcut_person(img, box)
    else:
        mask = S.thermal_person(img, box)
    dt = time.time() - t

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, os.path.splitext(os.path.basename(a.image))[0])
    cv2.imwrite(stem + '_mask.png', mask * 255)
    cv2.imwrite(stem + '_overlay.png', S.draw_result(img, mask, box))
    cs = S.contours_of(mask)
    with open(stem + '_boundary.txt', 'w') as f:
        f.write('# x y, one boundary pixel per line, in order around the outline\n')
        if cs:
            for x, y in cs[0][:, 0, :]:
                f.write('%d %d\n' % (x, y))

    print('%s  method=%s  %.2f s' % (os.path.basename(a.image), method, dt))
    print('  area %d px, boundary %d points' % (mask.sum(), len(cs[0]) if cs else 0))
    print('  wrote %s_{mask,overlay}.png and _boundary.txt' % stem)
    if a.ref:
        ref = cv2.imread(a.ref, 0)
        r = S.compare_masks(mask, ref > 0)
        print('  vs SAM2 (%s): IoU %.3f  Dice %.3f' % (
            os.path.basename(a.ref), r['iou'], r['dice']))


if __name__ == '__main__':
    main()

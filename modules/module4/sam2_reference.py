"""
sam2_reference.py -- make the SAM2 masks the classical methods get compared to.

SAM2 is a deep network, so it's not allowed as the method for this
assignment. It's only the yardstick. This script runs it once, offline, and
saves one PNG mask per image into data/sam2_masks/. The web app and
compare.py only ever read those PNGs, which is why torch isn't in
requirements.txt and the Streamlit deployment doesn't need a GPU or a 400 MB
checkpoint.

SAM2 gets the exact same box prompt as the classical methods (from
data/prompts.json), so both are answering the same question.

Setup (separate from the web app's requirements):
    pip install ultralytics torch torchvision
    # checkpoint: github.com/ultralytics/assets/releases -> sam2.1_l.pt
    # (ultralytics downloads it on first use if it can reach GitHub)

Run from this folder:
    python sam2_reference.py                    # all images, sam2.1_l.pt
    python sam2_reference.py --weights sam2.1_b.pt
    python sam2_reference.py --image data/thermal/FLIR_07583.png \
                             --box 232 210 322 468 --out my_mask.png

I used SAM2.1 Hiera-Large on CPU: about 22 s per image on a 2-core machine.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')


def run_sam2(model, image_path, box):
    """One box in, one binary mask out (0/1, image-sized)."""
    res = model.predict(image_path, bboxes=[list(map(int, box))], verbose=False)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        return np.zeros((h, w), np.uint8)
    m = res[0].masks.data[0].cpu().numpy()
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h),
                       interpolation=cv2.INTER_NEAREST)
    return (m > 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--weights', default='sam2.1_l.pt')
    ap.add_argument('--image', help='one image instead of the whole set')
    ap.add_argument('--box', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--out', help='where to save the mask (with --image)')
    args = ap.parse_args()

    from ultralytics import SAM          # imported here so --help is instant
    model = SAM(args.weights)

    if args.image:
        if not args.box:
            ap.error('--image needs --box')
        m = run_sam2(model, args.image, args.box)
        out = args.out or os.path.splitext(args.image)[0] + '_sam2.png'
        cv2.imwrite(out, m * 255)
        print('wrote', out, '(%d px)' % m.sum())
        return

    prompts = json.load(open(os.path.join(DATA, 'prompts.json')))
    for kind in ('rgb', 'thermal'):
        out_dir = os.path.join(DATA, 'sam2_masks', kind)
        os.makedirs(out_dir, exist_ok=True)
        for name, box in prompts[kind].items():
            t = time.time()
            m = run_sam2(model, os.path.join(DATA, kind, name), box)
            cv2.imwrite(os.path.join(out_dir, name), m * 255)
            print('%-8s %-20s %7d px  %.1f s' % (kind, name, m.sum(), time.time() - t))


if __name__ == '__main__':
    main()

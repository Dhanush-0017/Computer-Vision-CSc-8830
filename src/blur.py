"""
blur.py -- Module 3, blurring from the command line.

Same job as tab 1 of the web app, same code underneath (`filtering.py`), just
without Streamlit in the way. This is what I actually developed against before
wiring the UI up, and it is the quickest way to reproduce a result from the
report without clicking through the site.

Usage
-----
    cd src

    # 9x9 box blur, spatial route, written next to the input
    python blur.py --image ../data/module3_samples/sample_photo.jpg \
                   --filter box --size 9 --out ../data/module3_samples/blur_box9.png

    # Gaussian sigma=3, kernel size chosen automatically, Fourier route
    python blur.py --image path/to/photo.jpg --filter gaussian --sigma 3 \
                   --route fft --out blurred.png

    # Run both routes and print how far apart they are
    python blur.py --image path/to/photo.jpg --filter gaussian --sigma 2 --compare

Options that matter
-------------------
  --route     spatial | fft | separable   (default spatial)
  --boundary  wrap | zero | reflect | replicate  (default wrap)
              'wrap' is the DFT's own convention, so it is the one to use when
              comparing against --route fft. 'reflect' usually looks best.
  --gray      convert to grayscale first
  --compare   run the spatial and Fourier routes and report the difference
"""
import argparse
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filtering as F


def build_kernel(args):
    """Turn the command-line options into (2D kernel, 1D kernel or None, label)."""
    if args.filter == 'box':
        if args.size is None:
            raise SystemExit('--size is required for the box filter')
        return F.box_kernel(args.size), None, 'box %dx%d' % (args.size, args.size)
    size = args.size if args.size else F.suggested_size_for_sigma(args.sigma)
    k1 = F.gaussian_kernel_1d(size, args.sigma)
    return np.outer(k1, k1), k1, 'gaussian sigma=%.2f %dx%d' % (args.sigma, size, size)


def main():
    p = argparse.ArgumentParser(description='Blur an image by spatial convolution '
                                            'or by multiplying in the Fourier domain.')
    p.add_argument('--image', required=True)
    p.add_argument('--filter', choices=['box', 'gaussian'], default='gaussian')
    p.add_argument('--size', type=int, default=None,
                   help='kernel width, odd. Optional for gaussian (defaults to the '
                        '+/-3 sigma rule).')
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--route', choices=['spatial', 'fft', 'separable'], default='spatial')
    p.add_argument('--boundary', choices=list(F.BOUNDARIES), default='wrap')
    p.add_argument('--gray', action='store_true')
    p.add_argument('--out', default=None, help='where to write the blurred image')
    p.add_argument('--compare', action='store_true',
                   help='also run the other route and report the difference')
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit('could not read %s' % args.image)
    img = img.astype(np.float64)
    if args.gray:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    kernel, k1, label = build_kernel(args)
    print('image   : %s  %dx%d' % (args.image, img.shape[1], img.shape[0]))
    print('filter  : %s   (taps sum to %.12f)' % (label, kernel.sum()))

    if args.route == 'fft':
        out, secs = F.timed(F.convolve_fft_circular, img, kernel)
    elif args.route == 'separable':
        if k1 is None:
            raise SystemExit('--route separable needs the gaussian filter')
        out, secs = F.timed(F.convolve_separable, img, k1, args.boundary)
    else:
        out, secs = F.timed(F.convolve_spatial, img, kernel, args.boundary)
    print('route   : %s (%s boundary)   %.4f s' % (args.route, args.boundary, secs))

    if args.compare:
        a, ta = F.timed(F.convolve_spatial, img, kernel, 'wrap')
        b, tb = F.timed(F.convolve_fft_circular, img, kernel)
        m = F.compare(a, b)
        print()
        print('--- spatial (wrap) vs Fourier ---------------------------------')
        print('  spatial : %.4f s' % ta)
        print('  fourier : %.4f s' % tb)
        print('  max |A-B|  : %.6e   (%.0f ulps)' % (m['max_abs'], m['eps_multiples']))
        print('  mean |A-B| : %.6e' % m['mean_abs'])
        print('  RMSE       : %.6e' % m['rmse'])
        print('  PSNR       : %s' % ('inf' if np.isinf(m['psnr_db'])
                                     else '%.2f dB' % m['psnr_db']))
        print('  -> the two routes agree to floating-point round-off.'
              if m['max_abs'] < 1e-8 else
              '  -> UNEXPECTED: that difference is too large to be round-off.')

    if args.out:
        cv2.imwrite(args.out, F.to_uint8(out))
        print('wrote   : %s' % args.out)


if __name__ == '__main__':
    main()

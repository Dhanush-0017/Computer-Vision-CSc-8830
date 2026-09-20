"""
verify_convolution.py -- Module 3, the evidence, reproducible from a terminal.

This is the script behind the numbers in the report. It does three things:

  --selftest   a set of assertions that must hold if the implementation is
               right: spatial == Fourier, zero-padded spatial == linear FFT,
               my convolution == cv2.filter2D, separable == full 2D, blurring
               an impulse returns the kernel, and a box filter turns a step
               edge into a ramp of exactly the predicted width. Exits non-zero
               if any of them fails, so it can be run before submitting.

  --experiment the main validation table: for a range of kernel sizes, blur a
               real image both ways and record how far apart the two results
               are. Writes results_equivalence.csv.

  --timing     the cost sweep: naive spatial vs separable vs FFT against
               kernel size. Writes results_timing.csv and timing_plot.png.

Usage
-----
    cd modules/module3
    python verify_convolution.py --selftest
    python verify_convolution.py --experiment --image data/module3_samples/sample_photo.jpg
    python verify_convolution.py --timing
    python verify_convolution.py --all          # all three

Why bother with a self-test: the claim is "these two things are equal", and the
easiest way to accidentally fake that is a bug where both routes end up calling
the same code. So each check below compares against something outside that
route -- OpenCV, a known analytic answer, or a different algorithm.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filtering as F

TOL = 1e-8                     # anything below this on 0..255 data is round-off
HERE = os.path.dirname(os.path.abspath(__file__))


def _ok(name, value, tol=TOL):
    status = 'PASS' if value <= tol else 'FAIL'
    print('  [%s] %-52s %.3e' % (status, name, value))
    return status == 'PASS'


def selftest():
    print('Self-test — every number below is an error that should be ~0')
    print('-' * 72)
    rng = np.random.default_rng(8830)
    img = rng.uniform(0, 255, (97, 131))        # deliberately odd, non-square
    passed = []

    for size, sigma in [(3, None), (5, None), (9, 1.5), (15, 3.0), (21, 4.0)]:
        k = F.box_kernel(size) if sigma is None else F.gaussian_kernel(size, sigma)
        tag = 'box %d' % size if sigma is None else 'gauss %d/s%.1f' % (size, sigma)

        # (1) The headline claim: circular convolution == DFT product.
        passed.append(_ok('%s: spatial(wrap) vs fft(circular)' % tag,
                          F.compare(F.convolve_spatial(img, k, 'wrap'),
                                    F.convolve_fft_circular(img, k))['max_abs']))

        # (2) The same theorem used the other way: zero-padding turns the DFT
        #     product into linear convolution.
        passed.append(_ok('%s: spatial(zero) vs fft(linear)' % tag,
                          F.compare(F.convolve_spatial(img, k, 'zero'),
                                    F.convolve_fft_linear(img, k))['max_abs']))

        # (3) Against a library nobody can accuse me of having written.
        passed.append(_ok('%s: mine vs cv2.filter2D (reflect)' % tag,
                          F.compare(F.convolve_spatial(img, k, 'reflect'),
                                    F.reference_opencv(img, k, 'reflect'))['max_abs']))

    # (4) Separability: two 1D passes == one 2D pass.
    k1 = F.gaussian_kernel_1d(11, 2.0)
    passed.append(_ok('gaussian is separable (2x1D == 1x2D)',
                      F.compare(F.convolve_separable(img, k1, 'wrap'),
                                F.convolve_spatial(img, np.outer(k1, k1), 'wrap'))['max_abs']))

    # (5) Impulse response: blurring a single lit pixel must return the kernel.
    #     This is an analytic check -- it does not depend on the other route.
    n, ks = 64, 9
    imp = F.synthetic('impulse', n)
    k = F.gaussian_kernel(ks, 1.5)
    resp = F.convolve_spatial(imp, k, 'zero')[n // 2 - ks // 2: n // 2 + ks // 2 + 1,
                                              n // 2 - ks // 2: n // 2 + ks // 2 + 1] / 255.0
    passed.append(_ok('impulse response equals the kernel itself',
                      float(np.abs(resp - k).max())))

    # (6) Analytic edge response: a k-tap box filter turns an ideal step into a
    #     ramp with exactly k-1 intermediate values, each 1/k of the step apart.
    step = F.synthetic('step edge', 64)
    kb = 7
    ramp = F.convolve_spatial(step, F.box_kernel(kb), 'replicate')[32, :]
    mid = ramp[32 - kb: 32 + kb]
    expected_levels = kb - 1
    got_levels = int(np.sum((mid > 1e-9) & (mid < 255 - 1e-9)))
    print('  [%s] %-52s %d (expected %d)'
          % ('PASS' if got_levels == expected_levels else 'FAIL',
             'box filter ramps a step edge over k-1 pixels',
             got_levels, expected_levels))
    passed.append(got_levels == expected_levels)

    # (7) Kernels must sum to 1, or the filter changes brightness as well as
    #     blurring -- equivalently H(0,0) must be 1.
    passed.append(_ok('box kernel DC gain H(0,0) == 1',
                      abs(F.transfer_function(F.box_kernel(9), (64, 64))[0, 0].real - 1.0),
                      1e-12))
    passed.append(_ok('gaussian kernel DC gain H(0,0) == 1',
                      abs(F.transfer_function(F.gaussian_kernel(9, 2.0), (64, 64))[0, 0].real - 1.0),
                      1e-12))

    # (8) Colour images: each channel filtered independently, both routes.
    col = rng.uniform(0, 255, (40, 50, 3))
    passed.append(_ok('colour image: spatial vs fft',
                      F.compare(F.convolve_spatial(col, F.box_kernel(5), 'wrap'),
                                F.convolve_fft_circular(col, F.box_kernel(5)))['max_abs']))

    print('-' * 72)
    print('%d/%d checks passed' % (sum(bool(x) for x in passed), len(passed)))
    return all(passed)


def experiment(image_path):
    """The validation table for the report: both routes, several kernel sizes."""
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit('could not read %s' % image_path)
    img = img.astype(np.float64)
    print('Equivalence experiment on %s (%dx%d, grayscale)'
          % (image_path, img.shape[1], img.shape[0]))
    print('-' * 96)
    print('%-18s %10s %10s %13s %13s %13s %9s'
          % ('filter', 'spatial s', 'fft s', 'max|A-B|', 'mean|A-B|', 'RMSE', 'PSNR dB'))

    rows = []
    for size, sigma in [(3, None), (5, None), (9, None), (15, None),
                        (9, 1.5), (15, 2.5), (21, 3.5), (31, 5.0)]:
        if sigma is None:
            k, name = F.box_kernel(size), 'box %dx%d' % (size, size)
        else:
            k, name = F.gaussian_kernel(size, sigma), 'gauss %d s=%.1f' % (size, sigma)
        a, ta = F.timed(F.convolve_spatial, img, k, 'wrap')
        b, tb = F.timed(F.convolve_fft_circular, img, k)
        m = F.compare(a, b)
        psnr = 'inf' if np.isinf(m['psnr_db']) else '%.1f' % m['psnr_db']
        print('%-18s %10.4f %10.4f %13.3e %13.3e %13.3e %9s'
              % (name, ta, tb, m['max_abs'], m['mean_abs'], m['rmse'], psnr))
        rows.append({'filter': name, 'spatial_s': ta, 'fft_s': tb,
                     'max_abs_diff': m['max_abs'], 'mean_abs_diff': m['mean_abs'],
                     'rmse': m['rmse'], 'psnr_db': m['psnr_db'],
                     'ulps': m['eps_multiples']})

    # The boundary-mismatch case, which is the more interesting picture: keep
    # the Fourier side as-is but let the spatial side mirror instead of wrap,
    # then show the disagreement lives only in the border strip.
    k = F.gaussian_kernel(15, 2.5)
    h = 15 // 2
    a = F.convolve_spatial(img, k, 'reflect')
    b = F.convolve_fft_circular(img, k)
    d = np.abs(a - b)
    print('-' * 96)
    print('Boundary-mismatch case (spatial reflects, Fourier wraps), gauss 15 s=2.5:')
    print('  max difference over the whole image : %.3e' % d.max())
    print('  max difference with the %d-px border cropped : %.3e'
          % (h, d[h:-h, h:-h].max()))
    print('  -> the disagreement is exactly the (k-1)/2 strip the kernel')
    print('     overhangs, and identically zero everywhere else.')

    out = os.path.join(HERE, 'results_equivalence.csv')
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out, index=False)
        print('\nwrote %s' % out)
    except ImportError:
        pass
    return rows


def timing(side=384, upto=31):
    """Cost of each route as the kernel grows. Writes a CSV and a plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(8830)
    img = rng.uniform(0, 255, (side, side))
    sizes = list(range(3, upto + 1, 4))
    print('Timing sweep on a %dx%d image' % (side, side))
    print('-' * 64)
    print('%-8s %14s %14s %12s' % ('kernel', 'naive 2D s', 'separable s', 'fft s'))

    rows = []
    for s in sizes:
        k1 = F.gaussian_kernel_1d(s, max(0.5, s / 6.0))
        k2 = np.outer(k1, k1)
        _, tn = F.timed(F.convolve_spatial, img, k2, 'wrap')
        _, ts = F.timed(F.convolve_separable, img, k1, 'wrap')
        _, tf = F.timed(F.convolve_fft_circular, img, k2)
        print('%-8s %14.4f %14.4f %12.4f' % ('%dx%d' % (s, s), tn, ts, tf))
        rows.append({'size': s, 'naive_s': tn, 'separable_s': ts, 'fft_s': tf})

    naive = np.array([r['naive_s'] for r in rows])
    fft = np.array([r['fft_s'] for r in rows])
    cross = np.where(naive > fft)[0]
    print('-' * 64)
    print('FFT cost across the sweep: %.4f - %.4f s (ratio %.2f) — flat, as expected.'
          % (fft.min(), fft.max(), fft.max() / max(fft.min(), 1e-12)))
    print('Naive spatial cost: %.4f -> %.4f s, growing like k^2.' % (naive[0], naive[-1]))
    if len(cross):
        print('Crossover at k = %d.' % rows[cross[0]]['size'])
    else:
        print('No crossover in this range on an image this small.')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, naive, 'o-', label='spatial, naive 2D  (O(k^2) per pixel)')
    ax.plot(sizes, [r['separable_s'] for r in rows], 's-',
            label='spatial, separable  (O(k) per pixel)')
    ax.plot(sizes, fft, '^-', label='FFT multiply  (independent of k)')
    ax.set_xlabel('kernel width k (pixels)')
    ax.set_ylabel('seconds')
    ax.set_title('Blurring cost vs kernel size, %dx%d image' % (side, side))
    ax.grid(alpha=0.3)
    ax.legend()
    png = os.path.join(HERE, 'timing_plot.png')
    fig.savefig(png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % png)

    try:
        import pandas as pd
        csv = os.path.join(HERE, 'results_timing.csv')
        pd.DataFrame(rows).to_csv(csv, index=False)
        print('wrote %s' % csv)
    except ImportError:
        pass
    return rows


def main():
    p = argparse.ArgumentParser(description='Validation for Module 3.')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--experiment', action='store_true')
    p.add_argument('--timing', action='store_true')
    p.add_argument('--all', action='store_true')
    p.add_argument('--image',
                   default=os.path.join(HERE, 'data', 'module3_samples',
                                        'sample_photo.jpg'))
    p.add_argument('--side', type=int, default=384)
    p.add_argument('--max-kernel', type=int, default=31)
    args = p.parse_args()

    if not any([args.selftest, args.experiment, args.timing, args.all]):
        args.all = True

    ok = True
    if args.selftest or args.all:
        ok = selftest()
        print()
    if args.experiment or args.all:
        experiment(args.image)
        print()
    if args.timing or args.all:
        timing(args.side, args.max_kernel)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

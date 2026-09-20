"""
filtering.py

All the blurring math for Module 3, kept in one place so the Streamlit page
(`modules/module3/__init__.py`) and the command-line scripts (`blur.py`,
`verify_convolution.py`) are provably running the same code. Same reason
`common.py` exists for Module 2 -- if the app and the scripts each had their
own copy of the convolution, a "proof" that the two domains agree would only
be proving something about whichever copy I happened to run.

Nothing in here calls cv2.filter2D, cv2.blur or cv2.GaussianBlur to do the
actual work. The whole point of the assignment is the filtering itself, so
the spatial convolution is written out from scratch with numpy indexing, and
the frequency-domain version uses only the FFT. OpenCV shows up exactly once,
in `reference_opencv()`, which I use purely as an independent sanity check
that my from-scratch version is correct.

The one thing worth reading before the rest of this file: the DFT implements
*circular* convolution, not linear convolution. Multiplying two DFTs and
transforming back wraps the image around at the borders. So "does the spatial
result equal the Fourier result" only has a yes/no answer once you say which
boundary rule the spatial version used. Match the boundary handling
(`boundary='wrap'` vs `convolve_fft_circular`) and they agree to floating-point
noise; mismatch it and they differ in a border strip of exactly (k-1)/2 pixels
and nowhere else. Both cases are demonstrated in the app, because the second
one is the more instructive picture.

Convention note: this module computes true convolution, so the kernel is
flipped before it is applied. For the symmetric blur kernels here (box and
Gaussian) flipping changes nothing, but writing correlation and calling it
convolution would make the Fourier comparison an accident rather than a proof.
"""
import time

import numpy as np

# numpy pad mode for each boundary rule I expose in the UI.
#   wrap      -- circular; the one the DFT implicitly assumes
#   zero      -- pad with 0 (linear convolution); darkens the border
#   reflect   -- mirror across the edge pixel, OpenCV's BORDER_REFLECT_101
#   replicate -- repeat the edge pixel, OpenCV's BORDER_REPLICATE
_PAD_MODE = {
    'wrap': 'wrap',
    'zero': 'constant',
    'reflect': 'reflect',
    'replicate': 'edge',
}
BOUNDARIES = tuple(_PAD_MODE)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def box_kernel(size):
    """size x size mean filter, normalised so the image keeps its brightness.

    Every tap is 1/size^2. Summing to 1 is what makes this a *blur* rather
    than a blur plus a gain -- if the taps summed to S, the whole image would
    come out S times brighter, which in the frequency domain is just
    H(0,0) = S instead of 1.
    """
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError('kernel size must be a positive odd number, got %r' % size)
    return np.full((size, size), 1.0 / (size * size), dtype=np.float64)


def gaussian_kernel_1d(size, sigma):
    """1D Gaussian, sampled and then normalised to sum to 1.

    Normalising the *sampled* kernel rather than trusting the analytic
    1/(sigma*sqrt(2*pi)) factor matters: the continuous Gaussian has infinite
    support and I'm truncating it to `size` taps, so the samples I keep don't
    quite sum to 1 on their own. Dividing by the actual sum puts that back.
    """
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError('kernel size must be a positive odd number, got %r' % size)
    if sigma <= 0:
        raise ValueError('sigma must be positive, got %r' % sigma)
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_kernel(size, sigma):
    """2D Gaussian as the outer product of two 1D Gaussians.

    The 2D Gaussian is separable -- exp(-(x^2+y^2)/2s^2) factors into
    exp(-x^2/2s^2) * exp(-y^2/2s^2) -- which is why `convolve_separable`
    below can do the same job in O(k) work per pixel instead of O(k^2).
    """
    g = gaussian_kernel_1d(size, sigma)
    return np.outer(g, g)


def suggested_size_for_sigma(sigma):
    """Odd kernel width that captures ~99.7% of a Gaussian with this sigma.

    Cutting a Gaussian off at +/-3 sigma is the usual rule of thumb. Cutting
    it much tighter leaves a visible discontinuity at the kernel edge, which
    shows up as ringing in the transfer function.
    """
    size = int(2 * np.ceil(3.0 * float(sigma)) + 1)
    return max(3, size)


# ---------------------------------------------------------------------------
# Spatial domain
# ---------------------------------------------------------------------------
def convolve_spatial(image, kernel, boundary='wrap'):
    """2D convolution written out by hand. Works on 2D or 3D (colour) arrays.

    The direct definition is

        out[y, x] = sum_i sum_j  h[i, j] * f[y - i + cy, x - j + cx]

    with (cy, cx) the kernel centre. Rather than looping over pixels in
    Python -- which for a 12MP image and a 21x21 kernel is ~5 billion
    interpreted operations -- I loop over the *kernel taps* (at most a few
    hundred) and let numpy do each whole-image shift-and-accumulate as one
    vectorised operation. It is the same arithmetic in the same order, just
    with the loops transposed so the inner one runs in C.

    Boundary handling is done by padding first and then taking plain slices,
    so every output pixel is computed by the identical expression and there
    is no special case at the edges.
    """
    f = np.asarray(image, dtype=np.float64)
    h = np.asarray(kernel, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError('kernel must be 2D')
    if boundary not in _PAD_MODE:
        raise ValueError('boundary must be one of %s' % (BOUNDARIES,))

    if f.ndim == 3:            # colour: filter each channel independently
        return np.stack([convolve_spatial(f[..., c], h, boundary)
                         for c in range(f.shape[2])], axis=-1)

    kh, kw = h.shape
    cy, cx = kh // 2, kw // 2
    H, W = f.shape

    # Pad so that f[y - i + cy, x - j + cx] is always in range. Working
    # through the index limits: i runs 0..kh-1, so the row index reaches as
    # low as cy - (kh - 1) and as high as H - 1 + cy. Hence (kh-1-cy) rows
    # before and cy rows after.
    pad = ((kh - 1 - cy, cy), (kw - 1 - cx, cx))
    mode = _PAD_MODE[boundary]
    P = np.pad(f, pad, mode=mode) if mode != 'constant' else np.pad(f, pad, mode='constant', constant_values=0.0)

    # After padding, f[y - i + cy] sits at P[y + kh - 1 - i].
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(kh):
        r0 = kh - 1 - i
        for j in range(kw):
            w = h[i, j]
            if w == 0.0:
                continue
            c0 = kw - 1 - j
            out += w * P[r0:r0 + H, c0:c0 + W]
    return out


def convolve_separable(image, kernel_1d, boundary='wrap'):
    """Apply a separable kernel as two 1D passes (rows, then columns).

    Same result as convolving with np.outer(k, k), but the cost per pixel
    drops from k^2 multiply-adds to 2k. I include it because the speed
    comparison in the app is otherwise unfair to the spatial domain: the
    honest question is "FFT vs the *best* spatial method", not "FFT vs the
    naive one".
    """
    k = np.asarray(kernel_1d, dtype=np.float64).ravel()
    row = k.reshape(1, -1)
    col = k.reshape(-1, 1)
    return convolve_spatial(convolve_spatial(image, row, boundary), col, boundary)


_CV_BORDER = {
    'zero': 'BORDER_CONSTANT',
    'reflect': 'BORDER_REFLECT_101',
    'replicate': 'BORDER_REPLICATE',
}


def reference_opencv(image, kernel, boundary='reflect'):
    """cv2.filter2D for cross-checking only -- never used for the deliverable.

    Two details matter for this to be a fair check. First, cv2.filter2D
    computes *correlation*, so I flip the kernel to turn it into convolution;
    with symmetric blur kernels the flip is a no-op, but being sloppy here
    would defeat the purpose of the check. Second, OpenCV refuses
    BORDER_WRAP in filter2D, so the cross-check runs in 'reflect' mode --
    which is fine, since what I am testing is my convolution arithmetic, and
    the wrap case is already pinned down by the Fourier comparison itself.
    """
    import cv2
    if boundary not in _CV_BORDER:
        raise ValueError("OpenCV's filter2D does not support boundary=%r "
                         "(it rejects BORDER_WRAP); use one of %s"
                         % (boundary, tuple(_CV_BORDER)))
    f = np.asarray(image, dtype=np.float64)
    h = np.asarray(kernel, dtype=np.float64)
    return cv2.filter2D(f, -1, h[::-1, ::-1],
                        borderType=getattr(cv2, _CV_BORDER[boundary]))


# ---------------------------------------------------------------------------
# Frequency domain
# ---------------------------------------------------------------------------
def kernel_to_psf(kernel, shape):
    """Embed the kernel in an image-sized array with its centre at index (0,0).

    This is the step people usually get wrong. The DFT treats index 0 as the
    origin, but my kernel's origin is its middle tap. If I just drop the
    kernel into the top-left corner and transform it, the result is correct
    up to a linear phase ramp -- which comes back as the whole image being
    shifted diagonally by (cy, cx) pixels. Rolling the centre tap to (0, 0)
    first is what removes that shift.
    """
    h = np.asarray(kernel, dtype=np.float64)
    kh, kw = h.shape
    H, W = shape[:2]
    if kh > H or kw > W:
        raise ValueError('kernel is larger than the image')
    psf = np.zeros((H, W), dtype=np.float64)
    psf[:kh, :kw] = h
    return np.roll(psf, (-(kh // 2), -(kw // 2)), axis=(0, 1))


def transfer_function(kernel, shape):
    """H(u, v): the DFT of the padded, centred kernel. The filter's frequency response."""
    return np.fft.fft2(kernel_to_psf(kernel, shape))


def convolve_fft_circular(image, kernel):
    """Blur by multiplying in the Fourier domain: F(u,v) * H(u,v), then invert.

    This is the convolution theorem applied literally, and it is the exact
    counterpart of convolve_spatial(..., boundary='wrap') -- same arithmetic,
    different route. The imaginary part of the inverse transform is pure
    round-off (both inputs are real, so the spectrum is conjugate-symmetric
    and the result must be real); I return its size from `compare` so it can
    be reported rather than silently discarded.
    """
    f = np.asarray(image, dtype=np.float64)
    if f.ndim == 3:
        return np.stack([convolve_fft_circular(f[..., c], kernel)
                         for c in range(f.shape[2])], axis=-1)
    Hf = transfer_function(kernel, f.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(f) * Hf))


def convolve_fft_linear(image, kernel):
    """Fourier-domain convolution that reproduces *linear* (zero-padded) filtering.

    Circular wrap-around is an artefact of the DFT's periodicity, not of
    convolution. Zero-padding both signals to at least (H+kh-1, W+kw-1)
    gives the wrap-around nowhere to land, so the product of the DFTs now
    equals linear convolution -- matching convolve_spatial(..., 'zero')
    instead. Cropping back to the original size takes the 'same' region.
    """
    f = np.asarray(image, dtype=np.float64)
    if f.ndim == 3:
        return np.stack([convolve_fft_linear(f[..., c], kernel)
                         for c in range(f.shape[2])], axis=-1)
    h = np.asarray(kernel, dtype=np.float64)
    kh, kw = h.shape
    H, W = f.shape
    # next_fast_len would be faster still; plain H+kh-1 keeps the padding
    # arithmetic obvious, which matters more here than the last few ms.
    sh, sw = H + kh - 1, W + kw - 1
    F = np.fft.fft2(f, s=(sh, sw))
    Hf = np.fft.fft2(h, s=(sh, sw))
    full = np.real(np.fft.ifft2(F * Hf))
    cy, cx = kh // 2, kw // 2
    return full[cy:cy + H, cx:cx + W]


# ---------------------------------------------------------------------------
# Comparing the two routes
# ---------------------------------------------------------------------------
def compare(a, b, peak=255.0):
    """Error metrics between two versions of the same blurred image.

    max_abs is the number that actually settles the question: if the two
    domains are doing the same thing, it should sit at the level of
    double-precision round-off accumulated over the transform (~1e-10 or
    smaller for 8-bit-valued inputs), not at the level of a visible pixel
    difference. PSNR is included because it is the familiar image-quality
    number, but it saturates and is the less informative of the two here.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    return {
        'max_abs': float(d.max()),
        'mean_abs': float(d.mean()),
        'rmse': float(np.sqrt(mse)),
        'psnr_db': float('inf') if mse == 0 else float(10.0 * np.log10(peak * peak / mse)),
        'eps_multiples': float(d.max() / np.finfo(np.float64).eps),
    }


def difference_map(a, b, gain=None):
    """|a - b| stretched to 0..255 so a difference of 1e-13 is actually visible.

    Displayed without the stretch, an all-round-off difference image is just
    a black rectangle, which proves the point but shows nothing. `gain` is
    reported alongside the picture so the stretch is not mistaken for signal.
    """
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    if d.ndim == 3:
        d = d.mean(axis=2)
    peak = d.max()
    if gain is None:
        gain = 255.0 / peak if peak > 0 else 1.0
    return np.clip(d * gain, 0, 255).astype(np.uint8), gain


def log_spectrum(x, is_spectrum=False):
    """log(1 + |X|) with DC shifted to the middle -- the standard way to look at a spectrum.

    The raw magnitude spans several orders of magnitude between DC and the
    high frequencies, so on a linear scale every image looks like a single
    bright dot in the corner. The log compresses that; fftshift just moves
    DC from the corner to the centre so the picture is symmetric about the
    middle and easier to read.
    """
    X = np.asarray(x) if is_spectrum else np.fft.fft2(np.asarray(x, np.float64))
    mag = np.abs(np.fft.fftshift(X))
    out = np.log1p(mag)
    m = out.max()
    return (out / m * 255.0).astype(np.uint8) if m > 0 else out.astype(np.uint8)


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_seconds). perf_counter, not time()."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Test imagery (so the demo works with no uploads)
# ---------------------------------------------------------------------------
def synthetic(kind='checkerboard', size=256):
    """Generated test images, as float arrays in 0..255.

    Real photographs are the convincing demo, but synthetic patterns are the
    better *test*: a checkerboard is mostly high frequency, a step edge has a
    known analytic response to a box filter, and white noise has a flat
    expected spectrum, so I can predict what the filter should do to each and
    check that it did.
    """
    n = int(size)
    if kind == 'checkerboard':
        sq = max(1, n // 16)
        yy, xx = np.mgrid[0:n, 0:n]
        return np.where(((yy // sq) + (xx // sq)) % 2 == 0, 255.0, 0.0)
    if kind == 'step edge':
        img = np.zeros((n, n))
        img[:, n // 2:] = 255.0
        return img
    if kind == 'circle':
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.sqrt((yy - n / 2.0) ** 2 + (xx - n / 2.0) ** 2)
        return np.where(r < n / 3.5, 255.0, 30.0)
    if kind == 'white noise':
        rng = np.random.default_rng(8830)      # fixed seed: reruns are comparable
        return rng.uniform(0, 255, (n, n))
    if kind == 'sine grating':
        xx = np.arange(n)
        return 127.5 + 110.0 * np.sin(2 * np.pi * xx / 16.0)[None, :] * np.ones((n, 1))
    if kind == 'impulse':
        # The impulse response: blurring a single lit pixel returns the kernel
        # itself, which is the most direct demonstration that the filter *is*
        # the point spread function.
        img = np.zeros((n, n))
        img[n // 2, n // 2] = 255.0
        return img
    raise ValueError('unknown synthetic pattern %r' % kind)


SYNTHETIC_KINDS = ('checkerboard', 'step edge', 'circle',
                   'white noise', 'sine grating', 'impulse')


def to_uint8(x):
    """Clip to 0..255 and cast, for display only. Never feed this back into the math."""
    return np.clip(np.asarray(x), 0, 255).astype(np.uint8)


def to_gray(img):
    """Luminance, using the Rec.601 weights OpenCV uses. Input may be RGB or already gray."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        return a
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

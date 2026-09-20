"""
filtering.py -- all the blurring math for Module 3.

Kept in one file so the web page and the command-line scripts (blur.py,
verify_convolution.py) run the same code. If they each had their own copy of
the convolution, "the two domains agree" wouldn't prove much.

I don't use cv2.filter2D / cv2.blur / cv2.GaussianBlur to do the blurring --
writing the filter is the assignment. OpenCV is used once, in
reference_opencv(), only to check my version against something I didn't write.

Important: the DFT gives *circular* convolution (the image wraps at the
borders). So "spatial == Fourier" is only true if the spatial side wraps too.
Match them and they agree to ~1e-13; mismatch them and they differ only in a
(k-1)/2 border strip.
"""
import time

import numpy as np

# numpy pad mode for each boundary option in the UI.
#   wrap      = circular, what the DFT assumes
#   zero      = pad with 0 (linear convolution), darkens the border
#   reflect   = mirror at the edge, same as OpenCV BORDER_REFLECT_101
#   replicate = repeat the edge pixel, same as OpenCV BORDER_REPLICATE
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
    """size x size mean filter. Every tap is 1/size^2.

    Taps must sum to 1, otherwise the image also gets brighter/darker instead
    of just blurring.
    """
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError('kernel size must be a positive odd number, got %r' % size)
    return np.full((size, size), 1.0 / (size * size), dtype=np.float64)


def gaussian_kernel_1d(size, sigma):
    """1D Gaussian, sampled then divided by its own sum.

    I normalise by the actual sum instead of the analytic 1/(sigma*sqrt(2pi)):
    the real Gaussian is infinite and I'm cutting it off at `size` taps, so the
    samples I keep don't add up to 1 on their own.
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
    """2D Gaussian = outer product of two 1D Gaussians.

    Works because exp(-(x^2+y^2)/2s^2) splits into exp(-x^2/2s^2) *
    exp(-y^2/2s^2). That's also why convolve_separable() below works.
    """
    g = gaussian_kernel_1d(size, sigma)
    return np.outer(g, g)


def suggested_size_for_sigma(sigma):
    """Kernel width covering +/-3 sigma (~99.7% of the Gaussian).

    Standard rule of thumb. Cut it much tighter and the kernel ends with a
    visible jump, which causes ringing.
    """
    size = int(2 * np.ceil(3.0 * float(sigma)) + 1)
    return max(3, size)


# ---------------------------------------------------------------------------
# Spatial domain
# ---------------------------------------------------------------------------
def convolve_spatial(image, kernel, boundary='wrap'):
    """2D convolution, written out. Handles 2D (gray) and 3D (colour).

    The definition is:
        out[y, x] = sum_i sum_j  h[i, j] * f[y - i + cy, x - j + cx]

    Looping over pixels in Python would be far too slow, so I loop over the
    kernel taps instead (a few hundred at most) and let numpy do each
    whole-image shift as one operation. Same arithmetic, loops swapped.

    I pad the image first, so every output pixel uses the same expression and
    there's no special case at the edges.
    """
    f = np.asarray(image, dtype=np.float64)
    h = np.asarray(kernel, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError('kernel must be 2D')
    if boundary not in _PAD_MODE:
        raise ValueError('boundary must be one of %s' % (BOUNDARIES,))

    if f.ndim == 3:            # colour: do each channel separately
        return np.stack([convolve_spatial(f[..., c], h, boundary)
                         for c in range(f.shape[2])], axis=-1)

    kh, kw = h.shape
    cy, cx = kh // 2, kw // 2
    H, W = f.shape

    # How much padding: i goes 0..kh-1, so the row index reaches down to
    # cy-(kh-1) and up to H-1+cy. That's (kh-1-cy) rows before, cy rows after.
    pad = ((kh - 1 - cy, cy), (kw - 1 - cx, cx))
    mode = _PAD_MODE[boundary]
    P = np.pad(f, pad, mode=mode) if mode != 'constant' else np.pad(f, pad, mode='constant', constant_values=0.0)

    # After padding, f[y - i + cy] lives at P[y + kh - 1 - i].
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
    """Separable kernel done as two 1D passes: rows, then columns.

    Same answer as convolving with np.outer(k, k), but 2k multiply-adds per
    pixel instead of k^2. I added this so the speed comparison is fair -- the
    real question is FFT vs the *best* spatial method, not the naive one.
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
    """cv2.filter2D, used only to check my convolution. Not used for results.

    Two gotchas: filter2D does correlation, not convolution, so I flip the
    kernel (no-op for symmetric blur kernels, but the check should be honest).
    And filter2D rejects BORDER_WRAP, so this check runs in 'reflect' mode.
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
    """Put the kernel in an image-sized array with its centre at index (0,0).

    This is the step that's easy to get wrong. The DFT treats index 0 as the
    origin, but the kernel's origin is its middle tap. Drop the kernel in the
    top-left and transform it and the blur comes out shifted diagonally by
    (cy, cx) pixels -- easy to miss, because a shifted blur still looks blurry.
    np.roll fixes it.
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
    """H(u, v) -- the DFT of the padded kernel, i.e. the filter's frequency response."""
    return np.fft.fft2(kernel_to_psf(kernel, shape))


def convolve_fft_circular(image, kernel):
    """Blur by multiplying in the Fourier domain: F(u,v) * H(u,v), then invert.

    This is the convolution theorem used directly. It's the counterpart of
    convolve_spatial(..., boundary='wrap'). Both inputs are real so the result
    must be real; the imaginary part is round-off, so I take np.real.
    """
    f = np.asarray(image, dtype=np.float64)
    if f.ndim == 3:
        return np.stack([convolve_fft_circular(f[..., c], kernel)
                         for c in range(f.shape[2])], axis=-1)
    Hf = transfer_function(kernel, f.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(f) * Hf))


def convolve_fft_linear(image, kernel):
    """Fourier version that gives *linear* (zero-padded) convolution instead.

    The wrap-around is a side effect of the DFT being periodic, not of
    convolution itself. Zero-pad both to (H+kh-1, W+kw-1) and the wrap has
    nowhere to land, so the DFT product now equals linear convolution --
    matching convolve_spatial(..., 'zero'). Then crop back to the middle.
    """
    f = np.asarray(image, dtype=np.float64)
    if f.ndim == 3:
        return np.stack([convolve_fft_linear(f[..., c], kernel)
                         for c in range(f.shape[2])], axis=-1)
    h = np.asarray(kernel, dtype=np.float64)
    kh, kw = h.shape
    H, W = f.shape
    # next_fast_len would be a bit faster, but plain H+kh-1 keeps the padding
    # easy to follow.
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
    """Error metrics between the two blurred versions of the same image.

    max_abs is the one that answers the question. If both routes do the same
    thing it should be down at float64 round-off (~1e-13 here), not at the
    level of a pixel you could see. PSNR is here because it's the familiar
    number, but it saturates and says less.
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
    """|a - b| stretched to 0..255 so a 1e-13 difference is actually visible.

    Without the stretch the difference image is just a black rectangle. I
    return the gain too, so the brightness isn't mistaken for real signal.
    """
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    if d.ndim == 3:
        d = d.mean(axis=2)
    peak = d.max()
    if gain is None:
        gain = 255.0 / peak if peak > 0 else 1.0
    return np.clip(d * gain, 0, 255).astype(np.uint8), gain


def log_spectrum(x, is_spectrum=False):
    """log(1 + |X|) with DC moved to the middle -- the usual way to view a spectrum.

    Raw magnitudes span several orders of magnitude, so on a linear scale you
    just see one bright dot in the corner. The log compresses that, and
    fftshift moves DC to the centre so the picture is symmetric.
    """
    X = np.asarray(x) if is_spectrum else np.fft.fft2(np.asarray(x, np.float64))
    mag = np.abs(np.fft.fftshift(X))
    out = np.log1p(mag)
    m = out.max()
    return (out / m * 255.0).astype(np.uint8) if m > 0 else out.astype(np.uint8)


def timed(fn, *args, **kwargs):
    """Run fn, return (result, seconds). perf_counter, not time()."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Test images (so the demo works without uploading anything)
# ---------------------------------------------------------------------------
def synthetic(kind='checkerboard', size=256):
    """Generated test images as float arrays in 0..255.

    A photo is the nicer demo, but these are the better *test*: I know what a
    blur should do to each one, so I can check the output. A checkerboard is
    nearly all high frequency, a step edge has a known box-filter response,
    white noise has a flat spectrum.
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
        rng = np.random.default_rng(8830)      # fixed seed so reruns match
        return rng.uniform(0, 255, (n, n))
    if kind == 'sine grating':
        xx = np.arange(n)
        return 127.5 + 110.0 * np.sin(2 * np.pi * xx / 16.0)[None, :] * np.ones((n, 1))
    if kind == 'impulse':
        # Blurring one lit pixel gives back the kernel itself, which is the
        # simplest way to show the filter IS its point spread function.
        img = np.zeros((n, n))
        img[n // 2, n // 2] = 255.0
        return img
    raise ValueError('unknown synthetic pattern %r' % kind)


SYNTHETIC_KINDS = ('checkerboard', 'step edge', 'circle',
                   'white noise', 'sine grating', 'impulse')


def to_uint8(x):
    """Clip to 0..255 and cast. Display only -- don't feed this back into the math."""
    return np.clip(np.asarray(x), 0, 255).astype(np.uint8)


def to_gray(img):
    """Luminance using the Rec.601 weights OpenCV uses. Input may be RGB or gray."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        return a
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

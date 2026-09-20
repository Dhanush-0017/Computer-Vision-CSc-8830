"""
modules/module3 -- Module 3: Image Blurring & the Convolution Theorem.

The Module 3 page. app.py imports this; it doesn't run on its own.

Where each part of the assignment lives:
  "Implement image blurring"        -> tab 1 (box + Gaussian, my own
                                       convolution from filtering.py)
  "Show spatial == Fourier"         -> tab 2 (the experiment), tab 3
                                       (spectra), tab 5 (small worked
                                       example), tab 6 (the proof)
  "Use the implementation for
   evidence/validation"             -> every number here is computed live
                                       from the selected image

One thing to know up front: the DFT gives circular convolution, so
"spatial == Fourier" only holds once the spatial side wraps too. Matched, they
agree to ~1e-13. Mismatched, they only differ in a (k-1)/2 border strip. Tab 2
shows both, since the second case is arguably the better demonstration.
"""
import io
import os

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import filtering as F

# --- metadata read by app.py to build the navigation ------------------------
NUMBER = 3
TITLE = 'Blurring & the Convolution Theorem'
SUBTITLE = ('Box and Gaussian blurring by explicit 2D convolution, shown to be '
            'identical to multiplying by the filter in the Fourier domain.')
STATUS = 'complete'

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, 'data', 'module3_samples', 'sample_photo.jpg')

# Naive spatial convolution costs O(H*W*k^2). Past roughly a quarter of a
# megapixel with a large kernel the page stops feeling interactive, so the
# working image gets capped and the cap is stated in the UI rather than
# quietly applied.
MAX_SIDE = 512


# ---------------------------------------------------------------------------
# Image plumbing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_sample():
    """The bundled photo, as a float array. Borrowed from Module 2's data/ and
    downscaled -- Module 3 has no calibration requirement, so any image does;
    this one is here only so the demo runs with nothing uploaded."""
    import cv2
    img = cv2.imread(SAMPLE)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)


def _decode(uploaded):
    import cv2
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)


def _fit(img, max_side=MAX_SIDE):
    """Downscale so the long side is at most max_side. Returns (image, was_resized)."""
    import cv2
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, False
    s = max_side / float(max(h, w))
    out = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                     interpolation=cv2.INTER_AREA)
    return out.astype(np.float64), True


def _pick_image(key):
    """Sidebar-style image chooser, reused by several tabs via a unique key."""
    src = st.radio('Image source',
                   ['Sample photo', 'Generated test pattern', 'Upload your own'],
                   horizontal=True, key=key + '_src')
    img = None
    if src == 'Sample photo':
        img = _load_sample()
        if img is None:
            st.warning('Sample photo not found at data/module3_samples/ — '
                       'pick a generated pattern or upload an image.')
    elif src == 'Generated test pattern':
        kind = st.selectbox('Pattern', F.SYNTHETIC_KINDS, key=key + '_kind')
        img = F.synthetic(kind, 256)
        st.caption(_PATTERN_NOTE.get(kind, ''))
    else:
        up = st.file_uploader('Image file', type=['jpg', 'jpeg', 'png', 'bmp'],
                              key=key + '_up')
        if up is not None:
            img = _decode(up)
            if img is None:
                st.error('Could not decode that file.')
    if img is None:
        return None
    if st.checkbox('Convert to grayscale', value=(img.ndim == 2),
                   key=key + '_gray', disabled=(img.ndim == 2)):
        img = F.to_gray(img)
    img, resized = _fit(img)
    if resized:
        st.caption('Downscaled to %d x %d so the naive spatial convolution '
                   'stays interactive. The math is unaffected — it just runs '
                   'on fewer pixels.' % (img.shape[1], img.shape[0]))
    return img


_PATTERN_NOTE = {
    'checkerboard': 'Almost all high frequency, so a blur has a dramatic and '
                    'easily judged effect.',
    'step edge': 'A box filter turns a step into a ramp exactly (k+1) pixels '
                 'wide — an analytic result you can check against the output.',
    'circle': 'Curved edges make the isotropy of the Gaussian versus the '
              'squareness of the box filter visible.',
    'white noise': 'Flat expected spectrum, so the blurred spectrum shows the '
                   'transfer function |H(u,v)| almost directly.',
    'sine grating': 'A single spatial frequency. Blurring can only scale its '
                    'amplitude by |H| at that frequency — it cannot change '
                    'the pattern — which is the convolution theorem visible '
                    'in one picture.',
    'impulse': 'Blurring one lit pixel returns the kernel itself: the filter '
               'IS its impulse response (point spread function).',
}


def _kernel_controls(key, default_size=9):
    """Filter picker shared by the tabs. Returns (kernel, label, kernel_1d_or_None)."""
    c1, c2 = st.columns(2)
    kind = c1.selectbox('Filter', ['Gaussian', 'Box (mean)'], key=key + '_kind')
    if kind == 'Box (mean)':
        size = c2.slider('Kernel size (odd)', 3, 31, default_size, step=2,
                         key=key + '_size')
        return F.box_kernel(size), 'box %dx%d' % (size, size), None
    sigma = c2.slider('Sigma', 0.5, 8.0, 2.0, step=0.1, key=key + '_sigma')
    auto = st.checkbox('Size kernel automatically (±3σ)', value=True,
                       key=key + '_auto')
    if auto:
        size = F.suggested_size_for_sigma(sigma)
        st.caption('Kernel size %d x %d, from the ±3σ rule.' % (size, size))
    else:
        size = st.slider('Kernel size (odd)', 3, 41, default_size, step=2,
                         key=key + '_gsize')
    k1 = F.gaussian_kernel_1d(size, sigma)
    return np.outer(k1, k1), 'Gaussian σ=%.1f, %dx%d' % (sigma, size, size), k1


def _show(img, caption, width=None):
    st.image(F.to_uint8(img), caption=caption, use_container_width=(width is None),
             width=width, clamp=True)


def _fig(figure):
    """Render a matplotlib figure and close it, so reruns don't leak figures."""
    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(figure)
    st.image(buf.getvalue(), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 1 -- the blurring implementation
# ---------------------------------------------------------------------------
def _tab_blur():
    st.subheader('Blurring by explicit spatial convolution')
    st.markdown(
        'Both filters use my own convolution in `filtering.py`. It loops '
        'over the kernel taps and adds up shifted copies of the image, which '
        'is the definition of convolution with the per-pixel loop handed to '
        'numpy. `cv2.blur` and `cv2.GaussianBlur` are not used on this page.')

    img = _pick_image('t1')
    if img is None:
        return
    kernel, label, k1 = _kernel_controls('t1')
    boundary = st.selectbox('Boundary handling', F.BOUNDARIES, index=0,
                            key='t1_b',
                            help='How the filter treats pixels that fall off '
                                 'the edge of the image. "wrap" is the rule '
                                 'the DFT assumes, which is why it is the '
                                 'default here.')

    blurred, secs = F.timed(F.convolve_spatial, img, kernel, boundary)

    c1, c2 = st.columns(2)
    with c1:
        _show(img, 'Original')
    with c2:
        _show(blurred, 'Blurred — %s, %s boundary (%.3f s)' % (label, boundary, secs))

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('**The kernel**')
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        im = ax.imshow(kernel, cmap='viridis')
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
        _fig(fig)
        st.caption('Taps sum to %.10f. They have to sum to 1, or the filter '
                   'changes the overall brightness as well as blurring. In '
                   'frequency terms that is H(0,0) ≠ 1.' % kernel.sum())
    with c2:
        st.markdown('**Cost of this filter**')
        kh = kernel.shape[0]
        px = img.shape[0] * img.shape[1]
        st.markdown(
            '- Image: %d × %d = %s pixels\n'
            '- Kernel: %d × %d = %d taps\n'
            '- Multiply–adds, naive 2D: **%s**\n'
            '- Multiply–adds if run separably (%d + %d per pixel): **%s**'
            % (img.shape[1], img.shape[0], format(px, ','),
               kh, kh, kh * kh, format(px * kh * kh, ','),
               kh, kh, format(px * 2 * kh, ',')))
        if k1 is not None:
            sep, ssecs = F.timed(F.convolve_separable, img, k1, boundary)
            err = F.compare(sep, blurred)['max_abs']
            st.success('Separable run: %.3f s vs %.3f s, and it matches the '
                       'full 2D convolution to %.2e. So the 2D Gaussian does '
                       'split into two 1D passes.' % (ssecs, secs, err))
        else:
            st.info('The box filter is separable too — a 2D mean is a row '
                    'mean then a column mean. Switch to Gaussian to see the '
                    'separable timing side by side.')


# ---------------------------------------------------------------------------
# Tab 2 -- the main experiment
# ---------------------------------------------------------------------------
def _tab_equivalence():
    st.subheader('Spatial filtering vs. its Fourier-domain equivalent')
    st.markdown(
        'The convolution theorem says these two procedures must produce the '
        'same image:')
    c1, c2 = st.columns(2)
    c1.markdown('**Route A — spatial**\n\n'
                'Slide the kernel over the image and accumulate, '
                '`convolve_spatial`.')
    c2.markdown('**Route B — frequency**\n\n'
                'FFT the image, FFT the kernel, multiply the two pointwise, '
                'inverse FFT, `convolve_fft_circular`.')
    st.latex(r'f(x,y) * h(x,y) \;\;\Longleftrightarrow\;\; F(u,v)\,H(u,v)')

    img = _pick_image('t2')
    if img is None:
        return
    kernel, label, _ = _kernel_controls('t2')

    mode = st.radio(
        'Which pairing to test',
        ['Matched — circular convolution both ways',
         'Matched — linear (zero-padded) convolution both ways',
         'Mismatched — wrap in space vs. the plain DFT product'],
        key='t2_mode')

    if mode.startswith('Matched — circular'):
        a, ta = F.timed(F.convolve_spatial, img, kernel, 'wrap')
        b, tb = F.timed(F.convolve_fft_circular, img, kernel)
        note = ('Both routes are computing circular convolution: the spatial '
                'side wraps at the borders, and the DFT does that implicitly '
                'because it treats the image as one period of an infinitely '
                'tiled signal. Like for like.')
    elif mode.startswith('Matched — linear'):
        a, ta = F.timed(F.convolve_spatial, img, kernel, 'zero')
        b, tb = F.timed(F.convolve_fft_linear, img, kernel)
        note = ('Zero-padding both signals out to (H+k−1, W+k−1) before the '
                'transform leaves the wrap-around nowhere to land, so the '
                'product of the DFTs now equals *linear* convolution — the '
                'match to zero-padded spatial filtering.')
    else:
        a, ta = F.timed(F.convolve_spatial, img, kernel, 'reflect')
        b, tb = F.timed(F.convolve_fft_circular, img, kernel)
        note = ('Deliberately mismatched: the spatial side mirrors at the '
                'border while the Fourier side wraps. The theorem is not '
                'violated — the two are simply being asked different '
                'questions at the edges.')

    st.info(note)

    c1, c2, c3 = st.columns(3)
    with c1:
        _show(a, 'Route A — spatial (%.3f s)' % ta)
    with c2:
        _show(b, 'Route B — Fourier (%.3f s)' % tb)
    with c3:
        dmap, gain = F.difference_map(a, b)
        st.image(dmap, caption='|A − B|, brightness ×%.3g' % gain,
                 use_container_width=True)

    m = F.compare(a, b)
    st.markdown('**Measured difference between the two routes**')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Max |A − B|', '%.3e' % m['max_abs'])
    c2.metric('Mean |A − B|', '%.3e' % m['mean_abs'])
    c3.metric('RMSE', '%.3e' % m['rmse'])
    c4.metric('PSNR', '∞ dB' if np.isinf(m['psnr_db']) else '%.1f dB' % m['psnr_db'])

    if mode.startswith('Matched'):
        if m['max_abs'] < 1e-8:
            st.success(
                'Same image, to floating-point round-off. The biggest '
                'difference at any pixel is %.3e, about %.0f float64 rounding '
                'steps. On a 0–255 scale that is around one part in %.0e. So '
                'convolving in space and multiplying in frequency give the '
                'same result.'
                % (m['max_abs'], m['eps_multiples'], 255.0 / max(m['max_abs'], 1e-300)))
        else:
            st.error('Difference is %.3e, way too big to be round-off. '
                     'That would mean a bug in my code.' % m['max_abs'])
    else:
        h = kernel.shape[0] // 2
        if min(a.shape[0], a.shape[1]) > 2 * h + 2:
            d = np.abs(F.to_gray(a) - F.to_gray(b))
            interior = d[h:-h, h:-h] if h > 0 else d
            st.warning(
                'Max difference over the whole image: **%.3e**. But crop off '
                'the outer %d pixels — which is exactly (k−1)/2, how far the '
                'kernel hangs over the edge — and the max difference across '
                'the whole interior drops to **%.3e**. So the theorem is not '
                'failing. The two sides just handle the border differently, '
                'and they agree everywhere the border does not matter.'
                % (m['max_abs'], h, interior.max()))
            fig, ax = plt.subplots(figsize=(5.2, 2.4))
            ax.plot(d[d.shape[0] // 2, :], lw=0.9)
            ax.set_title('One horizontal scanline of |A − B|', fontsize=9)
            ax.set_xlabel('column'); ax.set_ylabel('abs difference')
            ax.axvspan(0, h, color='tab:red', alpha=0.15)
            ax.axvspan(d.shape[1] - h, d.shape[1], color='tab:red', alpha=0.15)
            _fig(fig)
            st.caption('Shaded: the (k−1)/2 border strips. Between them the '
                       'trace sits flat on zero.')

    with st.expander('Independent cross-check against OpenCV'):
        st.markdown(
            'My convolution being self-consistent with my FFT would not prove '
            'much if both were wrong in the same way, so here it is against a '
            'library implementation nobody can accuse me of having written. '
            '`cv2.filter2D` computes correlation and rejects BORDER_WRAP, so '
            'the check runs in reflect mode with the kernel flipped.')
        mine = F.convolve_spatial(img, kernel, 'reflect')
        theirs = F.reference_opencv(img, kernel, 'reflect')
        st.metric('Max |mine − cv2.filter2D|',
                  '%.3e' % F.compare(mine, theirs)['max_abs'])


# ---------------------------------------------------------------------------
# Tab 3 -- spectra
# ---------------------------------------------------------------------------
def _tab_spectra():
    st.subheader('What blurring looks like in the frequency domain')
    st.markdown(
        'If the theorem holds, the spectrum of the blurred image is the '
        'spectrum of the original multiplied, frequency by frequency, by the '
        'filter\'s transfer function. Blurring is then simply what a low-pass '
        'multiplier does: leave the low frequencies near DC alone, scale the '
        'high ones down.')

    img = _pick_image('t3')
    if img is None:
        return
    gray = F.to_gray(img)
    kernel, label, _ = _kernel_controls('t3')

    Fu = np.fft.fft2(gray)
    Hu = F.transfer_function(kernel, gray.shape)
    Gu = Fu * Hu
    blurred = np.real(np.fft.ifft2(Gu))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(F.log_spectrum(Fu, True), caption='|F(u,v)| — the image',
                 use_container_width=True)
    with c2:
        st.image(F.log_spectrum(Hu, True), caption='|H(u,v)| — the filter (%s)' % label,
                 use_container_width=True)
    with c3:
        st.image(F.log_spectrum(Gu, True), caption='|F·H| — the product',
                 use_container_width=True)
    st.caption('All three are log(1+|·|) with DC shifted to the centre. '
               'Bright centre, dark rim in the middle panel is exactly what a '
               'low-pass filter should look like.')

    st.divider()
    st.markdown('**Radial profile of the transfer function**')
    Hs = np.abs(np.fft.fftshift(Hu))
    cy, cx = Hs.shape[0] // 2, Hs.shape[1] // 2
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(Hs[cy, cx:], lw=1.2)
    ax.axhline(0, color='0.7', lw=0.6)
    ax.set_xlabel('spatial frequency (cycles per image width, from DC)')
    ax.set_ylabel('|H(u,0)|')
    ax.set_title('Horizontal slice through H, %s' % label, fontsize=9)
    _fig(fig)
    st.markdown(
        'H(0,0) = **%.6f**, which is the sum of the kernel taps. It being 1 '
        'is why the average brightness survives the blur. The shape also tells '
        'the two filters apart: a box filter is a rectangle in space, so its '
        'transform is a sinc that **rings** — it dips below zero and back, '
        'which is where the faint ripples near sharp edges come from. A '
        'Gaussian transforms into another Gaussian, so no ringing. Switch the '
        'filter above and watch the curve change.' % Hs[cy, cx])


# ---------------------------------------------------------------------------
# Tab 4 -- cost
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _timing_sweep(h, w, sizes, seed=8830):
    """Time the three routes across kernel sizes. Cached: the sweep is the slow
    part of this page and its result only depends on the arguments."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 255, (h, w))
    rows = []
    for s in sizes:
        k1 = F.gaussian_kernel_1d(s, max(0.5, s / 6.0))
        k2 = np.outer(k1, k1)
        _, t_naive = F.timed(F.convolve_spatial, img, k2, 'wrap')
        _, t_sep = F.timed(F.convolve_separable, img, k1, 'wrap')
        _, t_fft = F.timed(F.convolve_fft_circular, img, k2)
        rows.append({'kernel': '%dx%d' % (s, s), 'size': s,
                     'spatial_naive_s': t_naive,
                     'spatial_separable_s': t_sep,
                     'fft_s': t_fft})
    return rows


def _tab_timing():
    st.subheader('Why the frequency domain is worth the trouble')
    st.markdown(
        'The two routes give the same answer, so the choice between them is '
        'about cost. Spatial convolution does k² multiply–adds per pixel, so '
        'it grows with the **area** of the kernel. The FFT route costs '
        'O(HW log HW) no matter how big the kernel is — the kernel only has to '
        'be padded up to the image size, and after that a 3×3 and a 31×31 '
        'filter are the same amount of work. Somewhere there is a crossover.')

    c1, c2 = st.columns(2)
    side = c1.select_slider('Test image size', [128, 256, 384, 512], value=256,
                            key='t4_side')
    upto = c2.select_slider('Largest kernel', [15, 21, 31, 41], value=31,
                            key='t4_max')
    if not st.button('Run the timing sweep', key='t4_go'):
        st.caption('Runs on generated noise of the chosen size so the numbers '
                   'are reproducible and do not depend on which photo is loaded.')
        return

    sizes = [s for s in range(3, upto + 1, 4)]
    with st.spinner('Timing %d kernel sizes on a %d×%d image…' % (len(sizes), side, side)):
        rows = _timing_sweep(side, side, tuple(sizes))

    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(df['size'], df['spatial_naive_s'], 'o-', label='spatial, naive 2D')
    ax.plot(df['size'], df['spatial_separable_s'], 's-', label='spatial, separable')
    ax.plot(df['size'], df['fft_s'], '^-', label='FFT multiply')
    ax.set_xlabel('kernel width k (pixels)')
    ax.set_ylabel('seconds')
    ax.set_title('Cost vs kernel size, %d×%d image' % (side, side), fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _fig(fig)

    st.dataframe(df[['kernel', 'spatial_naive_s', 'spatial_separable_s', 'fft_s']],
                 use_container_width=True, hide_index=True)

    naive = df['spatial_naive_s'].to_numpy()
    fft = df['fft_s'].to_numpy()
    cross = np.where(naive > fft)[0]
    flat = fft.max() / max(fft.min(), 1e-12)
    st.markdown(
        '- The FFT trace is essentially **flat** (max/min ratio %.2f across the '
        'whole sweep) — the kernel size genuinely does not enter its cost.\n'
        '- The naive spatial trace grows like k², as predicted: from %.4f s at '
        '3×3 to %.4f s at %s.\n'
        '- %s'
        % (flat, naive[0], naive[-1], df["kernel"].iloc[-1],
           ('Crossover at **k = %d**. Past that the FFT route wins, even '
            'though it transforms the whole image twice.' % df['size'].iloc[cross[0]])
           if len(cross) else
           'No crossover in this range. On an image this small the FFT setup '
           'cost still dominates; make the image bigger and the crossover '
           'moves into range.'))
    st.caption('Separable filtering sits in the middle and is what real code '
               'normally uses at these kernel sizes. The FFT wins clearly once '
               'kernels get big, or when the same filter is reused on many '
               'images (transform the kernel once and keep it).')


# ---------------------------------------------------------------------------
# Tab 5 -- small worked example
# ---------------------------------------------------------------------------
def _tab_example():
    st.subheader('A small example you can check by hand')
    st.markdown(
        'The image experiments look convincing but you cannot check them by '
        'hand. So here is the same claim on an 8-sample 1D signal, small '
        'enough to see every number. The worked-out version is in '
        '`modules/module3/theory.md` and in the PDF.')

    f = np.array([0., 0., 0., 9., 9., 0., 0., 0.])
    st.markdown('**Signal** `f` (a rectangular pulse) and **filter** `h`, a '
                '3-tap mean filter:')
    st.code('f = [0, 0, 0, 9, 9, 0, 0, 0]\n'
            'h = [1/3, 1/3, 1/3]   centred on its middle tap', language='text')

    N = f.size
    h = np.array([1 / 3, 1 / 3, 1 / 3])
    hp = np.zeros(N); hp[:3] = h
    hp = np.roll(hp, -1)              # centre tap to index 0

    direct = np.array([sum(hp[(i) % N] * f[(n - i) % N] for i in range(N))
                       for n in range(N)])
    Ff, Hf = np.fft.fft(f), np.fft.fft(hp)
    viaf = np.real(np.fft.ifft(Ff * Hf))

    import pandas as pd
    st.markdown('**Route A — circular convolution, done term by term**')
    st.dataframe(pd.DataFrame({'n': range(N), 'f[n]': f,
                               '(f*h)[n]': np.round(direct, 6)}),
                 use_container_width=True, hide_index=True)

    st.markdown('**Route B — multiply the DFTs, then transform back**')
    st.dataframe(pd.DataFrame({
        'k': range(N),
        'F[k]': [('%+.3f%+.3fi' % (z.real, z.imag)) for z in Ff],
        'H[k]': [('%+.3f%+.3fi' % (z.real, z.imag)) for z in Hf],
        '(F·H)[k]': [('%+.3f%+.3fi' % (z.real, z.imag)) for z in Ff * Hf],
        'IDFT': np.round(viaf, 6)}),
        use_container_width=True, hide_index=True)

    st.success('Largest disagreement between the two columns: **%.2e**. '
               'The pulse of height 9 spread over 2 samples comes back as '
               '3, 6, 6, 3 — total signal energy conserved, edges ramped. '
               'Both routes agree on every sample.'
               % np.abs(direct - viaf).max())

    st.markdown('H[0] = %.3f, the DC gain, which is the sum of the filter '
                'taps again. H[k] only comes out real because I centred the '
                'kernel at index 0 first. Remove that `np.roll` and H gets a '
                'phase ramp, which shows up as the whole result shifted by one '
                'sample. That one line is the easiest thing to get wrong '
                'here.' % Hf[0].real)

    st.divider()
    st.markdown('**And the same thing in 2D, on a 4×4 image**')
    rng = np.random.default_rng(3)
    small = rng.integers(0, 10, (4, 4)).astype(float)
    k = F.box_kernel(3)
    sa = F.convolve_spatial(small, k, 'wrap')
    sb = F.convolve_fft_circular(small, k)
    c1, c2, c3 = st.columns(3)
    c1.markdown('f'); c1.dataframe(pd.DataFrame(small), hide_index=True)
    c2.markdown('spatial'); c2.dataframe(pd.DataFrame(np.round(sa, 4)), hide_index=True)
    c3.markdown('Fourier'); c3.dataframe(pd.DataFrame(np.round(sb, 4)), hide_index=True)
    st.caption('Max difference %.2e.' % F.compare(sa, sb)['max_abs'])


# ---------------------------------------------------------------------------
# Tab 6 -- the proof
# ---------------------------------------------------------------------------
def _tab_theory():
    st.subheader('Why it has to be true')
    st.markdown(
        'The experiments show the two routes agree. This is the argument for '
        'why they must. I am giving the discrete (DFT) version, because that '
        'is what the code actually computes — the continuous Fourier-transform '
        'proof is the same three steps with integrals in place of sums, and '
        'both are written out in `modules/module3/theory.md`.')

    st.markdown('**Setup.** An N×M image `f`, a filter `h` padded to the same '
                'size, and the 2D DFT:')
    st.latex(r'F(u,v)=\sum_{x=0}^{N-1}\sum_{y=0}^{M-1} f(x,y)\,'
             r'e^{-j2\pi\left(\frac{ux}{N}+\frac{vy}{M}\right)}')

    st.markdown('**Step 1.** Write down the DFT of the circular convolution '
                '`g = f ⊛ h`, and substitute the definition of the convolution:')
    st.latex(r'G(u,v)=\sum_{x}\sum_{y}\left[\sum_{m}\sum_{n} f(m,n)\,'
             r'h(x-m,\;y-n)\right]e^{-j2\pi\left(\frac{ux}{N}+\frac{vy}{M}\right)}')

    st.markdown('**Step 2.** Swap the order of summation (finite sums, so this '
                'is free) and split the exponential using '
                '`e^{a+b} = e^a · e^b`, writing `x = (x−m) + m`:')
    st.latex(r'G(u,v)=\sum_{m}\sum_{n} f(m,n)\,'
             r'e^{-j2\pi\left(\frac{um}{N}+\frac{vn}{M}\right)}'
             r'\sum_{x}\sum_{y} h(x-m,y-n)\,'
             r'e^{-j2\pi\left(\frac{u(x-m)}{N}+\frac{v(y-n)}{M}\right)}')

    st.markdown('**Step 3.** The inner double sum is just the DFT of `h`. '
                'The indices wrap (mod N and M), so shifting `h` by (m, n) '
                'only reorders which terms get added and the sum comes out the '
                'same. That means the inner sum equals `H(u,v)` no matter what '
                'm and n are, so it can come outside:')
    st.latex(r'G(u,v)=H(u,v)\sum_{m}\sum_{n} f(m,n)\,'
             r'e^{-j2\pi\left(\frac{um}{N}+\frac{vn}{M}\right)} = F(u,v)\,H(u,v)')

    st.markdown('---')
    st.markdown(
        '**What this proves, and what it does not.** It proves *circular* '
        'convolution equals multiplying the DFTs. Step 3 relied on the '
        'indices wrapping, so if the spatial filter mirrors or zero-fills at '
        'the border instead, it is doing something different near the edges '
        'and there is no reason for it to match there. That is what tab 2 '
        'shows: match the conventions and they agree to 1e-13, mismatch them '
        'and the difference is stuck in a (k−1)/2 border. Zero-padding both '
        'to N+k−1 first leaves the wrap region empty, which is how you use '
        'the same theorem to get *linear* convolution (`convolve_fft_linear`).')

    st.markdown(
        '**It works the other way round too.** Multiplying in frequency is '
        'convolving in space, so drawing a filter shape in the frequency '
        'domain and inverse-transforming it is the same thing as designing a '
        'kernel. This also explains the box filter ringing: a sharp rectangle '
        'in one domain is a sinc in the other, which oscillates and goes '
        'negative. You cannot have a filter that is small and sharp-edged in '
        'both domains at once. It is also why Gaussians are the usual choice '
        'for blurring: a Gaussian transforms into another Gaussian.')

    st.caption('Full derivation, the continuous-domain version, and the '
               'hand-worked numeric example: `modules/module3/theory.md`.')


# ---------------------------------------------------------------------------
def render():
    st.title('Module 3 — Image Blurring & the Convolution Theorem')
    st.caption('CSc 8830 · Dhanush Nagarajan · Georgia State University')
    st.markdown(
        'Blurring done as a spatial convolution I wrote myself, then shown to '
        'match multiplying by the same filter in the Fourier domain. Every '
        'number below is computed live from the image you pick — nothing here '
        'is a saved result.')

    t1, t2, t3, t4, t5, t6 = st.tabs([
        '1 · Blur it',
        '2 · Space = Frequency',
        '3 · Spectra',
        '4 · Cost',
        '5 · Worked example',
        '6 · The proof'])
    with t1:
        _tab_blur()
    with t2:
        _tab_equivalence()
    with t3:
        _tab_spectra()
    with t4:
        _tab_timing()
    with t5:
        _tab_example()
    with t6:
        _tab_theory()

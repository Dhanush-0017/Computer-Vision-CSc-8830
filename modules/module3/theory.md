# Theory: blurring as filtering, and why the spatial and Fourier routes agree

CSc 8830 (Computer Vision) — Module 3
Dhanush Nagarajan, Georgia State University

This is the written half of Module 3. The assignment asks for two things: an
implementation of image blurring by filtering, and a demonstration that doing
that filtering in the spatial domain gives the same result as the Fourier-domain
equivalent — that convolution in space is multiplication in frequency. The code
is in `src/filtering.py`, the live demonstration is tab 2 of the Module 3 page in
the web app, and the argument for *why* it has to come out that way is below,
along with a small example worked out by hand.

---

## 1. What blurring is

A blur replaces every pixel with a weighted average of its neighbours. The
weights are the kernel `h`, and the operation is convolution:

```
g(x, y) = (f * h)(x, y) = Σ_m Σ_n  f(m, n) · h(x − m, y − n)
```

Two kernels are used in this module.

**Box (mean) filter, k×k.** Every tap equals 1/k². The flattest possible
weighting: each neighbour in the window counts the same.

**Gaussian filter.** `h(x, y) ∝ exp(−(x² + y²) / 2σ²)`, sampled on the kernel
grid and then normalised so the taps sum to 1. I truncate at ±3σ, which keeps
about 99.7% of the mass; cutting much tighter leaves a step at the kernel's edge,
and section 6 explains why a step in one domain is expensive in the other.

Both kernels are normalised so that **Σ h = 1**. That is not cosmetic. Section 4
shows Σ h is exactly `H(0,0)`, the filter's gain at zero frequency — the DC term,
which is the image's average brightness. A kernel summing to 1.2 would brighten
the image by 20% as well as blurring it.

The Gaussian is **separable**: `exp(−(x²+y²)/2σ²) = exp(−x²/2σ²) · exp(−y²/2σ²)`,
so a 2D Gaussian blur is a 1D blur along the rows followed by a 1D blur down the
columns. That drops the cost from k² multiply–adds per pixel to 2k.
`convolve_separable()` implements this, and the self-test asserts it agrees with
the full 2D convolution to ~1e-13.

One more definition that makes the rest of this cleaner. Convolve an image that
is zero everywhere except a single pixel of value 1, and the result *is* the
kernel, printed into the image at that location. So the kernel is the filter's
**impulse response**, or in imaging terms its **point spread function**: the
picture of what the system does to a single point of light. The self-test checks
this too, and it comes out exactly equal, not approximately.

---

## 2. The claim

**Convolution theorem.** Convolution in one domain is pointwise multiplication in
the other:

```
f * h   ⟷   F · H
```

where `F = 𝔉{f}` and `H = 𝔉{h}`. Concretely, these two procedures produce the
same image:

| | Route A — spatial | Route B — frequency |
|---|---|---|
| 1 | — | `F = FFT2(f)` |
| 2 | — | `H = FFT2(h)`, h padded to the image size |
| 3 | slide `h` over `f`, accumulate | `G = F · H`, pointwise |
| 4 | — | `g = real(IFFT2(G))` |
| cost | O(H·W·k²) | O(H·W·log(H·W)) |

---

## 3. Proof, continuous case

Take the 2D Fourier transform of `g = f * h`:

```
G(u,v) = ∫∫ [ ∫∫ f(a,b) h(x−a, y−b) da db ] e^{−j2π(ux+vy)} dx dy
```

Swap the order of integration (allowed: both functions are absolutely integrable,
which any real image is, being bounded and of finite extent):

```
G(u,v) = ∫∫ f(a,b) [ ∫∫ h(x−a, y−b) e^{−j2π(ux+vy)} dx dy ] da db
```

Substitute `x′ = x − a`, `y′ = y − b` in the inner integral. The limits are
unchanged (they are infinite), `dx dy = dx′ dy′`, and the exponent splits:

```
e^{−j2π(ux+vy)} = e^{−j2π(ux′+vy′)} · e^{−j2π(ua+vb)}
```

The second factor has no `x′` or `y′` in it, so it comes out of the inner
integral:

```
inner = e^{−j2π(ua+vb)} ∫∫ h(x′,y′) e^{−j2π(ux′+vy′)} dx′ dy′
      = e^{−j2π(ua+vb)} · H(u,v)
```

`H(u,v)` no longer depends on `a` or `b`, so it comes out of the outer integral
as well:

```
G(u,v) = H(u,v) ∫∫ f(a,b) e^{−j2π(ua+vb)} da db = F(u,v) · H(u,v)      ∎
```

The whole proof is one substitution and one factorisation of an exponential. The
step that does the work is `e^{A+B} = e^A · e^B` — the reason the Fourier basis
functions are the right ones is precisely that shifting one of them only
multiplies it by a constant.

---

## 4. Proof, discrete case (what the code actually computes)

The code does not compute integrals; it computes DFTs of an N×M array. The
discrete statement needs one extra word, and that word matters for everything in
section 5.

The 2D DFT is

```
F(u,v) = Σ_{x=0}^{N−1} Σ_{y=0}^{M−1} f(x,y) e^{−j2π(ux/N + vy/M)}
```

and the relevant convolution is **circular**: indices are taken modulo N and M,
so the image is treated as one tile of an infinitely repeated pattern.

```
(f ⊛ h)(x,y) = Σ_{m=0}^{N−1} Σ_{n=0}^{M−1} f(m,n) · h((x−m) mod N, (y−n) mod M)
```

**Step 1.** Take the DFT of `g = f ⊛ h` and substitute:

```
G(u,v) = Σ_x Σ_y [ Σ_m Σ_n f(m,n) h(x−m, y−n) ] e^{−j2π(ux/N + vy/M)}
```

**Step 2.** Swap the summation order — these are finite sums, so no convergence
argument is needed — and write `x = (x−m) + m`, `y = (y−n) + n` in the exponent
so it factors:

```
G(u,v) = Σ_m Σ_n f(m,n) e^{−j2π(um/N + vn/M)}
                 · Σ_x Σ_y h(x−m, y−n) e^{−j2π(u(x−m)/N + v(y−n)/M)}
```

**Step 3.** The inner double sum looks like it depends on `m` and `n`, but it does
not. As `x` runs over a full period `0…N−1`, so does `x−m` modulo N — shifting
just permutes which terms get added, and addition does not care about order. So
the inner sum equals `H(u,v)` for every `m, n`, and pulls out:

```
G(u,v) = H(u,v) · Σ_m Σ_n f(m,n) e^{−j2π(um/N + vn/M)} = F(u,v) · H(u,v)      ∎
```

**Two corollaries worth noting.**

Setting `u = v = 0` gives `G(0,0) = F(0,0) · H(0,0)`, and `H(0,0) = Σ h` by the
definition above. Since `F(0,0)` is the sum of all pixel values, a kernel summing
to 1 is exactly the condition for the blur to preserve total brightness.

The `e^{−j2π(um/N + vn/M)}` factor that appeared in step 2 is the **shift
theorem**: displacing a signal multiplies its spectrum by a linear phase ramp and
leaves the magnitude alone. This is why `kernel_to_psf()` in the code rolls the
kernel's centre tap to index (0,0) before transforming. Skip that roll and `H`
picks up exactly such a ramp, and the blurred image comes back shifted
diagonally by (k−1)/2 pixels. It is the single most common way this experiment
goes wrong, and because the image is merely *shifted* rather than *wrong*, it is
easy to miss.

---

## 5. The one honest caveat: circular vs linear convolution

Step 3 used "shifting `h` wraps around" as a fact. It is a fact about the DFT,
not about photographs. The DFT has no way to represent an image that simply
stops at its border; it assumes the image tiles, so the left edge is the right
edge's neighbour.

So a spatial filter that pads with zeros, or mirrors, or repeats the edge pixel,
is computing something genuinely different near the border from what the DFT
product computes, and there is no reason for the two to agree there. This is not
a failure of the theorem — the theorem is about circular convolution and is
exactly true. It is a statement about which question each method is being asked
at the edges.

Two ways to get a clean match, both implemented and both demonstrated in the app:

**Match the conventions.** Use `boundary='wrap'` on the spatial side. Then both
routes compute circular convolution and agree to floating-point round-off
everywhere, borders included.

**Zero-pad out of the problem.** Pad both `f` and `h` with zeros to at least
(N+k−1, M+k−1) before transforming. The wrap-around region is then all zeros and
contributes nothing, so the DFT product equals *linear* convolution, matching
`boundary='zero'`. This is `convolve_fft_linear()`, and it is how the theorem is
used in practice when linear convolution is what you want.

And the third case, which is the most informative to look at: deliberately
mismatch them. The measurement (section 8) is that the difference is confined to
a border strip of exactly (k−1)/2 pixels — the distance the kernel overhangs the
image — and is *identically zero*, not merely small, across the whole interior.
That is a sharper piece of evidence than the matched case alone: it localises the
disagreement to precisely the region where the two definitions differ, and shows
the theorem holding perfectly everywhere else.

---

## 6. What the theorem says about blurring itself

Multiplication is pointwise: `G(u,v) = F(u,v) · H(u,v)` means each frequency in
the image is independently scaled by `H` at that frequency. So a filter cannot
create a frequency the image did not contain, and cannot move energy from one
frequency to another. All it can do is reweight.

A blur, then, is a filter whose `H` is near 1 at low frequencies and small at
high ones — a **low-pass filter**. Fine detail (high frequency) is attenuated,
broad structure (low frequency) survives, and `H(0,0) = 1` keeps the average
brightness. Tab 3 of the app shows `|F|`, `|H|`, and `|F·H|` side by side; the
middle panel is a bright centre fading to a dark rim, which is what that sentence
looks like as a picture.

This also explains the difference between the two kernels in a way the spatial
view does not:

- The **box filter** is a rectangle in space. The transform of a rectangle is a
  sinc, which oscillates and goes **negative**. Negative `H` at some frequency
  means that frequency comes back phase-inverted — and that is the origin of the
  faint ripples a box blur leaves near sharp edges. Section 7 shows this in
  numbers: for a 3-tap mean filter, `H` is negative at `k = 3, 4, 5`.
- The **Gaussian** transforms to another Gaussian: strictly positive, no side
  lobes, no ringing. It is the eigenfunction-like case, the function that is its
  own transform shape, and that is the real reason it is the default blur rather
  than any argument about statistics.

There is no kernel that is both compact and sharp-edged in both domains at once —
narrow in space forces wide in frequency. That is the uncertainty principle,
showing up in image processing as the reason you cannot have a blur that is both
perfectly local and perfectly band-limited.

---

## 7. Worked by hand: an 8-sample signal

The image experiments are convincing but not checkable with a pencil. Here is the
same claim on a signal small enough to verify by hand. Everything below was
worked out on paper first and then confirmed by the code; tab 5 of the app
recomputes it live.

### The setup

```
f = [0, 0, 0, 9, 9, 0, 0, 0]          N = 8, a rectangular pulse
h = [1/3, 1/3, 1/3]                    3-tap mean filter, centred on its middle tap
```

Centred means the taps sit at offsets −1, 0, +1. Placed into a length-8 array with
the centre at index 0 — the roll described in section 4 — that is:

```
h_p = [1/3, 1/3, 0, 0, 0, 0, 0, 1/3]
        n=0  n=1                n=7 ≡ n=−1
```

### Route A — circular convolution, term by term

`g[n] = Σ_i h_p[i] · f[(n−i) mod 8]`, which with only three non-zero taps is just
the average of each sample and its two neighbours, wrapping at the ends:

```
g[0] = (f[7] + f[0] + f[1])/3 = (0 + 0 + 0)/3 = 0
g[1] = (f[0] + f[1] + f[2])/3 = (0 + 0 + 0)/3 = 0
g[2] = (f[1] + f[2] + f[3])/3 = (0 + 0 + 9)/3 = 3
g[3] = (f[2] + f[3] + f[4])/3 = (0 + 9 + 9)/3 = 6
g[4] = (f[3] + f[4] + f[5])/3 = (9 + 9 + 0)/3 = 6
g[5] = (f[4] + f[5] + f[6])/3 = (9 + 0 + 0)/3 = 3
g[6] = (f[5] + f[6] + f[7])/3 = (0 + 0 + 0)/3 = 0
g[7] = (f[6] + f[7] + f[0])/3 = (0 + 0 + 0)/3 = 0
```

```
g = [0, 0, 3, 6, 6, 3, 0, 0]
```

Sanity check before going further: the input sums to 18 and so does the output,
which it must, because `H(0) = Σh = 1`. The sharp pulse has become a trapezoid —
edges ramped, total preserved. That is a blur.

### Route B — multiply the DFTs

`H[k]` can be written down in closed form rather than computed. With taps at
offsets −1, 0, +1 and weight 1/3 each:

```
H[k] = (1/3)(e^{+j2πk/8} + 1 + e^{−j2πk/8}) = (1 + 2cos(2πk/8)) / 3
```

The two complex exponentials are conjugates, so they combine into a cosine and
**H[k] is purely real** — a consequence of the kernel being symmetric about the
origin, and another sign the centring roll was done correctly. Evaluating:

| k | 2πk/8 | cos | H[k] = (1+2cos)/3 |
|---|---|---|---|
| 0 | 0 | +1.0000 | **+1.00000** |
| 1 | π/4 | +0.7071 | +0.80474 |
| 2 | π/2 | 0.0000 | +0.33333 |
| 3 | 3π/4 | −0.7071 | **−0.13807** |
| 4 | π | −1.0000 | **−0.33333** |
| 5 | 5π/4 | −0.7071 | **−0.13807** |
| 6 | 3π/2 | 0.0000 | +0.33333 |
| 7 | 7π/4 | +0.7071 | +0.80474 |

Read that column as the filter's frequency response. `H[0] = 1`: DC passes
untouched, brightness preserved. It falls monotonically to `k = 4` (the Nyquist
frequency, the fastest alternation the grid can hold), so high frequencies are
suppressed — this is a low-pass filter. And `H` is **negative at k = 3, 4, 5**:
those frequencies come back inverted, not merely reduced. That is the box
filter's ringing, visible as a number rather than as an artefact.

Now the DFT of the signal, `F[k] = Σ_n f[n] e^{−j2πkn/8}`, which with only two
non-zero samples is `9(e^{−j6πk/8} + e^{−j8πk/8})`:

| k | F[k] | H[k] | (F·H)[k] |
|---|---|---|---|
| 0 | +18.0000 + 0.0000j | +1.00000 | +18.0000 + 0.0000j |
| 1 | −15.3640 − 6.3640j | +0.80474 | −12.3640 − 5.1213j |
| 2 |  +9.0000 + 9.0000j | +0.33333 |  +3.0000 + 3.0000j |
| 3 |  −2.6360 − 6.3640j | −0.13807 |  +0.3640 + 0.8787j |
| 4 |   0.0000 + 0.0000j | −0.33333 |   0.0000 + 0.0000j |
| 5 |  −2.6360 + 6.3640j | −0.13807 |  +0.3640 − 0.8787j |
| 6 |  +9.0000 − 9.0000j | +0.33333 |  +3.0000 − 3.0000j |
| 7 | −15.3640 + 6.3640j | +0.80474 | −12.3640 + 5.1213j |

(The conjugate symmetry `F[k] = conj(F[8−k])` is the usual consequence of `f`
being real, and it survives the multiplication because `H` is real — which is why
the inverse transform below comes out real, as it must.)

Inverse transforming `F·H`:

```
IDFT(F·H) = [0, 0, 3, 6, 6, 3, 0, 0]
```

### The two routes

```
Route A (convolve in space)  : [0, 0, 3, 6, 6, 3, 0, 0]
Route B (multiply in freq.)  : [0, 0, 3, 6, 6, 3, 0, 0]
maximum difference           : 4.44e-16      (one unit in the last place of a float64)
```

Identical. Not similar — identical to the precision the arithmetic is carried in.

### The same thing in 2D

`src/verify_convolution.py --selftest` runs the 2D version on a 4×4 array with a
3×3 box kernel, and tab 5 of the app displays all three matrices side by side.
The max difference is ~1e-15. It behaves exactly as the 1D case does, because the
2D transform is just the 1D transform applied along rows and then along columns.

---

## 8. Experimental validation on real images

`python verify_convolution.py --experiment` blurs the sample photo (384×512,
grayscale) by both routes across a range of kernels. Measured:

| filter | spatial (s) | FFT (s) | max abs diff | mean abs diff | RMSE | PSNR |
|---|---|---|---|---|---|---|
| box 3×3 | 0.0019 | 0.0134 | 1.71e-13 | 2.58e-14 | 3.43e-14 | 317.4 dB |
| box 5×5 | 0.0043 | 0.0041 | 2.27e-13 | 3.61e-14 | 4.69e-14 | 314.7 dB |
| box 9×9 | 0.0104 | 0.0033 | 4.83e-13 | 6.73e-14 | 9.07e-14 | 309.0 dB |
| box 15×15 | 0.0288 | 0.0037 | 9.10e-13 | 1.95e-13 | 2.46e-13 | 300.3 dB |
| gauss 9, σ=1.5 | 0.0101 | 0.0034 | 2.84e-13 | 4.71e-14 | 6.11e-14 | 312.4 dB |
| gauss 15, σ=2.5 | 0.0272 | 0.0031 | 5.68e-13 | 7.33e-14 | 9.88e-14 | 308.2 dB |
| gauss 21, σ=3.5 | 0.0535 | 0.0035 | 7.11e-13 | 9.33e-14 | 1.23e-13 | 306.3 dB |
| gauss 31, σ=5.0 | 0.1169 | 0.0031 | 8.81e-13 | 1.27e-13 | 1.65e-13 | 303.8 dB |

Pixel values run 0–255. A worst-case disagreement of 1e-13 is about one part in
10¹⁵ — a few units in the last place of a double, which is what a few thousand
floating-point butterflies in an FFT accumulate. It is round-off, not
disagreement. PSNR above 300 dB is the same statement in the usual units (a
visually lossless JPEG sits around 40 dB).

The error growing slowly with kernel size is itself consistent with the
round-off explanation: larger kernels mean more terms summed in the spatial route
and larger intermediate values in the transform, so more accumulation. If the two
routes were computing genuinely different things, the error would scale with the
image content, not with the number of arithmetic operations.

**The boundary-mismatch experiment**, same script, Gaussian 15×15 σ=2.5, spatial
side reflecting while the Fourier side wraps:

```
max difference over the whole image        : 7.06e+01
max difference with the 7-px border cropped: 5.68e-13
```

70 grey levels of disagreement at the border; 5.7e-13 — i.e. none — everywhere
else. The border strip is 7 pixels, which is exactly (15−1)/2. Section 5 predicted
both the magnitude and the location.

**Cross-check against OpenCV.** Agreement between my two routes would prove
little if a shared bug made both wrong the same way, so the self-test also pins my
spatial convolution against `cv2.filter2D` (kernel flipped, since OpenCV computes
correlation): max difference 8.5e-14 to 5.4e-13 across all kernels tested. 21 of
21 assertions pass, including the two that are checked against analytic results
rather than against another implementation — the impulse response equalling the
kernel exactly, and a k-tap box filter ramping an ideal step edge over exactly
k−1 intermediate pixels.

---

## 9. Cost, and when each route is worth using

The two routes give the same answer, so the choice is purely about cost.
`python verify_convolution.py --timing`, 384×384 image:

| kernel | spatial, naive 2D | spatial, separable | FFT |
|---|---|---|---|
| 3×3 | 0.0016 s | 0.0010 s | 0.0049 s |
| 7×7 | 0.0046 s | 0.0014 s | 0.0027 s |
| 11×11 | 0.0111 s | 0.0019 s | 0.0026 s |
| 15×15 | 0.0208 s | 0.0025 s | 0.0027 s |
| 19×19 | 0.0333 s | 0.0030 s | 0.0027 s |
| 23×23 | 0.0488 s | 0.0036 s | 0.0027 s |
| 27×27 | 0.0671 s | 0.0041 s | 0.0026 s |
| 31×31 | 0.0886 s | 0.0046 s | 0.0027 s |

The FFT column is **flat** — 0.0026 to 0.0027 s once past the first
warm-up measurement — because the kernel is padded to the image size before
transforming, so a 3×3 and a 31×31 filter are literally the same amount of work.
The naive spatial column grows by 55× across the same range, tracking k² as
predicted (31²/3² ≈ 107, and the measured ratio is ~55, the gap being numpy's
per-operation overhead dominating at the small end).

The crossover is at **k = 7** against naive spatial convolution. Against the
separable implementation there is no crossover in this range: separable filtering
stays competitive up to 31×31 on an image this size, and is what production code
generally uses at these kernel sizes. The FFT route wins decisively when kernels
get genuinely large, when the kernel is not separable, or when the same filter is
applied to many images — transform the kernel once and reuse `H`.

Worth stating plainly: the FFT route is not faster because the transform is
clever about convolution. It is faster because it replaces an O(k²)-per-pixel
operation with an O(1)-per-pixel multiply, and pays a fixed O(N log N) to get
there and back. Below the crossover that fixed cost is not worth paying.

---

## 10. Summary

- Blurring is convolution with a normalised low-pass kernel; the kernel is the
  filter's impulse response.
- The convolution theorem says the spatial and Fourier routes must agree, and the
  proof is one change of variable plus `e^{A+B} = e^A·e^B`.
- In the discrete case the theorem is exactly true for *circular* convolution.
  Matching the boundary convention gives agreement at the 1e-13 level over the
  whole image; zero-padding to N+k−1 recovers linear convolution instead.
- Mismatching the conventions on purpose confines the disagreement to a
  (k−1)/2-pixel border and leaves the interior bit-for-bit identical, which is
  the sharper demonstration of the two.
- Measured on real images: max difference ~1e-13, PSNR > 300 dB, across box and
  Gaussian kernels from 3×3 to 31×31.
- The routes differ only in cost: O(k²) per pixel versus O(log N) per pixel, with
  the crossover here at k = 7.

**Files:** `src/filtering.py` (the implementation), `src/blur.py` (CLI),
`src/verify_convolution.py` (the self-test, the equivalence experiment, the
timing sweep), `src/modules/module3.py` (the web app page).

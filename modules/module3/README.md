# Module 3 — Image Blurring and the Convolution Theorem

CSc 8830, Computer Vision. Dhanush Nagarajan.

Blur an image two ways — by sliding a kernel over it, and by multiplying in the
frequency domain — and show the results are the same image, not just similar
ones.

## What the assignment asked for

1. Implement image blurring using a filtering approach.
2. Theory: show that spatial filtering gives the same outcome as the Fourier
   domain equivalent, and that convolution in space is the same as
   multiplication in frequency. Use the implementation and experiments as
   evidence.

## What's in this folder

| File | What it does |
|---|---|
| `__init__.py` | The Module 3 page in the web app — six tabs |
| `filtering.py` | The engine. All the blurring maths: kernels, spatial convolution, FFT convolution, comparison metrics. No UI in it |
| `blur.py` | Blur one image from the command line |
| `verify_convolution.py` | The evidence. 21 automatic checks, plus it generates the results tables |
| `theory.md` | The proof, for the PDF |
| `data/module3_samples/` | Three sample photos |
| `results_equivalence.csv` | How far apart the two methods came out, per filter |
| `results_timing.csv`, `timing_plot.png` | Speed of each method vs kernel size |

## What's implemented

Box and Gaussian kernels, both normalised to sum to 1, applied by a 2D
convolution I wrote out in `filtering.py`. `cv2.blur`, `cv2.GaussianBlur` and
`cv2.filter2D` don't produce any result here — writing the filter is the
assignment. OpenCV shows up once, in `reference_opencv()`, only so the self-test
can check my convolution against something I didn't write.

Four boundary rules are selectable: `wrap`, `zero`, `reflect`, `replicate`. The
Gaussian can also run separably as two 1D passes.

On the Fourier side there are both versions: `convolve_fft_circular()` (the plain
DFT product, which is circular convolution) and `convolve_fft_linear()`
(zero-padded to N+k−1, which gives linear convolution).

## Running it

Web app, from the repo root:

```
streamlit run app.py
```

Command line, from this folder:

```
cd modules/module3

# blur something
python blur.py --image data/module3_samples/text_journal.jpg \
               --filter gaussian --sigma 3 --out blurred.png

# blur it both ways and print how far apart they are
python blur.py --image data/module3_samples/text_journal.jpg \
               --filter gaussian --sigma 3 --compare

# the evidence: 21 checks, the equivalence table, the timing sweep
python verify_convolution.py --all
```

`--all` rewrites `results_equivalence.csv`, `results_timing.csv` and
`timing_plot.png`. Everything quoted in the PDF can be reproduced from it.

## The three sample photos

Each one shows something different, which is why there are three instead of one.

- **`text_journal.jpg`** — a handwritten page. Lots of thin strokes, so the blur
  is obvious immediately. The writing stops being readable well before the page
  stops looking like a page.
- **`brick_wall.jpg`** — repeating pattern. In the spectra tab a regular spacing
  shows up as distinct bright spots instead of a blob, and you can watch blurring
  dim those specific spots.
- **`building_sky.jpg`** — hard edges against flat sky. Good for comparing box vs
  Gaussian: the box filter rings near a sharp edge, the Gaussian doesn't. The sky
  itself barely changes, because there's no fine detail there to remove.

The assignment doesn't say anything about which image to use.

## What came out

Across box and Gaussian kernels from 3×3 to 31×31 on a 1160×1200 image, the two
methods agreed to a maximum of **1.1e-12** grey levels on a 0–255 scale, with
PSNR above 300 dB. That's double-precision round-off, not disagreement.

The self-test passes **21 out of 21**.

The thing that took longest to get right was kernel centring. The DFT treats index
0 as the origin, but a blur kernel's origin is its middle tap, so the kernel has
to be rolled by −(k//2) before transforming. Miss that and the result is a correct
blur that's *shifted* diagonally by (k−1)/2 pixels, which is easy to stare past
because a shifted blur still looks like a blur.

The other subtlety is more interesting than a bug. The DFT assumes the image is
periodic, so it computes circular convolution. A spatial filter that mirrors or
zero-fills at the border is doing something different there. Match the conventions
and the two agree everywhere. Mismatch them deliberately and the difference is 56
grey levels at the border but **4.8e-13 across the whole interior** — confined to
a strip of exactly (k−1)/2 pixels, which is how far the kernel overhangs the edge.
That's a sharper demonstration of the theorem than the matched case, so the app
shows both.

On cost: the FFT route's time is flat in kernel size (0.0023–0.0029 s across the
sweep) while naive spatial convolution grows like k², crossing over at **k = 7**
on a 384×384 image. Separable filtering stays competitive up to 31×31, which is
why it, not the FFT, is what normal code uses at ordinary blur radii.

The timing comparison isn't required by the assignment. I added it because it
answers the obvious follow-up: if both methods give the same image, why does the
Fourier one exist.

## Theory

`theory.md` has the convolution theorem proved in both the continuous and
discrete cases, the circular vs linear discussion, and a hand-worked 8-sample
example with the full DFT table — the same one tab 5 of the app recomputes live.

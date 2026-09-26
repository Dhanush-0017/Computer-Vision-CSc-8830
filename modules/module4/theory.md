# Module 4, Part 3: Edges and Regions in the Fourier Domain

CSc 8830 Computer Vision. Dhanush Nagarajan.

The question asks how edge detection and region segmentation can be done
with Fourier (frequency) domain analysis, with the maths derived. The short
version:

- **Edges are high frequencies.** Taking a derivative multiplies the spectrum
  by $j2\pi u$, which is a high-pass filter. So edge detection is high-pass
  filtering. Noise also lives at high frequencies, so in practice it has to
  be band-pass filtering.
- **Regions are low frequencies.** Inside a region the image changes slowly.
  A low-pass filter removes texture and noise and leaves the regions, which a
  threshold can then split. Where regions differ in texture instead of
  brightness, you measure how much energy each part of the image has in a
  chosen frequency band, and threshold that.

---

## 1. Setup

For an $M \times N$ image $f(x, y)$ the 2D DFT and its inverse are

$$
F(u,v) = \sum_{x=0}^{N-1}\sum_{y=0}^{M-1} f(x,y)\, e^{-j2\pi\left(\frac{ux}{N}+\frac{vy}{M}\right)},
\qquad
f(x,y) = \frac{1}{MN}\sum_{u}\sum_{v} F(u,v)\, e^{\,j2\pi\left(\frac{ux}{N}+\frac{vy}{M}\right)}
$$

and in the continuous case

$$
F(u,v) = \iint f(x,y)\, e^{-j2\pi(ux+vy)}\,dx\,dy .
$$

The one result I use everywhere comes from Module 3, the **convolution theorem**:

$$
(f * h)(x,y) \;\Longleftrightarrow\; F(u,v)\,H(u,v).
$$

So every linear, shift-invariant filter is completely described by its
*transfer function* $H(u,v)$: multiply the spectrum by $H$ and transform back.
The rest of this document is just picking the right $H$.

In the code, frequencies are in cycles per pixel (from `np.fft.fftfreq`), so
$u, v \in [-\tfrac12, \tfrac12)$ and the discrete formulas look the same as
the continuous ones.

---

## 2. Why an edge is a high-frequency thing

Take the simplest edge, a unit step in $x$: $f(x) = \mathbf{1}[x \ge 0]$.
Its derivative is an impulse, $f'(x) = \delta(x)$, and
$\mathcal{F}\{\delta\} = 1$. From the derivative theorem (next section),
$\mathcal{F}\{f'\} = j2\pi u\,F(u)$, so away from $u = 0$

$$
F(u) = \frac{1}{j2\pi u} \qquad\Rightarrow\qquad |F(u)| = \frac{1}{2\pi |u|}.
$$

The step's spectrum decays only like $1/|u|$, which is slow. A smooth ramp of
width $w$ decays much faster: it's a step convolved with a box of width $w$,
so its spectrum picks up an extra
$\operatorname{sinc}(wu) = \frac{\sin(\pi w u)}{\pi w u}$ factor, and that
dies off beyond $|u| \approx 1/w$. The sharper the edge, the more of its
energy sits at high frequency. In 2D, a straight edge at angle $\theta$
puts its energy on a line through the origin of the spectrum, perpendicular to
the edge. That's why an image's spectrum shows bright streaks through the
centre.

So "find the edges" is the same as "find where the image has high-frequency
content", which is filtering.

---

## 3. The derivative theorem

**Continuous.** Differentiate the inverse transform under the integral:

$$
f(x,y) = \iint F(u,v)\,e^{j2\pi(ux+vy)}\,du\,dv
\;\;\Rightarrow\;\;
\frac{\partial f}{\partial x} = \iint \big(j2\pi u\big)\,F(u,v)\,e^{j2\pi(ux+vy)}\,du\,dv .
$$

The right-hand side is the inverse transform of $j2\pi u\,F(u,v)$, so

$$
\boxed{\;\mathcal{F}\!\left\{\frac{\partial f}{\partial x}\right\} = j2\pi u\,F(u,v),
\qquad
\mathcal{F}\!\left\{\frac{\partial f}{\partial y}\right\} = j2\pi v\,F(u,v).\;}
$$

(The same result comes out of integrating by parts, with the boundary term
vanishing for any $f$ that decays.)

Applying it twice gives the Laplacian:

$$
\mathcal{F}\{\nabla^2 f\} = \big[(j2\pi u)^2 + (j2\pi v)^2\big]F
= -4\pi^2\,(u^2+v^2)\,F(u,v).
$$

**What this says.** $|H_x(u,v)| = 2\pi|u|$ is zero at DC and grows linearly
with frequency. $|H_{\nabla^2}| = 4\pi^2\rho^2$, with $\rho^2 = u^2+v^2$,
grows quadratically. Both are **high-pass filters**, and that's why
differentiation finds edges. An edge detector and a high-pass filter are the
same operation, seen from the two domains.

**Discrete versions.** Finite differences have their own transfer
functions. Put $f[x] = e^{j2\pi ux}$ into each one and see what factor comes out:

| Operator | Spatial | $H(u)$ | Small-$u$ behaviour |
|---|---|---|---|
| forward difference | $f[x{+}1]-f[x]$ | $e^{j2\pi u}-1 = 2j\,e^{j\pi u}\sin(\pi u)$ | $\approx j2\pi u$ |
| central difference | $\tfrac12(f[x{+}1]-f[x{-}1])$ | $j\sin(2\pi u)$ | $\approx j2\pi u$ |
| $[1,2,1]/4$ smoothing | | $\tfrac12(1+\cos 2\pi v) = \cos^2(\pi v)$ | $\approx 1$ |
| Sobel $x$ (unnormalised) | $[1,2,1]^T \otimes [-1,0,1]$ | $8j\,\sin(2\pi u)\cos^2(\pi v)$ | |

All of them match $j2\pi u$ at low frequency, which is why they all work as
derivatives. They differ near Nyquist: the central difference goes back to
zero at $u = \pm\tfrac12$, so it ignores pixel-level alternation. Sobel's
$\cos^2(\pi v)$ factor is a low-pass in the direction *along* the edge, so
Sobel is already a small derivative-of-smoothing filter. Multiplying by
$j2\pi u$ exactly (the *spectral* derivative) is exact for any sampled
sinusoid below Nyquist.

---

## 4. Noise: why a pure derivative isn't enough

Model the image as $f = s + n$, where $n$ is white noise with variance
$\sigma_n^2$. White noise has a flat power spectrum, $S_n(u,v) = \sigma_n^2$.
After a filter $H$, the noise variance is

$$
\operatorname{Var}[h * n] = \sigma_n^2 \iint |H(u,v)|^2\,du\,dv .
$$

For the derivative, $|H_x|^2 = 4\pi^2u^2$. That grows without bound, so the
noise is amplified most exactly where the filter is strongest. A plain
derivative of a noisy image mostly shows the noise.

The fix is to cut the high end off: multiply by a Gaussian low-pass
$G_\sigma(u,v) = e^{-2\pi^2\sigma^2(u^2+v^2)}$ (the transform of a unit-area
spatial Gaussian of standard deviation $\sigma$). Since multiplication
commutes, smoothing and then differentiating is one filter:

$$
H_{\partial_x G}(u,v) = j2\pi u\; e^{-2\pi^2\sigma^2(u^2+v^2)} .
$$

This is a **band-pass**: zero at DC (because of the $u$), zero at high
frequency (because of the Gaussian). Along $v = 0$ its magnitude is
$2\pi|u|\,e^{-2\pi^2\sigma^2u^2}$. Setting the derivative in $u$ to zero:

$$
\frac{d}{du}\Big[u\,e^{-2\pi^2\sigma^2u^2}\Big]
= e^{-2\pi^2\sigma^2u^2}\big(1 - 4\pi^2\sigma^2u^2\big) = 0
\;\;\Rightarrow\;\;
u^\star = \frac{1}{2\pi\sigma}.
$$

So **$\sigma$ picks which frequency, and therefore which size of edge, the
detector responds to most.** Bigger $\sigma$ means coarser edges and less noise.
The noise variance is now finite:

$$
\sigma_n^2 \iint 4\pi^2u^2\,e^{-4\pi^2\sigma^2(u^2+v^2)}\,du\,dv
= \frac{\sigma_n^2}{8\pi\,\sigma^4},
$$

which falls off like $\sigma^{-4}$. That's the trade-off in one line:
localisation versus noise.

**Gradient magnitude and direction.** With
$g_x = \mathcal{F}^{-1}\{H_{\partial_x G}F\}$ and
$g_y = \mathcal{F}^{-1}\{H_{\partial_y G}F\}$,

$$
|\nabla f| = \sqrt{g_x^2 + g_y^2}, \qquad \theta = \operatorname{atan2}(g_y, g_x).
$$

Thresholding $|\nabla f|$ (for example at Otsu's level) gives the edge map.

**Laplacian of Gaussian (Marr–Hildreth).** The same reasoning applied to the
Laplacian:

$$
H_{\text{LoG}}(u,v) = -4\pi^2\rho^2\,e^{-2\pi^2\sigma^2\rho^2},
\qquad
\frac{d}{d\rho}\big[\rho^2 e^{-2\pi^2\sigma^2\rho^2}\big]=0
\;\Rightarrow\;
\rho^\star = \frac{1}{\sqrt{2}\,\pi\sigma}.
$$

It's a ring-shaped band-pass, and it's isotropic, so it doesn't care about
edge direction. At an edge the first derivative peaks, so the second
derivative crosses zero there. The edges are the **zero crossings** of the
LoG response.

---

## 5. Explicit high-pass filters

Instead of a derivative, you can build the high-pass directly. With
$D = \sqrt{u^2+v^2}$ and cutoff $D_0$:

$$
H_{\text{ideal}} = \mathbf{1}[D > D_0], \qquad
H_{\text{Butterworth}} = \frac{1}{1 + (D_0/D)^{2n}}, \qquad
H_{\text{Gauss}} = 1 - e^{-D^2/2D_0^2}.
$$

The ideal filter looks cleanest but is the worst of the three. Its spatial
kernel is $\delta$ minus the transform of a disc, which is a jinc function
$J_1(2\pi D_0 r)/r$. That oscillates, so every edge comes back with rings
around it (the Gibbs phenomenon). Butterworth and Gaussian roll off
smoothly, so their kernels don't ring much.

A useful identity: $1 - G$ is a high-pass, so
$f - (g * f) = \mathcal{F}^{-1}\{(1-G)F\}$. **Subtracting a blurred copy of
an image is high-pass filtering.** That's exactly what the thermal method in
Q2 does when it subtracts its background estimate.

---

## 6. Region segmentation in the frequency domain

A region is a connected set of pixels that are *alike*, in brightness or in
texture. Both kinds of alikeness can be turned into something the frequency
domain can separate.

### 6a. Regions that differ in brightness: low-pass, then threshold

Model a region as a constant level $\mu_k$ plus noise and texture. The
constant is pure DC plus low frequencies (the region's shape). The
noise and texture are spread across all frequencies. A Gaussian low-pass keeps
the first and removes most of the second. From section 4 with $H = G_\sigma$:

$$
\operatorname{Var}[g_\sigma * n] = \sigma_n^2\iint e^{-4\pi^2\sigma^2(u^2+v^2)}\,du\,dv
= \frac{\sigma_n^2}{4\pi\sigma^2}.
$$

So the noise standard deviation drops by a factor $2\sqrt{\pi}\,\sigma$, while
the gap between region levels $|\mu_1 - \mu_2|$ stays the same away from the
boundary. The two peaks of the histogram get narrower but stay the same
distance apart, and Otsu's threshold,

$$
t^\star = \arg\max_t \; \omega_0(t)\,\omega_1(t)\,\big[\mu_0(t) - \mu_1(t)\big]^2 ,
$$

separates them cleanly.

The cost is that the boundary gets blurred over a width of about $\sigma$.
That's why a *region* result is usually refined with an *edge* result.

### 6b. Regions that differ in texture: band-pass energy

Two regions can have the same mean brightness and still look completely
different, like grass next to asphalt. The difference is in **which
frequencies they contain**. Parseval's theorem connects spatial energy to
spectral energy:

$$
\sum_{x,y} |r(x,y)|^2 = \frac{1}{MN}\sum_{u,v} |R(u,v)|^2 .
$$

If $r = \mathcal{F}^{-1}\{H_{\text{band}}F\}$ is the image band-passed to a
ring of frequencies, then $r^2$ averaged over a neighbourhood is the **local
energy of the image in that band**. Textured regions have a lot of it,
smooth regions very little. The recipe, all in the frequency domain:

$$
r = \mathcal{F}^{-1}\big\{(G_{\sigma_1} - G_{\sigma_2})\,F\big\}, \qquad
e = \mathcal{F}^{-1}\big\{G_{\sigma_e}\,\mathcal{F}\{r^2\}\big\}, \qquad
\text{mask} = \mathbf{1}[e > t_{\text{Otsu}}].
$$

$G_{\sigma_1} - G_{\sigma_2}$ is a difference-of-Gaussians band-pass, and
it's also a close approximation to the LoG. To tell *orientations* of texture
apart, replace it with a Gabor filter, which is a Gaussian centred at
$(u_0, v_0)$ instead of at the origin:
$H(u,v) = e^{-2\pi^2\sigma^2[(u-u_0)^2 + (v-v_0)^2]}$. A bank of these at
several $(u_0,v_0)$ gives each pixel a feature vector of band energies, and
regions are where that vector stays about the same.

### 6c. Removing the background: band-pass by object size

The thermal images show why this matters. The person is warm, but so is the
road near the camera, and the road's warmth changes smoothly down the frame.
That slow change is **low-frequency**. The person is a few tens of pixels
wide, which is a **middle** frequency, and sensor noise is **high**. So the
filter to use is the band-pass

$$
d = \mathcal{F}^{-1}\big\{(G_{\sigma_s} - G_{\sigma_\ell})\,F\big\},
\qquad \sigma_s \approx 1.5\text{ px}, \quad \sigma_\ell \approx \tfrac12\,\text{(person width)},
$$

followed by a threshold. The Q2 method does the same thing with a median
filter instead of $G_{\sigma_\ell}$. The median is nonlinear, so it has no
transfer function, but it plays the same role: it estimates everything wider
than the person, and subtracting that is a high-pass.

This only works when the object has a consistent brightness. In a colour
photo a person isn't reliably brighter or darker than the background: a red
shirt and a dark pair of jeans have opposite signs, so there is no frequency
band where "person" consistently lives. That's why Q1 uses a colour model
(GrabCut) instead of a filter.

---

## 7. Phase carries the location of edges

The magnitude $|F|$ says how much of each frequency there is. The phase
$\angle F$ says *where* those components line up. At a step edge, every
Fourier component is at the same phase at the edge location, which is why
they add up to a sharp jump there. If you swap magnitudes between two images
and keep the phases, the result still shows the edges of the image the
phases came from. For segmentation this means boundaries are encoded in the
phase, and magnitude-only features (like the band energy in 6b) are
position-free. They say *what kind* of region a pixel is in, not where the
region ends. That's why 6b needs the local averaging step: it gives up some
localisation to measure the texture.

---

## 8. Worked example by hand: an edge found by multiplying in frequency

An 8-sample signal with one step:

$$
f = [0,\,0,\,0,\,0,\,1,\,1,\,1,\,1].
$$

**DFT.** $F[k] = \sum_{n=4}^{7} e^{-j2\pi kn/8}$. For $k=0$ that's $4$. For
even $k \ne 0$ the four terms cancel, so $F[k] = 0$. For odd $k$, factor out
$e^{-j\pi k} = -1$ and sum the geometric series:

$$
F[k] = -\sum_{n=0}^{3} e^{-j\pi kn/4} = -\frac{1 - e^{-j\pi k}}{1 - e^{-j\pi k/4}}
= \frac{-2}{1 - e^{-j\pi k/4}} .
$$

Working it through gives $F[1] = -1 + j(1+\sqrt2)$ and
$F[3] = -1 + j(\sqrt2 - 1)$. $F[5]$ and $F[7]$ are the complex conjugates of
$F[3]$ and $F[1]$, as they must be for a real signal.

**Filter.** The central difference $g[n] = \tfrac12(f[n{+}1] - f[n{-}1])$
has transfer function $H[k] = j\sin(2\pi k/8)$.

| $k$ | $F[k]$ | $H[k]$ | $G[k] = H[k]F[k]$ |
|---|---|---|---|
| 0 | $4$ | $0$ | $0$ |
| 1 | $-1 + 2.4142j$ | $0.7071j$ | $-1.7071 - 0.7071j$ |
| 2 | $0$ | $j$ | $0$ |
| 3 | $-1 + 0.4142j$ | $0.7071j$ | $-0.2929 - 0.7071j$ |
| 4 | $0$ | $0$ | $0$ |
| 5 | $-1 - 0.4142j$ | $-0.7071j$ | $-0.2929 + 0.7071j$ |
| 6 | $0$ | $-j$ | $0$ |
| 7 | $-1 - 2.4142j$ | $-0.7071j$ | $-1.7071 + 0.7071j$ |

Check one entry: $G[1] = 0.7071j\,(-1 + 2.4142j) = -1.7071 - 0.7071j$. ✓

Notice $G[0] = 0$: the derivative threw away the DC term, which is the
average brightness, and kept only the change.

**Inverse DFT** of $G$:

$$
g = [-0.5,\; 0,\; 0,\; 0.5,\; 0.5,\; 0,\; 0,\; -0.5].
$$

**Directly in space**, $\tfrac12(f[n{+}1]-f[n{-}1])$ with wrap-around:

- $n=3$: $\tfrac12(1 - 0) = 0.5$
- $n=4$: $\tfrac12(1 - 0) = 0.5$
- $n=7$: $\tfrac12(0 - 1) = -0.5$
- $n=0$: $\tfrac12(0 - 1) = -0.5$
- every other $n$: $0$

The two routes give the same answer. The positive pair at $n = 3, 4$ is the
step up between samples 3 and 4. That's the edge, found by multiplying in
frequency. The negative pair at $n = 7, 0$ is the step *down* that the DFT
sees where the signal wraps around from the end back to the start. It's the
same circular-convolution effect Module 3 ran into at the image border.

---

## 9. Summary

| Goal | Transfer function $H(u,v)$ | Then |
|---|---|---|
| gradient edges | $j2\pi u\,G_\sigma,\; j2\pi v\,G_\sigma$ | magnitude, threshold |
| LoG edges | $-4\pi^2\rho^2\,G_\sigma$ | zero crossings |
| high-pass edges | Butterworth / Gaussian HP | magnitude, threshold |
| brightness regions | $G_\sigma$ | Otsu |
| texture regions | $G_{\sigma_1}-G_{\sigma_2}$ or Gabor, then $G_{\sigma_e}$ on $r^2$ | Otsu on energy |
| object-sized regions | $G_{\sigma_s}-G_{\sigma_\ell}$ | Otsu inside the box |

Edges come from filters that are **zero at DC and large at high frequency**.
Regions come from filters that **keep low frequency, or one chosen band, and
measure it**. Both are multiplications in the frequency domain, by the
convolution theorem from Module 3.

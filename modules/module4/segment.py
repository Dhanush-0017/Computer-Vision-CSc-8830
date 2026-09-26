"""
segment.py -- the Module 4 engine. Finding the outline of a person, no ML.

Everything here is classical: colour statistics, graph cuts, thresholds,
morphology, contours. No trained model, no learned weights, no
detector. The one input from outside is a rough box around the person (the
same kind of prompt SAM2 gets), so both methods are answering the same
question.

  RGB (Q1)      grabcut_person()
  thermal (Q2)  thermal_person()
  comparison    compare_masks()  -- IoU and Dice against SAM2's mask

No UI in this file. __init__.py (the page) and the command line scripts all
call into it.

Masks are uint8 arrays of 0/1, the same size as the image. Boxes are
(x0, y0, x1, y1) in pixels, inclusive-exclusive like a numpy slice.
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def clip_box(box, shape):
    h, w = shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    return x0, y0, x1, y1


def grow_box(box, frac, shape):
    """Box enlarged by frac of its size on every side, clipped to the image."""
    x0, y0, x1, y1 = box
    dx, dy = frac * (x1 - x0), frac * (y1 - y0)
    return clip_box((x0 - dx, y0 - dy, x1 + dx, y1 + dy), shape)


def box_mask(box, shape):
    m = np.zeros(shape[:2], np.uint8)
    x0, y0, x1, y1 = clip_box(box, shape)
    m[y0:y1, x0:x1] = 1
    return m


def to_gray_u8(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-9:
        return np.zeros(img.shape, np.uint8)
    return np.round(255 * (img - lo) / (hi - lo)).astype(np.uint8)


def fill_holes(mask):
    """Fill anything enclosed by the mask. Flood the background from the
    border; whatever the flood can't reach is a hole."""
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    flood = pad.copy()
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return (m | holes).astype(np.uint8)


def keep_components(mask, box=None, mode='largest', min_area=30):
    """Throw away blobs that aren't the person.

    mode='largest'  -> the single biggest blob (overlapping the box, if given)
    mode='overlap'  -> every blob that overlaps the central part of the box
    """
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    keep = []
    core = None
    if box is not None:
        x0, y0, x1, y1 = box
        # inner 60% of the box -- a blob that only touches the box edge is
        # usually a neighbour (the other pedestrian, a car door)
        cx0 = int(x0 + 0.2 * (x1 - x0)); cx1 = int(x1 - 0.2 * (x1 - x0))
        cy0 = int(y0 + 0.2 * (y1 - y0)); cy1 = int(y1 - 0.2 * (y1 - y0))
        core = np.zeros_like(m)
        core[cy0:max(cy1, cy0 + 1), cx0:max(cx1, cx0 + 1)] = 1
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if core is not None and not np.any(core[lab == i]):
            continue
        keep.append(i)
    if not keep:
        return np.zeros_like(m)
    if mode == 'largest':
        keep = [max(keep, key=lambda i: stats[i, cv2.CC_STAT_AREA])]
    return np.isin(lab, keep).astype(np.uint8)


def smooth_mask(mask, k=5):
    """Open then close with an ellipse: knocks off single-pixel spurs and
    seals 1-2 px cracks without rounding off fingers and feet too much."""
    if k <= 1:
        return (mask > 0).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_OPEN, se)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)


def contours_of(mask):
    """The boundary as polygons -- the 'exact boundary' the question asks
    for. CHAIN_APPROX_NONE keeps every boundary pixel, no simplification."""
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    return sorted(cs, key=cv2.contourArea, reverse=True)


def draw_result(img_bgr, mask, box=None, color=(0, 255, 0), alpha=0.35,
                thickness=2):
    """Tint the mask, trace the contour, draw the prompt box dashed-ish."""
    out = img_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    m = mask > 0
    tint = np.zeros_like(out)
    tint[:] = color
    out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
    cv2.drawContours(out, contours_of(mask), -1, color, thickness)
    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 255), 1)
    return out


# ---------------------------------------------------------------------------
# RGB (Q1) -- GrabCut (graph cut on colour), seeded from the box
# ---------------------------------------------------------------------------
def background_likelihood(img_bgr, box, ring=0.2, bins=16):
    """For each pixel, how much more its colour looks like the area just
    outside the box than the inside of the box. A ratio histogram:

        r(c) = H_ring(c) / (H_ring(c) + H_box(c))

    r near 1 -> this colour mostly lives outside the box, probably background.
    Plain counting, no fitting.
    """
    box = clip_box(box, img_bgr.shape)
    outer = grow_box(box, ring, img_bgr.shape)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    inside = box_mask(box, img_bgr.shape) > 0
    ringm = (box_mask(outer, img_bgr.shape) > 0) & ~inside
    q = (lab // (256 // bins)).astype(np.int32)
    idx = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
    hr = np.bincount(idx[ringm], minlength=bins ** 3).astype(np.float64)
    hb = np.bincount(idx[inside], minlength=bins ** 3).astype(np.float64)
    hr /= max(hr.sum(), 1.0)
    hb /= max(hb.sum(), 1.0)
    return (hr / (hr + hb + 1e-12))[idx]


def core_strip(box, core_w=0.10, top=0.08, bottom=0.50):
    """A thin vertical strip down the middle of the box, from just under the
    top edge to about the hips. For an upright person that strip is on the
    head/torso, so it's a safe 'definitely person' seed. Returns
    (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2
    half = max(1, int(core_w * bw / 2))
    return cx - half, y0 + int(top * bh), cx + half, y0 + int(bottom * bh)


def grabcut_person(img_bgr, box, iters=5, post_k=5, core_seed=True,
                   colour_prior=True, prior_thresh=0.5):
    """GrabCut seeded from the box, then cleaned up.

    What GrabCut does: every pixel gets a label cost from two colour models
    (person / not-person, each a 5-component Gaussian mixture fitted to *this
    image's* pixels), plus a cost for disagreeing with a similar-coloured
    neighbour. The min-cut of that graph is the best labelling. Re-fit the
    colour models to the new labels, cut again, `iters` times.

    Nothing is trained ahead of time -- the colour models are re-estimated
    from scratch for every image, so this is classical segmentation, not
    machine learning.

    Two additions over plain box-initialised GrabCut:
      core_seed     the core_strip() pixels are fixed as person. Stops GrabCut
                    losing a dark torso that happens to match the background.
      colour_prior  pixels inside the box whose colour mostly occurs *outside*
                    the box start as 'probably background' instead of
                    'probably person'. This is what stops the road between
                    the legs from being filled in.
    """
    box = clip_box(box, img_bgr.shape)
    x0, y0, x1, y1 = box
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros(img_bgr.shape[:2], np.uint8)
    mask = np.full(img_bgr.shape[:2], cv2.GC_BGD, np.uint8)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    if colour_prior:
        p = background_likelihood(img_bgr, box)
        inner = mask[y0:y1, x0:x1]
        inner[p[y0:y1, x0:x1] > prior_thresh] = cv2.GC_PR_BGD
    if core_seed:
        cx0, cy0, cx1, cy1 = core_strip(box)
        mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)  # the GMM init uses k-means, pin it so runs repeat
    cv2.grabCut(img_bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    fg = smooth_mask(fg, post_k)
    fg = keep_components(fg, box, mode='largest')
    return fill_holes(fg)


# ---------------------------------------------------------------------------
# Thermal (Q2) -- a person is a warm blob, so threshold, but carefully
# ---------------------------------------------------------------------------
def thermal_background(gray, box, scale=2.0):
    """Estimate what the scene would look like without the person: a median
    filter wider than the person. A median ignores anything narrower than
    half its window, so the person vanishes but slow changes -- the road
    getting warmer towards the camera, a warm wall -- stay. Computed at half
    resolution because OpenCV's median tops out at 255 px."""
    h, w = gray.shape
    k = int(scale * (box[2] - box[0]))
    k = max(3, min(255, (k // 2) | 1))
    small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small, k)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def thermal_person(img, box, blur_sigma=1.2, low_frac=0.7, margin=0.08,
                   post_k=5, bg_scale=2.0, return_stages=False):
    """Segment the warm body inside the box.

    Thermal intensity is (roughly) temperature, and a body sits a few degrees
    above most of a night scene, so the outline of a person is close to a
    level set of the image. That's why thresholding -- useless for people in
    RGB -- is the right tool here. The steps:

    1. Normalise to 0-255, light Gaussian blur (sensor noise).
    2. Subtract thermal_background(). Without this, a road that warms
       towards the bottom of the frame passes the threshold and the mask
       pours out of the feet.
    3. Hysteresis threshold inside the box (grown by `margin`). Otsu's level
       (maximises between-class variance of the warm/cool split) marks
       'definitely warm'; a lower level (low_frac x Otsu) keeps cooler
       pixels -- legs, clothing edges -- but only if they connect to
       something definitely warm. Same trick as Canny.
    4. Open/close, keep the blob through the middle of the box, fill holes.
    """
    g = to_gray_u8(img)
    h, w = g.shape
    box = clip_box(box, g.shape)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        empty = np.zeros((h, w), np.uint8)
        return (empty, {}) if return_stages else empty
    if blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    bg = thermal_background(g, box, bg_scale)
    d = np.clip(g.astype(np.int16) - bg.astype(np.int16), 0, 255).astype(np.uint8)

    rx0, ry0, rx1, ry1 = grow_box(box, margin, g.shape)
    patch = d[ry0:ry1, rx0:rx1]
    t_hi, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strong = (patch > t_hi).astype(np.uint8)
    weak = (patch > low_frac * t_hi).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    hit = np.unique(lab[strong > 0])
    m = np.isin(lab, hit[hit > 0]).astype(np.uint8)

    raw = np.zeros((h, w), np.uint8)
    raw[ry0:ry1, rx0:rx1] = m
    out = smooth_mask(raw, post_k)
    out = fill_holes(keep_components(out, box, mode='largest'))
    if return_stages:
        return out, {'blurred': g, 'background': bg, 'difference': d,
                     'threshold': float(t_hi), 'raw': raw}
    return out


# ---------------------------------------------------------------------------
# Comparing a mask with SAM2's
# ---------------------------------------------------------------------------
def compare_masks(pred, ref):
    """IoU = |A n B| / |A u B|  and  Dice = 2|A n B| / (|A| + |B|)."""
    p = pred > 0
    r = ref > 0
    inter = np.logical_and(p, r).sum()
    union = np.logical_or(p, r).sum()
    total = p.sum() + r.sum()
    return {'iou': float(inter / union) if union else 1.0,
            'dice': float(2 * inter / total) if total else 1.0}


def disagreement_image(pred, ref):
    """Green = both agree person, red = only pred, blue = only ref (BGR)."""
    p, r = pred > 0, ref > 0
    out = np.zeros(p.shape + (3,), np.uint8)
    out[p & r] = (80, 200, 80)
    out[p & ~r] = (60, 60, 230)
    out[~p & r] = (230, 140, 40)
    return out

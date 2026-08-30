"""Turn a heterogeneous source image into a normalized ink bitmap.

The source corpus mixes RGB / RGBA / L, sizes from 30x56 to 1500x1500, both
polarities, scans and *photographs* (some Bali samples are shot on grey or
blue card), and pencil as faint as a few grey levels.  Everything downstream
assumes a single representation:

    uint8 array, 0 = background, 255 = ink, tightly cropped to the ink,
    plus a *baseline* row index.

Scale is set per glyph from its **core band** -- the dense row-run holding the
base letter, excluding any vowel mark set above or below it.  Normalizing the
core band to a fixed height removes the scan-resolution spread in the source
corpus (Lontara alone runs 26-165 px for the same letter) while preserving the
proportion between a base and its marks, which is exactly the visual signal a
factored decoder has to pick up.  Genuine size variation is re-introduced
later, as a controlled jitter in the renderer, rather than inherited from
whatever DPI a contributor happened to scan at.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

#: Height, in pixels, every glyph's core band is scaled to.
DEFAULT_CORE_HEIGHT = 48

#: Clamp on the per-glyph scale factor, so a thumbnail-sized source image is
#: not blown up into a blurry giant.  Glyphs that hit the clamp are flagged.
MAX_UPSCALE = 6.0
MIN_CORE_PX = 4

#: Fraction of a glyph's ink mass the core band must contain.  Defined as a
#: mass share rather than a per-row density threshold: a single row crossing
#: many strokes spikes the maximum and makes any "fraction of the densest row"
#: rule collapse to two or three rows on handwritten input.
CORE_MASS_FRACTION = 0.72

#: Refuse glyphs whose ink is degenerate -- almost certainly a corrupt file.
MIN_INK_FRACTION = 1e-4
MAX_INK_FRACTION = 0.95


class GlyphLoadError(RuntimeError):
    pass


def _estimate_background(gray: np.ndarray) -> np.ndarray:
    """Smooth low-frequency background, for flattening uneven illumination.

    Estimated on a heavily downsampled copy: a max-filter there spans a large
    fraction of the image, so it climbs over the strokes and returns the
    paper, whether the paper is white, grey card, or a blue tablecloth.
    """
    h, w = gray.shape
    im = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8), mode="L")
    scale = 64 / max(h, w)
    sw, sh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    small = im.resize((sw, sh), Image.BILINEAR)
    small = small.filter(ImageFilter.MaxFilter(9))
    small = small.filter(ImageFilter.GaussianBlur(2.0))
    bg = small.resize((w, h), Image.BILINEAR)
    return np.asarray(bg, dtype=np.float32) / 255.0


def load_ink(path: str) -> np.ndarray:
    """Load an image as a float32 ink map in ``[0, 1]``, 1.0 = full ink."""
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            bg_img = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg_img, rgba).convert("L")
        else:
            im = im.convert("L")
        gray = np.asarray(im, dtype=np.float32) / 255.0

    if gray.size == 0:
        raise GlyphLoadError(f"empty image: {path}")

    # Polarity first: a dark 2px frame means light ink on dark paper.  Decide
    # before flattening, since the max-filter assumes the paper is the bright
    # side of the image.
    frame = np.concatenate([
        gray[:2].ravel(), gray[-2:].ravel(),
        gray[:, :2].ravel(), gray[:, -2:].ravel(),
    ])
    if float(np.median(frame)) < 0.5:
        gray = 1.0 - gray

    ink = np.clip(_estimate_background(gray) - gray, 0.0, None)

    # Scale by a high percentile rather than the max, so one dark speck does
    # not wash out faint pencil strokes.
    peak = float(np.percentile(ink, 99.5))
    if peak < 1e-3:
        raise GlyphLoadError(f"blank image: {path}")
    return np.clip(ink / peak, 0.0, 1.0)


def otsu(ink: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold on a [0,1] map, implemented directly (no skimage dep)."""
    hist, edges = np.histogram(ink, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    total_m = m0[-1]
    m1 = total_m - m0
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = np.where(w0 > 0, m0 / np.maximum(w0, 1e-12), 0.0)
        mu1 = np.where(w1 > 0, m1 / np.maximum(w1, 1e-12), 0.0)
    between = w0 * w1 * (mu0 - mu1) ** 2

    # Near-bimodal input makes the criterion flat across a wide plateau of
    # equally optimal thresholds.  argmax would return the lowest of them,
    # which under-thresholds and lets paper residue through; take the middle
    # of the plateau instead.
    best = np.flatnonzero(between >= between.max() - 1e-12)
    return float(centers[int(round(float(best.mean())))])


def ink_mask(ink: np.ndarray) -> np.ndarray:
    thr = otsu(ink)
    mask = ink >= max(thr, 0.15)
    frac = mask.mean()
    if frac < MIN_INK_FRACTION or frac > MAX_INK_FRACTION:
        mask = ink >= 0.5
    return mask


def _is_thin_line(cov, i: int, inward: int, coverage: float,
                  gap: float = 0.45, look: int = 3) -> bool:
    """True if index ``i`` is a thin high-coverage line, not part of the glyph.

    A frame edge is one to three pixels thick with blank paper just inside it.
    A glyph's own long horizontal stroke sits on a connected body, so the rows
    immediately inside it are also inked.  Checking the neighbour keeps the
    rule from eating real strokes.
    """
    if cov[i] < coverage:
        return False
    j = i + inward * look
    if not (0 <= j < len(cov)):
        return True
    return cov[j] < gap


def strip_frame(mask: np.ndarray, band: float = 0.18, coverage: float = 0.70,
                max_passes: int = 3):
    """Crop hard border lines -- a scan edge or the rim of a photographed card.

    Background flattening removes a *gradient*; it cannot remove a sharp dark
    edge, which then dominates the ink bounding box.  A frame line shows up as
    a row or column that is almost entirely ink and lies near the border, a
    combination glyph strokes essentially never produce.

    Returns the crop box ``(r0, c0, r1, c1)``.
    """
    h, w = mask.shape
    r0, c0, r1, c1 = 0, 0, h, w
    for _ in range(max_passes):
        sub = mask[r0:r1, c0:c1]
        sh, sw = sub.shape
        if sh < 8 or sw < 8:
            break
        rows = sub.mean(axis=1)
        cols = sub.mean(axis=0)
        nr, nc = max(1, int(sh * band)), max(1, int(sw * band))
        changed = False
        top = [i for i in range(nr) if _is_thin_line(rows, i, +1, coverage)]
        if top:
            r0 += top[-1] + 1
            changed = True
        bot = [i for i in range(sh - nr, sh) if _is_thin_line(rows, i, -1, coverage)]
        if bot:
            r1 = r0 + bot[0] - (r0 - (r1 - sh))
            r1 = min(r1, h)
            changed = True
        left = [i for i in range(nc) if _is_thin_line(cols, i, +1, coverage)]
        if left:
            c0 += left[-1] + 1
            changed = True
        right = [i for i in range(sw - nc, sw) if _is_thin_line(cols, i, -1, coverage)]
        if right:
            c1 = c0 + right[0]
            changed = True
        if not changed or r1 - r0 < 8 or c1 - c0 < 8:
            break
    return r0, c0, r1, c1


def trim(ink: np.ndarray, mask: np.ndarray, pad: int = 1):
    """Crop both maps to the ink bounding box.  Returns (ink, mask, bbox)."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise GlyphLoadError("no ink after thresholding")
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    r1, c1 = min(r1 + pad, ink.shape[0]), min(c1 + pad, ink.shape[1])
    return ink[r0:r1, c0:c1], mask[r0:r1, c0:c1], (r0, c0, r1, c1)


def core_band(mask: np.ndarray, mass_fraction: float = CORE_MASS_FRACTION,
              min_rows: int = 3):
    """Rows holding the glyph body, excluding marks above and below.

    An abugida glyph is a base letter plus optional vowel marks set above or
    below it.  The base holds most of the ink; the marks are thin.  So the
    body is the *shortest* run of rows containing ``mass_fraction`` of the
    total ink -- adding a mark's rows costs height and buys little mass, so
    the minimal window excludes them.

    Its bottom edge is what glyphs are aligned on.  Note this is an alignment
    reference, not a typographic baseline: on a vertically uniform body the
    minimal window is free to sit anywhere inside it.  What matters
    downstream is that the reference is computed the same way for every glyph,
    so a rendered line is internally consistent.
    """
    row = mask.sum(axis=1).astype(np.float64)
    height = mask.shape[0]
    total = row.sum()
    if total <= 0:
        return 0, height
    target = mass_fraction * total

    best_top, best_bottom, best_len = 0, height, height + 1
    start, acc = 0, 0.0
    for end in range(height):
        acc += row[end]
        while acc - row[start] >= target:
            acc -= row[start]
            start += 1
        if acc >= target and (end - start + 1) < best_len:
            best_len, best_top, best_bottom = end - start + 1, start, end + 1

    if best_bottom - best_top < min_rows:
        pad = (min_rows - (best_bottom - best_top) + 1) // 2
        best_top = max(0, best_top - pad)
        best_bottom = min(height, best_bottom + pad)
    return int(best_top), int(best_bottom)


def resize_ink(ink: np.ndarray, scale: float, max_side: int = 512) -> np.ndarray:
    h, w = ink.shape
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if max(nh, nw) > max_side:
        shrink = max_side / max(nh, nw)
        nh, nw = max(1, int(nh * shrink)), max(1, int(nw * shrink))
    if (nh, nw) == (h, w):
        return ink
    img = Image.fromarray((np.clip(ink, 0, 1) * 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize((nw, nh), Image.LANCZOS), dtype=np.float32) / 255.0


class PreparedGlyph:
    """A normalized glyph bitmap plus the geometry the renderer needs."""

    __slots__ = ("bitmap", "baseline", "core_top", "scale", "clamped",
                 "src_core_h", "src_h")

    def __init__(self, bitmap, baseline, core_top, scale, clamped,
                 src_core_h, src_h):
        self.bitmap = bitmap        # uint8, 0 = background, 255 = ink
        self.baseline = baseline    # row index: bottom of the core band
        self.core_top = core_top    # row index: top of the core band
        self.scale = scale
        self.clamped = clamped
        self.src_core_h = src_core_h
        self.src_h = src_h

    @property
    def size(self):
        return self.bitmap.shape  # (h, w)


def prepare_glyph(path: str, target_core_h: int = DEFAULT_CORE_HEIGHT,
                  core_fraction: float = CORE_MASS_FRACTION) -> PreparedGlyph:
    """Load, threshold, trim, scale by core band, and locate the baseline."""
    ink = load_ink(path)
    thr = otsu(ink)
    mask = ink_mask(ink)

    # Trim first, then look for a frame.  A drawn or photographed border is
    # often well inside the raw canvas, which puts it outside the border band
    # strip_frame examines; after trimming, the frame *is* the border.
    ink, mask, _ = trim(ink, mask, pad=0)

    # Detect the frame on a deliberately sensitive mask.  Many of these
    # borders are fainter than the binarization threshold, so they are absent
    # from the strict mask yet still visible once the glyph is composited.
    loose = ink >= max(0.45 * thr, 0.06)
    fr0, fc0, fr1, fc1 = strip_frame(loose)
    if (fr0, fc0, fr1, fc1) != (0, 0, *loose.shape):
        cropped = ink[fr0:fr1, fc0:fc1]
        if (cropped >= max(thr, 0.15)).any():
            ink = cropped
            mask = ink >= max(thr, 0.15)

    ink, mask, (r0, _, r1, _) = trim(ink, mask)
    src_h = ink.shape[0]

    top, bottom = core_band(mask, core_fraction)
    src_core_h = max(bottom - top, MIN_CORE_PX)

    scale = target_core_h / src_core_h
    clamped = scale > MAX_UPSCALE
    if clamped:
        scale = MAX_UPSCALE

    ink = resize_ink(ink, scale)

    # Zero the paper.  Background flattening leaves a low residue (a photo of
    # grey card does not come back perfectly white), and because glyphs are
    # max-blended onto the line canvas, any residue shows up as a grey box
    # behind the glyph.  The cut sits just under the binarization threshold,
    # which is where paper residue and faint borders live; real strokes are
    # above it by construction.
    floor = float(np.clip(0.80 * otsu(ink), 0.05, 0.50))
    ink = np.clip((ink - floor) / (1.0 - floor), 0.0, 1.0)

    mask2 = ink >= 0.35
    if not mask2.any():
        mask2 = ink >= max(ink.max() * 0.5, 1e-6)
    top2, bottom2 = core_band(mask2, core_fraction)

    bitmap = (np.clip(ink, 0, 1) * 255).astype(np.uint8)
    return PreparedGlyph(bitmap, int(bottom2), int(top2), float(scale),
                         bool(clamped), int(src_core_h), int(src_h))


def stroke_width(bitmap: np.ndarray, threshold: int = 96) -> float:
    """Mean stroke thickness in pixels, as a proxy for the pen used.

    Ink area divided by boundary length is roughly half the stroke width for
    a long thin stroke.  It needs no distance transform, so it costs one pass
    over the bitmap, and it separates a fine pencil from a marker well enough
    to keep one rendered line in a single hand.
    """
    mask = bitmap > threshold
    area = int(mask.sum())
    if area == 0:
        return 0.0
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                            & mask[1:-1, :-2] & mask[1:-1, 2:])
    boundary = int(mask.sum() - interior.sum())
    if boundary == 0:
        return float(2 * np.sqrt(area / np.pi))
    return float(2.0 * area / boundary)


def measure_core_height(path: str) -> tuple[int, int]:
    """``(core_band_height, full_ink_height)`` at native resolution."""
    ink = load_ink(path)
    mask = ink_mask(ink)
    _, mask, (r0, _, r1, _) = trim(ink, mask)
    top, bottom = core_band(mask)
    return bottom - top, r1 - r0

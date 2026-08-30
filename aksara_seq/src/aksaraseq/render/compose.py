"""Compose a line of text by placing real glyph bitmaps on a shared baseline.

Every glyph in a rendered line is an actual scan of someone's handwriting,
sampled from the split's own disjoint pool.  What the renderer adds is
*layout*: where each glyph sits relative to the baseline, how much it is
rotated and scaled, and how far the pen advanced.

Because these are abugidas, the syllable image is already the correct visual
unit -- for Javanese ``o`` it is a three-part cluster (taling before the
consonant, tarung after), stored whole.  So laying syllables out left to right
is orthographically faithful, not an approximation of one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class LayoutStyle:
    """All layout randomness, in units of the glyph core height."""

    core_height: int = 48

    # Gaps are *ink clearance*, not bounding-box padding: the distance from
    # the previous glyph's nearest ink to this one's, measured row by row.
    gap_mean: float = 0.09          # inter-syllable clearance
    gap_std: float = 0.04
    word_gap_mean: float = 0.42     # inter-word clearance
    word_gap_std: float = 0.10
    min_gap: float = 0.02           # floor on clearance; keeps ink separated

    rotation_std: float = 0.0       # degrees, per glyph
    rotation_max: float = 8.0
    scale_log_std: float = 0.0      # per-glyph size jitter, log-normal
    scale_min: float = 0.72
    scale_max: float = 1.40

    baseline_jitter_std: float = 0.0
    drift_amplitude: float = 0.0    # slow baseline wander across the line
    drift_periods: float = 1.4

    margin_x: float = 0.35
    margin_y: float = 0.30

    def px(self, v: float) -> float:
        return v * self.core_height


@dataclass
class PlacedGlyph:
    syllable: str
    pool_idx: int
    x: int
    y: int
    width: int
    height: int
    baseline: int
    angle: float
    scale: float


@dataclass
class ComposedLine:
    ink: np.ndarray                       # float32 in [0, 1], 1 = ink
    words: list
    placed: list = field(default_factory=list)
    baseline_y: int = 0
    pen: float = None

    @property
    def size(self):
        return self.ink.shape


def _rotate_glyph(bitmap: np.ndarray, baseline: int, angle: float):
    """Rotate about the image centre; return the bitmap and the new baseline.

    PIL's ``rotate(angle, expand=True)`` turns the image counter-clockwise
    about its centre and grows the canvas.  In image coordinates (y down) that
    is ``R = [[cos, sin], [-sin, cos]]``, so the baseline's new row follows
    analytically -- no need to re-detect it on the rotated bitmap.
    """
    if abs(angle) < 1e-3:
        return bitmap, float(baseline)

    h, w = bitmap.shape
    img = Image.fromarray(bitmap, mode="L")
    out = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
    rotated = np.asarray(out, dtype=np.uint8)
    H, W = rotated.shape

    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = 0.0, baseline - (h - 1) / 2.0            # taken at the glyph's centre column
    new_y = (H - 1) / 2.0 + (-sin_t * dx + cos_t * dy)
    return rotated, float(np.clip(new_y, 0, H - 1))


def _scale_glyph(bitmap: np.ndarray, baseline: float, factor: float):
    if abs(factor - 1.0) < 1e-3:
        return bitmap, baseline
    h, w = bitmap.shape
    nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    img = Image.fromarray(bitmap, mode="L").resize((nw, nh), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8), baseline * (nh / h)


def _ink_profiles(bitmap: np.ndarray, threshold: int = 32):
    """Per-row leftmost and rightmost ink column.

    Rows with no ink get ``(+inf, -inf)`` so they place no constraint on
    kerning.
    """
    mask = bitmap > threshold
    has_ink = mask.any(axis=1)
    left = np.argmax(mask, axis=1).astype(np.float64)
    right = (mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)).astype(np.float64)
    left[~has_ink] = np.inf
    right[~has_ink] = -np.inf
    return left, right


def _blend(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Max-blend ink, so overlapping strokes merge instead of overwriting."""
    h, w = patch.shape
    ch, cw = canvas.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    if x0 >= x1 or y0 >= y1:
        return
    sub = patch[y0 - y: y1 - y, x0 - x: x1 - x]
    np.maximum(canvas[y0:y1, x0:x1], sub, out=canvas[y0:y1, x0:x1])


def compose_line(words, pool, split: str, style: LayoutStyle, rng,
                 pen: float = None, pen_k_frac: float = 0.30) -> ComposedLine:
    """Render one line.  ``words`` is a list of syllable tuples.

    ``pen`` biases every glyph in the line toward one stroke width, so the
    line reads as written by a single person with a single pen.
    """
    # -- 1. draw the glyphs and their per-glyph transforms -------------------
    items = []
    for wi, word in enumerate(words):
        for si, syl in enumerate(word):
            idx = pool.sample_idx(syl, split, rng, pen=pen, k_frac=pen_k_frac)
            bmp = np.array(pool.bitmap(idx), dtype=np.uint8, copy=True)
            base = float(pool.baseline(idx))

            scale = 1.0
            if style.scale_log_std > 0:
                scale = float(np.clip(np.exp(rng.normal(0.0, style.scale_log_std)),
                                      style.scale_min, style.scale_max))
                bmp, base = _scale_glyph(bmp, base, scale)

            angle = 0.0
            if style.rotation_std > 0:
                angle = float(np.clip(rng.normal(0.0, style.rotation_std),
                                      -style.rotation_max, style.rotation_max))
                bmp, base = _rotate_glyph(bmp, base, angle)

            items.append({
                "syllable": syl, "idx": idx, "bmp": bmp, "base": base,
                "angle": angle, "scale": scale,
                "word_start": si == 0, "first": wi == 0 and si == 0,
            })

    # -- 2. lay them out along x, and record each baseline offset ------------
    gaps = []
    for it in items:
        if it["first"]:
            gaps.append(0.0)
        elif it["word_start"]:
            gaps.append(max(rng.normal(style.word_gap_mean, style.word_gap_std),
                            style.gap_mean))
        else:
            gaps.append(max(rng.normal(style.gap_mean, style.gap_std),
                            style.min_gap))

    # Provisional bounding-box advance.  Used only to give the baseline drift
    # an x coordinate; the real placement is kerned below.
    x = 0.0
    xs = []
    for it, gap in zip(items, gaps):
        x += style.px(gap)
        xs.append(x)
        x += it["bmp"].shape[1]
    line_width = x

    span = max(line_width, 1.0)
    # One phase for the whole line: drift is a slow wander of the writing
    # line, not per-glyph noise.  Per-glyph noise is baseline_jitter.
    drift_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    offsets = []
    for xi in xs:
        dy = 0.0
        if style.drift_amplitude > 0:
            phase = 2.0 * np.pi * style.drift_periods * (xi / span)
            dy += style.px(style.drift_amplitude) * np.sin(phase + drift_phase)
        if style.baseline_jitter_std > 0:
            dy += rng.normal(0.0, style.px(style.baseline_jitter_std))
        offsets.append(dy)

    # -- 3. canvas height, which depends only on the vertical placement ------
    tops = [off - it["base"] for it, off in zip(items, offsets)]
    bottoms = [off + (it["bmp"].shape[0] - it["base"]) for it, off in zip(items, offsets)]
    top, bottom = min(tops), max(bottoms)

    mx, my = style.px(style.margin_x), style.px(style.margin_y)
    height = max(int(np.ceil(bottom - top + 2 * my)), 1)
    baseline_y = my - top

    # -- 4. kerned placement -------------------------------------------------
    # Advancing by bounding-box width leaves a wide blank whenever a glyph's
    # ink sits away from its box edge -- which is common here, since a vowel
    # mark can extend the box well past the letter body.  Instead, slide each
    # glyph left until its closest ink comes within `gap` of the ink already
    # on the canvas, row by row.  Neighbours can then tuck under each other's
    # overhangs the way they do in real writing.
    right_profile = np.full(height, -np.inf, dtype=np.float64)
    positions, ys = [], []
    prev_x = prev_w = 0.0

    for i, (it, gap, off) in enumerate(zip(items, gaps, offsets)):
        bmp = it["bmp"]
        gh, gw = bmp.shape
        py = int(round(baseline_y + off - it["base"]))
        py = max(0, min(py, height - gh)) if gh <= height else 0

        left, right = _ink_profiles(bmp)
        rows = slice(py, min(py + gh, height))
        n_rows = rows.stop - rows.start
        clearance = style.px(gap)

        if i == 0:
            gx = 0.0
        else:
            occupied = right_profile[rows]
            need = occupied + clearance - left[:n_rows]
            finite = need[np.isfinite(need)]
            gx = float(finite.max()) if finite.size else prev_x + prev_w + clearance
            # Never let a glyph slide so far left that it disappears behind
            # its neighbour, however permissive the shapes happen to be.
            gx = max(gx, prev_x + 0.25 * prev_w)

        positions.append(gx)
        ys.append(py)
        placed_right = gx + right[:n_rows]
        np.maximum(right_profile[rows], placed_right, out=right_profile[rows])
        prev_x, prev_w = gx, float(gw)

    shift = -min(positions) if positions else 0.0
    ink_width = max((p + shift + it["bmp"].shape[1]
                     for p, it in zip(positions, items)), default=0.0)
    width = max(int(np.ceil(ink_width + 2 * mx)), 1)
    canvas = np.zeros((height, width), dtype=np.uint8)

    placed = []
    for it, gx, py in zip(items, positions, ys):
        px_ = int(round(mx + gx + shift))
        _blend(canvas, it["bmp"], px_, py)
        placed.append(PlacedGlyph(
            syllable=it["syllable"], pool_idx=it["idx"], x=px_, y=py,
            width=it["bmp"].shape[1], height=it["bmp"].shape[0],
            baseline=int(round(it["base"])), angle=it["angle"], scale=it["scale"],
        ))

    return ComposedLine(ink=canvas.astype(np.float32) / 255.0, words=list(words),
                        placed=placed, baseline_y=int(round(baseline_y)),
                        pen=pen)

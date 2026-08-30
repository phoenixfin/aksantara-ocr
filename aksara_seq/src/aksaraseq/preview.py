"""Contact sheets -- the only honest way to check normalization and rendering."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _paste_ink(canvas: np.ndarray, ink: np.ndarray, x: int, y: int) -> None:
    """Max-blend an ink patch into a canvas at (x, y), clipped to bounds."""
    h, w = ink.shape
    ch, cw = canvas.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    if x0 >= x1 or y0 >= y1:
        return
    sub = ink[y0 - y: y1 - y, x0 - x: x1 - x]
    np.maximum(canvas[y0:y1, x0:x1], sub, out=canvas[y0:y1, x0:x1])


def glyph_sheet(glyphs, labels, cols: int = 10, cell: tuple[int, int] = (150, 170),
                baseline_frac: float = 0.72, show_baseline: bool = True,
                title: str = "") -> Image.Image:
    """Grid of normalized glyphs, every baseline drawn at the same cell row.

    If the baseline estimate is sound the glyphs visibly sit on one line and
    marks above/below hang off it; if it is not, the row looks ragged.
    """
    cw, ch = cell
    rows = (len(glyphs) + cols - 1) // cols
    head = 26 if title else 0
    W, H = cols * cw, rows * ch + head

    canvas = np.zeros((H, W), dtype=np.uint8)
    base_y = int(ch * baseline_frac)

    for i, g in enumerate(glyphs):
        r, c = divmod(i, cols)
        gh, gw = g.bitmap.shape
        x = c * cw + (cw - gw) // 2
        y = head + r * ch + base_y - g.baseline
        _paste_ink(canvas, g.bitmap, x, y)

    img = Image.fromarray(255 - canvas, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    if title:
        draw.text((6, 6), title, fill=(0, 0, 0))
    for r in range(rows):
        y = head + r * ch + base_y
        if show_baseline:
            draw.line([(0, y), (W, y)], fill=(220, 60, 60), width=1)
        draw.line([(0, head + r * ch), (W, head + r * ch)], fill=(210, 210, 210), width=1)
    for i, lab in enumerate(labels):
        r, c = divmod(i, cols)
        draw.text((c * cw + 4, head + r * ch + 4), lab, fill=(40, 90, 200))
    return img


def line_sheet(images, captions, width: int = 1500, pad: int = 10,
               title: str = "") -> Image.Image:
    """Stack rendered line images vertically with their transcriptions.

    Reading this sheet is the acceptance test for the renderer: the glyphs
    should sit on a common writing line, the word gaps should be visibly
    wider than the syllable gaps, and the transcription should match what a
    reader of the script would read off the image.
    """
    scaled = []
    for im in images:
        if im.width > width:
            h = max(1, int(im.height * width / im.width))
            im = im.resize((width, h), Image.LANCZOS)
        scaled.append(im.convert("RGB"))

    head = 24 if title else 0
    total = head + sum(im.height + 20 + pad for im in scaled) + pad
    canvas = Image.new("RGB", (width, total), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((8, 6), title, fill=(0, 0, 0))

    y = head + pad
    for im, cap in zip(scaled, captions):
        draw.text((8, y), cap, fill=(30, 90, 190))
        y += 16
        canvas.paste(im, (8, y))
        draw.rectangle([7, y - 1, 8 + im.width, y + im.height], outline=(225, 225, 225))
        y += im.height + pad + 4
    return canvas

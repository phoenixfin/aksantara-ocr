"""Turn a clean ink map into something that looks photographed rather than pasted.

A CTC model reads a perfectly spaced, pure-black-on-pure-white line almost
trivially, and a near-zero error rate on such input measures the renderer, not
the model.  These presets exist so difficulty is an explicit, reported axis of
the dataset instead of an accident of how the compositor happened to work.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from .compose import LayoutStyle


@dataclass
class DegradeStyle:
    stroke_delta: float = 0.0        # >0 dilates the pen, <0 thins it (px)
    blur: float = 0.0                # gaussian sigma, px
    paper_gray: tuple = (1.0, 0.0)   # (mean, std) of the paper level
    ink_gray: tuple = (0.05, 0.0)    # (mean, std) of the darkest ink level
    texture: float = 0.0             # low-frequency paper mottling amplitude
    texture_scale: int = 12          # cells across the image; smaller = coarser
    noise: float = 0.0               # per-pixel gaussian sigma
    contrast: float = 1.0            # <1 washes the image out
    jpeg_quality: int = 0            # 0 disables the JPEG round trip


@dataclass
class RenderStyle:
    name: str = "clean"
    layout: LayoutStyle = field(default_factory=LayoutStyle)
    degrade: DegradeStyle = field(default_factory=DegradeStyle)

    def as_dict(self) -> dict:
        return {"name": self.name, "layout": asdict(self.layout),
                "degrade": asdict(self.degrade)}


def _low_freq_field(shape, cells: int, rng) -> np.ndarray:
    """Zero-mean smooth field, for paper mottling and uneven lighting."""
    h, w = shape
    ch = max(2, int(round(cells * h / max(h, w))))
    cw = max(2, int(round(cells * w / max(h, w))))
    small = rng.normal(0.0, 1.0, size=(ch, cw)).astype(np.float32)
    img = Image.fromarray(small, mode="F").resize((w, h), Image.BICUBIC)
    field_ = np.asarray(img, dtype=np.float32)
    return field_ - field_.mean()


def _stroke(ink: np.ndarray, delta: float) -> np.ndarray:
    if abs(delta) < 0.5:
        return ink
    k = int(2 * round(abs(delta)) + 1)
    img = Image.fromarray((np.clip(ink, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.MaxFilter(k) if delta > 0 else ImageFilter.MinFilter(k))
    return np.asarray(img, dtype=np.float32) / 255.0


def apply_degradation(ink: np.ndarray, style: DegradeStyle, rng) -> np.ndarray:
    """``ink`` is float32 in [0,1]; returns a uint8 grayscale page image."""
    x = np.clip(ink.astype(np.float32), 0.0, 1.0)

    x = _stroke(x, style.stroke_delta)

    if style.blur > 0:
        img = Image.fromarray((x * 255).astype(np.uint8), mode="L")
        x = np.asarray(img.filter(ImageFilter.GaussianBlur(style.blur)),
                       dtype=np.float32) / 255.0

    paper = float(np.clip(rng.normal(*style.paper_gray), 0.35, 1.0))
    ink_level = float(np.clip(rng.normal(*style.ink_gray), 0.0, 0.75))
    if ink_level >= paper:                      # keep ink darker than paper
        ink_level = max(0.0, paper - 0.15)

    page = np.full(x.shape, paper, dtype=np.float32)
    if style.texture > 0:
        page += style.texture * _low_freq_field(x.shape, style.texture_scale, rng)

    out = page * (1.0 - x) + ink_level * x

    if style.contrast != 1.0:
        out = 0.5 + (out - 0.5) * style.contrast

    if style.noise > 0:
        out = out + rng.normal(0.0, style.noise, size=out.shape).astype(np.float32)

    out = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

    if style.jpeg_quality:
        buf = io.BytesIO()
        Image.fromarray(out, mode="L").save(buf, format="JPEG",
                                            quality=int(style.jpeg_quality))
        buf.seek(0)
        with Image.open(buf) as im:
            out = np.asarray(im.convert("L"), dtype=np.uint8)
    return out


def make_presets(core_height: int = 48) -> dict:
    """The four difficulty levels, all expressed relative to the core height."""

    def layout(**kw) -> LayoutStyle:
        return LayoutStyle(core_height=core_height, **kw)

    return {
        # Pasted glyphs, even spacing.  The upper bound on any model's score,
        # and the control condition for the difficulty ablation.
        "clean": RenderStyle(
            "clean",
            layout(gap_mean=0.10, gap_std=0.02, min_gap=0.045),
            DegradeStyle(),
        ),
        # A tidy hand on clean paper.
        "light": RenderStyle(
            "light",
            layout(gap_mean=0.09, gap_std=0.04, min_gap=0.035,
                   rotation_std=1.5,
                   scale_log_std=0.05, baseline_jitter_std=0.03,
                   drift_amplitude=0.04),
            DegradeStyle(blur=0.4, paper_gray=(0.97, 0.02),
                         ink_gray=(0.08, 0.03), texture=0.012, noise=0.006),
        ),
        # A normal hand, scanned.
        "medium": RenderStyle(
            "medium",
            layout(gap_mean=0.08, gap_std=0.045, word_gap_mean=0.38,
                   min_gap=0.028, rotation_std=3.0, scale_log_std=0.10,
                   baseline_jitter_std=0.06, drift_amplitude=0.10),
            DegradeStyle(stroke_delta=0.6, blur=0.7, paper_gray=(0.93, 0.04),
                         ink_gray=(0.12, 0.05), texture=0.030, noise=0.012,
                         contrast=0.94, jpeg_quality=88),
        ),
        # A cramped hand, photographed under uneven light.  Still never
        # overlapping: every preset floors clearance above zero, so each
        # syllable keeps a separable ink box.
        "heavy": RenderStyle(
            "heavy",
            layout(gap_mean=0.07, gap_std=0.05, word_gap_mean=0.32,
                   word_gap_std=0.12, min_gap=0.022, rotation_std=5.0,
                   rotation_max=11.0, scale_log_std=0.16,
                   baseline_jitter_std=0.10, drift_amplitude=0.20,
                   drift_periods=1.9),
            DegradeStyle(stroke_delta=1.2, blur=1.1, paper_gray=(0.86, 0.07),
                         ink_gray=(0.20, 0.08), texture=0.065, texture_scale=7,
                         noise=0.022, contrast=0.86, jpeg_quality=72),
        ),
    }

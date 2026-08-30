"""Generate a rendered line corpus: images, transcriptions, and provenance.

One sample is produced by a seed derived from ``(seed, script, split, index)``,
so any single sample can be regenerated on its own and the whole corpus is
reproducible from the config alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..glyphs.pool import GlyphPool
from .compose import compose_line
from .degrade import apply_degradation, make_presets
from .lexicon import (
    PseudoWordSampler, WordMixer, load_lexicon, line_text, line_tokens, WORD_SEP,
)

SPLITS = ("train", "val", "test")


@dataclass
class CorpusConfig:
    name: str = "aksaraline_v1"
    glyph_cache: Path = Path("aksara_seq/build/glyphs")
    out: Path = Path("aksara_seq/build/corpus/v1")
    scripts: list = field(default_factory=lambda: ["Sunda", "Jawa", "Bali", "Lontara"])
    seed: int = 20260830
    core_height: int = 48
    output_height: int = 96          # 0 keeps the composed resolution
    samples: dict = field(default_factory=lambda: {"train": 20000, "val": 2000, "test": 2000})
    words_per_line: dict = field(default_factory=lambda: {1: 0.45, 2: 0.25, 3: 0.16, 4: 0.09, 5: 0.05})
    style_mix: dict = field(default_factory=lambda: {"clean": 0.15, "light": 0.30, "medium": 0.35, "heavy": 0.20})
    length_weights: dict = field(default_factory=dict)
    vowel_weights: dict = field(default_factory=dict)
    lexicon_dir: Path = Path("aksara_seq/lexicons")
    lexicon_fraction: float = 0.0
    include_digits: bool = False
    image_format: str = "png"
    # Fraction of each syllable's instances a line may draw from, taken
    # nearest its own pen width.  Keeps one line in one hand.  1.0 disables
    # and samples every glyph independently.
    pen_k_frac: float = 0.30

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k in ("glyph_cache", "out", "lexicon_dir"):
            d[k] = str(d[k])
        return d


def _weighted_keys(mapping):
    keys = list(mapping)
    w = np.array([float(mapping[k]) for k in keys], dtype=np.float64)
    return keys, w / w.sum()


def _script_seed(seed: int, script: str, split: str, index: int) -> np.random.Generator:
    # blake2b, not hash(): Python randomizes string hashing per process, which
    # would make the corpus irreproducible across runs.
    tag = int.from_bytes(
        hashlib.blake2b(f"{script}/{split}".encode("utf-8"), digest_size=4).digest(),
        "big")
    return np.random.default_rng([seed, tag, index])


def build_word_source(pool: GlyphPool, cfg: CorpusConfig, script: str, split: str):
    """Word sampler restricted to syllables this split can actually draw."""
    available = {s for s in pool.syllables(split)
                 if cfg.include_digits or not s.startswith("<")}
    onsets = sorted({e.onset for e in pool.entries if e.syllable in available})
    vowels = sorted({e.vowel for e in pool.entries
                     if e.syllable in available and e.vowel})

    pseudo = PseudoWordSampler(
        available, onsets=onsets, vowels=vowels,
        vowel_weights=cfg.vowel_weights or None,
        length_weights={int(k): v for k, v in cfg.length_weights.items()} or None,
    )

    lex_words, rejected = [], []
    if cfg.lexicon_fraction > 0:
        path = Path(cfg.lexicon_dir) / f"{script.lower()}.txt"
        if path.exists():
            loaded = load_lexicon(path, available)
            lex_words, rejected = loaded.words, loaded.rejected
    return WordMixer(pseudo, lex_words, cfg.lexicon_fraction), rejected


def charset(pool: GlyphPool, cfg: CorpusConfig) -> dict:
    syls = sorted({s for s in pool.syllables("train")
                   if cfg.include_digits or not s.startswith("<")})
    by_syl = factor_lookup(pool)
    onsets = sorted({by_syl[s][0] for s in syls})
    vowels = sorted({by_syl[s][1] for s in syls if by_syl[s][1]})
    return {
        "script": pool.script,
        "word_separator": WORD_SEP,
        "syllables": syls,
        "onsets": onsets,
        "vowels": vowels,
        "syllable_to_factors": {s: {"onset": by_syl[s][0], "vowel": by_syl[s][1]}
                                for s in syls},
        "n_syllables": len(syls),
        "n_onsets": len(onsets),
        "n_vowels": len(vowels),
    }


def _scale_box(g, sx: float, sy: float, max_w: int, max_h: int) -> list:
    """A placed glyph's ink box, in saved-image pixels, clipped to the image."""
    x0 = int(round(g.x * sx))
    y0 = int(round(g.y * sy))
    x1 = int(round((g.x + g.width) * sx))
    y1 = int(round((g.y + g.height) * sy))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(max_w, x1), min(max_h, y1)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _union_box(boxes) -> list:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def render_sample(pool: GlyphPool, mixer: WordMixer, cfg: CorpusConfig,
                  presets: dict, split: str, index: int):
    """Render one line.  Returns ``(PIL image, record dict)``."""
    rng = _script_seed(cfg.seed, pool.script, split, index)

    n_keys, n_p = _weighted_keys(cfg.words_per_line)
    n_words = int(n_keys[rng.choice(len(n_keys), p=n_p)])
    s_keys, s_p = _weighted_keys(cfg.style_mix)
    style_name = s_keys[rng.choice(len(s_keys), p=s_p)]
    style = presets[style_name]

    pen = pool.sample_pen(split, rng) if cfg.pen_k_frac < 1.0 else None
    words = mixer.sample_line(rng, n_words)
    line = compose_line(words, pool, split, style.layout, rng, pen=pen,
                        pen_k_frac=cfg.pen_k_frac)
    page = apply_degradation(line.ink, style.degrade, rng)

    img = Image.fromarray(page, mode="L")
    composed_w, composed_h = img.width, img.height
    resize_scale = 1.0
    if cfg.output_height and img.height != cfg.output_height:
        resize_scale = cfg.output_height / img.height
        new_w = max(1, int(round(img.width * resize_scale)))
        img = img.resize((new_w, cfg.output_height), Image.LANCZOS)

    # Boxes must be expressed in the *saved* image's pixels.  compose_line
    # works in composed coordinates, and the line is then rescaled to
    # output_height, so the two differ whenever output_height is set.  Derive
    # the factors from the actual sizes rather than from resize_scale, since
    # the width was rounded to an integer.
    sx = img.width / composed_w
    sy = img.height / composed_h
    boxes = [_scale_box(g, sx, sy, img.width, img.height) for g in line.placed]

    tokens = line_tokens(words)
    lookup = factor_lookup(pool)
    factors = pool_factors(pool, tokens)
    record = {
        "script": pool.script,
        "split": split,
        "index": index,
        "text": line_text(words),
        "words": [list(w) for w in words],
        "tokens": tokens,
        "onsets": factors["onsets"],
        "vowels": factors["vowels"],
        "n_syllables": sum(len(w) for w in words),
        "style": style_name,
        "pen": round(pen, 3) if pen is not None else None,
        "width": img.width,
        "height": img.height,
        "resize_scale": round(resize_scale, 5),
        # bbox is [x, y, w, h] in this image's pixels, tight to the glyph's
        # ink before degradation (dilation and blur can spill a pixel or two).
        "bbox_format": "xywh",
        "glyphs": [{
            "syllable": g.syllable,
            "onset": lookup.get(g.syllable, ("", ""))[0],
            "vowel": lookup.get(g.syllable, ("", ""))[1],
            "word": wi, "pos": pi,
            "bbox": box,
            "pool_idx": g.pool_idx, "source": pool.entries[g.pool_idx].path,
            "angle": round(g.angle, 3), "scale": round(g.scale, 4),
        } for (wi, pi, g), box in zip(_index_glyphs(words, line.placed), boxes)],
        "word_boxes": _word_boxes(words, boxes),
        "line_bbox": _union_box(boxes) if boxes else None,
    }
    return img, record


def _index_glyphs(words, placed):
    """Pair each placed glyph with its (word index, position in word)."""
    out, k = [], 0
    for wi, word in enumerate(words):
        for pi in range(len(word)):
            out.append((wi, pi, placed[k]))
            k += 1
    return out


def _word_boxes(words, boxes) -> list:
    out, k = [], 0
    for wi, word in enumerate(words):
        span = boxes[k: k + len(word)]
        k += len(word)
        if span:
            out.append({"word": wi, "text": "-".join(word),
                        "bbox": _union_box(span)})
    return out


def factor_lookup(pool: GlyphPool) -> dict:
    """``syllable -> (onset, vowel)``, built once and cached on the pool.

    Rebuilding this per sample would walk all ~28k Sunda entries for every one
    of the 25k lines.
    """
    cached = getattr(pool, "_factor_lookup", None)
    if cached is None:
        cached = {}
        for e in pool.entries:
            cached.setdefault(e.syllable, (e.onset, e.vowel))
        pool._factor_lookup = cached
    return cached


def pool_factors(pool: GlyphPool, tokens) -> dict:
    lookup = factor_lookup(pool)
    onsets, vowels = [], []
    for t in tokens:
        if t == WORD_SEP:
            onsets.append(None)
            vowels.append(None)
        else:
            o, v = lookup.get(t, ("", ""))
            onsets.append(o)
            vowels.append(v)
    return {"onsets": onsets, "vowels": vowels}


def generate(cfg: CorpusConfig, progress=None) -> dict:
    presets = make_presets(cfg.core_height)
    for name in cfg.style_mix:
        if name not in presets:
            raise KeyError(f"unknown style preset {name!r}; have {sorted(presets)}")

    out_root = Path(cfg.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    for script in cfg.scripts:
        pool = GlyphPool(cfg.glyph_cache, script)
        sdir = out_root / script
        sdir.mkdir(parents=True, exist_ok=True)
        cs = charset(pool, cfg)
        (sdir / "charset.json").write_text(
            json.dumps(cs, indent=2, ensure_ascii=False), encoding="utf-8")

        script_summary = {"charset": {k: cs[k] for k in
                                      ("n_syllables", "n_onsets", "n_vowels")}}
        for split in SPLITS:
            n = int(cfg.samples.get(split, 0))
            if n <= 0:
                continue
            mixer, rejected = build_word_source(pool, cfg, script, split)
            if rejected and split == "train":
                script_summary["lexicon_rejected"] = [
                    {"word": w, "unrenderable": bad} for w, bad in rejected]

            split_dir = sdir / split
            img_dir = split_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            label_path = split_dir / "labels.jsonl"

            style_counts, syl_counts, widths = {}, {}, []
            with open(label_path, "w", encoding="utf-8") as fh:
                for i in range(n):
                    img, rec = render_sample(pool, mixer, cfg, presets, split, i)
                    fname = f"{i:06d}.{cfg.image_format}"
                    img.save(img_dir / fname)
                    rec["id"] = f"{script}/{split}/{i:06d}"
                    rec["file"] = f"images/{fname}"
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    style_counts[rec["style"]] = style_counts.get(rec["style"], 0) + 1
                    widths.append(rec["width"])
                    for t in rec["tokens"]:
                        if t != WORD_SEP:
                            syl_counts[t] = syl_counts.get(t, 0) + 1
                    if progress:
                        progress(script, split, i + 1, n)

            script_summary[split] = {
                "n_samples": n,
                "styles": style_counts,
                "syllable_types_seen": len(syl_counts),
                "syllable_tokens": int(sum(syl_counts.values())),
                "min_syllable_count": min(syl_counts.values()) if syl_counts else 0,
                "width_mean": float(np.mean(widths)) if widths else 0.0,
                "width_max": int(max(widths)) if widths else 0,
            }
        summary[script] = script_summary

    meta = {"config": cfg.as_dict(), "scripts": summary,
            "notes": {
                "renderable_syllables": "open CV only -- the source corpus "
                                        "contributes no coda marks (pasangan, "
                                        "gantungan, virama, final consonants)",
                "pool_disjointness": "glyph bitmaps are partitioned across "
                                     "train/val/test before rendering; no line "
                                     "shares ink with a line in another split",
                "writer_ids": "absent from the source corpus, so splits are "
                              "instance-disjoint, not writer-disjoint",
                "within_line_hand": "each line draws a pen width and prefers "
                                    "glyph instances nearest it, so a line "
                                    "reads as one pen. Stroke width is only "
                                    "one dimension of a hand; slant and "
                                    "proportion still vary within a line",
                "layout": "composed, not written: no real handwritten word or "
                          "line appears in this corpus, and there is no "
                          "transcribed real text to validate against",
            }}
    (out_root / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta

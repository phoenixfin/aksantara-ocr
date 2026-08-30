"""Instance-disjoint glyph pools and the normalized-bitmap cache.

The source corpus carries no writer ids, so a writer-disjoint split is not
available.  The weaker guarantee we *can* enforce absolutely is that no glyph
bitmap is ever reused across splits: the pools are disjoint at the level of
individual source files, decided once, before a single line is rendered.  A
rendered line therefore shares no ink with any line in another split.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .normalize import (
    DEFAULT_CORE_HEIGHT, GlyphLoadError, prepare_glyph, stroke_width,
)

SPLITS = ("train", "val", "test")

INDEX_FIELDS = [
    "idx", "split", "script", "syllable", "onset", "vowel", "kind", "variant",
    "raw_class", "group", "renamed_from", "height", "width", "baseline",
    "core_top", "scale", "clamped", "stroke_width", "content_hash", "path",
]


@dataclass
class PoolConfig:
    target_core_h: int = DEFAULT_CORE_HEIGHT
    ratios: tuple = (0.70, 0.15, 0.15)
    seed: int = 20260830
    # Jawa's "variations" group holds murda (honorific) letterforms, not free
    # allographs.  Dropping them keeps rendered words orthographically
    # ordinary; enable only for an explicit allograph experiment.
    include_variants: bool = False

    def as_dict(self) -> dict:
        return {
            "target_core_h": self.target_core_h,
            "ratios": list(self.ratios),
            "seed": self.seed,
            "include_variants": self.include_variants,
        }


def assign_splits(records, cfg: PoolConfig) -> dict:
    """Map source path -> split, disjointly, per (script, syllable) class.

    Every class with at least three instances contributes to all three splits,
    so no syllable is unevaluable; smaller classes fill train first.
    """
    rng = np.random.default_rng(cfg.seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[(r.script, r.syllable)].append(r)

    out = {}
    tr, va, _ = cfg.ratios
    for key in sorted(by_class):
        items = sorted(by_class[key], key=lambda r: r.path)
        n = len(items)
        order = rng.permutation(n)
        if n == 1:
            counts = (1, 0, 0)
        elif n == 2:
            counts = (1, 0, 1)
        else:
            n_val = max(1, int(round(n * va)))
            n_test = max(1, int(round(n * (1.0 - tr - va))))
            n_train = n - n_val - n_test
            if n_train < 1:                      # tiny class: protect train
                n_train, n_val, n_test = n - 2, 1, 1
            counts = (n_train, n_val, n_test)
        bounds = np.cumsum(counts)
        for rank, pos in enumerate(order):
            split = SPLITS[int(np.searchsorted(bounds, rank, side="right"))]
            out[items[pos].path] = split
    return out


def _prepare_one(job):
    """Worker entry point: must stay module-level so it can be pickled."""
    path, target_core_h = job
    try:
        g = prepare_glyph(path, target_core_h)
    except (GlyphLoadError, OSError, ValueError) as exc:
        return path, None, 0, 0, 0.0, False, 0.0, repr(exc)
    return (path, g.bitmap, g.baseline, g.core_top, g.scale, g.clamped,
            stroke_width(g.bitmap), None)


def build_cache(records, out_dir: Path, cfg: PoolConfig, progress=None,
                workers: int = 1) -> dict:
    """Normalize every glyph and write one memory-mappable cache per script."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    usable = [r for r in records if cfg.include_variants or not r.variant]
    splits = assign_splits(usable, cfg)

    summary = {}
    failures = []

    for script in sorted({r.script for r in usable}):
        recs = [r for r in usable if r.script == script]
        recs.sort(key=lambda r: (r.syllable, r.path))
        sdir = out_dir / script
        sdir.mkdir(parents=True, exist_ok=True)

        buffers, rows, offsets = [], [], []
        cursor = 0

        jobs = [(r.path, cfg.target_core_h) for r in recs]
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_prepare_one, jobs, chunksize=64))
        else:
            results = [_prepare_one(j) for j in jobs]

        for r, (_, bitmap, baseline, core_top, scale, clamped, stroke,
                err) in zip(recs, results):
            if err is not None:
                failures.append({"path": r.path, "script": script, "error": err})
                continue
            h, w = bitmap.shape
            buffers.append(bitmap.ravel())
            offsets.append(cursor)
            cursor += h * w
            rows.append({
                "idx": len(rows), "split": splits[r.path], "script": script,
                "syllable": r.syllable, "onset": r.onset, "vowel": r.vowel,
                "kind": r.kind, "variant": int(r.variant),
                "raw_class": r.raw_class, "group": r.group,
                "renamed_from": r.renamed_from,
                "height": h, "width": w, "baseline": baseline,
                "core_top": core_top, "scale": round(scale, 4),
                "clamped": int(clamped), "stroke_width": round(stroke, 3),
                "content_hash": r.content_hash,
                "path": r.path,
            })
            if progress:
                progress(script)

        flat = np.concatenate(buffers) if buffers else np.zeros(0, dtype=np.uint8)
        np.save(sdir / "bitmaps.npy", flat)
        np.savez(
            sdir / "geometry.npz",
            offset=np.asarray(offsets, dtype=np.int64),
            height=np.asarray([r["height"] for r in rows], dtype=np.int32),
            width=np.asarray([r["width"] for r in rows], dtype=np.int32),
            baseline=np.asarray([r["baseline"] for r in rows], dtype=np.int32),
            core_top=np.asarray([r["core_top"] for r in rows], dtype=np.int32),
            stroke_width=np.asarray([r["stroke_width"] for r in rows],
                                    dtype=np.float32),
        )
        with open(sdir / "index.csv", "w", newline="", encoding="utf-8") as fh:
            wtr = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
            wtr.writeheader()
            wtr.writerows(rows)

        summary[script] = {
            "n_glyphs": len(rows),
            "n_syllables": len({r["syllable"] for r in rows}),
            "per_split": {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS},
            "bytes": int(flat.nbytes),
            "clamped": sum(r["clamped"] for r in rows),
        }
        buffers.clear()

    meta = {"config": cfg.as_dict(), "scripts": summary, "n_failures": len(failures)}
    (out_dir / "cache_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        with open(out_dir / "failures.csv", "w", newline="", encoding="utf-8") as fh:
            wtr = csv.DictWriter(fh, fieldnames=["path", "script", "error"])
            wtr.writeheader()
            wtr.writerows(failures)
    return meta


@dataclass
class GlyphEntry:
    idx: int
    syllable: str
    onset: str
    vowel: str
    split: str
    height: int
    width: int
    baseline: int
    path: str


class GlyphPool:
    """Read-only, memory-mapped access to one script's normalized glyphs."""

    def __init__(self, cache_dir: Path, script: str):
        self.script = script
        self.dir = Path(cache_dir) / script
        self.bitmaps = np.load(self.dir / "bitmaps.npy", mmap_mode="r")
        geo = np.load(self.dir / "geometry.npz")
        self.offset, self.height = geo["offset"], geo["height"]
        self.width, self.baseline_arr = geo["width"], geo["baseline"]
        self.stroke = (geo["stroke_width"] if "stroke_width" in geo
                       else np.zeros(len(self.width), dtype=np.float32))

        self.entries = []
        with open(self.dir / "index.csv", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.entries.append(GlyphEntry(
                    idx=int(row["idx"]), syllable=row["syllable"],
                    onset=row["onset"], vowel=row["vowel"], split=row["split"],
                    height=int(row["height"]), width=int(row["width"]),
                    baseline=int(row["baseline"]), path=row["path"],
                ))

        self.by_split = {s: defaultdict(list) for s in SPLITS}
        for e in self.entries:
            self.by_split[e.split][e.syllable].append(e.idx)

        # Per (split, syllable), instances sorted by stroke width, so a line
        # can draw everything from one pen weight.
        self.by_stroke = {sp: {} for sp in SPLITS}
        for sp in SPLITS:
            for syl, idxs in self.by_split[sp].items():
                arr = np.asarray(idxs, dtype=np.int64)
                widths = np.asarray(self.stroke, dtype=np.float64)[arr]
                order = np.argsort(widths, kind="stable")
                self.by_stroke[sp][syl] = (arr[order], widths[order])

    def bitmap(self, idx: int) -> np.ndarray:
        off = int(self.offset[idx])
        h, w = int(self.height[idx]), int(self.width[idx])
        return np.asarray(self.bitmaps[off: off + h * w]).reshape(h, w)

    def baseline(self, idx: int) -> int:
        return int(self.baseline_arr[idx])

    def syllables(self, split: str = "train") -> list:
        return sorted(self.by_split[split])

    def has(self, syllable: str, split: str) -> bool:
        return bool(self.by_split[split].get(syllable))

    def sample_pen(self, split: str, rng) -> float:
        """A stroke width to write a whole line in, drawn from this split."""
        widths = [w for _, ws in self.by_stroke[split].values() for w in ws]
        if not widths:
            return 0.0
        return float(widths[rng.integers(len(widths))])

    def sample_idx(self, syllable: str, split: str, rng,
                   pen: float = None, k_frac: float = 0.30,
                   k_min: int = 2) -> int:
        """Draw a glyph, preferring instances written with a similar pen.

        Sampling independently puts fine pencil next to marker inside a single
        line, which no real hand does.  Discrete stroke-weight bands do not
        work here: Jawa carries only ~22 instances per non-*a* syllable, so a
        third of draws would find the band empty and fall back to the whole
        pool, undoing the effect.  Instead take the ``k`` instances nearest the
        line's pen width, which always returns something and degrades smoothly
        as a syllable gets rarer.
        """
        entry = self.by_stroke[split].get(syllable)
        if not entry or entry[0].size == 0:
            raise KeyError(f"{self.script}: no '{syllable}' glyph in split '{split}'")
        idxs, widths = entry
        n = idxs.size
        if pen is None or n <= k_min:
            return int(idxs[rng.integers(n)])

        k = max(k_min, int(round(k_frac * n)))
        if k >= n:
            return int(idxs[rng.integers(n)])
        lo = int(np.clip(np.searchsorted(widths, pen) - k // 2, 0, n - k))
        return int(idxs[lo + rng.integers(k)])

    def coverage(self, split: str) -> dict:
        return {syl: len(idxs) for syl, idxs in sorted(self.by_split[split].items())}

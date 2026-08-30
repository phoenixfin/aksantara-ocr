"""Walk the cleaned character corpus and build a parsed glyph index."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from ..scripts_def import SCRIPTS, GlyphClass, parse_class_path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class GlyphRecord:
    path: str
    script: str
    raw_class: str
    group: str
    kind: str
    onset: str
    vowel: str
    syllable: str
    variant: bool
    renamed_from: str = ""
    content_hash: str = ""

    @classmethod
    def from_class(cls, path: Path, gc: GlyphClass) -> "GlyphRecord":
        return cls(
            path=str(path), script=gc.script, raw_class=gc.raw_class,
            group=gc.group, kind=gc.kind, onset=gc.onset, vowel=gc.vowel,
            syllable=gc.syllable, variant=gc.variant,
            renamed_from=gc.renamed_from,
        )

    def as_row(self) -> dict:
        return asdict(self)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def scan_script(data_root: Path, script: str, hash_files: bool = True) -> list[GlyphRecord]:
    """Index every image under ``data_root/<script>``, parsing its class path."""
    sdef = SCRIPTS[script]
    base = Path(data_root) / sdef.dirname
    if not base.is_dir():
        raise FileNotFoundError(f"no such script directory: {base}")

    records: list[GlyphRecord] = []
    by_dir: dict[Path, GlyphClass] = {}

    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parent = path.parent
        gc = by_dir.get(parent)
        if gc is None:
            parts = parent.relative_to(base).parts
            gc = parse_class_path(script, parts)
            by_dir[parent] = gc
        rec = GlyphRecord.from_class(path, gc)
        if hash_files:
            rec.content_hash = _md5(path)
        records.append(rec)

    return records


def scan_all(data_root: Path, scripts, hash_files: bool = True) -> list[GlyphRecord]:
    out: list[GlyphRecord] = []
    for s in scripts:
        out.extend(scan_script(data_root, s, hash_files=hash_files))
    return out


# --- verification ----------------------------------------------------------

def grid_report(records: list[GlyphRecord], script: str) -> dict:
    """Onsets x vowels occupancy for one script, plus the holes."""
    syl = [r for r in records if r.script == script and r.kind == "syllable"]
    onsets = sorted({r.onset for r in syl}, key=lambda o: (o == "", o))
    vowels = sorted({r.vowel for r in syl})
    counts: dict[tuple[str, str], int] = Counter((r.onset, r.vowel) for r in syl)
    missing = [(o, v) for o in onsets for v in vowels if (o, v) not in counts]
    classes = {(r.onset, r.vowel, r.variant) for r in syl}
    digits = {r.syllable for r in records if r.script == script and r.kind == "digit"}
    return {
        "script": script,
        "onsets": onsets,
        "vowels": vowels,
        "cells_filled": len(counts),
        "cells_total": len(onsets) * len(vowels),
        "missing_cells": missing,
        "n_classes": len(classes) + len(digits),
        "n_images": len(syl) + sum(1 for r in records
                                   if r.script == script and r.kind == "digit"),
        "n_digit_classes": len(digits),
        "per_cell_min": min(counts.values()) if counts else 0,
        "per_cell_median": sorted(counts.values())[len(counts) // 2] if counts else 0,
        "per_cell_max": max(counts.values()) if counts else 0,
    }


def duplicate_hashes(records: list[GlyphRecord]) -> dict[str, list[str]]:
    """Byte-identical images -- they would break instance-disjoint pooling."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r.content_hash:
            by_hash[r.content_hash].append(r.path)
    return {h: ps for h, ps in by_hash.items() if len(ps) > 1}

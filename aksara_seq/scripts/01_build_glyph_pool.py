"""Step 1 -- scan the character corpus, verify the grids, build the glyph cache.

    python aksara_seq/scripts/01_build_glyph_pool.py \
        --data-root data/clean --out aksara_seq/build/glyphs --workers 8

Writes, per script: ``bitmaps.npy`` (normalized ink, memory-mappable),
``geometry.npz`` (offsets, sizes, baselines) and ``index.csv`` (one row per
glyph, carrying its split assignment and full provenance).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksaraseq.glyphs.pool import PoolConfig, build_cache  # noqa: E402
from aksaraseq.glyphs.scan import (  # noqa: E402
    duplicate_hashes, grid_report, scan_all,
)
from aksaraseq.scripts_def import DEFAULT_SCRIPTS, SCRIPTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/clean"))
    ap.add_argument("--out", type=Path, default=Path("aksara_seq/build/glyphs"))
    ap.add_argument("--scripts", nargs="+", default=list(DEFAULT_SCRIPTS),
                    choices=sorted(SCRIPTS))
    ap.add_argument("--core-height", type=int, default=48,
                    help="pixel height every glyph's core band is scaled to")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--include-variants", action="store_true",
                    help="include Jawa murda letterforms (off by default)")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip md5 of every source file (faster, less provenance)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify-only", action="store_true",
                    help="report the grids and exit without writing a cache")
    args = ap.parse_args()

    t0 = time.time()
    print(f"scanning {args.data_root} ...", flush=True)
    records = scan_all(args.data_root, args.scripts, hash_files=not args.no_hash)
    print(f"  {len(records)} images in {time.time() - t0:.1f}s\n")

    ok = True
    for script in args.scripts:
        rep = grid_report(records, script)
        expected = SCRIPTS[script].expected_classes
        status = "OK" if rep["n_classes"] == expected else f"MISMATCH (expected {expected})"
        if rep["n_classes"] != expected:
            ok = False
        print(f"{script:9s} {rep['n_images']:6d} images  {rep['n_classes']:4d} classes  {status}")
        print(f"          grid {len(rep['onsets'])} onsets x {len(rep['vowels'])} vowels "
              f"= {rep['cells_total']} cells, {rep['cells_filled']} filled")
        print(f"          per-cell images  min {rep['per_cell_min']} / "
              f"median {rep['per_cell_median']} / max {rep['per_cell_max']}")
        if rep["missing_cells"]:
            holes = ", ".join(f"{o or '<null>'}+{v}" for o, v in rep["missing_cells"])
            print(f"          holes: {holes}")
        print()

    if not args.no_hash:
        dupes = duplicate_hashes(records)
        if dupes:
            print(f"WARNING: {len(dupes)} byte-identical image groups -- these would "
                  f"break instance-disjoint pooling\n")
        else:
            print("no byte-identical duplicates\n")

    renamed = [r for r in records if r.renamed_from]
    if renamed:
        pairs = sorted({(r.script, r.renamed_from, r.raw_class) for r in renamed})
        for script, was, now in pairs:
            n = sum(1 for r in renamed if r.renamed_from == was and r.script == script)
            print(f"applied class-name fix: {script} {was!r} -> {now!r} ({n} images)")
        print()

    if args.verify_only:
        return 0 if ok else 1

    cfg = PoolConfig(target_core_h=args.core_height, ratios=tuple(args.ratios),
                     seed=args.seed, include_variants=args.include_variants)

    seen = {"n": 0, "t": time.time()}

    def progress(script: str) -> None:
        seen["n"] += 1
        if seen["n"] % 2000 == 0:
            rate = seen["n"] / max(time.time() - seen["t"], 1e-9)
            print(f"  normalized {seen['n']} glyphs ({rate:.0f}/s)", flush=True)

    print(f"normalizing to core height {args.core_height}px "
          f"with {args.workers} worker(s) ...", flush=True)
    t1 = time.time()
    meta = build_cache(records, args.out, cfg,
                       progress=None if args.workers > 1 else progress,
                       workers=args.workers)
    print(f"  done in {time.time() - t1:.1f}s\n")

    total_bytes = 0
    for script, s in meta["scripts"].items():
        ps = s["per_split"]
        total_bytes += s["bytes"]
        print(f"{script:9s} {s['n_glyphs']:6d} glyphs  {s['n_syllables']:4d} syllables  "
              f"train/val/test = {ps['train']}/{ps['val']}/{ps['test']}  "
              f"({s['bytes'] / 1e6:.0f} MB, {s['clamped']} upscale-clamped)")
    print(f"\ncache {total_bytes / 1e6:.0f} MB -> {args.out}")
    if meta["n_failures"]:
        print(f"{meta['n_failures']} glyphs failed -- see {args.out / 'failures.csv'}")
    print(json.dumps(meta["config"], indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

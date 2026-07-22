"""Pre-resize the dataset once into a decode-cheap cache.

Why this exists: the source images are 512x512 PNGs. Decoding and resizing one
costs ~6-9 ms per core, so two Colab worker processes sustain roughly 300
images/s. Every epoch of every run pays that cost again, and the GPU spends most
of its time waiting on the data loader — a faster GPU buys nothing.

Resizing once to the largest resolution any experiment needs drops per-image
decode to ~0.6 ms (measured: 11x faster), which moves the bottleneck back onto
the GPU where it belongs.

    python scripts/00b_build_cache.py --data-root data/raw --out data/cache --size 224

**Methodological note for the paper.** With a 224px cache, the image-size
ablation downsamples 512 -> 224 -> N rather than 512 -> N directly. Double
resampling is standard practice and the difference is small, but it is a real
processing choice: either state it, or pass --size 512 to disable the resize and
keep only the format normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _cache_name(relative: Path) -> Path:
    """Destination path for a cached image.

    Cached files are always PNG, but the *name* must stay unique. Simply
    replacing the extension merges "1.jpg" and "1.png" into one file; this
    corpus contains 83 such pairs (Batak/na, Kawi/cha, Jawi/...), so that
    silently loses 83 images. Appending ".png" instead of replacing it keeps
    every name distinct while leaving already-PNG files untouched.
    """
    if relative.suffix.lower() == ".png":
        return relative
    return relative.with_name(relative.name + ".png")


def _resize_one(job: tuple[str, str, int, bool]) -> str | None:
    src, dst, size, grayscale = job
    destination = Path(dst)
    if destination.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as img:
            # Grayscale first, then resize: converting after would resample three
            # identical channels and cost 3x the work for the same result.
            out = img.convert("L" if grayscale else "RGB")
            if out.size != (size, size):
                out = out.resize((size, size), Image.LANCZOS)
            # optimize=False keeps this step fast; these are small files and the
            # cache is disposable, so compression ratio does not matter.
            out.save(destination, format="PNG", optimize=False)
    except Exception as exc:  # noqa: BLE001 — report and continue
        return f"{src}: {exc}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Cache resolution. Use the largest image_size any config needs "
             "(224 covers the transformers).",
    )
    parser.add_argument("--rgb", action="store_true", help="Store RGB instead of grayscale.")
    parser.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise SystemExit(f"No such directory: {args.data_root}")

    print(f"Scanning {args.data_root} ...")
    sources = [
        p for p in args.data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not sources:
        raise SystemExit(f"No images found under {args.data_root}")

    jobs = [
        (str(p), str(args.out / _cache_name(p.relative_to(args.data_root))),
         args.size, not args.rgb)
        for p in sources
    ]

    # Two source files must never claim one destination. Replacing the suffix
    # with ".png" collapses "1.jpg" and "1.png" onto the same name, and the
    # skip-if-exists check below then drops one of them silently — the cache
    # ends up smaller than the dataset with nothing to indicate why.
    claims: dict[str, list[str]] = {}
    for source_path, destination, _, _ in jobs:
        claims.setdefault(destination, []).append(source_path)
    collisions = {d: s for d, s in claims.items() if len(s) > 1}
    if collisions:
        print(f"\nERROR: {len(collisions)} destination(s) claimed by multiple sources:")
        for destination, source_paths in list(collisions.items())[:5]:
            print(f"  {destination}")
            for source_path in source_paths:
                print(f"      <- {source_path}")
        raise SystemExit(
            f"{sum(len(s) - 1 for s in collisions.values())} image(s) would be lost. "
            "This is a bug in the cache naming scheme — please report it."
        )

    todo = [j for j in jobs if not Path(j[1]).exists()]
    print(f"{len(sources)} image(s); {len(jobs) - len(todo)} already cached, {len(todo)} to do.")

    if not todo:
        print(f"Cache already complete -> {args.out}")
        return 0

    workers = args.workers or None
    errors: list[str] = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, error in enumerate(pool.map(_resize_one, todo, chunksize=64), start=1):
            if error:
                errors.append(error)
            if i % 2000 == 0 or i == len(todo):
                rate = i / (time.time() - start)
                eta = (len(todo) - i) / rate / 60
                print(f"\r  {i}/{len(todo)}  ({rate:.0f} img/s, ETA {eta:.1f} min)", end="")
    print()

    elapsed = time.time() - start
    print(f"\nCached {len(todo) - len(errors)} image(s) in {elapsed / 60:.1f} min -> {args.out}")

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for e in errors[:10]:
            print(f"  {e}")

    source_mb = sum(p.stat().st_size for p in sources) / 1e6
    # Count only this run's own destinations. Globbing the output directory
    # counts anything else already there — point two runs at one cache and the
    # arithmetic goes nonsensical rather than reporting what this run produced.
    expected_paths = [Path(destination) for _, destination, _, _ in jobs]
    present = [p for p in expected_paths if p.exists()]
    cache_mb = sum(p.stat().st_size for p in present) / 1e6
    print(f"\nimages: {len(sources)} source -> {len(present)} cached")
    print(f"size  : {source_mb:.0f} MB -> {cache_mb:.0f} MB")

    if len(present) != len(sources):
        missing = [p for p in expected_paths if not p.exists()]
        print(f"\nWARNING: {len(missing)} image(s) missing from the cache, e.g.:")
        for path in missing[:5]:
            print(f"  {path}")

    # Downscaling can make distinct originals byte-identical. Those pairs are a
    # leakage path if they straddle the train/test boundary, and the count
    # depends on the cache resolution — so it is measured here rather than
    # assumed, and 01_prepare_data.py --drop-duplicates removes them.
    digests: dict[str, int] = {}
    for path in present:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        digests[digest] = digests.get(digest, 0) + 1
    merged = sum(count - 1 for count in digests.values() if count > 1)
    if merged:
        print(
            f"\nNOTE: {merged} image(s) became byte-identical to another at "
            f"{args.size}px (they differ in the source).\n"
            "      Downscaling merged them. They are a leakage path if split "
            "across train/test —\n"
            "      pass --drop-duplicates to 01_prepare_data.py to remove them."
        )

    print(f"\nNext:\n  python scripts/01_prepare_data.py --data-root {args.out} --drop-duplicates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

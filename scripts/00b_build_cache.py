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
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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
        (str(p), str(args.out / p.relative_to(args.data_root).with_suffix(".png")),
         args.size, not args.rgb)
        for p in sources
    ]
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
    cached = list(args.out.rglob("*.png"))
    cache_mb = sum(p.stat().st_size for p in cached) / 1e6
    print(f"\nsize: {source_mb:.0f} MB -> {cache_mb:.0f} MB")
    print(f"\nNext:\n  python scripts/01_prepare_data.py --data-root {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

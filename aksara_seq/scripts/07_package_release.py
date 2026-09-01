"""Package the rendered corpus for publication (Mendeley Data).

    python aksara_seq/scripts/07_package_release.py --version v1

Produces one archive per script plus a dataset card and checksums:

    <out>/README.md
    <out>/CHECKSUMS.sha256
    <out>/dataset_meta.json
    <out>/AksaraLine_<Script>.zip

Two things change on the way out, because the corpus as generated is written
for local use rather than for publication:

* the per-glyph ``source`` path is dropped. It carries the upstream corpus's
  internal directory naming, which the published artifact has no reason to
  expose. ``pool_idx`` is kept, so lines can still be grouped by the source
  scan they drew from.
* every archive is checksummed, so a truncated download is detectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SCRIPTS = ("Sunda", "Jawa", "Bali", "Lontara")
SPLITS = ("train", "val", "test")


def rewrite_labels(text: str) -> tuple:
    """Strip per-glyph source paths; returns (text, n_records, n_stripped).

    ``pool_idx`` is kept: it still groups lines by the exact source scan they
    drew from, so split disjointness stays checkable from the labels alone,
    without publishing the upstream corpus's internal directory naming.
    """
    out, n, changed = [], 0, 0
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for g in rec.get("glyphs", []):
            if g.pop("source", None) is not None:
                changed += 1
        out.append(json.dumps(rec, ensure_ascii=False))
        n += 1
    return "\n".join(out) + "\n", n, changed


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def package_script(corpus: Path, script: str, out: Path) -> dict:
    archive = out / f"AksaraLine_{script}.zip"
    src = corpus / script
    n_images = n_records = n_rewritten = 0
    t0 = time.time()

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{script}/charset.json",
                   (src / "charset.json").read_text(encoding="utf-8"))
        for split in SPLITS:
            labels = src / split / "labels.jsonl"
            if not labels.exists():
                continue
            text, n, changed = rewrite_labels(labels.read_text(encoding="utf-8"))
            n_records += n
            n_rewritten += changed
            z.writestr(f"{script}/{split}/labels.jsonl", text)
            for img in sorted((src / split / "images").glob("*.png")):
                z.write(img, arcname=f"{script}/{split}/images/{img.name}")
                n_images += 1

    return {"script": script, "archive": archive.name,
            "bytes": archive.stat().st_size, "images": n_images,
            "records": n_records, "sources_stripped": n_rewritten,
            "sha256": sha256(archive), "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("aksara_seq/build/corpus/v1"))
    ap.add_argument("--out", type=Path, default=Path("aksara_seq/build/release"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--scripts", nargs="+", default=list(SCRIPTS))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta_src = args.corpus / "dataset_meta.json"
    meta = json.loads(meta_src.read_text(encoding="utf-8")) if meta_src.exists() else {}

    parts = []
    for script in args.scripts:
        print(f"packaging {script} ...", flush=True)
        info = package_script(args.corpus, script, args.out)
        parts.append(info)
        print(f"  {info['images']} images, {info['records']} records, "
              f"{info['bytes'] / 1e6:.0f} MB, {info['seconds']}s "
              f"({info['sources_stripped']} source paths stripped)")

    (args.out / "dataset_meta.json").write_text(
        json.dumps({"version": args.version, "generator": meta, "archives": parts},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"{p['sha256']}  {p['archive']}" for p in parts]
    (args.out / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")

    total = sum(p["bytes"] for p in parts)
    print(f"\n{len(parts)} archives, {total / 1e9:.2f} GB -> {args.out}")
    print("checksums written; write README.md before uploading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

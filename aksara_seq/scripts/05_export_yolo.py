"""Export a rendered corpus to YOLO detection format.

    python aksara_seq/scripts/05_export_yolo.py --script Bali --classes syllable

Produces the layout ultralytics expects::

    <out>/data.yaml
    <out>/images/{train,val,test}/<id>.png
    <out>/labels/{train,val,test}/<id>.txt      # class cx cy w h, normalized

``--classes`` chooses what a box is labelled with:

* ``syllable`` -- one class per syllable (120-170).  The detection analogue of
  the monolithic label space.
* ``onset`` / ``vowel`` -- the factored alternatives, 17-26 and 6-7 classes.
  Running both and combining their predictions is the detection counterpart of
  the factored head, and it needs far fewer examples per class.

A caveat worth carrying into any comparison: this corpus renders glyphs with a
*positive* minimum ink clearance, so syllables never overlap.  Detection is
therefore much easier here than on real manuscript hands, where aksara touch
and connect.  A strong YOLO score is partly a property of the renderer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPLITS = ("train", "val", "test")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # NTFS hard link: no second copy on disk
    except OSError:
        shutil.copy2(src, dst)


def class_names(charset: dict, mode: str) -> list:
    if mode == "syllable":
        return list(charset["syllables"])
    if mode == "onset":
        return list(charset["onsets"])
    if mode == "vowel":
        return list(charset["vowels"])
    raise ValueError(mode)


def class_of(glyph: dict, mode: str) -> str:
    return {"syllable": glyph["syllable"], "onset": glyph["onset"],
            "vowel": glyph["vowel"]}[mode]


def export(corpus: Path, script: str, out: Path, mode: str,
           copy_images: bool = True) -> dict:
    charset = json.loads((corpus / script / "charset.json").read_text(encoding="utf-8"))
    names = class_names(charset, mode)
    index = {n: i for i, n in enumerate(names)}

    out.mkdir(parents=True, exist_ok=True)
    counts, boxes_total, skipped = {}, 0, 0

    for split in SPLITS:
        labels_path = corpus / script / split / "labels.jsonl"
        if not labels_path.exists():
            continue
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        with open(labels_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                stem = rec["id"].replace("/", "_")
                w, h = rec["width"], rec["height"]

                rows = []
                for g in rec["glyphs"]:
                    name = class_of(g, mode)
                    if name not in index:
                        skipped += 1
                        continue
                    bx, by, bw, bh = g["bbox"]
                    cx = (bx + bw / 2.0) / w
                    cy = (by + bh / 2.0) / h
                    rows.append(f"{index[name]} {cx:.6f} {cy:.6f} "
                                f"{bw / w:.6f} {bh / h:.6f}")
                (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n",
                                                     encoding="utf-8")
                boxes_total += len(rows)
                if copy_images:
                    link_or_copy(corpus / script / split / rec["file"],
                                 img_dir / f"{stem}.png")
                n += 1
        counts[split] = n

    yaml = [
        f"# {script}: YOLO export of the AksaraLine corpus ({mode} classes)",
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    return {"script": script, "mode": mode, "n_classes": len(names),
            "images": counts, "boxes": boxes_total, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("aksara_seq/build/corpus/v1"))
    ap.add_argument("--script", default="Bali")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--classes", default="syllable",
                    choices=["syllable", "onset", "vowel"])
    ap.add_argument("--no-images", action="store_true",
                    help="write labels only (images already linked elsewhere)")
    args = ap.parse_args()

    out = args.out or Path("aksara_seq/build/yolo") / f"{args.script}_{args.classes}"
    info = export(args.corpus, args.script, out, args.classes,
                  copy_images=not args.no_images)

    print(f"{info['script']}  {info['mode']} classes: {info['n_classes']}")
    for split, n in info["images"].items():
        print(f"  {split:5s} {n} images")
    print(f"  {info['boxes']} boxes"
          + (f", {info['skipped']} skipped (class not in charset)"
             if info["skipped"] else ""))
    print(f"\n-> {out / 'data.yaml'}")
    print("\ntrain with:\n"
          f"  yolo detect train data={(out / 'data.yaml').as_posix()} "
          f"model=yolo11s.pt imgsz=960 epochs=50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

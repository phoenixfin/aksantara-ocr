"""Integrity check for a rendered corpus.

The project's central claim is that no rendered line shares ink with a line in
another split.  That is a property worth testing rather than asserting, so this
re-derives it from the written labels instead of trusting the generator.

    python aksara_seq/scripts/03_verify_corpus.py --corpus aksara_seq/build/corpus/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksaraseq.render.lexicon import WORD_SEP  # noqa: E402

SPLITS = ("train", "val", "test")


def load_labels(split_dir: Path):
    path = split_dir / "labels.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check_script(script_dir: Path, check_files: bool) -> list:
    problems = []
    charset = json.loads((script_dir / "charset.json").read_text(encoding="utf-8"))
    known = set(charset["syllables"])
    factors = charset["syllable_to_factors"]

    pool_use = defaultdict(set)      # pool_idx -> splits it appears in
    coverage = {s: set() for s in SPLITS}
    counts = {}

    for split in SPLITS:
        records = load_labels(script_dir / split)
        counts[split] = len(records)
        for rec in records:
            for g in rec["glyphs"]:
                pool_use[g["pool_idx"]].add(split)

            syls = [t for t in rec["tokens"] if t != WORD_SEP]
            coverage[split].update(syls)

            unknown = sorted(set(syls) - known)
            if unknown:
                problems.append(f"{rec['id']}: tokens outside charset: {unknown}")

            # the factored labels must agree with the charset decomposition
            for tok, onset, vowel in zip(rec["tokens"], rec["onsets"], rec["vowels"]):
                if tok == WORD_SEP:
                    if onset is not None or vowel is not None:
                        problems.append(f"{rec['id']}: separator carries factors")
                    continue
                expect = factors.get(tok)
                if expect and (expect["onset"], expect["vowel"]) != (onset, vowel):
                    problems.append(
                        f"{rec['id']}: {tok!r} factored as ({onset},{vowel}), "
                        f"charset says ({expect['onset']},{expect['vowel']})")

            # boxes must be inside the image, non-degenerate, ordered, and
            # consistent with the word grouping
            w, h = rec["width"], rec["height"]
            prev_x = -1
            for g in rec["glyphs"]:
                bx, by, bw, bh = g["bbox"]
                if bw <= 0 or bh <= 0:
                    problems.append(f"{rec['id']}: degenerate bbox {g['bbox']}")
                if bx < 0 or by < 0 or bx + bw > w or by + bh > h:
                    problems.append(f"{rec['id']}: bbox {g['bbox']} outside "
                                    f"{w}x{h} image")
                if bx < prev_x:
                    problems.append(f"{rec['id']}: bbox order not left-to-right")
                prev_x = bx

            n_words = len(rec["words"])
            if len(rec.get("word_boxes", [])) != n_words:
                problems.append(f"{rec['id']}: {len(rec.get('word_boxes', []))} "
                                f"word boxes for {n_words} words")
            for wb in rec.get("word_boxes", []):
                members = [g["bbox"] for g in rec["glyphs"] if g["word"] == wb["word"]]
                if not members:
                    problems.append(f"{rec['id']}: word box {wb['word']} has no glyphs")
                    continue
                x0 = min(b[0] for b in members)
                x1 = max(b[0] + b[2] for b in members)
                if not (wb["bbox"][0] == x0
                        and wb["bbox"][0] + wb["bbox"][2] == x1):
                    problems.append(f"{rec['id']}: word box {wb['word']} does not "
                                    f"span its glyphs")

            if len(syls) != rec["n_syllables"]:
                problems.append(f"{rec['id']}: n_syllables disagrees with tokens")
            if len(rec["glyphs"]) != rec["n_syllables"]:
                problems.append(f"{rec['id']}: glyph count disagrees with n_syllables")

            if check_files and not (script_dir / split / rec["file"]).exists():
                problems.append(f"{rec['id']}: missing image {rec['file']}")

    shared = {idx: sorted(sp) for idx, sp in pool_use.items() if len(sp) > 1}
    if shared:
        sample = list(shared.items())[:5]
        problems.append(f"POOL LEAK: {len(shared)} glyph bitmaps used in more "
                        f"than one split, e.g. {sample}")

    print(f"  samples          train/val/test = "
          f"{counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('test', 0)}")
    print(f"  charset          {len(known)} syllables "
          f"= {charset['n_onsets']} onsets x {charset['n_vowels']} vowels")
    for split in SPLITS:
        if counts.get(split):
            missing = len(known - coverage[split])
            print(f"  {split:5s} coverage   {len(coverage[split])}/{len(known)} syllables"
                  + (f"  ({missing} never rendered)" if missing else ""))
    print(f"  distinct glyphs  {len(pool_use)} bitmaps used, "
          f"{len(shared)} shared across splits")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("aksara_seq/build/corpus/v1"))
    ap.add_argument("--no-file-check", action="store_true",
                    help="skip existence check for every image (much faster)")
    args = ap.parse_args()

    meta_path = args.corpus / "dataset_meta.json"
    if not meta_path.exists():
        print(f"no dataset_meta.json under {args.corpus}")
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    all_problems = []
    for script in meta["config"]["scripts"]:
        script_dir = args.corpus / script
        if not script_dir.is_dir():
            all_problems.append(f"{script}: directory missing")
            continue
        print(f"\n{script}")
        all_problems.extend(check_script(script_dir, not args.no_file_check))

    print()
    if all_problems:
        print(f"{len(all_problems)} problem(s):")
        for p in all_problems[:40]:
            print(f"  - {p}")
        if len(all_problems) > 40:
            print(f"  ... and {len(all_problems) - 40} more")
        return 1

    print("OK -- splits are instance-disjoint, labels agree with the charset, "
          "every bounding box is in-bounds and consistent with its word, "
          "and every referenced image exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

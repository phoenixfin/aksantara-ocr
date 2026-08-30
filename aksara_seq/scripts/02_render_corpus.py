"""Step 2 -- render the line corpus from the glyph cache.

    # look before you generate
    python aksara_seq/scripts/02_render_corpus.py --preview 6

    # generate
    python aksara_seq/scripts/02_render_corpus.py --config aksara_seq/configs/corpus_v1.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from aksaraseq.glyphs.pool import GlyphPool  # noqa: E402
from aksaraseq.preview import line_sheet  # noqa: E402
from aksaraseq.render.corpus import (  # noqa: E402
    CorpusConfig, build_word_source, generate, render_sample,
)
from aksaraseq.render.degrade import make_presets  # noqa: E402


def load_config(path: Path | None, overrides: dict) -> CorpusConfig:
    data = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("glyph_cache", "out", "lexicon_dir"):
        if key in data:
            data[key] = Path(data[key])
    if "words_per_line" in data:
        data["words_per_line"] = {int(k): float(v)
                                  for k, v in data["words_per_line"].items()}
    return CorpusConfig(**data)


def _draw_boxes(img, rec):
    """Overlay per-syllable (red) and per-word (blue) boxes."""
    from PIL import ImageDraw
    vis = img.convert("RGB")
    d = ImageDraw.Draw(vis)
    for wb in rec.get("word_boxes", []):
        x, y, w, h = wb["bbox"]
        d.rectangle([x, y, x + w - 1, y + h - 1], outline=(60, 140, 255))
    for g in rec["glyphs"]:
        x, y, w, h = g["bbox"]
        d.rectangle([x, y, x + w - 1, y + h - 1], outline=(230, 60, 60))
    return vis


def do_preview(cfg: CorpusConfig, n: int, out_path: Path,
               boxes: bool = False) -> None:
    presets = make_presets(cfg.core_height)
    sheets = []
    for script in cfg.scripts:
        pool = GlyphPool(cfg.glyph_cache, script)
        mixer, rejected = build_word_source(pool, cfg, script, "train")
        if rejected:
            print(f"{script}: {len(rejected)} lexicon words unrenderable, e.g. "
                  f"{rejected[:3]}")
        images, captions = [], []
        for name in ("clean", "light", "medium", "heavy"):
            if name not in presets:
                continue
            single = CorpusConfig(**{**cfg.__dict__, "style_mix": {name: 1.0}})
            for i in range(n):
                img, rec = render_sample(pool, mixer, single, presets, "train",
                                         i + 1000 * list(presets).index(name))
                images.append(_draw_boxes(img, rec) if boxes else img)
                captions.append(f"[{name}] {rec['text']}   "
                                f"({rec['width']}x{rec['height']}px, "
                                f"{rec['n_syllables']} syllables)")
        sheets.append(line_sheet(images, captions, title=f"{script} - rendered lines"))

    width = max(s.width for s in sheets)
    height = sum(s.height + 16 for s in sheets)
    combined = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for s in sheets:
        combined.paste(s, (0, y))
        y += s.height + 16
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f"preview -> {out_path}  ({combined.width}x{combined.height})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--glyph-cache", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--scripts", nargs="+")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--output-height", type=int)
    ap.add_argument("--lexicon-fraction", type=float)
    ap.add_argument("--train", type=int, help="samples per script, train split")
    ap.add_argument("--val", type=int)
    ap.add_argument("--test", type=int)
    ap.add_argument("--preview", type=int, metavar="N",
                    help="render N lines per style per script to a sheet, then exit")
    ap.add_argument("--boxes", action="store_true",
                    help="overlay syllable and word bounding boxes on the preview")
    ap.add_argument("--preview-out", type=Path,
                    default=Path("aksara_seq/build/preview_lines.png"))
    args = ap.parse_args()

    overrides = {
        "glyph_cache": args.glyph_cache, "out": args.out,
        "scripts": args.scripts, "seed": args.seed,
        "output_height": args.output_height,
        "lexicon_fraction": args.lexicon_fraction,
    }
    cfg = load_config(args.config, overrides)
    if any(v is not None for v in (args.train, args.val, args.test)):
        samples = dict(cfg.samples)
        for key, val in (("train", args.train), ("val", args.val), ("test", args.test)):
            if val is not None:
                samples[key] = val
        cfg.samples = samples

    if args.preview:
        do_preview(cfg, args.preview, args.preview_out, boxes=args.boxes)
        return 0

    total = len(cfg.scripts) * sum(cfg.samples.values())
    print(f"rendering {total} lines -> {cfg.out}", flush=True)
    state = {"n": 0, "t": time.time()}

    def progress(script, split, i, n):
        state["n"] += 1
        if state["n"] % 2000 == 0:
            rate = state["n"] / max(time.time() - state["t"], 1e-9)
            eta = (total - state["n"]) / max(rate, 1e-9)
            print(f"  {state['n']}/{total}  {script}/{split} {i}/{n}  "
                  f"{rate:.0f}/s  eta {eta / 60:.1f}min", flush=True)

    t0 = time.time()
    meta = generate(cfg, progress=progress)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min\n")

    for script, s in meta["scripts"].items():
        cs = s["charset"]
        print(f"{script:9s} {cs['n_syllables']} syllables "
              f"= {cs['n_onsets']} onsets x {cs['n_vowels']} vowels")
        for split in ("train", "val", "test"):
            if split not in s:
                continue
            d = s[split]
            print(f"          {split:5s} {d['n_samples']:6d} lines  "
                  f"{d['syllable_tokens']:8d} syllable tokens  "
                  f"{d['syllable_types_seen']:4d} types  "
                  f"rarest seen {d['min_syllable_count']}x  "
                  f"mean width {d['width_mean']:.0f}px")
    print(f"\nmetadata -> {Path(cfg.out) / 'dataset_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

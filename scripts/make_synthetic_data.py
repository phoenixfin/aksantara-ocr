"""Generate a fake dataset with the expected layout, for testing the pipeline.

This is scaffolding, not research data — the "characters" are random geometric
doodles. Its only job is to let you exercise the full pipeline before your real
images are ready, and to keep the smoke test runnable in CI.

    python scripts/make_synthetic_data.py --out data/synthetic
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def draw_character(class_seed: int, sample_seed: int, size: int = 64) -> Image.Image:
    """A shape determined by ``class_seed``, perturbed by ``sample_seed``.

    The split matters: every sample of a character must be a *variation of one
    shape*, not an independent doodle. Otherwise the classes are unlearnable,
    the smoke test sits at chance accuracy, and it would report exactly the same
    numbers if the training loop were broken — which makes it worthless as a
    check. Learnable synthetic classes are what give the smoke test a signal.
    """
    shape_rng = random.Random(class_seed)
    jitter_rng = random.Random(sample_seed)

    strokes = []
    for _ in range(shape_rng.randint(2, 5)):
        strokes.append(
            {
                "kind": shape_rng.choice(["line", "arc", "ellipse"]),
                "box": [
                    shape_rng.randint(5, size // 2),
                    shape_rng.randint(5, size // 2),
                    shape_rng.randint(size // 2, size - 5),
                    shape_rng.randint(size // 2, size - 5),
                ],
                "width": shape_rng.randint(2, 4),
                "angles": (shape_rng.randint(0, 180), shape_rng.randint(180, 360)),
            }
        )

    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    # Per-sample wobble, small enough that the class identity survives it.
    dx, dy = jitter_rng.randint(-3, 3), jitter_rng.randint(-3, 3)

    for stroke in strokes:
        x0, y0, x1, y1 = stroke["box"]
        box = [
            x0 + dx + jitter_rng.randint(-2, 2),
            y0 + dy + jitter_rng.randint(-2, 2),
            x1 + dx + jitter_rng.randint(-2, 2),
            y1 + dy + jitter_rng.randint(-2, 2),
        ]
        # PIL requires x1 >= x0 and y1 >= y0; jitter can invert a tight box.
        box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]

        if stroke["kind"] == "line":
            draw.line(box, fill=0, width=stroke["width"])
        elif stroke["kind"] == "arc":
            draw.arc(box, stroke["angles"][0], stroke["angles"][1], fill=0, width=3)
        else:
            draw.ellipse(box, outline=0, width=2)
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--scripts", type=int, default=4)
    parser.add_argument("--characters", type=int, default=20)
    parser.add_argument("--writers", type=int, default=10)
    parser.add_argument("--samples-per-writer", type=int, default=3)
    args = parser.parse_args()

    total = 0
    for s in range(args.scripts):
        script_name = f"script_{s:02d}"
        for c in range(args.characters):
            char_dir = args.out / script_name / f"char_{c:02d}"
            char_dir.mkdir(parents=True, exist_ok=True)
            base_seed = s * 1000 + c

            for w in range(args.writers):
                for k in range(args.samples_per_writer):
                    # class_seed is constant across every sample of this
                    # character; only the jitter seed varies.
                    image = draw_character(base_seed, w * 10 + k)
                    # Writer id goes in the filename — this is the naming
                    # convention the grouped splitter looks for.
                    image.save(char_dir / f"w{w:02d}_{k:02d}.png")
                    total += 1

    print(f"Wrote {total} images to {args.out}")
    print(f"  {args.scripts} scripts x {args.characters} characters x "
          f"{args.writers} writers x {args.samples_per_writer} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Step 3 -- train the CRNN+CTC baseline.

    # quick smoke test on the small corpus, CPU is fine
    python aksara_seq/scripts/04_train_recognizer.py --corpus aksara_seq/build/corpus/smoke \
        --script Bali --head factored --epochs 2 --limit-train 200 --batch-size 8

    # real run
    python aksara_seq/scripts/04_train_recognizer.py --script Sunda --head plain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksaraseq.recog.train import TrainConfig, train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("aksara_seq/build/corpus/v1"))
    ap.add_argument("--script", default="Sunda")
    ap.add_argument("--head", default="plain", choices=["plain", "factored"])
    ap.add_argument("--out", type=Path, default=Path("aksara_seq/build/recog"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--holdout-cells", nargs="*", default=[],
                    help="syllables to withhold from training entirely")
    args = ap.parse_args()

    cfg = TrainConfig(
        corpus=args.corpus, script=args.script, head=args.head, out=args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        height=args.height, seed=args.seed, num_workers=args.num_workers,
        limit_train=args.limit_train, limit_eval=args.limit_eval,
        augment=not args.no_augment, amp=not args.no_amp,
        holdout_cells=list(args.holdout_cells),
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

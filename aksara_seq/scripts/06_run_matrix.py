"""Run the {script x head} matrix and collect one results table.

    python aksara_seq/scripts/06_run_matrix.py --epochs 30

Every cell trains the same encoder on the same corpus and differs only in the
label space, which is the comparison the project exists to make.  Completed
cells are detected by their result.json and skipped, so an interrupted session
resumes by re-running the same command.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksaraseq.recog.train import TrainConfig, train  # noqa: E402
from aksaraseq.recog.vocab import Vocab, diagonal_holdout  # noqa: E402

FIELDS = ["script", "head", "seed", "n_parameters", "best_epoch",
          "val_ser", "test_ser", "test_wer", "test_line_acc",
          "onset_error", "vowel_error", "minutes",
          "ser_clean", "ser_light", "ser_medium", "ser_heavy",
          "n_holdout", "holdout_recall", "seen_recall", "holdout_tokens",
          "train_lines", "train_frac", "epochs"]


def row_from(result: dict) -> dict:
    cfg, test = result["config"], result["test"]
    by_style = test.get("by_style", {})
    row = {
        "script": cfg["script"], "head": cfg["head"], "seed": cfg["seed"],
        "n_parameters": result["n_parameters"],
        "best_epoch": result["best_epoch"],
        "val_ser": round(result["best_val_ser"], 5),
        "test_ser": round(test["ser"], 5),
        "test_wer": round(test["wer"], 5),
        "test_line_acc": round(test["line_accuracy"], 5),
        "onset_error": round(test.get("onset_error", float("nan")), 5),
        "vowel_error": round(test.get("vowel_error", float("nan")), 5),
        "minutes": round(result["minutes"], 2),
    }
    for style in ("clean", "light", "medium", "heavy"):
        v = by_style.get(style, {}).get("ser")
        row[f"ser_{style}"] = round(v, 5) if v is not None else ""
    row["n_holdout"] = len(cfg.get("holdout_cells", []))
    for k in ("holdout_recall", "seen_recall", "holdout_tokens"):
        v = test.get(k)
        row[k] = round(v, 5) if isinstance(v, float) else (v if v is not None else "")
    row["train_lines"] = result.get("train_lines", "")
    row["train_frac"] = cfg.get("train_frac", 1.0)
    row["epochs"] = cfg["epochs"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("aksara_seq/build/corpus/v1"))
    ap.add_argument("--out", type=Path, default=Path("aksara_seq/build/recog"))
    ap.add_argument("--scripts", nargs="+",
                    default=["Sunda", "Jawa", "Bali", "Lontara"])
    ap.add_argument("--heads", nargs="+", default=["plain", "factored"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="train on this fraction of the training lines. Epochs "
                         "are scaled up so every fraction gets a comparable "
                         "optimisation budget, otherwise a small fraction is "
                         "simply undertrained and the comparison is confounded")
    ap.add_argument("--max-epochs", type=int, default=90,
                    help="cap on the scaled epoch count")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="withhold this fraction of (onset,vowel) cells from "
                         "training, to test whether a head can read a syllable "
                         "it never saw")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = [(s, h, sd) for s in args.scripts for h in args.heads
             for sd in args.seeds]
    print(f"{len(cells)} runs: {args.scripts} x {args.heads} x seeds {args.seeds}")
    if args.dry_run:
        for s, h, sd in cells:
            print(f"  {s:9s} {h:9s} seed {sd}")
        return 0

    rows, failures, t0 = [], [], time.time()
    for i, (script, head, seed) in enumerate(cells, start=1):
        tag = f"{script}_{head}" + (f"_s{seed}" if seed else "")
        run_dir = Path(args.out) / tag
        done = run_dir / "result.json"
        if done.exists():
            print(f"[{i}/{len(cells)}] {tag}: already done, skipping")
            rows.append(row_from(json.loads(done.read_text(encoding="utf-8"))))
            continue

        print(f"\n[{i}/{len(cells)}] {tag}")
        limit_train = args.limit_train
        epochs = args.epochs
        if args.train_frac < 1.0:
            n_train = sum(1 for _ in open(
                args.corpus / script / "train" / "labels.jsonl", encoding="utf-8"))
            limit_train = max(1, int(round(args.train_frac * n_train)))
            epochs = min(args.max_epochs,
                         max(args.epochs, int(round(args.epochs / args.train_frac))))
            print(f"  train_frac {args.train_frac}: {limit_train} lines, "
                  f"{epochs} epochs")

        holdout = []
        if args.holdout_frac > 0:
            vocab = Vocab.from_charset(args.corpus / script / "charset.json")
            holdout = diagonal_holdout(vocab, args.holdout_frac)
            print(f"  withholding {len(holdout)} cells: {holdout}")

        cfg = TrainConfig(corpus=args.corpus, script=script, head=head,
                          out=args.out, epochs=epochs, seed=seed,
                          batch_size=args.batch_size, height=args.height,
                          num_workers=args.num_workers,
                          limit_train=limit_train,
                          holdout_cells=holdout, tag=tag)
        cfg.train_frac = args.train_frac
        try:
            result = train(cfg)
        except Exception as exc:                       # keep the matrix going
            print(f"  FAILED: {exc!r}")
            failures.append((tag, repr(exc)))
            continue
        rows.append(row_from(result))

        report = Path(args.out) / "matrix.csv"
        with open(report, "w", newline="", encoding="utf-8") as fh:
            wtr = csv.DictWriter(fh, fieldnames=FIELDS)
            wtr.writeheader()
            wtr.writerows(rows)

    print(f"\n{'script':9s} {'head':9s} {'test SER':>9s} {'WER':>8s} "
          f"{'line-acc':>9s} {'onset':>8s} {'vowel':>8s}")
    for r in rows:
        print(f"{r['script']:9s} {r['head']:9s} {r['test_ser']:9.4f} "
              f"{r['test_wer']:8.4f} {r['test_line_acc']:9.4f} "
              f"{r['onset_error']:8.4f} {r['vowel_error']:8.4f}")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min "
          f"-> {Path(args.out) / 'matrix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

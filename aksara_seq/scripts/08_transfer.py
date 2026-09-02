"""Cross-script transfer: does an encoder trained on one abugida help another?

    python aksara_seq/scripts/08_transfer.py --source Sunda --targets Jawa Bali Lontara

The low-resource question the four scripts are actually good for. Sunda has
28k source glyphs, Lontara 6.4k; if the convolutional encoder learns something
about abugida shape in general rather than about Sunda specifically, it should
transfer.

Only the encoder moves. The head is vocabulary-specific -- a different script
has a different syllable inventory -- so it is always learned from scratch,
and the comparison is therefore about visual features, not about labels.

The scratch baseline is *not* re-run here: it is exactly what the data-scaling
kernel at the same fraction and seed already measured, so this script only
trains the source models and the transferred ones, and expects the baseline to
be merged in afterwards.
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

FIELDS = ["role", "source", "script", "head", "seed", "train_frac", "epochs",
          "train_lines", "test_ser", "test_wer", "test_line_acc",
          "onset_error", "vowel_error", "best_epoch", "minutes", "init_from"]


def row_from(result: dict, role: str, source: str) -> dict:
    cfg, test = result["config"], result["test"]
    return {
        "role": role, "source": source,
        "script": cfg["script"], "head": cfg["head"], "seed": cfg["seed"],
        "train_frac": cfg.get("train_frac", 1.0), "epochs": cfg["epochs"],
        "train_lines": result.get("train_lines", ""),
        "test_ser": round(test["ser"], 5),
        "test_wer": round(test["wer"], 5),
        "test_line_acc": round(test["line_accuracy"], 5),
        "onset_error": round(test.get("onset_error", float("nan")), 5),
        "vowel_error": round(test.get("vowel_error", float("nan")), 5),
        "best_epoch": result["best_epoch"],
        "minutes": round(result["minutes"], 2),
        "init_from": cfg.get("init_from", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("aksara_seq/build/corpus/v1"))
    ap.add_argument("--out", type=Path, default=Path("aksara_seq/build/transfer"))
    ap.add_argument("--source", default="Sunda")
    ap.add_argument("--targets", nargs="+", default=["Jawa", "Bali", "Lontara"])
    ap.add_argument("--heads", nargs="+", default=["plain", "factored"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-epochs", type=int, default=30)
    ap.add_argument("--target-frac", type=float, default=0.10)
    ap.add_argument("--target-epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = ([(args.source, h, "source") for h in args.heads]
            + [(t, h, "transfer") for t in args.targets for h in args.heads])
    print(f"{len(plan)} runs: {len(args.heads)} source + "
          f"{len(args.targets) * len(args.heads)} transfer")
    for script, head, role in plan:
        print(f"  {role:9s} {script:9s} {head}")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    source_ckpt = {}

    # 1. train the source models on the full corpus
    for head in args.heads:
        tag = f"source_{args.source}_{head}"
        done = args.out / tag / "result.json"
        if done.exists():
            print(f"\n[source] {tag}: already done")
            result = json.loads(done.read_text(encoding="utf-8"))
        else:
            print(f"\n[source] {tag}")
            cfg = TrainConfig(corpus=args.corpus, script=args.source, head=head,
                              out=args.out, epochs=args.source_epochs,
                              seed=args.seed, batch_size=args.batch_size,
                              num_workers=args.num_workers, tag=tag)
            result = train(cfg)
        source_ckpt[head] = args.out / tag / "best.pt"
        rows.append(row_from(result, "source", args.source))

    # 2. transfer that encoder into each target, at low resource
    n_target = {t: sum(1 for _ in open(args.corpus / t / "train" / "labels.jsonl",
                                       encoding="utf-8"))
                for t in args.targets}
    for target in args.targets:
        limit = max(1, int(round(args.target_frac * n_target[target])))
        for head in args.heads:
            tag = f"transfer_{args.source}_to_{target}_{head}"
            done = args.out / tag / "result.json"
            if done.exists():
                print(f"\n[transfer] {tag}: already done")
                rows.append(row_from(json.loads(done.read_text(encoding="utf-8")),
                                     "transfer", args.source))
                continue
            print(f"\n[transfer] {tag}  ({limit} lines, "
                  f"{args.target_epochs} epochs)")
            cfg = TrainConfig(corpus=args.corpus, script=target, head=head,
                              out=args.out, epochs=args.target_epochs,
                              seed=args.seed, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              limit_train=limit, train_frac=args.target_frac,
                              init_from=str(source_ckpt[head]), tag=tag)
            try:
                result = train(cfg)
            except Exception as exc:
                print(f"  FAILED: {exc!r}")
                continue
            rows.append(row_from(result, "transfer", args.source))

            report = args.out / "transfer.csv"
            with open(report, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

    report = args.out / "transfer.csv"
    with open(report, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'role':9s} {'script':9s} {'head':9s} {'SER':>9s} {'line-acc':>9s}")
    for r in rows:
        print(f"{r['role']:9s} {r['script']:9s} {r['head']:9s} "
              f"{r['test_ser']:9.4f} {r['test_line_acc']:9.4f}")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min -> {report}")
    print("compare 'transfer' rows against the scratch baseline from the "
          "data-scaling run at the same fraction and seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

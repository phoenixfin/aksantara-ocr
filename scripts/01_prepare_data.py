"""Scan the dataset, report its health, and write manifest + splits.

Run this once. Everything downstream reads its outputs, so the splits every
model trains on are identical by construction.

    python scripts/01_prepare_data.py --data-root /path/to/dataset --split-strategy grouped
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksara.data.scan import find_duplicates, scan_directory  # noqa: E402
from aksara.data.splits import make_splits, save_splits, summarize_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--split-strategy", choices=["grouped", "stratified"], default="grouped")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-writer-parse", action="store_true")
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=0,
        help="Drop classes with fewer than N images. Excluded classes are listed "
             "and written to excluded_classes.csv so the paper can report exactly "
             "what was removed.",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Keep only the first image of each byte-identical group. Removes a "
             "leakage path; changes the dataset size, so report it.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.data_root} ...")
    manifest, warnings = scan_directory(args.data_root, writer_from_filename=not args.no_writer_parse)

    print(f"\n  images     : {len(manifest)}")
    print(f"  scripts    : {manifest['script'].nunique()}")
    print(f"  characters : {manifest['label'].nunique()} (script/character pairs)")

    writers = manifest["writer_id"].notna().sum()
    print(f"  with writer id: {writers}/{len(manifest)}")

    counts = manifest["label"].value_counts()
    print(f"  images per character: min={counts.min()} median={int(counts.median())} max={counts.max()}")

    thin = counts[counts < 10]
    if len(thin):
        print(f"\n  WARNING: {len(thin)} characters have <10 images; per-class metrics will be noisy.")
        print(f"           e.g. {list(thin.index[:5])}")

    duplicates = find_duplicates(manifest)
    if len(duplicates):
        print(f"\n  WARNING: {len(duplicates)} images are byte-identical duplicates of another image.")
        print(f"           Left in place, these can leak across the train/test boundary.")
        duplicates.to_csv(args.out_dir / "duplicates.csv", index=False)
        print(f"           Written to {args.out_dir / 'duplicates.csv'} for review.")

    if warnings:
        print(f"\n  {len(warnings)} scan warning(s):")
        for w in warnings[:10]:
            print(f"    - {w}")
        if len(warnings) > 10:
            print(f"    ... and {len(warnings) - 10} more")
        (args.out_dir / "scan_warnings.txt").write_text("\n".join(warnings), encoding="utf-8")

    manifest_path = args.out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest -> {manifest_path}")

    # Filtering happens after the manifest is written, so the manifest always
    # describes the dataset as published and the filters stay auditable.
    if args.drop_duplicates:
        before = len(manifest)
        manifest = manifest.drop_duplicates(subset="content_hash", keep="first")
        print(f"\nDropped {before - len(manifest)} duplicate image(s); {len(manifest)} remain.")

    if args.min_per_class > 0:
        sizes = manifest["label"].value_counts()
        small = sizes[sizes < args.min_per_class]
        if len(small):
            excluded = manifest[manifest["label"].isin(small.index)]
            excluded_path = args.out_dir / "excluded_classes.csv"
            small.rename("images").to_csv(excluded_path, header=True)
            manifest = manifest[~manifest["label"].isin(small.index)]
            print(
                f"\nExcluded {len(small)} class(es) with <{args.min_per_class} images "
                f"({len(excluded)} images) -> {excluded_path}"
            )
            for label, count in small.head(10).items():
                print(f"    {label}  ({count})")
            if len(small) > 10:
                print(f"    ... and {len(small) - 10} more")

    strategy = args.split_strategy
    if strategy == "grouped" and writers == 0:
        print(
            "\n  Requested grouped (writer-disjoint) splitting, but no writer ids were found.\n"
            "  Falling back to stratified splitting.\n"
            "  This means samples by the same writer may appear in both train and test,\n"
            "  so reported accuracy is an upper bound. State this limitation in the paper,\n"
            "  or encode writer ids in filenames (e.g. 'w03_sample01.png') and re-run."
        )
        strategy = "stratified"

    print(f"\nSplitting (strategy={strategy}, seed={args.seed}) ...")
    try:
        split = make_splits(
            manifest,
            strategy=strategy,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    except ValueError as exc:
        # These are dataset-shape problems with actionable fixes, not bugs — a
        # traceback would bury the message that tells the user what to change.
        print(f"\nCannot split this dataset:\n\n{exc}\n")
        print(f"The manifest was still written to {manifest_path}; inspect it to "
              "see the per-class and per-writer counts.")
        return 1
    summary = summarize_splits(manifest, split)
    print(json.dumps(summary, indent=2)[:2000])

    if summary["classes_absent_from_train"]:
        print(
            f"\n  WARNING: {len(summary['classes_absent_from_train'])} classes have no training "
            "examples and can never be predicted correctly."
        )

    splits_path = args.out_dir / "splits.csv"
    save_splits(
        splits_path,
        manifest,
        split,
        meta={
            "strategy": strategy,
            "requested_strategy": args.split_strategy,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "data_root": str(args.data_root.resolve()),
            "summary": summary,
        },
    )
    print(f"\nSplits -> {splits_path}")
    print("\nNext: python scripts/02_run_matrix.py --config configs/full_benchmark.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

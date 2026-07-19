"""Full dataset audit — composition table plus the integrity checks a reviewer runs.

Produces the "dataset description" numbers a data paper needs, and separately
flags the problems that undermine reported accuracy if left unstated:
duplicates, cross-label collisions, thin classes, and inconsistent image formats.

    python scripts/04_audit_dataset.py --manifest artifacts/manifest.csv --out artifacts/audit

Run it on the *raw* manifest, before any resizing — a resize cache normalizes
image dimensions and would hide format inconsistencies in the source data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def composition(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-script composition — Table 1 material for a dataset paper."""
    rows = []
    for script, group in manifest.groupby("script"):
        per_char = group.groupby("character").size()
        rows.append(
            {
                "script": script,
                "characters": group["character"].nunique(),
                "images": len(group),
                "min_per_char": int(per_char.min()),
                "median_per_char": int(per_char.median()),
                "max_per_char": int(per_char.max()),
                "balanced": bool(per_char.nunique() == 1),
                "image_sizes": ", ".join(
                    f"{w}x{h}" for w, h in sorted(set(zip(group.width, group.height)))[:3]
                ),
                "modes": ", ".join(sorted(group["mode"].unique())),
                "MB": round(group["size_bytes"].sum() / 1e6, 1) if "size_bytes" in group else None,
            }
        )
    return pd.DataFrame(rows).sort_values("images", ascending=False)


def duplicate_report(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Classify byte-identical images by how far apart their labels are.

    The three tiers matter differently:
      - same character  : inflates the dataset size, leaks across the split
      - across characters: contradictory labels — unlearnable, and a data bug
      - across scripts   : the same image filed under two different scripts
    """
    counts = manifest["content_hash"].value_counts()
    dupes = manifest[manifest["content_hash"].isin(counts[counts > 1].index)]

    records, summary = [], {"same_character": 0, "cross_character": 0, "cross_script": 0}
    for content_hash, group in dupes.groupby("content_hash"):
        scripts = group["script"].nunique()
        labels = group["label"].nunique()
        if scripts > 1:
            tier = "cross_script"
        elif labels > 1:
            tier = "cross_character"
        else:
            tier = "same_character"
        summary[tier] += 1
        records.append(
            {
                "tier": tier,
                "content_hash": content_hash,
                "n_copies": len(group),
                "labels": " | ".join(sorted(group["label"].unique())),
                "files": " | ".join(os.path.basename(p) for p in group["path"]),
            }
        )

    report = pd.DataFrame(records)
    if not report.empty:
        order = {"cross_script": 0, "cross_character": 1, "same_character": 2}
        report = report.sort_values(["tier", "n_copies"], key=lambda s: s.map(order).fillna(s))
    summary["duplicate_groups"] = len(report)
    summary["images_involved"] = len(dupes)
    summary["unique_images_after_dedup"] = int(manifest["content_hash"].nunique())
    return report, summary


def shared_character_names(manifest: pd.DataFrame) -> pd.DataFrame:
    """Character names appearing in more than one script.

    Relevant to the unified task: 'ha' in Javanese and 'ha' in Balinese are
    distinct classes here (labels are script-qualified), which is the right
    choice — but it is worth stating explicitly in the paper.
    """
    pairs = manifest[["script", "character"]].drop_duplicates()
    counts = pairs.groupby("character")["script"].agg(["count", lambda s: ", ".join(sorted(s))])
    counts.columns = ["n_scripts", "scripts"]
    return counts[counts["n_scripts"] > 1].sort_values("n_scripts", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/audit"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)

    if "size_bytes" not in manifest.columns:
        manifest["size_bytes"] = [
            os.path.getsize(p) if os.path.exists(p) else 0 for p in manifest["path"]
        ]

    print("=" * 78)
    print("OVERALL")
    print("=" * 78)
    per_char = manifest.groupby("label").size()
    overall = {
        "images": len(manifest),
        "scripts": manifest["script"].nunique(),
        "classes": manifest["label"].nunique(),
        "unique_characters": manifest["character"].nunique(),
        "images_per_class_min": int(per_char.min()),
        "images_per_class_median": int(per_char.median()),
        "images_per_class_max": int(per_char.max()),
        "total_MB": round(manifest["size_bytes"].sum() / 1e6, 1),
        "distinct_resolutions": int(manifest.groupby(["width", "height"]).ngroups),
        "modes": sorted(manifest["mode"].unique()),
        "writer_ids_present": int(manifest["writer_id"].notna().sum()),
    }
    for key, value in overall.items():
        print(f"  {key:26} {value}")

    print("\n" + "=" * 78)
    print("COMPOSITION BY SCRIPT")
    print("=" * 78)
    comp = composition(manifest)
    print(comp.to_string(index=False))
    comp.to_csv(args.out / "composition.csv", index=False)

    print("\n" + "=" * 78)
    print("DUPLICATES")
    print("=" * 78)
    dupes, summary = duplicate_report(manifest)
    for key, value in summary.items():
        print(f"  {key:26} {value}")

    if not dupes.empty:
        dupes.to_csv(args.out / "duplicates.csv", index=False)
        serious = dupes[dupes["tier"] != "same_character"]
        if not serious.empty:
            print(f"\n  {len(serious)} contradictory-label group(s) — identical pixels, "
                  "different labels:")
            for _, row in serious.head(20).iterrows():
                print(f"    [{row['tier']}] {row['labels']}  <-  {row['files']}")
            if len(serious) > 20:
                print(f"    ... and {len(serious) - 20} more (see duplicates.csv)")

        pct = 100 * summary["images_involved"] / len(manifest)
        print(f"\n  {summary['images_involved']} images ({pct:.2f}%) are duplicates of another.")
        print(f"  Deduplicated dataset size: {summary['unique_images_after_dedup']} images.")

    print("\n" + "=" * 78)
    print("CHARACTER NAMES SHARED ACROSS SCRIPTS")
    print("=" * 78)
    shared = shared_character_names(manifest)
    if shared.empty:
        print("  none")
    else:
        print(f"  {len(shared)} character name(s) appear in multiple scripts.")
        print("  Labels are script-qualified, so these remain distinct classes.")
        print(shared.head(15).to_string())
        shared.to_csv(args.out / "shared_character_names.csv")

    print("\n" + "=" * 78)
    print("FORMAT CONSISTENCY")
    print("=" * 78)
    resolutions = manifest.groupby(["width", "height"]).size().sort_values(ascending=False)
    print(f"  {len(resolutions)} distinct resolution(s):")
    for (w, h), n in resolutions.head(10).items():
        print(f"    {w}x{h}: {n} images")
    modes = manifest["mode"].value_counts()
    print(f"  modes: {modes.to_dict()}")
    if len(resolutions) > 1 or len(modes) > 1:
        print("\n  Mixed formats. Harmless for training (everything is resized and"
              "\n  converted), but worth reporting accurately in the paper.")

    (args.out / "summary.json").write_text(
        json.dumps({"overall": overall, "duplicates": summary}, indent=2), encoding="utf-8"
    )
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

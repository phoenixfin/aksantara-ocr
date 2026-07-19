"""Produce a cleaned copy of the dataset from an explicit rules file.

Design constraints, both deliberate:

**Never mutates the source.** It writes a new tree and leaves the original
untouched, so a bad rule costs a re-run rather than a re-download — and the
published v2 stays reproducible from its DOI.

**Every change is declared in a YAML rules file, not in code.** A reviewer (or
you, in six months) can read exactly what was removed and why. The script emits
a changelog listing every affected file, which is the provenance record for
publishing a v3.

    python scripts/06_clean_dataset.py \\
        --data-root data/full --out data/clean --rules configs/cleaning_rules.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
import yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def file_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8")) or {}

    sources = sorted(
        p for p in args.data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"{len(sources)} image(s) under {args.data_root}\n")

    # relative path -> (action, reason). Default: keep, unchanged location.
    plan: dict[Path, tuple[str, str, Path]] = {}
    for src in sources:
        rel = src.relative_to(args.data_root)
        plan[rel] = ("keep", "", rel)

    # --- 1. Drop files, by exact path or glob -------------------------------
    # Both forms exist because these filenames contain literal square brackets
    # ("rarangken[e]_18.jpg"), which fnmatch reads as a character class. A glob
    # pattern for such a file silently matches nothing, so surgical deletions
    # must use `path:` for an exact literal comparison.
    unmatched: list[str] = []
    for rule in rules.get("drop_files", []):
        reason = rule.get("reason", "")
        if "path" in rule:
            target_path, literal = rule["path"], True
        elif "pattern" in rule:
            target_path, literal = rule["pattern"], False
        else:
            raise SystemExit(f"drop_files rule needs 'path' or 'pattern': {rule}")

        n = 0
        for rel in plan:
            if plan[rel][0] != "keep":
                continue
            posix = rel.as_posix()
            hit = (posix == target_path) if literal else fnmatch(posix, target_path)
            if hit:
                plan[rel] = ("drop", reason or f"matched drop_files: {target_path}", rel)
                n += 1

        kind = "path" if literal else "glob"
        print(f"drop_files  [{kind}] {target_path!r}: {n} file(s)")
        if n == 0:
            unmatched.append(f"drop_files [{kind}] {target_path!r}")

    # --- 2. Drop whole folders ---------------------------------------------
    for rule in rules.get("drop_folders", []):
        prefix, reason = rule["path"].rstrip("/"), rule.get("reason", "")
        n = 0
        for rel in plan:
            if plan[rel][0] == "keep" and rel.as_posix().startswith(prefix + "/"):
                plan[rel] = ("drop", reason or f"inside dropped folder: {prefix}", rel)
                n += 1
        print(f"drop_folders {prefix!r}: {n} file(s)")
        if n == 0:
            unmatched.append(f"drop_folders {prefix!r}")

    # --- 3. Relocate folders ------------------------------------------------
    for rule in rules.get("move_folders", []):
        source_prefix = rule["from"].rstrip("/")
        target_prefix = rule["to"].rstrip("/")
        reason = rule.get("reason", "")
        n = 0
        for rel in plan:
            if plan[rel][0] != "keep":
                continue
            posix = rel.as_posix()
            if posix.startswith(source_prefix + "/"):
                remainder = posix[len(source_prefix) + 1:]
                plan[rel] = ("move", reason, Path(target_prefix) / remainder)
                n += 1
        print(f"move_folders {source_prefix!r} -> {target_prefix!r}: {n} file(s)")
        if n == 0:
            unmatched.append(f"move_folders {source_prefix!r}")

    # --- 4. Deduplicate ------------------------------------------------------
    dedupe = rules.get("dedupe") or {}
    if dedupe.get("enabled"):
        scope = dedupe.get("scope", "global")  # global | within_class
        print(f"\nhashing {sum(1 for a in plan.values() if a[0] != 'drop')} file(s) for dedupe ...")

        by_hash: dict[tuple, list[Path]] = defaultdict(list)
        for rel, (action, _, target) in plan.items():
            if action == "drop":
                continue
            digest = file_hash(args.data_root / rel)
            key = (digest, target.parent.as_posix()) if scope == "within_class" else (digest,)
            by_hash[key].append(rel)

        dropped_dupes = 0
        contradictions = []
        for key, group in by_hash.items():
            if len(group) < 2:
                continue
            classes = {plan[r][2].parent.as_posix() for r in group}
            if len(classes) > 1:
                # Identical pixels under different labels. Not a dedupe case —
                # one of the labels is wrong, and picking arbitrarily would bake
                # a known error into the cleaned data.
                contradictions.append((sorted(classes), [r.as_posix() for r in group]))
                if dedupe.get("drop_contradictions"):
                    for rel in group:
                        plan[rel] = ("drop", f"contradictory label: {sorted(classes)}", plan[rel][2])
                        dropped_dupes += 1
                continue
            for rel in sorted(group)[1:]:
                plan[rel] = ("drop", f"duplicate of {sorted(group)[0].as_posix()}", plan[rel][2])
                dropped_dupes += 1

        print(f"dedupe ({scope}): dropped {dropped_dupes} file(s)")
        if contradictions:
            verb = "dropped" if dedupe.get("drop_contradictions") else "KEPT (review these)"
            print(f"\n{len(contradictions)} contradictory-label group(s) — {verb}:")
            for classes, files in contradictions[:20]:
                print(f"  {classes}")
                for f in files:
                    print(f"      {f}")

    # A rule that matches nothing is almost always a typo or a stale path, and
    # it fails silently — the cleaned data simply still contains the problem.
    # Refuse to write in that state rather than produce a quietly-wrong dataset.
    if unmatched:
        print(f"\n{'!' * 66}")
        print(f"{len(unmatched)} rule(s) matched NOTHING:")
        for rule_desc in unmatched:
            print(f"  {rule_desc}")
        print("Fix or remove them. Note that [] in a filename is a glob character\n"
              "class — use 'path:' for literal filenames containing brackets.")
        print(f"{'!' * 66}")
        if not args.dry_run:
            raise SystemExit("Refusing to write with unmatched rules. Re-run with --dry-run to iterate.")

    # --- Report and execute --------------------------------------------------
    changelog = pd.DataFrame(
        [
            {
                "source": rel.as_posix(),
                "action": action,
                "destination": target.as_posix() if action != "drop" else "",
                "reason": reason,
            }
            for rel, (action, reason, target) in sorted(plan.items())
        ]
    )
    counts = changelog["action"].value_counts().to_dict()
    kept = counts.get("keep", 0) + counts.get("move", 0)
    print(f"\n{'=' * 66}")
    print(f"  keep {counts.get('keep', 0)}   move {counts.get('move', 0)}   "
          f"drop {counts.get('drop', 0)}")
    print(f"  {len(sources)} -> {kept} images")
    print(f"{'=' * 66}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        out_log = args.rules.parent / "cleaning_changelog_dryrun.csv"
        changelog.to_csv(out_log, index=False)
        print(f"changelog -> {out_log}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for rel, (action, _, target) in sorted(plan.items()):
        if action == "drop":
            continue
        destination = args.out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.data_root / rel, destination)
        written += 1
        if written % 5000 == 0:
            print(f"  copied {written}/{kept}")

    log_path = args.out.parent / "cleaning_changelog.csv"
    changelog.to_csv(log_path, index=False)
    print(f"\nWrote {written} image(s) -> {args.out}")
    print(f"changelog -> {log_path}")
    print(f"\nNext:\n  python scripts/01_prepare_data.py --data-root {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

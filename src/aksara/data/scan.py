"""Scan a nested dataset directory into a flat manifest.

Layout::

    <root>/<script>/<...any depth.../><class folder>/<image files>

The nesting depth is **not** assumed to be uniform. Real script corpora mix
flat alphabets with syllabaries, so one archive may be
``<script>/<character>/`` while another is
``<script>/<vowel>/<consonant>/`` or deeper. A fixed-depth scan silently drops
every image below the assumed level, which looks like a smaller dataset rather
than like a bug.

A *class folder* is therefore any directory that directly contains images. Its
path relative to the script becomes the character label, so
``Bali/Letter/Vowel A/ha`` and ``Bali/Number/one`` coexist without collision.

Everything downstream (splits, datasets, experiments) reads the manifest CSV
rather than touching the directory tree, so layout quirks are handled once here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Matches a writer/source id embedded in a filename, e.g. "w03_sample12.png",
# "writer-7_a.png". Falls back to None when absent.
WRITER_PATTERNS = [
    re.compile(r"(?:^|[_-])w(?:riter)?[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|[_-])s(?:ubject)?[_-]?(\d+)", re.IGNORECASE),
]


@dataclass
class ImageRecord:
    path: str
    script: str
    character: str  # class-folder path relative to the script, e.g. "Vowel A/ba"
    label: str      # "<script>/<character>" — the unified-task label
    group: str      # first intermediate level ("Letter", "Number", ""), for analysis
    depth: int      # directory levels below the script folder
    writer_id: str | None
    width: int
    height: int
    mode: str
    content_hash: str


def _infer_writer_id(filename: str) -> str | None:
    for pattern in WRITER_PATTERNS:
        match = pattern.search(filename)
        if match:
            return match.group(1)
    return None


def _hash_file(path: Path, chunk_size: int = 1 << 16) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_directory(
    root: Path,
    writer_from_filename: bool = True,
    verify_images: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Walk ``root`` and build a manifest DataFrame.

    Returns the manifest plus a list of human-readable warnings — unreadable
    files, empty character folders, and so on. Warnings are returned rather
    than raised so a single bad file doesn't abort a scan of 50k images.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root}")

    records: list[ImageRecord] = []
    warnings: list[str] = []

    script_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not script_dirs:
        raise ValueError(f"No script subdirectories found under {root}")

    for script_dir in script_dirs:
        # Any directory holding images directly is a class folder, at whatever
        # depth it happens to sit.
        class_dirs = sorted(
            {
                p.parent
                for p in script_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            }
        )
        if not class_dirs:
            warnings.append(f"Script '{script_dir.name}' contains no images — skipped.")
            continue

        # Case-only collisions are invisible on Windows/macOS but become two
        # separate classes on Linux, where the benchmark actually runs. Catching
        # this here prevents a silent train/test discrepancy across platforms.
        lowered: dict[str, list[str]] = {}
        for class_dir in class_dirs:
            key = str(class_dir.relative_to(script_dir)).lower()
            lowered.setdefault(key, []).append(str(class_dir.relative_to(script_dir)))
        for key, variants in lowered.items():
            if len(set(variants)) > 1:
                warnings.append(
                    f"Case-only collision in '{script_dir.name}': {sorted(set(variants))} "
                    "— these are one folder on Windows but two on Linux."
                )

        for class_dir in class_dirs:
            image_paths = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            relative = class_dir.relative_to(script_dir)
            character = relative.as_posix()
            depth = len(relative.parts)
            group = relative.parts[0] if depth > 1 else ""

            # A folder holding both images and subfolders is ambiguous: the loose
            # images may be strays, or a class that lost its own folder.
            if any(p.is_dir() for p in class_dir.iterdir()):
                warnings.append(
                    f"'{script_dir.name}/{character}' holds {len(image_paths)} image(s) "
                    "alongside subfolders — treated as its own class; verify this is intended."
                )

            for image_path in image_paths:
                width = height = -1
                mode = "unknown"
                if verify_images:
                    try:
                        with Image.open(image_path) as img:
                            img.verify()
                        # verify() consumes the file object, so reopen for metadata.
                        with Image.open(image_path) as img:
                            width, height = img.size
                            mode = img.mode
                    except Exception as exc:  # noqa: BLE001 — report, don't crash the scan
                        warnings.append(f"Unreadable image {image_path}: {exc}")
                        continue

                records.append(
                    ImageRecord(
                        path=str(image_path.resolve()),
                        script=script_dir.name,
                        character=character,
                        label=f"{script_dir.name}/{character}",
                        group=group,
                        depth=depth,
                        writer_id=_infer_writer_id(image_path.name) if writer_from_filename else None,
                        width=width,
                        height=height,
                        mode=mode,
                        content_hash=_hash_file(image_path),
                    )
                )

    if not records:
        raise ValueError(f"Scan of {root} produced zero usable images.")

    manifest = pd.DataFrame([asdict(r) for r in records])
    return manifest, warnings


def find_duplicates(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose pixel content is byte-identical to another row.

    Exact duplicates that straddle a train/test boundary leak the test set.
    Reviewers ask about this, so surface it before training rather than after.
    """
    counts = manifest["content_hash"].value_counts()
    duplicated_hashes = counts[counts > 1].index
    return manifest[manifest["content_hash"].isin(duplicated_hashes)].sort_values("content_hash")

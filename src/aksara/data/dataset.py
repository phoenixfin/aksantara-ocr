"""Torch datasets built on top of the manifest + split files.

Two implementations:

``AksaraDataset``
    Decodes each image from disk on access. Correct for any resolution, but at
    small input sizes the decode cost dominates and the GPU starves.

``PreloadedAksaraDataset``
    Decodes and resizes the whole split once into a uint8 array, then serves
    from memory. At 64px the full corpus is ~400 MB, which removes the loader
    bottleneck entirely — measured decode drops from ~6.7 ms/img to ~0.02 ms.
    The array is cached to disk and memory-mapped, so every subsequent run in
    the matrix shares one copy through the OS page cache instead of rebuilding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class AksaraDataset(Dataset):
    """Character images with a configurable label column.

    ``label_column`` selects the task:
      - ``"label"``     unified classification over every (script, character) pair
      - ``"script"``    script identification
      - ``"character"`` character classification within a single script
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform=None,
        label_column: str = "label",
        grayscale: bool = False,
    ):
        missing = set(frame[label_column]) - set(class_to_idx)
        if missing:
            raise ValueError(f"Labels absent from class_to_idx: {sorted(missing)[:10]}")

        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.label_column = label_column
        self.mode = "L" if grayscale else "RGB"

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        # Convert unconditionally: source images mix RGB, RGBA, L and 1-bit modes,
        # and a mixed-channel batch fails to collate.
        with Image.open(row["path"]) as img:
            image = img.convert(self.mode)

        if self.transform is not None:
            image = self.transform(image)

        target = self.class_to_idx[row[self.label_column]]
        return image, torch.tensor(target, dtype=torch.long)


def estimate_preload_bytes(n_images: int, image_size: int, grayscale: bool) -> int:
    return n_images * image_size * image_size * (1 if grayscale else 3)


def _array_cache_path(cache_dir: Path, paths: list[str], image_size: int, grayscale: bool) -> Path:
    """Cache key covers the exact file list, so a different split or a cleaned
    dataset never silently reuses a stale array."""
    digest = hashlib.md5()
    digest.update(f"{image_size}:{grayscale}:{len(paths)}".encode())
    for p in paths:
        digest.update(p.encode())
    return Path(cache_dir) / f"preload_{digest.hexdigest()[:16]}.npy"


def build_image_array(
    paths: list[str],
    image_size: int,
    grayscale: bool,
    cache_dir: Path | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Decode and resize every image into one uint8 array.

    When ``cache_dir`` is given the array is written once and memory-mapped
    thereafter. Across a matrix of runs this turns a repeated multi-minute
    decode into a near-instant mmap that all runs share via the page cache.
    """
    channels = 1 if grayscale else 3
    cache_path = _array_cache_path(cache_dir, paths, image_size, grayscale) if cache_dir else None

    if cache_path is not None and cache_path.exists():
        array = np.load(cache_path, mmap_mode="r")
        expected = (len(paths), image_size, image_size, channels)
        if array.shape == expected:
            return array
        # Hash collision or an interrupted write — rebuild rather than serve
        # data that does not match the requested split.
        cache_path.unlink()

    array = np.zeros((len(paths), image_size, image_size, channels), dtype=np.uint8)
    mode = "L" if grayscale else "RGB"
    for i, path in enumerate(tqdm(paths, desc=f"preload {image_size}px", disable=not progress)):
        with Image.open(path) as img:
            resized = img.convert(mode).resize((image_size, image_size), Image.BILINEAR)
        array[i] = np.asarray(resized, dtype=np.uint8).reshape(image_size, image_size, channels)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp name then rename: a session killed mid-write must not
        # leave a truncated array that a later run would happily memory-map.
        temp_path = cache_path.with_suffix(".npy.tmp")
        # Write through an open handle: np.save appends ".npy" to a path that
        # does not already end in it, which would silently produce
        # "...npy.tmp.npy" and break the rename below.
        with temp_path.open("wb") as handle:
            np.save(handle, array)
        temp_path.replace(cache_path)
        array = np.load(cache_path, mmap_mode="r")

    return array


class PreloadedAksaraDataset(Dataset):
    """Serves images from an in-memory uint8 array instead of decoding on access."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        image_size: int,
        transform=None,
        label_column: str = "label",
        grayscale: bool = False,
        cache_dir: Path | None = None,
        progress: bool = True,
    ):
        missing = set(frame[label_column]) - set(class_to_idx)
        if missing:
            raise ValueError(f"Labels absent from class_to_idx: {sorted(missing)[:10]}")

        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.label_column = label_column
        self.grayscale = grayscale
        self.images = build_image_array(
            self.frame["path"].tolist(), image_size, grayscale, cache_dir, progress
        )
        self.targets = np.array(
            [class_to_idx[v] for v in self.frame[label_column]], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        array = self.images[index]
        # Transforms are the torchvision PIL pipeline, shared with the on-disk
        # dataset so augmentation behaviour is identical between the two paths.
        image = Image.fromarray(array.squeeze(-1) if self.grayscale else array,
                                mode="L" if self.grayscale else "RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.targets[index], dtype=torch.long)


def build_class_index(frame: pd.DataFrame, label_column: str = "label") -> dict[str, int]:
    """Stable, sorted label -> index mapping.

    Sorted rather than order-of-appearance so the mapping is identical across
    runs, machines, and splits — confusion matrices from different experiments
    stay directly comparable.
    """
    return {label: i for i, label in enumerate(sorted(frame[label_column].unique()))}


def load_split_frame(splits_csv: Path, manifest_csv: Path) -> pd.DataFrame:
    """Join the split assignment back onto the full manifest."""
    splits = pd.read_csv(splits_csv)
    manifest = pd.read_csv(manifest_csv)
    merged = manifest.merge(splits[["path", "split"]], on="path", how="inner")
    if len(merged) != len(splits):
        raise ValueError(
            f"Split file references {len(splits)} images but only {len(merged)} "
            "matched the manifest. Rebuild the manifest and splits together."
        )
    return merged

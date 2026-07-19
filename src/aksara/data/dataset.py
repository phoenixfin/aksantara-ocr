"""Torch dataset built on top of the manifest + split files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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

"""Dataset and batching for line recognition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .vocab import Vocab


def load_records(corpus: Path, script: str, split: str) -> list:
    path = Path(corpus) / script / split / "labels.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class LineDataset(Dataset):
    """Rendered lines, resized to a fixed height, with CTC targets.

    ``holdout`` removes every line containing one of the given syllables, for
    the held-out-cell experiment: a factored head can still represent those
    syllables, a monolithic head cannot.
    """

    def __init__(self, corpus: Path, script: str, split: str, vocab: Vocab,
                 height: int = 64, max_width: int = 1600,
                 holdout: set = None, augment: bool = False,
                 limit: int = 0):
        self.root = Path(corpus) / script / split
        self.vocab = vocab
        self.height = height
        self.max_width = max_width
        self.augment = augment

        records = load_records(corpus, script, split)
        self.holdout = set(holdout or ())
        if self.holdout:
            records = [r for r in records
                       if not (self.holdout & {t for t in r["tokens"]})]
        if limit:
            records = records[:limit]
        self.records = records
        # Width after resizing to `height`, for the length-bucketing sampler.
        self.widths = [max(1, int(round(r["width"] * height / r["height"])))
                       for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rec) -> np.ndarray:
        with Image.open(self.root / rec["file"]) as im:
            im = im.convert("L")
            w = max(1, int(round(im.width * self.height / im.height)))
            w = min(w, self.max_width)
            im = im.resize((w, self.height), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        # ink positive, roughly zero-centred
        return 1.0 - arr

    def __getitem__(self, i: int):
        rec = self.records[i]
        x = self._load_image(rec)

        if self.augment:
            rng = np.random.default_rng()
            x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.06, 0.06)
            if rng.random() < 0.3:
                x = x + rng.normal(0, 0.02, size=x.shape).astype(np.float32)
            x = np.clip(x, 0.0, 1.0)

        target = self.vocab.encode(rec["tokens"])
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x))[None],  # (1,H,W)
            "target": torch.tensor(target, dtype=torch.long),
            "width": x.shape[1],
            "text": rec["text"],
            "style": rec["style"],
            "id": rec["id"],
        }


def collate(batch, pad_value: float = 0.0) -> dict:
    """Right-pad images to the batch's widest line; concatenate targets."""
    widths = [b["width"] for b in batch]
    max_w = max(widths)
    h = batch[0]["image"].shape[1]

    images = torch.full((len(batch), 1, h, max_w), pad_value, dtype=torch.float32)
    for i, b in enumerate(batch):
        images[i, :, :, : b["width"]] = b["image"]

    targets = torch.cat([b["target"] for b in batch])
    target_lengths = torch.tensor([len(b["target"]) for b in batch], dtype=torch.long)
    return {
        "images": images,
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": [b["text"] for b in batch],
        "styles": [b["style"] for b in batch],
        "ids": [b["id"] for b in batch],
    }


class WidthBucketSampler(Sampler):
    """Group similar widths into a batch, so padding stays small.

    Line widths here span roughly 150-1500 px.  Batching at random would pad
    most of a batch to the longest line in it and waste the majority of the
    compute on blank columns.
    """

    def __init__(self, widths, batch_size: int, shuffle: bool = True,
                 pool_multiplier: int = 32, seed: int = 0):
        self.widths = list(widths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool = batch_size * pool_multiplier
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.widths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.widths)
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(n) if self.shuffle else np.arange(n)

        batches = []
        for start in range(0, n, self.pool):
            chunk = order[start: start + self.pool]
            chunk = chunk[np.argsort([self.widths[i] for i in chunk], kind="stable")]
            for b in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b: b + self.batch_size].tolist())
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

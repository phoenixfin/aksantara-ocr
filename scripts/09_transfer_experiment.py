"""Leave-one-script-out transfer — does the corpus work as a pretraining resource?

The strongest "value of this dataset" claim: can a model pretrained on the other
12 scripts learn a held-out 13th from very few examples? For a held-out script H:

  1. BASE: train ResNet-18 on the unified task over the 12 non-H scripts.
  2. For each k in --shots: fine-tune on k examples/class of H, evaluate on H's
     full test set, under two initializations:
       transfer  — backbone from the 12-script BASE model (this corpus)
       imagenet  — backbone from ImageNet (the usual off-the-shelf start)
  The gap (transfer − imagenet) at small k measures how much in-corpus
  pretraining helps a new Indonesian script.

Run one held-out script per invocation (they are independent):
    python scripts/09_transfer_experiment.py --held-out Sunda --shots 10 50 100

Suggested three (one large syllabary, one compact alphabet, one hard):
    Lontara (138-class syllabary), Lampung (20-class alphabet), Jawi (hard).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksara.data.dataset import AksaraDataset, PreloadedAksaraDataset, build_class_index  # noqa: E402
from aksara.data.transforms import build_transform  # noqa: E402
from aksara.data.dataset import load_split_frame  # noqa: E402
from aksara.engine.train import TrainConfig, train_model  # noqa: E402
from aksara.models.registry import build_model  # noqa: E402
from aksara.utils.seed import seed_worker, set_seed  # noqa: E402


def make_loaders(frame, class_to_idx, image_size, aug, cfg, device, seed,
                 train_subset=None, preload=True):
    """Build train/val/test loaders. ``train_subset`` overrides the train rows
    (for k-shot). Val/test always come from ``frame``'s own splits."""
    loaders = {}
    for split in ("train", "val", "test"):
        rows = train_subset if (split == "train" and train_subset is not None) \
            else frame[frame["split"] == split]
        if rows.empty:
            raise ValueError(f"empty split {split!r}")
        is_train = split == "train"
        tf = build_transform(aug, image_size, is_train, grayscale=False)
        common = dict(transform=tf, label_column="label", grayscale=False)
        if preload and len(rows) * image_size * image_size * 3 < 2e9:
            ds = PreloadedAksaraDataset(rows, class_to_idx, image_size=image_size,
                                        cache_dir=None, progress=False, **common)
        else:
            ds = AksaraDataset(rows, class_to_idx, **common)
        g = torch.Generator(); g.manual_seed(seed)
        loaders[split] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=is_train,
                                    num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                                    worker_init_fn=seed_worker, generator=g)
    return loaders


def load_backbone(target: nn.Module, source_state: dict):
    """Copy all weights that match by name and shape (i.e. the backbone; the
    classifier head differs in size and is left freshly initialized)."""
    tgt = target.state_dict()
    keep = {k: v for k, v in source_state.items() if k in tgt and v.shape == tgt[k].shape}
    tgt.update(keep)
    target.load_state_dict(tgt)
    return len(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--shots", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--epochs-base", type=int, default=30)
    ap.add_argument("--epochs-ft", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = args.out or args.artifacts / "results" / f"transfer_{args.held_out}"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    frame = load_split_frame(args.artifacts / "splits.csv", args.artifacts / "manifest.csv")
    if args.held_out not in set(frame["script"]):
        raise SystemExit(f"Unknown script {args.held_out!r}. Have: {sorted(set(frame['script']))}")
    base = frame[frame["script"] != args.held_out].reset_index(drop=True)
    held = frame[frame["script"] == args.held_out].reset_index(drop=True)
    print(f"held-out {args.held_out}: {len(held)} images, {held['label'].nunique()} classes | "
          f"base: {len(base)} images, {base['label'].nunique()} classes")

    # ---- 1. BASE: train on the 12 non-held-out scripts --------------------
    base_idx = build_class_index(base, "label")
    base_cfg = TrainConfig(epochs=args.epochs_base, batch_size=128, num_workers=4,
                           early_stopping_patience=6)
    base_model, _ = build_model("resnet18", len(base_idx), pretrained=True, image_size=args.image_size)
    print(f"\n[BASE] training resnet18 on {len(base_idx)} classes ({args.epochs_base} ep) ...")
    base_loaders = make_loaders(base, base_idx, args.image_size, "medium", base_cfg, device, args.seed)
    base_res, _, _ = train_model(base_model, base_loaders["train"], base_loaders["val"],
                                 base_loaders["test"], list(base_idx), base_cfg, device, progress=True)
    base_state = {k: v.cpu() for k, v in base_model.state_dict().items()}
    print(f"[BASE] test macro-F1 {base_res.test_metrics['macro_f1']*100:.2f}")

    # ---- 2. k-shot fine-tuning on the held-out script ---------------------
    held_idx = build_class_index(held, "label")
    ft_cfg = TrainConfig(epochs=args.epochs_ft, batch_size=64, lr=5e-4, num_workers=4,
                         early_stopping_patience=8)
    import pandas as pd
    held_train = held[held["split"] == "train"]
    rows = []
    for k in args.shots:
        # Concat per-group samples: preserves all columns across pandas versions
        # (groupby.apply with group_keys=False can drop the grouping column).
        kshot = pd.concat([g.sample(n=min(k, len(g)), random_state=args.seed)
                           for _, g in held_train.groupby("label")]).reset_index(drop=True)
        print(f"\n[k={k}] {len(kshot)} train images ({kshot['label'].nunique()} classes)")
        for init in ("transfer", "imagenet"):
            set_seed(args.seed)
            model, _ = build_model("resnet18", len(held_idx),
                                   pretrained=(init == "imagenet"), image_size=args.image_size)
            if init == "transfer":
                n = load_backbone(model, base_state)
                print(f"  transfer: loaded {n} backbone tensors from BASE")
            loaders = make_loaders(held, held_idx, args.image_size, "medium", ft_cfg, device,
                                   args.seed, train_subset=kshot)
            res, _, _ = train_model(model, loaders["train"], loaders["val"], loaders["test"],
                                    list(held_idx), ft_cfg, device, progress=False)
            m = res.test_metrics
            rows.append({"held_out": args.held_out, "k": k, "init": init,
                         "accuracy": m["accuracy"]*100, "macro_f1": m["macro_f1"]*100})
            print(f"  {init:9} k={k:3}: acc {m['accuracy']*100:.2f}  macro-F1 {m['macro_f1']*100:.2f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "transfer_results.csv", index=False)
    print("\n=== transfer vs imagenet (macro-F1) ===")
    piv = df.pivot(index="k", columns="init", values="macro_f1")
    piv["gain"] = piv["transfer"] - piv["imagenet"]
    print(piv.to_string())
    (out / "base_metrics.json").write_text(json.dumps(base_res.test_metrics, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

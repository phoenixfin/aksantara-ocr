"""Near-duplicate leakage audit.

Byte-hash dedup (01_prepare_data / 06_clean) only catches *identical* files. A
sheared, rescaled, or re-saved sibling of the same handwritten glyph has a
different hash but is effectively the same sample — and if it straddles the
train/test boundary it leaks the test item into training, inflating every
metric. This is a tighter, more damaging variant of writer leakage, and it is
invisible to a hash.

Method: embed every image, then for each TEST image find its highest cosine
similarity to a TRAIN image *of the same class* (augmentation preserves the
label, so a leaked twin is same-class). The distribution of that max-similarity
has a normal within-class body plus a high-similarity tail — the tail is the
leakage. Count how much of the test set sits above a threshold.

    python scripts/07_leakage_audit.py --artifacts artifacts_local_full --embedder resnet18

Embedders:
  hog       reuse cached HOG features if present (fast, but not geometry-
            invariant, so it is a LOWER bound — misses sheared/rotated twins)
  resnet18  ImageNet-pretrained ResNet-18 penultimate (512-d). More invariant,
            catches augmented siblings HOG cannot. --checkpoint loads a
            fine-tuned model instead (the ideal embedder).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksara.data.dataset import load_split_frame  # noqa: E402


def embed_cnn(paths, image_size, checkpoint=None, batch=128, progress=True):
    """Penultimate-layer embeddings from ResNet-18 (pretrained or fine-tuned)."""
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
    from tqdm.auto import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None if checkpoint else ResNet18_Weights.IMAGENET1K_V1)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        # tolerate a classifier head of a different size
        model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()
                               and v.shape == model.state_dict()[k].shape}, strict=False)
    model.fc = nn.Identity()  # penultimate = 512-d avgpool output
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    out = np.zeros((len(paths), 512), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch), disable=not progress, desc="embed"):
            chunk = paths[start:start + batch]
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            out[start:start + len(chunk)] = model(imgs).cpu().numpy()
    return out


def load_hog(artifacts, tr, te):
    d = Path(artifacts) / "results" / "classical"
    ftr = d / f"_feat_hog_64_train_{len(tr)}.npy"
    fte = d / f"_feat_hog_64_test_{len(te)}.npy"
    if not (ftr.exists() and fte.exists()):
        raise SystemExit(
            f"No cached HOG features at {d}. Run 03_run_classical.py first, or use "
            "--embedder resnet18."
        )
    return np.load(ftr), np.load(fte)


def max_same_class_sim(Xtr, Xte, tr_labels, te_labels):
    """For each test row, max cosine similarity to same-class train rows."""
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    Xte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    by = defaultdict(list)
    for i, l in enumerate(tr_labels):
        by[l].append(i)
    maxsim = np.full(len(Xte), -1.0, dtype=np.float32)
    argmax = np.full(len(Xte), -1, dtype=np.int64)
    for j in range(len(Xte)):
        idx = by.get(te_labels[j])
        if not idx:
            continue
        sims = Xtr[idx] @ Xte[j]
        k = int(np.argmax(sims))
        maxsim[j], argmax[j] = sims[k], idx[k]
    return maxsim, argmax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts_local_full"))
    ap.add_argument("--embedder", choices=["hog", "resnet18"], default="resnet18")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Fine-tuned ResNet-18 weights (the ideal embedder).")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.artifacts / f"leakage_{args.embedder}"
    out.mkdir(parents=True, exist_ok=True)

    frame = load_split_frame(args.artifacts / "splits.csv", args.artifacts / "manifest.csv")
    tr = frame[frame.split == "train"].reset_index(drop=True)
    te = frame[frame.split == "test"].reset_index(drop=True)
    print(f"{len(tr)} train, {len(te)} test, {frame['label'].nunique()} classes")

    if args.embedder == "hog":
        Xtr, Xte = load_hog(args.artifacts, tr, te)
    else:
        print(f"embedding with resnet18 ({'fine-tuned' if args.checkpoint else 'ImageNet-pretrained'}) ...")
        Xtr = embed_cnn(tr["path"].tolist(), args.image_size, args.checkpoint)
        Xte = embed_cnn(te["path"].tolist(), args.image_size, args.checkpoint)

    maxsim, argmax = max_same_class_sim(Xtr, Xte, tr["label"].values, te["label"].values)
    valid = maxsim[maxsim >= 0]

    print(f"\n== test->train same-class max cosine ({args.embedder}) ==")
    for p in (50, 75, 90, 95, 99):
        print(f"  {p}th pct: {np.percentile(valid, p):.3f}")
    print("\n  fraction of test set with a near-duplicate train twin:")
    for thr in (0.90, 0.95, 0.98, 0.99):
        n = int((valid >= thr).sum())
        print(f"    >= {thr}: {n:5} ({100 * n / len(valid):5.2f}% of test)")

    np.save(out / "maxsim.npy", maxsim)
    np.save(out / "argmax.npy", argmax)
    # dump the flagged pairs for review / for building a clean test set
    import csv
    thr = 0.95
    with (out / f"leaked_pairs_thr{thr}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_path", "train_twin_path", "label", "cosine"])
        for j in np.where(maxsim >= thr)[0]:
            w.writerow([te.iloc[j]["path"], tr.iloc[argmax[j]]["path"], te.iloc[j]["label"],
                        f"{maxsim[j]:.4f}"])
    print(f"\nflagged pairs (>= {thr}) -> {out / f'leaked_pairs_thr{thr}.csv'}")
    print(f"arrays -> {out}")
    print("\nUse leaked_pairs CSV to build a near-dup-clean test set and re-measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

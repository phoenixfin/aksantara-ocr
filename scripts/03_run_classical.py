"""Classical (HOG/pixel + sklearn) baselines on the same splits as the deep models.

Kept separate from the deep matrix because these need no GPU and no epochs —
they run to completion on a CPU-only machine in minutes.

    python scripts/03_run_classical.py --task unified
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksara.data.dataset import load_split_frame  # noqa: E402
from aksara.engine import metrics as metrics_mod  # noqa: E402
from aksara.models.classical import DEFAULT_MODELS, build_classical, extract_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--task", choices=["unified", "script_id"], default="unified")
    # HOG only by default: it is the meaningful representation, and adding raw
    # pixels doubles the (already hours-long) runtime for a weaker number.
    parser.add_argument("--features", nargs="+", default=["hog"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Cap training samples (stratified subsample). Use to bound runtime "
             "on a CPU; results are reported with the actual n used.",
    )
    args = parser.parse_args()

    frame = load_split_frame(args.artifacts / "splits.csv", args.artifacts / "manifest.csv")
    label_column = "label" if args.task == "unified" else "script"
    class_names = sorted(frame[label_column].unique())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    out_dir = args.artifacts / "results" / "classical"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subsample the TRAIN split up front, before extraction, so a cap actually
    # saves the (expensive) feature extraction rather than only the fit. One-vs-
    # rest linear models train one classifier per class, so at 889 classes the
    # full 70k train set is intractable on a CPU; a stratified cap keeps every
    # class present and is a standard, reportable choice for a reference baseline.
    split_frames = {"test": frame[frame["split"] == "test"]}
    train_frame = frame[frame["split"] == "train"]
    if args.max_train and len(train_frame) > args.max_train:
        from sklearn.model_selection import train_test_split
        train_frame, _ = train_test_split(
            train_frame, train_size=args.max_train,
            stratify=train_frame[label_column], random_state=42
        )
        print(f"train subsampled to {len(train_frame)} (stratified) for tractability")
    split_frames["train"] = train_frame

    rows = []
    for feature in args.features:
        # Extraction is the expensive part (~15 min for HOG over 83k images) and
        # is model-independent. Cache to disk keyed by (feature, size, split, n)
        # so re-runs and each model reuse it instead of re-extracting.
        cache = {}
        for split, subset in split_frames.items():
            y = np.array([class_to_idx[v] for v in subset[label_column]])
            cache_file = out_dir / f"_feat_{feature}_{args.image_size}_{split}_{len(subset)}.npy"
            if cache_file.exists():
                print(f"  reusing cached {feature} features for {split}: {cache_file.name}")
                x = np.load(cache_file)
            else:
                x = extract_features(subset["path"].tolist(), args.image_size, feature)
                np.save(cache_file, x)
            cache[split] = (x, y)

        x_train, y_train = cache["train"]
        x_test, y_test = cache["test"]
        print(f"{feature}: train={x_train.shape} test={x_test.shape}")

        for model_name in args.models:
            for seed in args.seeds:
                print(f"  fitting {model_name} (seed={seed}) ...")
                model = build_classical(model_name, seed=seed)
                model.fit(x_train, y_train)

                # Not all sklearn estimators expose predict_proba; fall back to
                # one-hot so the shared metrics code still works (top-5 becomes
                # equal to top-1 for those, which is noted in the output).
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(x_test)
                    has_scores = True
                else:
                    preds = model.predict(x_test)
                    scores = np.eye(len(class_names))[preds]
                    has_scores = False

                result = metrics_mod.compute_metrics(scores, y_test, class_names)
                rows.append(
                    {
                        "model": f"{feature}_{model_name}",
                        "feature": feature,
                        "classifier": model_name,
                        "task": args.task,
                        "image_size": args.image_size,
                        "seed": seed,
                        "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"],
                        "balanced_accuracy": result["balanced_accuracy"],
                        "top5_accuracy": result["top5_accuracy"] if has_scores else None,
                    }
                )
                print(f"    acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")
                (out_dir / f"{feature}_{model_name}_s{seed}.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "classical_results.csv", index=False)
    print(f"\n-> {out_dir / 'classical_results.csv'}")
    print(table.sort_values("macro_f1", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

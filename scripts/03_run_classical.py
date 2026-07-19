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
from aksara.models.classical import build_classical, extract_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--task", choices=["unified", "script_id"], default="unified")
    parser.add_argument("--features", nargs="+", default=["hog", "pixels"])
    parser.add_argument("--models", nargs="+", default=["svm_rbf", "knn", "random_forest", "logreg"])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    frame = load_split_frame(args.artifacts / "splits.csv", args.artifacts / "manifest.csv")
    label_column = "label" if args.task == "unified" else "script"
    class_names = sorted(frame[label_column].unique())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    out_dir = args.artifacts / "results" / "classical"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for feature in args.features:
        # Feature extraction is the expensive part and is model-independent,
        # so it happens once per feature type rather than once per (feature, model).
        cache = {}
        for split in ("train", "test"):
            subset = frame[frame["split"] == split]
            cache[split] = (
                extract_features(subset["path"].tolist(), args.image_size, feature),
                np.array([class_to_idx[v] for v in subset[label_column]]),
            )

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

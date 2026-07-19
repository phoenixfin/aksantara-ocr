"""Train/val/test splitting.

Splits are computed **once**, written to disk, and reused by every model in the
experiment matrix. This is what makes cross-model comparisons in the paper
honest — otherwise a model can win by getting a luckier split.

Two strategies:

``stratified``
    Class-balanced random split. Correct only when each image is an independent
    sample. If several images of one character were written by the same person,
    this leaks writer style across the split boundary and inflates test scores.

``grouped``
    Class-balanced *and* writer-disjoint: no writer appears in more than one
    fold. This measures what a dataset paper should claim — generalization to
    unseen handwriting. Requires ``writer_id`` in the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

SPLIT_NAMES = ("train", "val", "test")


def _fractions_to_folds(val_frac: float, test_frac: float) -> tuple[int, int, int]:
    """Express the requested fractions as k-fold counts.

    Using k-fold machinery (rather than shuffling indices) keeps the split
    class-balanced even for characters with very few samples.
    """
    if not 0 < test_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("val_frac and test_frac must each be in (0, 1).")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must leave room for a training set.")

    outer_k = int(round(1 / test_frac))
    # val_frac is expressed relative to the whole dataset, so rescale it to the
    # train+val remainder that the inner split actually operates on.
    inner_frac = val_frac / (1 - test_frac)
    inner_k = int(round(1 / inner_frac))
    return outer_k, inner_k, 0


def _preflight(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    strategy: str,
    outer_k: int,
    inner_k: int,
    val_frac: float,
    test_frac: float,
    label_column: str,
) -> None:
    """Check the split is achievable before handing off to sklearn.

    Both k-fold splitters require at least ``n_splits`` members per stratum, and
    the grouped variant additionally needs at least ``n_splits`` distinct
    writers. Violating either raises deep inside sklearn with a message that
    names neither the offending class nor the fraction that caused it — so the
    conditions are checked here where an actionable error can be produced.
    """
    k = max(outer_k, inner_k)
    label_counts = pd.Series(labels).value_counts()
    thin = label_counts[label_counts < k]

    if len(thin):
        smallest = int(label_counts.min())
        max_frac = 1 / smallest if smallest else 1.0
        raise ValueError(
            f"Cannot build a {1 - val_frac - test_frac:.0%}/{val_frac:.0%}/{test_frac:.0%} "
            f"split: it needs at least {k} samples of every class, but "
            f"{len(thin)} classes have fewer (smallest: '{thin.index[0]}' with "
            f"{int(thin.iloc[0])}).\n"
            f"Options:\n"
            f"  - collect more samples for the thin classes (best);\n"
            f"  - raise --val-frac/--test-frac to at most {max_frac:.2f} each;\n"
            f"  - drop classes below a minimum count before splitting."
        )

    if strategy == "grouped":
        n_writers = manifest["writer_id"].nunique()
        if n_writers < k:
            raise ValueError(
                f"Writer-disjoint splitting needs at least {k} distinct writers "
                f"to carve {1 / test_frac:.0f} folds, but the dataset has {n_writers}.\n"
                f"Options:\n"
                f"  - raise --val-frac/--test-frac to at most {1 / n_writers:.2f} each;\n"
                f"  - use --split-strategy stratified and report the writer-leakage "
                f"limitation in the paper."
            )

        # A writer who wrote every sample of some character makes that character
        # unlearnable under a disjoint split — it lands wholly in one fold.
        per_class_writers = manifest.groupby(label_column)["writer_id"].nunique()
        single = per_class_writers[per_class_writers < 2]
        if len(single):
            raise ValueError(
                f"{len(single)} classes have samples from only one writer "
                f"(e.g. '{single.index[0]}'). Under writer-disjoint splitting those "
                f"classes land entirely in one fold and can never be both trained "
                f"and tested.\n"
                f"Options:\n"
                f"  - collect those characters from more writers;\n"
                f"  - drop them before splitting;\n"
                f"  - use --split-strategy stratified (and state the limitation)."
            )


def make_splits(
    manifest: pd.DataFrame,
    strategy: str = "grouped",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    label_column: str = "label",
) -> pd.Series:
    """Return a Series of split assignments aligned to ``manifest``'s index."""
    if strategy not in {"stratified", "grouped"}:
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    labels = manifest[label_column].to_numpy()
    outer_k, inner_k, _ = _fractions_to_folds(val_frac, test_frac)
    _preflight(manifest, labels, strategy, outer_k, inner_k, val_frac, test_frac, label_column)

    if strategy == "grouped":
        if "writer_id" not in manifest.columns or manifest["writer_id"].isna().all():
            raise ValueError(
                "strategy='grouped' requires a populated 'writer_id' column. "
                "No writer ids were parsed from the dataset. Either encode writer "
                "ids in filenames (e.g. 'w03_sample01.png') or switch to "
                "strategy='stratified' and state the limitation in the paper."
            )
        if manifest["writer_id"].isna().any():
            missing = int(manifest["writer_id"].isna().sum())
            raise ValueError(
                f"{missing} rows have no writer_id. Grouped splitting needs a writer "
                "for every sample; fix the filenames or drop those rows explicitly."
            )
        groups = manifest["writer_id"].to_numpy()
        outer = StratifiedGroupKFold(n_splits=outer_k, shuffle=True, random_state=seed)
        inner_cls = StratifiedGroupKFold
    else:
        groups = None
        outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)
        inner_cls = StratifiedKFold

    split = pd.Series("train", index=manifest.index, dtype=object)

    trainval_idx, test_idx = next(iter(outer.split(np.zeros(len(labels)), labels, groups)))
    split.iloc[test_idx] = "test"

    inner = inner_cls(n_splits=inner_k, shuffle=True, random_state=seed)
    inner_groups = groups[trainval_idx] if groups is not None else None
    sub_train_idx, sub_val_idx = next(
        iter(inner.split(np.zeros(len(trainval_idx)), labels[trainval_idx], inner_groups))
    )
    split.iloc[trainval_idx[sub_val_idx]] = "val"

    return split


def summarize_splits(manifest: pd.DataFrame, split: pd.Series) -> dict:
    """Diagnostics worth printing before you trust a split."""
    summary = {
        "counts": split.value_counts().to_dict(),
        "classes_per_split": {
            name: int(manifest.loc[split == name, "label"].nunique()) for name in SPLIT_NAMES
        },
        "total_classes": int(manifest["label"].nunique()),
    }

    # A class missing from train is unlearnable; missing from test is unmeasured.
    train_classes = set(manifest.loc[split == "train", "label"])
    summary["classes_absent_from_train"] = sorted(
        set(manifest["label"]) - train_classes
    )

    if "writer_id" in manifest.columns and manifest["writer_id"].notna().any():
        writers = {
            name: set(manifest.loc[split == name, "writer_id"].dropna())
            for name in SPLIT_NAMES
        }
        summary["writer_overlap"] = {
            "train_test": sorted(writers["train"] & writers["test"]),
            "train_val": sorted(writers["train"] & writers["val"]),
            "val_test": sorted(writers["val"] & writers["test"]),
        }
    return summary


def save_splits(path: Path, manifest: pd.DataFrame, split: pd.Series, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = manifest[["path", "label"]].copy()
    out["split"] = split
    out.to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

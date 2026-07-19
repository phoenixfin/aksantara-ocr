"""Evaluation metrics.

Macro-F1 is the headline number, not accuracy: character classes in handwritten
corpora are rarely balanced, and accuracy on an imbalanced test set flatters
models that ignore rare characters.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def top_k_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    k = min(k, logits.shape[1])
    top_k = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([t in row for t, row in zip(targets, top_k)]))


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
) -> dict:
    preds = logits.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=np.arange(len(class_names)), zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(targets, preds)),
        "top5_accuracy": top_k_accuracy(logits, targets, k=5),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }


def confusion(logits: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(targets, logits.argmax(axis=1), labels=np.arange(num_classes))


def most_confused_pairs(cm: np.ndarray, class_names: list[str], top_n: int = 20) -> list[dict]:
    """Off-diagonal hot spots.

    For a script dataset this is the qualitative result readers actually want:
    which character pairs are genuinely hard to tell apart by hand.
    """
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true": class_names[i],
                        "predicted": class_names[j],
                        "count": int(cm[i, j]),
                        "rate": float(cm[i, j] / max(cm[i].sum(), 1)),
                    }
                )
    return sorted(pairs, key=lambda p: -p["count"])[:top_n]


def text_report(logits: np.ndarray, targets: np.ndarray, class_names: list[str]) -> str:
    return classification_report(
        targets,
        logits.argmax(axis=1),
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

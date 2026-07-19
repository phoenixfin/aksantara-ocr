"""Turn the raw results table into paper-ready aggregates.

Everything here reports mean +/- std across seeds. A single-seed number in a
benchmark table is not a result — it is one sample from a distribution whose
spread, for small character datasets, is routinely larger than the gaps between
the models being compared.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GROUP_KEYS = ["model", "task", "image_size", "augmentation", "pretrained", "script_filter"]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "top5_accuracy"]


def aggregate_seeds(results: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Collapse the seed axis into mean/std/n per configuration."""
    metrics = metrics or METRICS
    keys = [k for k in GROUP_KEYS if k in results.columns]

    # script_filter is None for non-per-script runs; groupby drops NaN keys by
    # default, which would silently discard most of the table.
    frame = results.copy()
    if "script_filter" in frame.columns:
        frame["script_filter"] = frame["script_filter"].fillna("-")

    agg = frame.groupby(keys, dropna=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        n_seeds=("seed", "nunique"),
        params=("num_params", "first"),
        seconds=("train_seconds", "mean"),
    ).reset_index()

    return agg.sort_values("macro_f1_mean", ascending=False)


def format_pm(mean: float, std: float, decimals: int = 2, scale: float = 100.0) -> str:
    """'97.31 ± 0.42'. NaN std (single seed) renders without the ± term."""
    if pd.isna(std):
        return f"{mean * scale:.{decimals}f}"
    return f"{mean * scale:.{decimals}f} ± {std * scale:.{decimals}f}"


def main_benchmark_table(results: pd.DataFrame, task: str = "unified") -> pd.DataFrame:
    """Table 1 of the paper: every model on the headline task."""
    agg = aggregate_seeds(results[results["task"] == task])
    if agg.empty:
        return agg

    return pd.DataFrame(
        {
            "Model": agg["model"],
            "Pretrained": agg["pretrained"].map({True: "Yes", False: "No"}),
            "Params (M)": (agg["params"] / 1e6).round(2),
            "Accuracy (%)": [format_pm(m, s) for m, s in zip(agg["accuracy_mean"], agg["accuracy_std"])],
            "Macro-F1 (%)": [format_pm(m, s) for m, s in zip(agg["macro_f1_mean"], agg["macro_f1_std"])],
            "Top-5 (%)": [format_pm(m, s) for m, s in zip(agg["top5_accuracy_mean"], agg["top5_accuracy_std"])],
            "Seeds": agg["n_seeds"],
        }
    )


def ablation_table(results: pd.DataFrame, axis: str, task: str = "unified") -> pd.DataFrame:
    """Pivot one ablation axis (``augmentation``, ``image_size``, ``pretrained``)
    against models, holding everything else pooled."""
    if axis not in results.columns:
        raise KeyError(f"No such ablation axis: {axis!r}")

    subset = results[results["task"] == task]
    pivot = subset.pivot_table(index="model", columns=axis, values="macro_f1", aggfunc="mean")
    return (pivot * 100).round(2)


def per_script_table(results: pd.DataFrame) -> pd.DataFrame:
    """How hard is each script? Directly answers 'is this dataset uniform?'"""
    subset = results[results["task"] == "per_script"]
    if subset.empty:
        return pd.DataFrame()

    pivot = subset.pivot_table(index="script_filter", columns="model", values="macro_f1", aggfunc="mean")
    pivot = (pivot * 100).round(2)
    pivot["mean"] = pivot.mean(axis=1).round(2)
    return pivot.sort_values("mean")


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    body = table.to_latex(index=False, escape=True, column_format="l" * len(table.columns))
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )


def write_all(results: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "raw_results.csv", index=False)
    aggregate_seeds(results).to_csv(out_dir / "aggregated.csv", index=False)

    tables = {
        "main_benchmark": (main_benchmark_table(results), "Classification performance on the unified task."),
        "ablation_augmentation": (ablation_table(results, "augmentation"), "Macro-F1 (\\%) by augmentation policy."),
        "ablation_image_size": (ablation_table(results, "image_size"), "Macro-F1 (\\%) by input resolution."),
        "ablation_pretrained": (ablation_table(results, "pretrained"), "Macro-F1 (\\%) with and without ImageNet initialization."),
        "per_script": (per_script_table(results), "Per-script macro-F1 (\\%)."),
    }

    for name, (table, caption) in tables.items():
        if table is None or table.empty:
            continue
        reset = table.reset_index() if table.index.name else table
        reset.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.tex").write_text(to_latex(reset, caption, name), encoding="utf-8")

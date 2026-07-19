"""Experiment matrix expansion and execution.

An *experiment* is one (model, task, image_size, augmentation, pretrained, seed)
point. The runner expands a matrix spec into experiments, skips ones already on
disk, and appends each result to a single results directory.

Every run is resumable. Colab and Kaggle sessions get killed mid-matrix as a
matter of routine, and a matrix that has to restart from zero is unusable there.
"""

from __future__ import annotations

import itertools
import json
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import AksaraDataset, build_class_index
from ..data.transforms import build_transform
from ..engine import metrics as metrics_mod
from ..engine.train import TrainConfig, train_model
from ..models.registry import REGISTRY, build_model
from ..utils.seed import seed_worker, set_seed


@dataclass(frozen=True)
class Experiment:
    model: str
    task: str = "unified"      # unified | script_id | per_script
    image_size: int = 64
    augmentation: str = "medium"
    pretrained: bool = True
    seed: int = 0
    script_filter: str | None = None  # set when task == "per_script"

    @property
    def run_id(self) -> str:
        parts = [
            self.model,
            self.task,
            f"sz{self.image_size}",
            f"aug-{self.augmentation}",
            "pt" if self.pretrained else "scratch",
            f"s{self.seed}",
        ]
        if self.script_filter:
            parts.insert(2, self.script_filter)
        return "__".join(parts)


@dataclass
class MatrixSpec:
    models: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=lambda: ["unified"])
    image_sizes: list[int] = field(default_factory=lambda: [64])
    augmentations: list[str] = field(default_factory=lambda: ["medium"])
    pretrained: list[bool] = field(default_factory=lambda: [True])
    seeds: list[int] = field(default_factory=lambda: [0])
    scripts: list[str] = field(default_factory=list)


def expand(spec: MatrixSpec) -> list[Experiment]:
    """Cross-product the matrix, dropping combinations that are meaningless.

    Two kinds get filtered:
      - ``pretrained=True`` for from-scratch architectures (no weights exist)
      - non-224 image sizes for transformers (fixed positional embeddings)
    Emitting them and letting them crash would pollute the results table with
    failures that look like real negative findings.

    **Ordering is seed-major**: every configuration is run at seed 0 before any
    runs at seed 1. When a session dies partway through — the normal case on a
    hosted runtime — this leaves one complete seed across the whole matrix
    rather than three seeds across an arbitrary prefix of it. A single-seed
    complete table is a usable result; a three-seed partial table is not.
    """
    experiments: list[Experiment] = []

    for seed, model, task, size, aug, pretrained in itertools.product(
        spec.seeds, spec.models, spec.tasks, spec.image_sizes, spec.augmentations, spec.pretrained
    ):
        if model not in REGISTRY:
            raise KeyError(f"Matrix references unknown model {model!r}")
        model_spec = REGISTRY[model]

        if model_spec.family == "scratch" and pretrained:
            continue
        if model_spec.requires_fixed_size and size != model_spec.default_image_size:
            continue

        if task == "per_script":
            if not spec.scripts:
                raise ValueError("task 'per_script' requires spec.scripts to be populated.")
            for script in spec.scripts:
                experiments.append(
                    Experiment(model, task, size, aug, pretrained, seed, script_filter=script)
                )
        else:
            experiments.append(Experiment(model, task, size, aug, pretrained, seed))

    return experiments


def _task_columns(task: str) -> tuple[str, str | None]:
    """(label_column, filter_column) for a task."""
    if task == "unified":
        return "label", None
    if task == "script_id":
        return "script", None
    if task == "per_script":
        return "character", "script"
    raise ValueError(f"Unknown task: {task!r}")


def run_one(
    exp: Experiment,
    frame: pd.DataFrame,
    train_cfg: TrainConfig,
    device: torch.device,
    results_dir: Path,
    progress: bool = True,
) -> dict:
    """Train and evaluate a single experiment; write artifacts; return a summary row."""
    set_seed(exp.seed)
    label_column, filter_column = _task_columns(exp.task)

    data = frame
    if filter_column and exp.script_filter:
        data = data[data[filter_column] == exp.script_filter]
        if data.empty:
            raise ValueError(f"No rows for script {exp.script_filter!r}")

    class_to_idx = build_class_index(data, label_column)
    class_names = list(class_to_idx)
    grayscale = REGISTRY[exp.model].grayscale

    loaders = {}
    for split in ("train", "val", "test"):
        subset = data[data["split"] == split]
        if subset.empty:
            raise ValueError(f"Split {split!r} is empty for {exp.run_id}")

        is_train = split == "train"
        dataset = AksaraDataset(
            subset,
            class_to_idx,
            transform=build_transform(exp.augmentation, exp.image_size, is_train, grayscale),
            label_column=label_column,
            grayscale=grayscale,
        )
        generator = torch.Generator()
        generator.manual_seed(exp.seed)
        loaders[split] = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=is_train,
            num_workers=train_cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    model, _ = build_model(
        exp.model,
        num_classes=len(class_names),
        pretrained=exp.pretrained,
        image_size=exp.image_size,
    )

    result, test_logits, test_targets = train_model(
        model, loaders["train"], loaders["val"], loaders["test"],
        class_names, train_cfg, device, progress=progress,
    )

    run_dir = results_dir / exp.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cm = metrics_mod.confusion(test_logits, test_targets, len(class_names))
    np.save(run_dir / "confusion_matrix.npy", cm)

    # Save the image path for every prediction row. Without it, predictions from
    # different runs cannot be joined per-image, which is exactly what
    # hierarchical (script -> character) evaluation requires. DataLoader runs
    # with shuffle=False on test, so row order matches the frame order here.
    test_paths = data[data["split"] == "test"]["path"].to_numpy()
    if len(test_paths) != len(test_targets):
        raise RuntimeError(
            f"Path/prediction length mismatch ({len(test_paths)} vs {len(test_targets)}) "
            f"for {exp.run_id} — cannot align predictions to images."
        )
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        logits=test_logits,
        targets=test_targets,
        paths=test_paths,
        class_names=np.array(class_names),
    )
    (run_dir / "classification_report.txt").write_text(
        metrics_mod.text_report(test_logits, test_targets, class_names), encoding="utf-8"
    )

    payload = {
        "experiment": asdict(exp),
        "run_id": exp.run_id,
        "num_classes": len(class_names),
        "class_names": class_names,
        "n_train": int((data["split"] == "train").sum()),
        "n_val": int((data["split"] == "val").sum()),
        "n_test": int((data["split"] == "test").sum()),
        "test_metrics": result.test_metrics,
        "best_val_metrics": result.best_val_metrics,
        "history": result.history,
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "train_seconds": result.train_seconds,
        "num_params": result.num_params,
        "most_confused": metrics_mod.most_confused_pairs(cm, class_names),
        "train_config": asdict(train_cfg),
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
        },
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_matrix(
    experiments: list[Experiment],
    frame: pd.DataFrame,
    train_cfg: TrainConfig,
    device: torch.device,
    results_dir: Path,
    skip_existing: bool = True,
    progress: bool = True,
    time_budget_hours: float | None = None,
) -> pd.DataFrame:
    """Run the matrix, optionally stopping before ``time_budget_hours`` elapses.

    The budget is checked between runs, using the mean duration so far to
    predict whether the next run fits. Stopping one run early is much better
    than being killed mid-run on a runtime with no persistent storage, where an
    interrupted run leaves nothing behind at all.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    failures_path = results_dir / "failures.jsonl"
    budget_seconds = time_budget_hours * 3600 if time_budget_hours else None
    started = time.time()
    durations: list[float] = []

    for i, exp in enumerate(experiments, start=1):
        run_dir = results_dir / exp.run_id
        if skip_existing and (run_dir / "result.json").exists():
            print(f"[{i}/{len(experiments)}] skip (done): {exp.run_id}")
            continue

        if budget_seconds is not None and durations:
            elapsed = time.time() - started
            predicted = sum(durations) / len(durations)
            if elapsed + predicted > budget_seconds:
                remaining = len(experiments) - i + 1
                print(
                    f"\nStopping: {elapsed / 3600:.2f}h elapsed of {time_budget_hours}h budget, "
                    f"and the next run needs ~{predicted / 60:.1f}min.\n"
                    f"{remaining} run(s) not started. Completed results are intact."
                )
                break

        print(f"[{i}/{len(experiments)}] run: {exp.run_id}")
        run_started = time.time()
        try:
            payload = run_one(exp, frame, train_cfg, device, results_dir, progress)
            m = payload["test_metrics"]
            print(f"    acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
            durations.append(time.time() - run_started)
            # Rewrite the rollup after every run. On a runtime with no persistent
            # storage the notebook's cell output is the only thing the user
            # reliably keeps, so results must never exist solely in memory.
            collect_results(results_dir).to_csv(results_dir / "results_running.csv", index=False)
        except Exception as exc:  # noqa: BLE001 — one bad config must not kill the matrix
            print(f"    FAILED: {exc}")
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "run_id": exp.run_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }) + "\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if durations:
        print(
            f"\n{len(durations)} run(s) in {(time.time() - started) / 3600:.2f}h "
            f"(mean {sum(durations) / len(durations) / 60:.1f}min/run)"
        )
    return collect_results(results_dir)


def collect_results(results_dir: Path) -> pd.DataFrame:
    """Flatten every result.json under ``results_dir`` into one table."""
    rows = []
    for result_file in sorted(Path(results_dir).glob("*/result.json")):
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        exp = payload["experiment"]
        test = payload["test_metrics"]
        rows.append(
            {
                "run_id": payload["run_id"],
                **{k: v for k, v in exp.items()},
                "num_classes": payload["num_classes"],
                "n_train": payload["n_train"],
                "n_test": payload["n_test"],
                "accuracy": test["accuracy"],
                "balanced_accuracy": test["balanced_accuracy"],
                "macro_f1": test["macro_f1"],
                "weighted_f1": test["weighted_f1"],
                "top5_accuracy": test["top5_accuracy"],
                "cohen_kappa": test["cohen_kappa"],
                "val_macro_f1": payload["best_val_metrics"].get("macro_f1"),
                "num_params": payload["num_params"],
                "epochs_run": payload["epochs_run"],
                "train_seconds": payload["train_seconds"],
            }
        )
    return pd.DataFrame(rows)

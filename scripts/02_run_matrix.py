"""Execute an experiment matrix defined in YAML.

Resumable: completed runs are detected by their result.json and skipped, so a
killed Colab session is restarted by re-running the identical command.

    python scripts/02_run_matrix.py --config configs/full_benchmark.yaml
    python scripts/02_run_matrix.py --config configs/smoke_test.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aksara.data.dataset import load_split_frame  # noqa: E402
from aksara.engine.train import TrainConfig  # noqa: E402
from aksara.experiments.runner import MatrixSpec, expand, run_matrix  # noqa: E402
from aksara.reporting.tables import write_all  # noqa: E402


def _filter_kwargs(cls, data: dict) -> dict:
    """Drop unknown YAML keys with a warning rather than a TypeError deep in a
    2-hour run."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        print(f"  ignoring unknown {cls.__name__} keys: {sorted(unknown)}")
    return {k: v for k, v in data.items() if k in known}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="List experiments and exit.")
    parser.add_argument("--rerun", action="store_true", help="Re-run even if results exist.")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Wall-clock budget in hours. Stops cleanly between runs rather than "
             "being killed mid-run. Essential on runtimes without persistent storage.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results_dir = args.results or args.artifacts / "results" / args.config.stem

    frame = load_split_frame(args.artifacts / "splits.csv", args.artifacts / "manifest.csv")
    print(f"Loaded {len(frame)} images, {frame['label'].nunique()} classes, "
          f"{frame['script'].nunique()} scripts.")

    matrix_cfg = dict(config.get("matrix", {}))
    if matrix_cfg.get("scripts") == "all":
        matrix_cfg["scripts"] = sorted(frame["script"].unique())
    if matrix_cfg.get("models") == "all":
        from aksara.models.registry import REGISTRY
        matrix_cfg["models"] = sorted(REGISTRY)

    spec = MatrixSpec(**_filter_kwargs(MatrixSpec, matrix_cfg))
    train_cfg = TrainConfig(**_filter_kwargs(TrainConfig, config.get("train", {})))

    experiments = expand(spec)
    print(f"\nMatrix expands to {len(experiments)} experiments -> {results_dir}")

    if args.dry_run:
        for exp in experiments:
            print(f"  {exp.run_id}")
        est_min = len(experiments) * train_cfg.epochs * 0.5
        print(f"\n(very rough estimate: {est_min:.0f} GPU-minutes at ~30s/epoch)")
        return 0

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  WARNING: no GPU detected. The transformer runs will be impractically slow.")

    results = run_matrix(
        experiments,
        frame,
        train_cfg,
        device,
        results_dir,
        skip_existing=not args.rerun,
        time_budget_hours=args.time_budget,
    )

    if results.empty:
        print("\nNo results produced. Check failures.jsonl in the results directory.")
        return 1

    report_dir = results_dir / "report"
    write_all(results, report_dir)
    print(f"\n{len(results)} runs complete. Tables -> {report_dir}")

    top = results.sort_values("macro_f1", ascending=False).head(10)
    print("\nTop 10 by test macro-F1:")
    for _, row in top.iterrows():
        print(f"  {row['macro_f1']:.4f}  {row['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

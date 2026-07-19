"""End-to-end evaluation of the layered classifier.

The layered system is: stage 1 predicts the script, stage 2 applies that
script's character model. Reporting the two stages separately overstates the
system — stage-2 accuracy is measured with the true script handed to it, which
the deployed pipeline never has.

    python scripts/05_hierarchical_eval.py --results artifacts/results/tier1

**Why this is exactly computable.** Labels are script-qualified, so an image
routed to the wrong script can never produce the correct label no matter what
stage 2 says. End-to-end correctness is therefore exactly

    stage1(x) == script(x)   AND   model_{script(x)}(x) == character(x)

and both terms are already measured: stage 2's model for script S is evaluated
on every test image of S. No extra inference is needed, and nothing is
approximated.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(run_dir: Path) -> dict | None:
    """Load one run's predictions, keyed by image path."""
    npz_path = run_dir / "test_predictions.npz"
    result_path = run_dir / "result.json"
    if not npz_path.exists() or not result_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=True)
    if "paths" not in data:
        return None  # produced before path saving existed

    result = json.loads(result_path.read_text(encoding="utf-8"))
    class_names = [str(c) for c in data["class_names"]]
    predictions = data["logits"].argmax(axis=1)

    return {
        "experiment": result["experiment"],
        "paths": [str(p) for p in data["paths"]],
        "predicted": [class_names[i] for i in predictions],
        "true": [class_names[i] for i in data["targets"]],
    }


def collect(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """Split runs into script-identification and per-script character runs."""
    script_runs, character_runs = [], []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run is None:
            continue
        task = run["experiment"].get("task")
        if task == "script_id":
            script_runs.append(run)
        elif task == "per_script":
            character_runs.append(run)
    return script_runs, character_runs


def evaluate(script_run: dict, character_runs: list[dict]) -> dict:
    """Combine one stage-1 run with the matching stage-2 runs."""
    stage1 = dict(zip(script_run["paths"], script_run["predicted"]))
    true_script = dict(zip(script_run["paths"], script_run["true"]))

    # Stage 2: one model per script. Each contributes predictions only for its
    # own script's test images, which is exactly the correctly-routed subset.
    stage2: dict[str, str] = {}
    covered_scripts = set()
    for run in character_runs:
        covered_scripts.add(run["experiment"].get("script_filter"))
        stage2.update(zip(run["paths"], run["predicted"]))
    true_character = {}
    for run in character_runs:
        true_character.update(zip(run["paths"], run["true"]))

    rows = []
    for path, predicted_script in stage1.items():
        actual_script = true_script[path]
        routed = predicted_script == actual_script
        # Stage 2 is only consulted when routing was correct; otherwise the
        # script-qualified label is already unreachable.
        char_correct = routed and stage2.get(path) == true_character.get(path)
        rows.append(
            {
                "path": path,
                "true_script": actual_script,
                "predicted_script": predicted_script,
                "routed_correctly": routed,
                "end_to_end_correct": bool(char_correct),
                "stage2_available": path in stage2,
            }
        )

    frame = pd.DataFrame(rows)
    missing = frame[~frame["stage2_available"] & frame["routed_correctly"]]

    stage1_acc = float(frame["routed_correctly"].mean())
    end_to_end = float(frame["end_to_end_correct"].mean())
    routed = frame[frame["routed_correctly"]]
    stage2_given_routed = float(routed["end_to_end_correct"].mean()) if len(routed) else float("nan")

    per_script = (
        frame.groupby("true_script")
        .agg(
            n=("path", "size"),
            script_acc=("routed_correctly", "mean"),
            end_to_end=("end_to_end_correct", "mean"),
        )
        .reset_index()
        .sort_values("end_to_end")
    )

    return {
        "stage1_script_accuracy": stage1_acc,
        "stage2_accuracy_given_correct_routing": stage2_given_routed,
        "end_to_end_accuracy": end_to_end,
        "error_from_routing": stage1_acc - end_to_end,
        "n_test_images": len(frame),
        "scripts_covered_by_stage2": sorted(s for s in covered_scripts if s),
        "images_missing_stage2": int(len(missing)),
        "per_script": per_script.to_dict("records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    script_runs, character_runs = collect(args.results)
    if not script_runs:
        raise SystemExit(
            f"No script_id runs with saved paths under {args.results}.\n"
            "Run a config with task 'script_id' first (configs/layered.yaml)."
        )
    if not character_runs:
        raise SystemExit(
            f"No per_script runs with saved paths under {args.results}.\n"
            "Run a config with task 'per_script' first."
        )

    out_dir = args.out or args.results / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for script_run in script_runs:
        seed = script_run["experiment"].get("seed")
        # Pair stage 1 with the stage-2 runs from the same seed, so the reported
        # system is one coherent pipeline rather than a mix across seeds.
        matched = [r for r in character_runs if r["experiment"].get("seed") == seed]
        if not matched:
            print(f"seed {seed}: no matching per_script runs — skipped")
            continue

        summary = evaluate(script_run, matched)
        summary["seed"] = seed
        summary["stage1_model"] = script_run["experiment"].get("model")
        summaries.append(summary)

        print(f"\n{'=' * 66}\nseed {seed}  (stage 1: {summary['stage1_model']})\n{'=' * 66}")
        print(f"  stage 1 — script accuracy      : {summary['stage1_script_accuracy']:.4f}")
        print(f"  stage 2 — given correct routing: {summary['stage2_accuracy_given_correct_routing']:.4f}")
        print(f"  END-TO-END                     : {summary['end_to_end_accuracy']:.4f}")
        print(f"  accuracy lost to misrouting    : {summary['error_from_routing']:.4f}")
        if summary["images_missing_stage2"]:
            print(f"  WARNING: {summary['images_missing_stage2']} correctly-routed images have no "
                  "stage-2 prediction (a per_script run is missing); they count as errors.")

        per_script = pd.DataFrame(summary["per_script"])
        print("\n  per script (sorted by end-to-end):")
        print(per_script.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not summaries:
        raise SystemExit("No seed had both stages available.")

    frame = pd.DataFrame([
        {k: v for k, v in s.items() if k not in ("per_script", "scripts_covered_by_stage2")}
        for s in summaries
    ])
    frame.to_csv(out_dir / "hierarchical.csv", index=False)
    (out_dir / "hierarchical.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    if len(frame) > 1:
        print(f"\n{'=' * 66}\nacross {len(frame)} seed(s): mean ± std\n{'=' * 66}")
        for column in ["stage1_script_accuracy", "stage2_accuracy_given_correct_routing",
                       "end_to_end_accuracy"]:
            print(f"  {column:38} {frame[column].mean():.4f} ± {frame[column].std():.4f}")

    print(f"\nWritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

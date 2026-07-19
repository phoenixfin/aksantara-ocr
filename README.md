# Aksara OCR — benchmark harness for Indonesian handwritten script classification

A reproducible experiment harness for a dataset paper: scan a folder of
handwritten character images, build fixed splits, and run a full model matrix
(from-scratch CNNs, pretrained CNNs, vision transformers, classical baselines)
across several ablation axes — producing paper-ready CSV and LaTeX tables.

## Expected dataset layout

```
<data-root>/
  javanese/
    ha/  w01_00.png  w01_01.png  w02_00.png  ...
    na/  ...
  balinese/
    ha/  ...
```

`<script>/<character>/<images>`. The `w01_` filename prefix is the **writer id**
— see the warning below.

### Fetching from Mendeley Data

```bash
python scripts/00_fetch_mendeley.py --doi 10.17632/abcd1234ef.1 --list-only  # inspect
python scripts/00_fetch_mendeley.py --doi 10.17632/abcd1234ef.1 --out data/raw
```

Accepts a DOI, a dataset URL, or a bare id; resolves the latest version when
none is given, unpacks archives, and skips files already downloaded. No
authentication — but **published datasets only**, since Mendeley's public API
serves no drafts or embargoed datasets.

## Quick start

```bash
pip install -r requirements.txt

# Optional: verify the pipeline works before your real data is ready
python scripts/make_synthetic_data.py --out data/synthetic
python scripts/01_prepare_data.py --data-root data/synthetic
python scripts/02_run_matrix.py --config configs/smoke_test.yaml

# Real run — fetch from Mendeley Data, or skip if the images are already local
python scripts/00_fetch_mendeley.py --doi 10.17632/xxxxxxxxxx.1 --out data/raw
python scripts/01_prepare_data.py --data-root data/raw
python scripts/02_run_matrix.py --config configs/full_benchmark.yaml
python scripts/02_run_matrix.py --config configs/ablations.yaml
python scripts/02_run_matrix.py --config configs/per_script.yaml
python scripts/03_run_classical.py
```

Use `--dry-run` to list what a matrix will execute, and roughly how long, before
committing GPU hours to it.

## ⚠️ Writer leakage — read before reporting numbers

If the same person wrote several samples of a character and those samples land
on both sides of the train/test boundary, the model can recognize *handwriting
style* rather than *character shape*. Reported accuracy is then inflated, and it
is the first thing a reviewer will probe in a handwritten-dataset paper.

`01_prepare_data.py` defaults to `--split-strategy grouped`, which keeps every
writer entirely within one fold. It needs a writer id, parsed from filenames
matching `w03_...`, `writer-3_...`, or `s03_...`.

**If no writer ids are found, the script prints a warning and falls back to
stratified splitting.** That fallback is usable, but the resulting numbers are an
upper bound and the limitation belongs in the paper. Encoding writer ids in your
filenames before you publish is much the better option — it also makes the
dataset more useful to everyone who downloads it.

The script additionally flags byte-identical duplicate images (another leakage
route) and characters with fewer than 10 samples (per-class metrics get noisy).

## What the harness varies

| Axis | Values |
|---|---|
| Architecture | LeNet-5, SimpleCNN (depth 2/3/4, ±BN), ResNet-18/50, EfficientNet-B0, MobileNetV3, DenseNet-121, ConvNeXt-Tiny, ViT-Tiny/Small, DeiT-Small, Swin-Tiny |
| Initialization | ImageNet-pretrained vs. from scratch |
| Input size | 32 / 48 / 64 / 96 / 128 / 224 px |
| Augmentation | none / light / medium / heavy |
| Task | unified (script×character) · per-script · script identification |
| Seeds | 3 by default |
| Classical | HOG or raw pixels × SVM / kNN / RandomForest / LogReg |

Invalid combinations are dropped automatically (pretrained weights for
from-scratch models; non-224 sizes for transformers with fixed positional
embeddings) rather than crashing mid-matrix.

## Design decisions that matter for the paper

**Splits are computed once and shared.** Every model trains and evaluates on
byte-identical splits, so cross-model differences can't come from split luck.

**One training loop for all architectures.** A ResNet-vs-ViT gap reflects the
architecture, not two different training recipes.

**Model selection on validation macro-F1; the test set is touched once**, after
training, with the selected weights.

**Macro-F1 is the headline metric,** not accuracy — character classes are rarely
balanced, and accuracy flatters models that ignore rare characters.

**Multi-seed by default.** Tables report mean ± std. On datasets this size the
seed-to-seed spread is routinely wider than the gaps between models, so a
single-seed number isn't a result.

**Every run is resumable.** Completed runs are detected and skipped, so a killed
Colab session resumes by re-running the same command.

## Outputs

```
artifacts/
  manifest.csv                     # every image + script/character/writer/hash
  splits.csv                       # the shared train/val/test assignment
  duplicates.csv                   # flagged if any exist
  results/<config>/
    <run_id>/result.json           # metrics, history, per-class, environment
    <run_id>/confusion_matrix.npy
    <run_id>/test_predictions.npz  # raw logits, for post-hoc analysis
    failures.jsonl                 # any runs that errored, with tracebacks
    report/
      main_benchmark.{csv,tex}     # Table 1
      ablation_*.{csv,tex}
      per_script.{csv,tex}
      raw_results.csv, aggregated.csv
```

`most_confused` in each `result.json` lists the character pairs the model mixes
up most — usually the most interesting qualitative result in a script paper.

## Running on a cloud GPU

Use [`notebooks/run_on_gpu.ipynb`](notebooks/run_on_gpu.ipynb) — a thin driver
that clones this repo, built for Colab Pro with **no persistent storage**.

Three features exist specifically for that constraint:

**Seed-major ordering.** Every configuration runs at seed 0 before any runs at
seed 1. A session killed at 60% then leaves one complete seed across the whole
matrix — a usable table — rather than three seeds across an arbitrary prefix,
which is not one.

**`--time-budget <hours>`.** The runner predicts whether the next run fits in the
remaining budget and stops cleanly if it doesn't. Being killed mid-run wastes it
entirely; stopping one run early costs almost nothing.

**Continuous output.** Every run's metrics print as it completes, and
`results_running.csv` is rewritten after each one. Nothing exists only in memory.

If you *can* mount Drive for output, pointing `--artifacts` there restores full
resume — completed runs are detected by their `result.json` and skipped.

### Budget the matrix before running it

| Config set | Runs |
|---|---|
| `tier1_*` (benchmark + 2 ablations + per-script) | **177** |
| `full_benchmark` + `ablations` + `per_script` | **870** |

Counts are for a 20-script × 20-character dataset; `per_script` scales with the
number of scripts. Start with tier 1 — it supports every claim a dataset paper
needs. Run `--dry-run` to get the count for *your* data, then time one run and
multiply before committing GPU hours.

## Layout

```
configs/          experiment matrices (YAML)
scripts/          CLI entry points
src/aksara/
  data/           scan, splits, dataset, transforms
  models/         registry, CNN baselines, classical
  engine/         training loop, metrics
  experiments/    matrix expansion + runner
  reporting/      aggregation, LaTeX tables
```

Adding a model is one `register(ModelSpec(...))` line in
`src/aksara/models/registry.py` plus its name in a config — the training loop
never changes.

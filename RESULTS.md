# Aksara OCR — Consolidated Results

Everything needed to write the experimental sections of the dataset paper.
All numbers are from the committed result files; figures are in `figures_preview/`
(PNG + PDF, 300 dpi) and reproducible via `notebooks/make_figures.ipynb`.

Repository: https://github.com/phoenixfin/aksantara-ocr
Dataset (Mendeley): 10.17632/vfj32bpjsf — v2 raw, v3 cleaned.

---

## 1. Dataset

A corpus of handwritten characters from **13 traditional Indonesian scripts**.
After cleaning (§1.2): **97,383 images, 889 classes** (script × character),
704 unique character names.

### 1.1 Composition (cleaned v3)

| Script | Images | Classes |
|---|---:|---:|
| Sunda | 28,388 | 170 |
| Bali | 13,460 | 137 |
| Jawa | 13,256 | 128 |
| Lontara | 6,407 | 138 |
| Kawi | 6,246 | 31 |
| Dunging-Iban | 5,900 | 59 |
| Lampung | 4,996 | 20 |
| Pallawa | 3,914 | 33 |
| Ogan | 3,679 | 23 |
| Batak | 3,534 | 19 |
| Jawi | 3,201 | 34 |
| Minangkabau | 2,250 | 75 |
| Bima | 2,152 | 22 |
| **Total** | **97,383** | **889** |

Several scripts are syllabaries (consonant × vowel), so class counts vary widely
(19–170 per script) and are not flat alphabets. Images are grayscale/RGB/RGBA,
sizes ranging from 30×56 to 1500×1500, resized to a common resolution at
training time. Class distribution is imbalanced (per-class counts 19–725),
which is why **macro-F1 is the headline metric** rather than accuracy.

### 1.2 Cleaning (v2 → v3)

The published v2 was audited and cleaned to v3. Changelog: **97,263 kept,
120 relocated, 3,117 removed** (100,500 → 97,383 images). Removals/moves:

- **3,109 byte-identical duplicate images** (would leak across the split).
- **4 mislabelled copies** — identical pixels under two labels; the wrong copy
  removed after visual inspection (e.g. Bima `pa`/`wa`, Jawi/Sunda pairs).
- **4 stray non-character files** (photo-editor exports).
- **6 misfiled folders** in Sunda: syllable folders nested under the wrong
  vowel group, relocated (verified: every S-syllable folder went 80 → 100
  images, matching its siblings).

Every change is recorded in `configs/cleaning_rules.yaml` (with evidence) and a
per-file changelog. v3 was independently verified byte-for-byte identical to the
local cleaning output.

### 1.3 Splits and protocol

- **Stratified** train/val/test = **70/15/15** (69,552 / 13,911 / 13,911 after
  dropping 9 images that become identical at cache resolution).
- Splits computed **once** and shared by every model, so cross-model differences
  cannot come from split luck.
- Single training loop for all architectures (a ResNet-vs-ViT gap reflects the
  model, not the recipe). Model selection on validation macro-F1; the test set
  is read once.
- **3 seeds** unless noted; tables report mean ± std.
- Training: AdamW, cosine schedule, label smoothing 0.1, early stopping on val
  macro-F1 (patience 8), mixed precision. Augmentation "medium" unless ablated.

---

## 2. Main result — layered (hierarchical) classifier

Two stages: (1) identify the script (13-way), (2) apply that script's character
model. Because labels are script-qualified, a misrouted image can never yield
the correct label, so **end-to-end accuracy = P(correct script) × P(correct
character | correct script)** is exactly computable, no extra inference.

ResNet-18 @64px, pretrained, 3 seeds:

| Stage | Metric |
|---|---:|
| Stage 1 — script identification (13-way) | **99.91% ± 0.01** accuracy |
| Stage 2 — character, given correct routing | 98.57% ± 0.05 |
| **End-to-end** | **98.49% ± 0.04** |
| Accuracy lost to misrouting | 1.43 |

**Finding.** Script identification is essentially solved (99.91%); only ~1.4
points are lost to routing. The difficulty is entirely *intra-script*
character recognition. This justifies the layered design and localizes the
open problem.

### 2.1 Per-script end-to-end accuracy (mean, 3 seeds)

| Script | End-to-end % | | Script | End-to-end % |
|---|---:|---|---|---:|
| Lampung | 99.91 | | Bali | 99.43 |
| Kawi | 99.89 | | Sunda | 98.44 |
| Batak | 99.80 | | Bima | 97.29 |
| Ogan | 99.62 | | **Jawa** | **95.33** |
| Dunging-Iban | 99.60 | | **Jawi** | **95.12** |
| Pallawa | 99.52 | | | |
| Minangkabau | 99.48 | | | |
| Lontara | 99.45 | | | |

**Finding.** Jawa (largest syllabary, 128 classes) and Jawi (cursive,
Arabic-derived) are hardest. Simple, geometrically distinct scripts (Lampung,
Kawi, Batak) are nearly perfect.

### 2.2 Confusion analysis (Jawi)

The Jawi confusion matrix (Fig 5) surfaces linguistically real errors — visually
similar Arabic-derived letters:

| True → Predicted | Rate |
|---|---:|
| lam → dal | 26% |
| H Kecil → dal | 12% |
| tsa → ta | 10% |
| dhal → ta | 10% |
| kha → H Kecil | 7% |

---

## 3. Backbone comparison (unified 895-class task)

CNNs at 64px (near their accuracy ceiling per §5, far cheaper); transformers at
224px (fixed positional embeddings). ResNet-18 has both 64px and 224px numbers
as the same-resolution bridge. macro-F1 %:

| Model | Family | Res | Params | Macro-F1 | Seeds |
|---|---|---:|---:|---:|---:|
| swin_tiny | Transformer | 224 | 28.2M | **99.35** | 1 |
| resnet18 | CNN | 224 | 11.6M | 99.09 ± 0.11 | 2 |
| vit_tiny | Transformer | 224 | 5.7M | 98.93 | 1 |
| convnext_tiny | CNN | 64 | 28.5M | 98.67 ± 0.01 | 3 |
| densenet121 | CNN | 64 | 7.9M | 98.59 ± 0.12 | 3 |
| efficientnet_b0 | CNN | 64 | 5.1M | 98.02 ± 0.60 | 3 |
| resnet18 | CNN | 64 | 11.6M | 97.85 ± 0.43 | 3 |
| mobilenetv3_small | CNN | 64 | 2.4M | 94.61 ± 0.15 | 3 |

**Findings.** At matched 224px, **swin_tiny is best** (99.35), with vit_tiny ≈
resnet18. Among CNNs at 64px, **convnext_tiny leads** (98.67), and the
efficiency-oriented mobilenetv3_small is clearly weakest (94.61). All strong
models cluster in 98–99.4%, so architecture choice matters little above a
threshold — the dataset is learnable by any competent modern backbone.

*Note: transformers are single-seed (swin ~5.4 h/run made 3 seeds impractical);
this does not affect the ranking.*

---

## 4. Ablation — input resolution (ResNet-18, aug=medium, pretrained)

| Resolution | Macro-F1 % |
|---:|---:|
| 32px | 94.05 ± 0.13 |
| 48px | 97.02 ± 0.15 |
| 64px | 97.85 ± 0.43 |
| 96px | 98.79 ± 0.16 |
| 128px | 99.02 ± 0.17 |
| 224px | 99.13 ± 0.10 |

**Finding.** Accuracy rises steeply to ~96px then plateaus; 128px and 224px are
within noise (99.02 vs 99.13) despite 3× the pixels. **96–128px captures
essentially all the accuracy** — practical guidance for dataset users and the
reason the backbone CNNs were run at 64px.

---

## 5. Ablation — augmentation strength (ResNet-18 @64px)

Macro-F1 %:

| Augmentation | Pretrained | From scratch | Pretrain gain |
|---|---:|---:|---:|
| none | 97.98 ± 0.20 | 96.70 ± 0.13 | +1.28 |
| light | **98.19 ± 0.09** | 97.50 ± 0.12 | +0.69 |
| medium | 97.85 ± 0.43 | 97.50 ± 0.09 | +0.35 |
| heavy | 97.87 ± 0.19 | 97.32 ± 0.32 | +0.55 |

**Findings.** (1) **Augmentation barely helps** — `light` is marginally best;
`medium`/`heavy` do not improve on it. With ~70k training images the model
already sees enough variation. (2) **Pretraining and augmentation partly
substitute for each other**: the ImageNet advantage is largest without
augmentation (+1.28) and shrinks with it (+0.35 at medium). Both supply
invariance; when one is present the other matters less.

---

## 6. Classical baselines (HOG features)

| Method | Train | Accuracy % | Macro-F1 % |
|---|---|---:|---:|
| kNN (k=5) | 70k | 64.22 | **46.27** |
| RandomForest (100 trees) | 70k | 63.55 | 45.44 |
| linear-SVM (SGD) | 10k* | 49.37 | 31.69 |

*linear-SVM capped at 10k training samples: one-vs-rest over 889 classes is
intractable at the full 70k on CPU. kNN and RandomForest are native multiclass
and use the full set.

**Findings.** Best classical (kNN, 46.27% macro-F1) is **~52 points below the
deep layered result (98.49%)**. Macro-F1 sits far below accuracy (46 vs 64),
i.e. classical methods collapse on the long tail of rare classes. Together these
show the fine-grained 889-way task genuinely **requires learned features** — HOG
+ a shallow classifier is not enough.

---

## 7. Figures

All in `figures_preview/` (PNG + PDF, 300 dpi):

| File | Content |
|---|---|
| fig1_size_ablation | Macro-F1 vs input resolution (plateau after 96px) |
| fig2_augmentation | Augmentation × pretraining grouped bars |
| fig3_per_script | Per-script end-to-end accuracy (sorted) |
| fig4_classical_vs_deep | Classical vs deep macro-F1 (the gap) |
| fig5_jawi_confusion | Jawi confusion heatmap (3 seeds pooled) |
| fig6_backbones | Backbone comparison by family |

---

## 8. Limitations

1. **Writer leakage.** Filenames carry no writer ID, so the split is stratified,
   not writer-disjoint. Reported accuracy is therefore an **upper bound on
   generalization to unseen handwriting** — the same person's samples may appear
   in both train and test. This is the most important caveat and cannot be fixed
   without writer metadata.
2. **Single-seed transformers.** swin_tiny/vit_tiny report seed 0 only (compute
   cost); their ranking as top performers is stable, but they lack a variance
   estimate.
3. **Classical linear-SVM cap.** Trained on a 10k stratified subsample (OvR over
   889 classes is intractable at full scale); kNN/RF use the full set.
4. **Class imbalance.** Per-class counts range 19–725; macro-F1 is reported
   throughout to avoid rewarding models that ignore rare classes.

---

## 9. Reproducibility

- **Data:** `scripts/00_fetch_mendeley.py` (by DOI) → `06_clean_dataset.py`
  (rules-driven, non-destructive) → `00b_build_cache.py` → `01_prepare_data.py`.
- **Experiments:** `02_run_matrix.py` over YAML configs in `configs/`
  (resumable, seed-major, wall-clock budgeted).
- **Evaluation:** `05_hierarchical_eval.py` (end-to-end), `03_run_classical.py`,
  `04_audit_dataset.py`.
- **Notebooks (Kaggle):** `run_on_kaggle.ipynb` (ablations),
  `run_backbones_kaggle.ipynb`, `make_figures.ipynb`.
- All result metrics committed under `artifacts_kaggle/`, `artifacts_colab/`,
  `artifacts_local/`.

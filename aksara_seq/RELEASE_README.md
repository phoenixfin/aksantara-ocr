# AksaraLine v1 — a rendered line corpus for Indonesian abugida sequence OCR

100,000 labelled text-line images in four traditional Indonesian scripts —
**Sundanese, Javanese, Balinese and Lontara (Buginese)** — composed from real
handwritten syllable images, with transcriptions, factored (onset, vowel)
labels, and bounding boxes at syllable, word and line level.

**License: CC BY 4.0**, inherited from the source corpus (see *Provenance*).

## What this is, and what it is not

Every glyph in every line is a genuine scan of someone's handwriting, taken
from the character corpus this dataset derives from. What is synthetic is the
*layout*: lines are composed, not written. There is no real handwritten word or
line here, and no transcribed real text to validate against. Please read
*Limitations* before using these numbers to support a claim about reading real
manuscripts.

## Why these four scripts

All four are written as complete **(onset × vowel) grids**, and the source
corpus stores classes at *syllable* level, so the vowel sign is already drawn
by the writer in its correct position. For Javanese `o` — a circumfix vowel,
taling before the consonant and tarung after — the stored image is the whole
three-part cluster. Laying syllables out left to right is therefore
orthographically faithful rather than an approximation of one.

| script | onsets | vowels | syllables | lines (train / val / test) | archive |
|---|---:|---:|---:|---|---:|
| Sunda | 26 | 7 | 170 | 20,000 / 2,500 / 2,500 | 370 MB |
| Jawa | 20 | 6 | 120 | 20,000 / 2,500 / 2,500 | 480 MB |
| Bali | 19 | 7 | 127 | 20,000 / 2,500 / 2,500 | 449 MB |
| Lontara | 23 | 6 | 138 | 20,000 / 2,500 / 2,500 | 518 MB |

Every syllable type occurs in every split of every script.

## Contents

```
AksaraLine_<Script>.zip
  <Script>/charset.json                     syllables, onsets, vowels, factor map
  <Script>/{train,val,test}/labels.jsonl
  <Script>/{train,val,test}/images/000000.png ...
CHECKSUMS.sha256                            sha256 of each archive
dataset_meta.json                           full generator configuration
```

Images are 8-bit grayscale PNG, 96 px tall, variable width. Verify a download
with `sha256sum -c CHECKSUMS.sha256`.

## Label schema

One JSON object per line of `labels.jsonl`:

```json
{"id": "Bali/test/000000",
 "file": "images/000000.png",
 "script": "Bali", "split": "test",
 "text": "ba-ra",
 "words": [["ba", "ra"]],
 "tokens": ["ba", "ra"],
 "onsets": ["b", "r"],
 "vowels": ["a", "a"],
 "n_syllables": 2,
 "style": "medium", "pen": 3.41,
 "width": 305, "height": 96,
 "bbox_format": "xywh",
 "glyphs": [{"syllable": "ba", "onset": "b", "vowel": "a",
             "word": 0, "pos": 0,
             "bbox": [12, 10, 150, 75],
             "pool_idx": 2699,
             "angle": -1.2, "scale": 1.03}],
 "word_boxes": [{"word": 0, "text": "ba-ra", "bbox": [12, 10, 280, 78]}],
 "line_bbox": [12, 8, 280, 80]}
```

`tokens` uses `<sp>` between words. `onsets`/`vowels` are the factored form of
the same sequence, aligned position-for-position with `tokens` (`null` at a
separator) — this is what makes the corpus usable for studying factored versus
monolithic label spaces.

`pool_idx` identifies the source scan a glyph was drawn from, within its
script. It is stable across the dataset, so lines can be grouped by shared
source material and the split guarantee below can be verified independently.

Boxes are `[x, y, w, h]` in the saved image's pixels, tight to each glyph's ink
*before* degradation; stroke dilation and blur can spill a pixel or two past
the box at the `heavy` setting. Measured across the corpus, 99.1–100% of line
ink falls inside a glyph box, and adjacent boxes are effectively disjoint (mean
IoU 0.008–0.056 — they overlap only where kerning lets one glyph tuck under a
neighbour's overhang). The corpus therefore supports detection and
segment-then-classify baselines as well as segmentation-free CTC.

## Splits

The source corpus carries **no writer identifiers**, so writer-disjoint
splitting is not available. What is enforced instead, absolutely, is that no
glyph *bitmap* is reused across splits: the pools are partitioned at the level
of individual source files before a single line is rendered. A test line
therefore shares no ink with any training line.

This was re-derived from the written labels rather than assumed: **0 glyph
bitmaps shared across splits** in all four scripts. It is checkable from
`pool_idx` in the published labels.

## Difficulty levels

Each line is rendered at one of four settings, recorded in `style`, mixed
15 / 30 / 35 / 20:

| style | models |
|---|---|
| `clean` | even spacing, white paper — the control condition |
| `light` | a tidy hand on clean paper |
| `medium` | a normal hand, scanned |
| `heavy` | a cramped hand photographed under uneven light |

Difficulty is a reported axis, not an accident: in baseline CRNN+CTC models,
syllable error rate rises 1.7–6.3× from `clean` to `heavy` depending on script.

## Reference results

A CRNN+CTC baseline (3 seeds, 30 epochs) on this corpus, syllable error rate:

| script | monolithic head | factored head |
|---|---:|---:|
| Sunda | 0.0872 ± 0.0021 | 0.0810 ± 0.0020 |
| Jawa | 0.0667 ± 0.0024 | 0.0589 ± 0.0025 |
| Bali | 0.0461 ± 0.0008 | 0.0444 ± 0.0012 |
| Lontara | 0.0227 ± 0.0011 | 0.0223 ± 0.0004 |

Withholding ~10% of (onset, vowel) cells from training entirely, a monolithic
head recalls **0.000** of the unseen syllables in all runs, while a factored
head — scoring a syllable as onset + vowel, both learned from other cells —
recalls 0.41–0.66 of them.

Code and configurations: https://github.com/phoenixfin/aksantara-ocr
(branch `aksara-seq`).

## How it was generated

```bash
python aksara_seq/scripts/01_build_glyph_pool.py --data-root <character corpus>
python aksara_seq/scripts/02_render_corpus.py --config aksara_seq/configs/corpus_v1.yaml
python aksara_seq/scripts/03_verify_corpus.py
```

Generation is deterministic from `seed: 20260830` — each sample is seeded from
`(seed, script, split, index)`, so any single line can be regenerated alone.
The corpus was rebuilt from source on separate hardware and reproduced
identical per-script token counts.

Layout uses shape-aware kerning on ink clearance rather than bounding-box
advance, with a positive clearance floor so glyphs never overlap, and each line
draws its glyphs from a narrow stroke-width band so one line reads as written
with one pen.

## Limitations

- **Open CV syllables only.** The source corpus contributes no coda machinery
  — no pasangan, gantungan, virama, or final-consonant marks — so closed
  syllables and consonant clusters cannot be written. Real Javanese and
  Balinese text is full of them. This bounds what the corpus represents.
- **Words are not real words.** Syllable strings are sampled from each script's
  own grid under a vowel prior, not drawn from a lexicon. There is no aligned
  text corpus for these four scripts, and a generator that silently resembled
  real language would be more misleading than one that does not. Do not use
  this corpus to measure language-model benefit.
- **Instance-disjoint, not writer-disjoint.** See *Splits*.
- **Composed, not written.** Results obtained here are synthetic-to-synthetic
  and do not transfer directly to manuscript photographs.
- **Glyphs never overlap**, by construction. Real cursive hands touch and
  connect, so detection and segmentation are easier here than on real material.

## Provenance

Derived from the cleaned (v3) handwritten character corpus **"Indonesian Local
Script Characters"**, Mendeley Data
[10.17632/vfj32bpjsf.3](https://doi.org/10.17632/vfj32bpjsf.3), CC BY 4.0.
61,511 source images across the four scripts were used.

One upstream labelling correction is applied and recorded: in Lontara, the
class directory `Vowel E/Mpa` is misnamed — every other vowel column carries
exactly one `Mp` cell, and the E column has no `Mpe` but a second `Mpa` — and
is read here as `Mpe` (30 images).

## Citation

```
[AUTHORS]. AksaraLine v1: a rendered line corpus for Indonesian abugida
sequence OCR. Mendeley Data, [YEAR]. doi:[ASSIGNED ON PUBLICATION]
```

Please also cite the source corpus:

```
Ihsan, A. F. Indonesian Local Script Characters. Mendeley Data, v3.
doi:10.17632/vfj32bpjsf.3
```

## License

**CC BY 4.0** (Creative Commons Attribution 4.0 International),
http://creativecommons.org/licenses/by/4.0 — matching the source corpus, which
is also CC BY 4.0. You may share and adapt this dataset, including
commercially, provided you give appropriate credit and indicate changes.

# AksaraLine — a rendered line corpus for Indonesian abugida sequence recognition

Builds a word/line-level OCR dataset for four Indonesian scripts — **Sunda,
Jawa, Bali, Lontara** — out of an isolated-syllable handwriting corpus. Every
glyph in every rendered line is a real scan of someone's handwriting; what this
code adds is layout, degradation, and labels.

Self-contained subproject: it reads `data/clean/` and writes only under
`aksara_seq/`.

## Why this is possible at all

The source corpus stores classes at **syllable** level, not letter level —
`Vowel I/bi`, `Vokal O/no` — so the vowel sign is already drawn by the writer in
its correct position. For Javanese `o` (a circumfix vowel: *taling* before the
consonant, *tarung* after) the stored image is the complete three-part cluster.
Laying syllables out left to right is therefore orthographically faithful, not
an approximation of one.

All four scripts are **complete (onset × vowel) grids**, verified against the
known class counts:

| Script | Onsets | Vowels | Grid cells | Filled | Images | Images / non-*a* syllable |
|---|---:|---|---:|---:|---:|---:|
| Sunda | 26 | a e eu i o u é | 182 | 170 | 28,388 | ~100 |
| Jawa | 20 | a e i o u è | 120 | 120 | 13,256 | ~22 |
| Bali | 19 | a e i o u é ā | 133 | 127 (+10 digits) | 13,460 | 60 |
| Lontara | 23 | a e i o u é | 138 | 138 | 6,407 | 30 |

The only holes are the loanword consonants (Sunda `kh`, `sy`) and Bali `h`
outside the *a* column. Grid completeness is what makes factored
(onset, vowel) decoding and held-out-cell generalization testable; the wide
spread in the last column gives a four-point data-scarcity axis for free.

## Quick start

```bash
python aksara_seq/scripts/01_build_glyph_pool.py --workers 8   # ~10 min
python aksara_seq/scripts/02_render_corpus.py --preview 3      # look first
python aksara_seq/scripts/02_render_corpus.py --config aksara_seq/configs/corpus_v1.yaml
```

`01` prints the grid report and exits without writing anything under
`--verify-only`.

## Step 1 — the glyph pool

`scripts/01_build_glyph_pool.py` parses every class directory into
`(script, onset, vowel)`, normalizes each image, and writes a
memory-mappable cache per script.

**Normalization** has to cope with a corpus that mixes RGB/RGBA/L, sizes from
30×56 to 1500×1500, both polarities, scans *and* photographs (some Bali samples
are shot on grey or blue card), and pencil only a few grey levels above the
paper:

1. Composite alpha, decide polarity from the border, flatten the background
   with a max-filter estimated on a heavily downsampled copy.
2. Trim to the ink, then strip hard border frames — detected on a deliberately
   sensitive mask, since many are fainter than the binarization threshold, and
   guarded so a glyph's own long stroke is never mistaken for a frame.
3. Zero the paper just below the Otsu threshold. Glyphs are max-blended onto
   the line canvas, so any residue would show as a grey box behind the glyph.
4. Scale by the **core band** — the shortest run of rows holding 72% of the
   ink, i.e. the base letter without its marks. Normalizing the core band
   removes the scan-resolution spread (Lontara alone runs 26–165 px for the
   same letter) while preserving the base-to-mark proportion, which is exactly
   the signal a factored decoder needs. Real size variation is re-introduced as
   controlled jitter at render time rather than inherited from whatever DPI a
   contributor scanned at.
5. Record the baseline (bottom of the core band) for line alignment.

**Splits are decided here, before any line exists.** The source corpus has no
writer ids, so writer-disjoint splitting is unavailable; what is enforced
absolutely is that no glyph bitmap is reused across splits. A rendered test
line shares no ink with any training line.

Jawa's `variations` group (murda / honorific letterforms) is **excluded by
default** — those are context-specific forms, not free allographs, and mixing
them in would make ordinary words orthographically wrong. `--include-variants`
re-enables them for an explicit allograph experiment.

## Step 2 — the rendered corpus

`scripts/02_render_corpus.py` composes lines and writes images, labels and
provenance.

**Word sources.** Default is pseudo-words: syllables sampled from the script's
own grid under a configurable vowel prior and word-length distribution. They
make no linguistic claim, which is the point — there is no aligned text corpus
behind these four scripts, and a generator that silently looked like real
language would be the more dishonest option. A lexicon path exists and is
validated against the grid (`ra-hi-na` is rejected for Bali, which has no `hi`),
but `lexicon_fraction` defaults to `0.0`; see *Limitations*.

**Layout** places each glyph on a shared baseline with per-glyph rotation and
scale jitter, a slow baseline drift across the line, and optional overlap.
Rotation transforms the baseline analytically rather than re-detecting it.

Two things keep a line from reading as pasted-together boxes:

*Kerning.* Every preset floors ink clearance above zero, so glyphs never
overlap and each syllable keeps a separable box. Advancing by bounding-box width leaves a wide blank whenever a
glyph's ink sits away from its box edge — common here, since a vowel mark can
extend the box well past the letter body. Instead each glyph slides left until
its nearest ink comes within the sampled gap of the ink already on the canvas,
compared row by row, so neighbours tuck under each other's overhangs. Gaps are
therefore specified as *ink clearance*, not box padding. This cut the 90th
percentile inter-glyph gap from 17–22 px to 10–12 and the worst case from 59 px
to 39.

*One line, one pen.* Sampling glyphs independently puts fine pencil beside
marker inside a single line, which no real hand does. Each line draws a pen
width and then prefers, for every syllable, the instances nearest that width —
a continuous criterion rather than discrete stroke-weight bands, which fail on
exactly the script that needs them most: Jawa has ~22 instances per non-*a*
syllable, so a third of banded draws found the band empty and fell back to the
whole pool. Nearest-width sampling always returns something and degrades
smoothly. Within-line stroke-width CV drops from 0.20–0.43 to 0.12–0.24, and
`pen_k_frac` tunes the strength (1.0 disables it).

**Degradation** is a four-level ladder — `clean / light / medium / heavy` —
covering pen weight, blur, paper level and mottling, ink darkness, noise,
contrast and JPEG artifacts. `clean` exists as the control: a perfectly spaced,
pure-black-on-white line is read almost trivially, and a near-zero error rate
there measures the renderer, not the model. Difficulty is an explicit, reported
axis rather than an accident of the compositor.

### Output layout

```
build/corpus/v1/
  dataset_meta.json                 # full config, per-split stats, known limitations
  <Script>/
    charset.json                    # syllables, onsets, vowels, syllable -> (onset, vowel)
    {train,val,test}/
      images/000000.png
      labels.jsonl
```

Each `labels.jsonl` record carries the transcription (`"ku-la ra-ma"`), the
token sequence with explicit `<sp>` separators, the **factored** onset and vowel
sequences, the style and pen width, and full per-glyph annotation:

```json
{"bbox_format": "xywh",
 "glyphs": [{"syllable": "pa", "onset": "p", "vowel": "a",
             "word": 0, "pos": 0, "bbox": [7, 36, 45, 32],
             "pool_idx": 9727, "source": "data/clean/Bali/.../Pa_176.jpg",
             "angle": -2.139, "scale": 1.1646}, "..."],
 "word_boxes": [{"word": 0, "text": "pa-pu-yé", "bbox": [7, 34, 124, 56]}],
 "line_bbox": [7, 6, 291, 84]}
```

Boxes are `[x, y, w, h]` in the **saved image's** pixels — the compositor works
at a different resolution and the line is then rescaled to `output_height`, so
the two are not the same coordinate system. They are tight to each glyph's ink
before degradation; stroke dilation and blur can spill a pixel or two past the
box at `heavy`, which costs about 1% of ink coverage. Measured on the corpus:
99.1–100% of line ink falls inside a glyph box, and adjacent boxes are
effectively disjoint (mean IoU 0.008–0.056 — they overlap slightly only where
kerning lets one glyph tuck under another's overhang).

So the corpus supports detection and segment-then-classify baselines as well as
segmentation-free CTC, and `pool_idx`/`source` trace every stroke back to the
file it came from.

Preview them with `--preview N --boxes`.

Every sample is seeded from `(seed, script, split, index)` via blake2b, so any
single line regenerates on its own and the corpus is reproducible from the
config alone.

## Step 3 — the CRNN+CTC baseline

`scripts/04_train_recognizer.py` trains one model; `06_run_matrix.py` runs the
{script × head} matrix and writes `matrix.csv`.

```bash
python aksara_seq/scripts/06_run_matrix.py --epochs 30            # 8 runs
python aksara_seq/scripts/04_train_recognizer.py --script Jawa --head factored
```

**The encoder is deliberately ordinary** — conv stack, height collapsed to one,
2-layer BiLSTM, CTC. It is identical in every condition, because the comparison
under study is the *label space*, and an encoder that differed between arms
would confound it.

**Two heads, one difference.**

*Plain* gives every syllable its own parameter vector: 120–170 classes, each
learnable only from its own examples. Jawa carries ~22 instances per non-*a*
syllable.

*Factored* scores a syllable as **onset score + vowel score**. A consonant is
then trained by every vowel column it appears in and a vowel sign by every
consonant it attaches to — the statistical sharing an abugida's structure
actually offers. There is deliberately no per-syllable free parameter, which
has a second consequence: a syllable that never appeared in training still
receives a well-defined score. That is what makes the held-out-cell experiment
answerable rather than impossible by construction, and `--holdout-cells`
withholds chosen syllables from training for exactly that.

Blank and the word separator are not syllables and keep their own vectors in
both heads.

**Metrics** are computed over syllable tokens, not romanized letters — the
syllable is what the model emits and what the script is written in; scoring
letters would flatter scripts with longer transliterations. Reported per run:
SER, WER, exact-line accuracy, **SER broken down by degradation preset**, and
onset-vs-vowel error rates on length-matched predictions, which says *which
factor* a model is getting wrong.

Batching buckets lines by width — they span roughly 150–1500 px, so random
batches would spend most of their compute on padding.

## Second baseline: detection (YOLO)

`scripts/05_export_yolo.py` converts a corpus to ultralytics format, with
`--classes syllable | onset | vowel` — the detection counterpart of the two
heads above. Boxes round-trip exactly, and images are hard-linked rather than
copied.

```bash
python aksara_seq/scripts/05_export_yolo.py --script Bali --classes syllable
yolo detect train data=aksara_seq/build/yolo/Bali_syllable/data.yaml model=yolo11s.pt imgsz=960
```

Worth stating plainly in any comparison: this corpus renders glyphs with a
*positive* minimum ink clearance, so syllables never overlap and are always
separable. Detection is therefore much easier here than on real manuscript
hands, where aksara touch and connect. A strong YOLO score is partly a property
of the renderer, and the honest way to report it is as a measure of how
segmentable the synthetic corpus is — which is itself a useful number, since it
bounds how much the corpus can say about real cursive text.

## Limitations

Recorded in `dataset_meta.json`, not just here.

- **Open CV syllables only.** The source corpus contributes no coda machinery —
  no pasangan, gantungan, virama, or final-consonant marks — so closed
  syllables and consonant clusters cannot be written. Real Javanese and
  Balinese text is full of them. This bounds what the corpus represents.
- **Instance-disjoint, not writer-disjoint.** No writer ids exist upstream.
  Style-clustered pseudo-writer splits are the natural next hardening step.
- **Synthetic layout.** Lines are composed, not written. There is no real
  handwritten word or line in this dataset and no transcribed real text to
  validate against; a small real evaluation set is the single highest-value
  addition available.
- **Pen consistency is a proxy, not a writer.** Nearest-stroke-width sampling
  keeps one line in one pen weight, but stroke width is only one dimension of
  a hand — slant, curvature and proportion still vary within a line. A richer
  style embedding would tighten this, and is the same machinery pseudo-writer
  splits would need.
- **Seed lexicons are unverified.** `lexicons/*.txt` were assembled without
  native-speaker review and are placeholders, which is why generation defaults
  to pseudo-words only.

## One upstream data fix

`Lontara / Vowel E / Mpa` is misnamed: every other vowel column carries exactly
one `Mp` cell (Mpa, Mpi, Mpo, Mpu), the E column has no `Mpe` but does have a
second `Mpa`, which cannot belong to the E group. It is read as `Mpe` through an
explicit, documented entry in `CLASS_NAME_FIXES` (`src/aksaraseq/scripts_def.py`)
rather than a silent heuristic, and every affected row keeps `renamed_from` in
the index.

## Layout

```
configs/          corpus recipes (YAML)
lexicons/         seed word lists (unverified placeholders)
scripts/          01_build_glyph_pool.py, 02_render_corpus.py
src/aksaraseq/
  scripts_def.py  path -> (script, onset, vowel), grid definitions, class fixes
  preview.py      contact sheets for glyphs and rendered lines
  glyphs/         scan.py, normalize.py, pool.py
  render/         lexicon.py, compose.py, degrade.py, corpus.py
```

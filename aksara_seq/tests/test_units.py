"""Unit checks for the parts where a silent bug would corrupt the dataset.

Plain asserts, no pytest dependency:

    python aksara_seq/tests/test_units.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from aksaraseq.glyphs.normalize import core_band, otsu, strip_frame  # noqa: E402
from aksaraseq.glyphs.pool import PoolConfig, assign_splits  # noqa: E402
from aksaraseq.render.compose import LayoutStyle, compose_line  # noqa: E402
from aksaraseq.render.lexicon import (  # noqa: E402
    PseudoWordSampler, line_text, line_tokens,
)
from aksaraseq.scripts_def import parse_class_path, split_onset  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# --- class-path parsing ----------------------------------------------------

def test_parsing() -> None:
    print("class-path parsing")
    cases = [
        ("Sunda", ("Vowel Eu", "Beu"), "b", "eu", "beu"),
        ("Sunda", ("Vowel A", "a"), "", "a", "a"),
        ("Sunda", ("Vowel I", "Ngi"), "ng", "i", "ngi"),
        ("Jawa", ("all_class", "Vokal È", "dhè"), "dh", "è", "dhè"),
        ("Bali", ("Letter", "Vowel Ā", "nyā"), "ny", "ā", "nyā"),
        ("Lontara", ("Vowel É", "Ngke'"), "ngk", "é", "ngké"),
        ("Lontara", ("Vowel A", "YA"), "y", "a", "ya"),
    ]
    for script, parts, onset, vowel, syllable in cases:
        gc = parse_class_path(script, parts)
        check(f"{script} {'/'.join(parts)}",
              (gc.onset, gc.vowel, gc.syllable) == (onset, vowel, syllable),
              f"got ({gc.onset}, {gc.vowel}, {gc.syllable})")

    gc = parse_class_path("Bali", ("Number", "seven"))
    check("Bali digit", gc.kind == "digit" and gc.digit == 7 and gc.syllable == "<7>")

    gc = parse_class_path("Jawa", ("variations", "ka"))
    check("Jawa murda flagged", gc.variant and gc.vowel == "a")

    # The documented upstream fix must fire, and be traceable.
    gc = parse_class_path("Lontara", ("Vowel E", "Mpa"))
    check("Lontara Mpa -> Mpe", gc.syllable == "mpe" and gc.renamed_from == "Mpa",
          f"got {gc.syllable} from {gc.renamed_from!r}")

    # A vowel suffix that is not there must raise, not guess.
    try:
        split_onset("Xyz", "i")
        check("unparseable class raises", False)
    except ValueError:
        check("unparseable class raises", True)


# --- image geometry --------------------------------------------------------

def test_core_band() -> None:
    print("core band / baseline")
    # A dense body in rows 10..30 with a thin mark above at rows 2..4:
    # the band must be the body, so the baseline lands at its bottom.
    mask = np.zeros((40, 20), dtype=bool)
    mask[10:30, 2:18] = True
    mask[2:5, 9:11] = True
    top, bottom = core_band(mask)
    check("mark above excluded", top >= 6 and bottom <= 30, f"got ({top}, {bottom})")

    # Same, with the mark below instead.
    mask = np.zeros((40, 20), dtype=bool)
    mask[6:26, 2:18] = True
    mask[33:37, 9:11] = True
    top, bottom = core_band(mask)
    check("mark below excluded", top >= 4 and bottom <= 28, f"got ({top}, {bottom})")

    # A single spiking row must not collapse the band (the old failure mode).
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:25, 5:12] = True
    mask[15, :] = True
    top, bottom = core_band(mask)
    check("dense row does not collapse band", bottom - top >= 10,
          f"got height {bottom - top}")


def test_strip_frame() -> None:
    print("frame stripping")
    mask = np.zeros((60, 60), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True   # 1px frame
    mask[25:35, 25:35] = True                                    # the glyph
    r0, c0, r1, c1 = strip_frame(mask)
    check("frame removed", r0 >= 1 and c0 >= 1 and r1 <= 59 and c1 <= 59,
          f"got ({r0}, {c0}, {r1}, {c1})")

    # A thick glyph body touching the border is not a frame.
    solid = np.zeros((60, 60), dtype=bool)
    solid[0:30, :] = True
    box = strip_frame(solid)
    check("thick body kept", box[0] == 0, f"got {box}")


def test_otsu() -> None:
    print("otsu")
    x = np.concatenate([np.zeros(800), np.ones(200)]).reshape(40, 25).astype(np.float32)
    t = otsu(x)
    check("threshold separates modes", 0.05 < t < 0.95, f"got {t}")


# --- splits ----------------------------------------------------------------

@dataclass
class FakeRecord:
    path: str
    script: str
    syllable: str
    variant: bool = False


def test_assign_splits() -> None:
    print("instance-disjoint splits")
    recs = [FakeRecord(f"/x/{s}/{i}", "S", s)
            for s in ("ba", "ka") for i in range(20)]
    recs += [FakeRecord("/x/rare/0", "S", "rare"),
             FakeRecord("/x/rare/1", "S", "rare"),
             FakeRecord("/x/rare/2", "S", "rare")]
    splits = assign_splits(recs, PoolConfig())

    check("every instance assigned", len(splits) == len(recs))
    for syl in ("ba", "ka"):
        got = {splits[r.path] for r in recs if r.syllable == syl}
        check(f"'{syl}' present in all splits", got == {"train", "val", "test"},
              f"got {sorted(got)}")
    rare = {splits[r.path] for r in recs if r.syllable == "rare"}
    check("3-instance class still spans splits", rare == {"train", "val", "test"},
          f"got {sorted(rare)}")

    # Determinism: same config, same assignment.
    again = assign_splits(recs, PoolConfig())
    check("assignment is deterministic", again == splits)


# --- word sampling and layout ----------------------------------------------

def test_pseudo_words() -> None:
    print("pseudo-word sampling")
    # A grid with a deliberate hole at ("h", "i"), as in Bali.
    onsets, vowels = ["", "b", "h", "k"], ["a", "i"]
    syllables = {o + v for o in onsets for v in vowels} - {"hi"}
    sampler = PseudoWordSampler(syllables, onsets=onsets, vowels=vowels)
    rng = np.random.default_rng(0)

    words = [sampler.sample(rng) for _ in range(400)]
    flat = [s for w in words for s in w]
    check("never samples the grid hole", "hi" not in flat)
    check("only samples real syllables", set(flat) <= syllables,
          f"stray: {set(flat) - syllables}")
    check("null onset only word-initial",
          all(s not in ("a", "i") for w in words for s in w[1:]))

    check("tokens carry separators",
          line_tokens([("ku",), ("la", "ra")]) == ["ku", "<sp>", "la", "ra"])
    check("text round-trips", line_text([("ku", "la"), ("ra",)]) == "ku-la ra")


class FakePool:
    """Two syllables, one bitmap each, with a known baseline."""

    script = "Fake"

    def __init__(self):
        bmp = np.zeros((30, 20), dtype=np.uint8)
        bmp[5:25, 4:16] = 255
        self._bmp = bmp

    def sample_idx(self, syllable, split, rng, pen=None, k_frac=0.3, k_min=2):
        return 0

    def bitmap(self, idx):
        return self._bmp

    def baseline(self, idx):
        return 25


def test_compose() -> None:
    print("line composition")
    pool = FakePool()
    rng = np.random.default_rng(0)
    style = LayoutStyle(core_height=48)
    line = compose_line([("ka", "ba"), ("ta",)], pool, "train", style, rng)

    check("one placement per syllable", len(line.placed) == 3)
    check("ink present", line.ink.max() > 0.9)
    check("placements ordered left to right",
          all(a.x < b.x for a, b in zip(line.placed, line.placed[1:])))

    # With no jitter every baseline must land on exactly the same row.
    rows = {p.y + p.baseline for p in line.placed}
    check("baselines coincide without jitter", len(rows) == 1, f"got rows {rows}")

    # The word gap must exceed the syllable gap.
    syl_gap = line.placed[1].x - (line.placed[0].x + line.placed[0].width)
    word_gap = line.placed[2].x - (line.placed[1].x + line.placed[1].width)
    check("word gap wider than syllable gap", word_gap > syl_gap,
          f"syllable {syl_gap}px vs word {word_gap}px")

    # Nothing may be clipped by the canvas.
    check("no glyph clipped",
          all(p.x >= 0 and p.y >= 0
              and p.x + p.width <= line.ink.shape[1]
              and p.y + p.height <= line.ink.shape[0] for p in line.placed))


def main() -> int:
    for fn in (test_parsing, test_core_band, test_strip_frame, test_otsu,
               test_assign_splits, test_pseudo_words, test_compose):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

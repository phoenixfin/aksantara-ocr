"""Where the syllable strings to render come from.

Two sources, mixable:

*Lexicon* -- real words, written as hyphen-separated syllables (``ku-la``).
Every syllable is checked against the script's actual grid, so a word using a
syllable the corpus cannot draw is rejected loudly rather than rendered wrong.

*Pseudo-words* -- syllables sampled from a configurable prior.  These make no
linguistic claim, which is the point: they give uniform coverage of the grid
without pretending to be a language model.  The four scripts have no aligned
text corpus behind them, so a generator that silently looked like real
language would be the more dishonest option.

None of the four scripts contributes a coda inventory (no pasangan, gantungan,
virama, or final-consonant marks), so only open CV syllables are renderable.
That restriction is a property of the source corpus and is recorded in the
dataset metadata rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORD_SEP = "<sp>"

#: Default vowel prior.  Weighted toward /a/, which dominates in all four
#: languages; anything not listed shares the remaining mass equally.
DEFAULT_VOWEL_WEIGHTS = {"a": 0.34, "i": 0.18, "u": 0.16, "e": 0.13, "o": 0.12}

#: Default syllables-per-word distribution: these are predominantly
#: disyllabic languages.
DEFAULT_LENGTH_WEIGHTS = {1: 0.08, 2: 0.44, 3: 0.33, 4: 0.12, 5: 0.03}


@dataclass
class LexiconLoad:
    words: list = field(default_factory=list)      # list[tuple[str, ...]]
    rejected: list = field(default_factory=list)   # list[(word, bad_syllables)]

    @property
    def n_kept(self) -> int:
        return len(self.words)


def load_lexicon(path: Path, valid: set) -> LexiconLoad:
    """Read a hyphen-separated syllable lexicon, dropping unrenderable words."""
    out = LexiconLoad()
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        syls = tuple(s for s in line.split("-") if s)
        if not syls:
            continue
        bad = [s for s in syls if s not in valid]
        if bad:
            out.rejected.append((line, bad))
        else:
            out.words.append(syls)
    return out


class PseudoWordSampler:
    """Sample syllable strings from the script's own (onset x vowel) grid."""

    def __init__(self, syllables, onsets=None, vowels=None,
                 vowel_weights=None, length_weights=None,
                 null_onset_initial_only: bool = True):
        self.syllables = set(syllables)

        # Recover the grid from the syllables the pool can actually draw, so a
        # hole (Bali has no "hi") is never sampled.
        self.cells = {}     # (onset, vowel) -> syllable
        for onset in (onsets or []):
            for vowel in (vowels or []):
                syl = onset + vowel
                if syl in self.syllables:
                    self.cells[(onset, vowel)] = syl

        self.onsets = sorted({o for o, _ in self.cells})
        self.vowels = sorted({v for _, v in self.cells})
        self.null_onset_initial_only = null_onset_initial_only

        vw = dict(DEFAULT_VOWEL_WEIGHTS if vowel_weights is None else vowel_weights)
        listed = [v for v in self.vowels if v in vw]
        unlisted = [v for v in self.vowels if v not in vw]
        spare = max(0.0, 1.0 - sum(vw[v] for v in listed))
        for v in unlisted:
            vw[v] = spare / len(unlisted) if unlisted else 0.0
        weights = np.array([max(vw.get(v, 0.0), 1e-9) for v in self.vowels], dtype=np.float64)
        self.vowel_p = weights / weights.sum()

        lw = dict(DEFAULT_LENGTH_WEIGHTS if length_weights is None else length_weights)
        self.lengths = np.array(sorted(lw), dtype=np.int64)
        lp = np.array([lw[int(k)] for k in self.lengths], dtype=np.float64)
        self.length_p = lp / lp.sum()

        # Onsets available at a non-initial position (null onset excluded when
        # independent vowel letters are word-initial only).
        self._medial_onsets = [o for o in self.onsets
                               if o or not self.null_onset_initial_only]
        if not self._medial_onsets:
            self._medial_onsets = list(self.onsets)

    def _sample_syllable(self, rng, initial: bool) -> str:
        onsets = self.onsets if initial else self._medial_onsets
        for _ in range(32):
            onset = onsets[rng.integers(len(onsets))]
            vowel = self.vowels[rng.choice(len(self.vowels), p=self.vowel_p)]
            syl = self.cells.get((onset, vowel))
            if syl is not None:
                return syl
        # Grid hole in every draw: fall back to any syllable with this onset.
        cands = [s for (o, _), s in self.cells.items() if o == onset]
        if not cands:
            cands = sorted(self.syllables)
        return cands[rng.integers(len(cands))]

    def sample(self, rng) -> tuple:
        n = int(self.lengths[rng.choice(len(self.lengths), p=self.length_p)])
        return tuple(self._sample_syllable(rng, initial=(i == 0)) for i in range(n))


class WordMixer:
    """Draw words from a lexicon and a pseudo-word sampler in a fixed ratio."""

    def __init__(self, pseudo: PseudoWordSampler, lexicon_words=None,
                 lexicon_fraction: float = 0.0):
        self.pseudo = pseudo
        self.lexicon_words = list(lexicon_words or [])
        self.lexicon_fraction = (lexicon_fraction if self.lexicon_words else 0.0)

    def sample(self, rng) -> tuple:
        if self.lexicon_words and rng.random() < self.lexicon_fraction:
            return self.lexicon_words[rng.integers(len(self.lexicon_words))]
        return self.pseudo.sample(rng)

    def sample_line(self, rng, n_words: int) -> list:
        return [self.sample(rng) for _ in range(n_words)]


def line_tokens(words) -> list:
    """Flatten words into a token sequence with explicit word separators."""
    out = []
    for i, w in enumerate(words):
        if i:
            out.append(WORD_SEP)
        out.extend(w)
    return out


def line_text(words) -> str:
    """Human-readable transcription: ``ku-la ra-ma``."""
    return " ".join("-".join(w) for w in words)

"""Label spaces for sequence recognition.

Two ways to name the same target sequence, and the comparison between them is
the point of the project:

*Monolithic* -- one class per syllable.  120-170 classes per script, each with
its own free parameters, so a syllable is only learnable from its own examples.
Jawa carries ~22 instances of each non-*a* syllable, which is thin.

*Factored* -- a syllable's score is the sum of an onset score and a vowel
score.  A consonant is then learned from every vowel column it appears in and a
vowel sign from every consonant it attaches to, which is the statistical
sharing an abugida's structure actually offers.  It also makes a syllable that
never appeared in training *representable*, which is what the held-out-cell
experiment needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BLANK = "<blank>"       # CTC blank, always index 0
WORD_SEP = "<sp>"


@dataclass
class Vocab:
    """Maps syllables to class indices, and classes to their (onset, vowel)."""

    syllables: list          # class order, excluding blank
    onsets: list
    vowels: list
    onset_of: list           # per class index: onset index, or -1 for specials
    vowel_of: list           # per class index: vowel index, or -1 for specials
    special_of: list         # per class index: special index, or -1

    @property
    def n_classes(self) -> int:
        """Including the CTC blank at index 0."""
        return len(self.syllables) + 1

    @property
    def n_specials(self) -> int:
        return sum(1 for s in self.special_of if s >= 0)

    def index(self, token: str) -> int:
        return self._lookup[token]

    def token(self, index: int) -> str:
        return BLANK if index == 0 else self.syllables[index - 1]

    def encode(self, tokens) -> list:
        return [self._lookup[t] for t in tokens]

    def decode(self, indices) -> list:
        return [self.syllables[i - 1] for i in indices if i > 0]

    def __post_init__(self):
        self._lookup = {s: i + 1 for i, s in enumerate(self.syllables)}
        self._lookup[BLANK] = 0

    @classmethod
    def from_charset(cls, path: Path, include_word_sep: bool = True) -> "Vocab":
        cs = json.loads(Path(path).read_text(encoding="utf-8"))
        factors = cs["syllable_to_factors"]

        specials = [WORD_SEP] if include_word_sep else []
        syllables = specials + list(cs["syllables"])
        onsets = list(cs["onsets"])
        vowels = list(cs["vowels"])
        o_idx = {o: i for i, o in enumerate(onsets)}
        v_idx = {v: i for i, v in enumerate(vowels)}

        onset_of, vowel_of, special_of = [-1], [-1], [-1]   # index 0 = blank
        for s in syllables:
            if s in specials:
                onset_of.append(-1)
                vowel_of.append(-1)
                special_of.append(specials.index(s))
                continue
            f = factors[s]
            onset_of.append(o_idx[f["onset"]])
            vowel_of.append(v_idx[f["vowel"]])
            special_of.append(-1)

        return cls(syllables=syllables, onsets=onsets, vowels=vowels,
                   onset_of=onset_of, vowel_of=vowel_of, special_of=special_of)

    def cells(self) -> list:
        """All (onset, vowel) pairs the vocabulary actually contains."""
        out = []
        for i, s in enumerate(self.syllables, start=1):
            if self.special_of[i] < 0:
                out.append((self.onsets[self.onset_of[i]],
                            self.vowels[self.vowel_of[i]]))
        return out

    def syllable_for_cell(self, onset: str, vowel: str):
        for i, s in enumerate(self.syllables, start=1):
            if (self.special_of[i] < 0
                    and self.onsets[self.onset_of[i]] == onset
                    and self.vowels[self.vowel_of[i]] == vowel):
                return s
        return None

    def summary(self) -> str:
        return (f"{self.n_classes} classes (blank + {self.n_specials} special "
                f"+ {len(self.syllables) - self.n_specials} syllables) "
                f"= {len(self.onsets)} onsets x {len(self.vowels)} vowels")


def diagonal_holdout(vocab: Vocab, frac: float = 0.10) -> list:
    """Pick (onset, vowel) cells to withhold, spread across the grid.

    Chosen on a rotating diagonal rather than at random, and never taking the
    last cell of an onset or of a vowel: the factored head can only score an
    unseen cell if *both* of its factors were trained somewhere else, so a
    holdout that strands a factor would test nothing. The result is
    deterministic, so every seed and both heads withhold exactly the same
    syllables and the comparison stays paired.
    """
    cells = {}
    for i, s in enumerate(vocab.syllables, start=1):
        if vocab.special_of[i] >= 0:
            continue
        cells[(vocab.onsets[vocab.onset_of[i]],
               vocab.vowels[vocab.vowel_of[i]])] = s

    per_onset, per_vowel = {}, {}
    for o, v in cells:
        per_onset[o] = per_onset.get(o, 0) + 1
        per_vowel[v] = per_vowel.get(v, 0) + 1

    onsets = sorted({o for o, _ in cells})
    vowels = sorted({v for _, v in cells})
    target = max(1, int(round(frac * len(cells))))

    held = []
    for shift in range(len(vowels)):
        for i, o in enumerate(onsets):
            if len(held) >= target:
                return sorted(held)
            v = vowels[(i + shift) % len(vowels)]
            syl = cells.get((o, v))
            if syl is None or syl in held:
                continue
            if per_onset[o] <= 2 or per_vowel[v] <= 2:
                continue
            held.append(syl)
            per_onset[o] -= 1
            per_vowel[v] -= 1
    return sorted(held)

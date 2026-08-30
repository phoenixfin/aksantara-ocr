"""CTC decoding and error rates.

Error rates are computed over *syllable* tokens, not characters of the
romanization: the syllable is the unit the model emits and the unit an abugida
is written in, so `ku-la` is two tokens.  Reporting over romanized letters
would make scripts with longer transliterations look artificially better.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .vocab import WORD_SEP


def greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor) -> list:
    """Best-path CTC decode: argmax, collapse repeats, drop blanks."""
    best = logits.argmax(dim=-1)                     # (B,T)
    out = []
    for row, length in zip(best, input_lengths):
        seq, prev = [], -1
        for t in range(int(length)):
            k = int(row[t])
            if k != prev and k != 0:
                seq.append(k)
            prev = k
        out.append(seq)
    return out


def levenshtein(a, b) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def split_words(tokens) -> list:
    words, cur = [], []
    for t in tokens:
        if t == WORD_SEP:
            if cur:
                words.append(tuple(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        words.append(tuple(cur))
    return words


class ErrorRates:
    """Accumulates syllable- and word-level errors, sliceable by a tag."""

    def __init__(self):
        self.edits = 0
        self.length = 0
        self.word_edits = 0
        self.word_length = 0
        self.exact = 0
        self.n = 0
        self.by_tag = defaultdict(lambda: [0, 0, 0])   # edits, length, n

    def update(self, pred_tokens, ref_tokens, tag: str = None) -> None:
        d = levenshtein(pred_tokens, ref_tokens)
        self.edits += d
        self.length += len(ref_tokens)
        self.n += 1
        self.exact += int(d == 0)

        pw, rw = split_words(pred_tokens), split_words(ref_tokens)
        self.word_edits += levenshtein(pw, rw)
        self.word_length += len(rw)

        if tag is not None:
            slot = self.by_tag[tag]
            slot[0] += d
            slot[1] += len(ref_tokens)
            slot[2] += 1

    @property
    def ser(self) -> float:
        """Syllable error rate."""
        return self.edits / max(self.length, 1)

    @property
    def wer(self) -> float:
        return self.word_edits / max(self.word_length, 1)

    @property
    def line_accuracy(self) -> float:
        return self.exact / max(self.n, 1)

    def tagged(self) -> dict:
        return {t: {"ser": e / max(l, 1), "n": n}
                for t, (e, l, n) in sorted(self.by_tag.items())}

    def summary(self) -> dict:
        return {"ser": self.ser, "wer": self.wer,
                "line_accuracy": self.line_accuracy, "n_lines": self.n,
                "by_style": self.tagged()}


def factor_errors(pred_tokens, ref_tokens, vocab) -> tuple:
    """Onset and vowel error counts, on the aligned portion of the sequences.

    Only defined where the two sequences have equal length; otherwise the
    factor-level comparison would need its own alignment and would double-count
    insertion and deletion errors already reflected in the syllable rate.
    """
    if len(pred_tokens) != len(ref_tokens):
        return 0, 0, 0
    onset_err = vowel_err = counted = 0
    for p, r in zip(pred_tokens, ref_tokens):
        if r == WORD_SEP or p == WORD_SEP:
            continue
        pi, ri = vocab.index(p), vocab.index(r)
        if vocab.special_of[ri] >= 0:
            continue
        counted += 1
        onset_err += int(vocab.onset_of[pi] != vocab.onset_of[ri])
        vowel_err += int(vocab.vowel_of[pi] != vocab.vowel_of[ri])
    return onset_err, vowel_err, counted

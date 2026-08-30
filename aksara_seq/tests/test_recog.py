"""Unit checks for the recognition layer.

    python aksara_seq/tests/test_recog.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from aksaraseq.recog.metrics import (  # noqa: E402
    ErrorRates, greedy_decode, levenshtein, split_words,
)
from aksaraseq.recog.model import build_model  # noqa: E402
from aksaraseq.recog.vocab import Vocab  # noqa: E402

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def toy_charset(tmp: Path, onsets=None, vowels=None, hole=("k", "i")) -> Path:
    """A grid with one deliberate hole, like Bali's missing 'hi'."""
    onsets = onsets or ["", "b", "k", "g", "m", "n", "ng", "ny"]
    vowels = vowels or ["a", "i", "u", "e", "o"]
    syllables, factors = [], {}
    for o in onsets:
        for v in vowels:
            if (o, v) == hole:
                continue
            s = o + v
            syllables.append(s)
            factors[s] = {"onset": o, "vowel": v}
    cs = {"script": "Toy", "word_separator": "<sp>",
          "syllables": sorted(syllables), "onsets": onsets, "vowels": vowels,
          "syllable_to_factors": factors,
          "n_syllables": len(syllables), "n_onsets": len(onsets),
          "n_vowels": len(vowels)}
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "charset.json"
    path.write_text(json.dumps(cs), encoding="utf-8")
    return path


def test_vocab(path: Path) -> Vocab:
    print("vocab")
    v = Vocab.from_charset(path)
    check("blank is index 0", v.token(0) == "<blank>")
    check("<sp> present and factor-free",
          "<sp>" in v.syllables and v.onset_of[v.index("<sp>")] == -1)
    check("class count", v.n_classes == len(v.syllables) + 1,
          f"got {v.n_classes}")

    i = v.index("bi")
    check("factorization of 'bi'",
          v.onsets[v.onset_of[i]] == "b" and v.vowels[v.vowel_of[i]] == "i")
    check("grid hole absent from vocab", v.syllable_for_cell("k", "i") is None)
    check("encode/decode round-trip",
          v.decode(v.encode(["ba", "<sp>", "ka"])) == ["ba", "<sp>", "ka"])
    return v


def test_factored_head(v: Vocab) -> None:
    print("factored head")
    torch.manual_seed(0)
    m = build_model(v, head="factored", conv_dim=16, rnn_hidden=8, dropout=0.0)
    m.eval()
    h = torch.randn(1, 3, 16)                       # (B,T,D) into the head
    logits = m.head(h)
    onset_logits, vowel_logits = m.head.factor_logits(h)

    # a syllable's score must be exactly onset + vowel, with no extra term --
    # that is what makes an unseen cell representable
    ok = True
    for s in v.syllables:
        i = v.index(s)
        if v.special_of[i] >= 0:
            continue
        expect = (onset_logits[..., v.onset_of[i]]
                  + vowel_logits[..., v.vowel_of[i]])
        ok &= torch.allclose(logits[..., i], expect, atol=1e-5)
    check("syllable logit == onset + vowel", ok)

    # blank and <sp> must NOT be built from the factor projections
    blank = logits[..., 0]
    check("blank has its own parameters",
          not torch.allclose(blank, onset_logits[..., 0] + vowel_logits[..., 0],
                             atol=1e-5))

    # the head can score a cell the vocabulary does not contain: 'ki' is a hole,
    # yet onset 'k' and vowel 'i' both have parameters, so the sum is defined
    k = v.onsets.index("k")
    i_ = v.vowels.index("i")
    unseen = onset_logits[..., k] + vowel_logits[..., i_]
    check("held-out cell is representable", torch.isfinite(unseen).all())

    plain = build_model(v, head="plain", conv_dim=16, rnn_hidden=8)
    n_plain = sum(p.numel() for p in plain.head.parameters())
    n_fact = sum(p.numel() for p in m.head.parameters())
    check("factored head is smaller on a real-sized grid", n_fact < n_plain,
          f"plain {n_plain} vs factored {n_fact}")


def test_head_scaling(tmp: Path) -> None:
    """The point of factoring: head size tracks factors, not syllables."""
    print("head scaling")

    def head_sizes(onsets, vowels):
        path = toy_charset(tmp / f"g{len(onsets)}x{len(vowels)}", onsets, vowels,
                           hole=(onsets[-1], vowels[-1]))
        vv = Vocab.from_charset(path)
        sizes = {}
        for head in ("plain", "factored"):
            m = build_model(vv, head=head, conv_dim=16, rnn_hidden=8)
            sizes[head] = sum(p.numel() for p in m.head.parameters())
        return len(vv.syllables), sizes

    small_n, small = head_sizes(["", "b", "k"], ["a", "i"])
    large_n, large = head_sizes(["", "b", "k", "g", "m", "n", "ng", "ny", "p", "r"],
                                ["a", "i", "u", "e", "o"])
    check("grid grew", large_n > 3 * small_n, f"{small_n} -> {large_n}")
    check("plain head grows with syllable count",
          large["plain"] > 3 * small["plain"],
          f"{small['plain']} -> {large['plain']}")
    check("factored head grows only with onsets+vowels",
          large["factored"] < 2.5 * small["factored"],
          f"{small['factored']} -> {large['factored']}")


def test_decode(v: Vocab) -> None:
    print("ctc decode")
    n = v.n_classes
    a, b = v.index("ba"), v.index("ka")
    # blank-separated repeats must collapse to two distinct emissions
    frames = [a, a, 0, a, b, b, 0]
    logits = torch.full((1, len(frames), n), -10.0)
    for t, k in enumerate(frames):
        logits[0, t, k] = 10.0
    out = greedy_decode(logits, torch.tensor([len(frames)]))
    check("collapse repeats, drop blanks",
          [v.token(k) for k in out[0]] == ["ba", "ba", "ka"],
          f"got {[v.token(k) for k in out[0]]}")

    # input_lengths must truncate padding
    out = greedy_decode(logits, torch.tensor([2]))
    check("respects input length",
          [v.token(k) for k in out[0]] == ["ba"],
          f"got {[v.token(k) for k in out[0]]}")


def test_metrics() -> None:
    print("metrics")
    check("levenshtein identity", levenshtein(["a", "b"], ["a", "b"]) == 0)
    check("levenshtein substitution", levenshtein(["a", "b"], ["a", "c"]) == 1)
    check("levenshtein deletion", levenshtein(["a", "b", "c"], ["a", "c"]) == 1)

    check("split_words on separators",
          split_words(["ku", "la", "<sp>", "ra"]) == [("ku", "la"), ("ra",)])

    r = ErrorRates()
    r.update(["ku", "la"], ["ku", "la"], tag="clean")
    r.update(["ku", "xx"], ["ku", "la"], tag="heavy")
    check("SER over tokens", abs(r.ser - 0.25) < 1e-9, f"got {r.ser}")
    check("line accuracy", abs(r.line_accuracy - 0.5) < 1e-9)
    check("per-style slicing",
          r.tagged()["clean"]["ser"] == 0.0 and r.tagged()["heavy"]["ser"] == 0.5)

    # a word is wrong if any syllable in it is wrong
    r2 = ErrorRates()
    r2.update(["ku", "la", "<sp>", "ra"], ["ku", "xx", "<sp>", "ra"])
    check("WER counts whole words", abs(r2.wer - 0.5) < 1e-9, f"got {r2.wer}")


def test_shapes(v: Vocab) -> None:
    print("model shapes")
    for head in ("plain", "factored"):
        m = build_model(v, head=head, conv_dim=32, rnn_hidden=16)
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(2, 1, 64, 256))
        check(f"{head} output shape", tuple(y.shape) == (2, 64, v.n_classes),
              f"got {tuple(y.shape)}")
        lengths = m.input_lengths(torch.tensor([256, 128]))
        check(f"{head} input_lengths", lengths.tolist() == [64, 32],
              f"got {lengths.tolist()}")
        check(f"{head} lengths fit the time axis",
              int(lengths.max()) <= y.shape[1])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = test_vocab(toy_charset(root))
        test_factored_head(v)
        test_head_scaling(root)
        test_decode(v)
        test_metrics()
        test_shapes(v)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Script definitions and the folder-path -> (script, onset, vowel) parser.

Every one of the four scripts is stored as a (consonant x vowel) grid of
directories.  The grid is what makes factored decoding possible, so parsing it
correctly -- and *verifying* the parse against the known class counts -- is the
foundation of everything downstream.

Path shapes actually on disk::

    Sunda/Vowel A/Ca/...                 vowel group at depth 1
    Jawa/all_class/Vokal A/ba/...        container dir, then vowel group
    Jawa/variations/ka/...               alternate forms of the a-vowel glyphs
    Bali/Letter/Vowel A/a/...            container dir, then vowel group
    Bali/Number/one/...                  digits, no vowel
    Lontara/Vowel A/Ba/...               vowel group at depth 1
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- vowels ----------------------------------------------------------------

#: Canonical vowel keys.  Lexicons and labels use exactly these.
VOWELS = ("a", "e", "eu", "i", "o", "u", "é", "è", "ā")

#: Suffixes a class-directory name may use for a given vowel, longest first.
#: Lontara writes é as ``e'`` (Be', Nge'); Bali writes the long a as ``ā``.
VOWEL_SUFFIXES: dict[str, tuple[str, ...]] = {
    "a": ("a",),
    "e": ("e",),
    "eu": ("eu",),
    "i": ("i",),
    "o": ("o",),
    "u": ("u",),
    "é": ("é", "e'", "e"),
    "è": ("è", "e"),
    "ā": ("ā", "aa", "a"),
}

_VOWEL_GROUP_RE = re.compile(r"^(?:vowel|vokal)\s+(.+)$", re.IGNORECASE)

#: Directory components that only group things and carry no label information.
_CONTAINER_DIRS = {"all_class", "letter"}

#: Digit directory names -> value, for Bali's ``Number`` group.
DIGIT_NAMES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

NULL_ONSET = ""  # independent vowel letter (Sunda "a", Lontara "E", Bali "o", ...)

#: Class directories misnamed in the source corpus, keyed by
#: ``(script, group, raw_class) -> corrected_class``.  Kept explicit rather
#: than repaired by heuristic so the change is auditable.
#:
#: * ``Lontara / Vowel E / Mpa`` -- the Lontara vowel grid is complete at 23
#:   cells per column, and every other column carries exactly one ``Mp`` cell
#:   (Mpa, Mpi, Mpo, Mpu).  The E column has no ``Mpe`` but does have a second
#:   ``Mpa``, which cannot belong to the E group.  Read as ``Mpe``.
CLASS_NAME_FIXES: dict[tuple[str, str, str], str] = {
    ("Lontara", "Vowel E", "Mpa"): "Mpe",
}


@dataclass(frozen=True)
class ScriptDef:
    name: str          # directory name under data/clean, and the label we use
    lang: str          # ISO-ish code, used to pick a lexicon file
    expected_classes: int   # sanity check against the cleaned corpus

    @property
    def dirname(self) -> str:
        return self.name


SCRIPTS: dict[str, ScriptDef] = {
    "Sunda":   ScriptDef("Sunda",   "su",  170),
    "Jawa":    ScriptDef("Jawa",    "jv",  128),
    "Bali":    ScriptDef("Bali",    "ban", 137),
    "Lontara": ScriptDef("Lontara", "bug", 138),
}

DEFAULT_SCRIPTS = ("Sunda", "Jawa", "Bali", "Lontara")


@dataclass(frozen=True)
class GlyphClass:
    """One leaf directory, parsed."""

    script: str
    raw_class: str        # the leaf directory name, verbatim
    group: str            # the path between script and leaf, for traceability
    kind: str             # "syllable" | "digit"
    onset: str            # consonant cluster, "" for an independent vowel
    vowel: str            # canonical vowel key, "" for digits
    variant: bool         # True for Jawa's alternate a-vowel forms
    digit: int | None = None
    renamed_from: str = ""  # set when CLASS_NAME_FIXES rewrote raw_class

    @property
    def syllable(self) -> str:
        """Canonical romanized label, e.g. ``ngké``, ``a``, ``<0>``."""
        if self.kind == "digit":
            return f"<{self.digit}>"
        return self.onset + self.vowel

    @property
    def key(self) -> str:
        return f"{self.script}/{self.syllable}"


def normalize_vowel(token: str) -> str:
    """``"A"`` -> ``"a"``, ``"Eu"`` -> ``"eu"``, ``"É"`` -> ``"é"``."""
    v = token.strip().lower()
    if v not in VOWEL_SUFFIXES:
        raise ValueError(f"unknown vowel group {token!r}")
    return v


def split_onset(raw_class: str, vowel: str) -> str:
    """Strip the vowel suffix off a class name, leaving the onset.

    ``("Ngka", "a") -> "ngk"``, ``("Be'", "é") -> "b"``, ``("A", "a") -> ""``.
    """
    name = raw_class.strip().lower()
    for suffix in VOWEL_SUFFIXES[vowel]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"class {raw_class!r} does not end in vowel {vowel!r}")


def parse_class_path(script: str, parts: tuple[str, ...]) -> GlyphClass:
    """Parse the path components *below* the script directory.

    ``parts[-1]`` is the class directory; everything before it is grouping.
    """
    if not parts:
        raise ValueError("empty class path")
    raw_class, groups = parts[-1], parts[:-1]

    vowel = ""
    variant = False
    kind = "syllable"

    for comp in groups:
        low = comp.strip().lower()
        if low in _CONTAINER_DIRS:
            continue
        if low == "variations":
            variant = True
            vowel = "a"
            continue
        if low == "number":
            kind = "digit"
            continue
        m = _VOWEL_GROUP_RE.match(comp.strip())
        if m:
            vowel = normalize_vowel(m.group(1))
            continue
        raise ValueError(f"{script}: unrecognised group component {comp!r} in {parts}")

    group = "/".join(groups)

    if kind == "digit":
        name = raw_class.strip().lower()
        if name not in DIGIT_NAMES:
            raise ValueError(f"{script}: unknown digit name {raw_class!r}")
        return GlyphClass(script, raw_class, group, "digit", "", "", False,
                          digit=DIGIT_NAMES[name])

    if not vowel:
        raise ValueError(f"{script}: no vowel group found in {parts}")

    fixed = CLASS_NAME_FIXES.get((script, group, raw_class))
    label_class = fixed or raw_class

    onset = split_onset(label_class, vowel)
    return GlyphClass(script, label_class, group, "syllable", onset, vowel, variant,
                      renamed_from=raw_class if fixed else "")

"""Deterministic answer grading and abstention detection.

Grading is rule-based on purpose: it is reproducible, free, and has no self-preference
bias. Known limitation (documented in the README): alias matching misses valid
paraphrases ("JFK" vs "John F. Kennedy" when the alias list lacks one), so measured
accuracy is a slight under-estimate — equally for every system compared.
"""

from __future__ import annotations

import re
import string
from typing import Literal

Outcome = Literal["correct", "incorrect", "abstain"]

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation + "‘’“”"})

# Phrases that count as an abstention. Matched on the normalised prediction.
_ABSTAIN_PATTERNS = [
    r"\bi (?:do not|dont|don t) know\b",
    r"\bi(?: am|m| m) not (?:sure|certain|confident)\b",
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\bno idea\b",
    r"\bi (?:cannot|can not|cant|can t) (?:answer|say|tell|determine)\b",
    r"\bunknown\b$",
    r"\bi (?:do not|dont|don t) have (?:that|this|enough) information\b",
    r"\bunable to (?:answer|determine)\b",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS))


def normalize(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation and articles, squash spaces."""
    text = text.lower().replace("'", "").translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def is_abstention(prediction: str) -> bool:
    norm = normalize(prediction)
    if not norm:
        return True  # an empty answer commits to nothing
    return bool(_ABSTAIN_RE.search(norm))


def matches_any(prediction: str, aliases: list[str]) -> bool:
    """True if a normalised alias equals the prediction or appears in it on word boundaries."""
    pred = normalize(prediction)
    if not pred:
        return False
    padded = f" {pred} "
    for alias in aliases:
        a = normalize(alias)
        if not a:
            continue
        if pred == a or f" {a} " in padded:
            return True
    return False


def grade(prediction: str, aliases: list[str]) -> Outcome:
    if is_abstention(prediction):
        return "abstain"
    return "correct" if matches_any(prediction, aliases) else "incorrect"


def clean_answer(text: str) -> str:
    """Trim a raw generation to its first line and strip wrapping quotes/whitespace."""
    first = text.strip().split("\n", 1)[0].strip()
    return first.strip("\"'` ").strip()

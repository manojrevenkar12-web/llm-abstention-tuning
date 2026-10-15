"""Build SFT and DPO training data from probe records.

Design choices (see README "Method"):

* Answers in the training data are the model's *own* correct phrasings, not the gold
  string. Fine-tuning a model on facts it does not already know is linked to more
  hallucination (Gekhman et al., 2024), so SFT only teaches *when* to answer, never *what*.
* SFT and DPO use disjoint question sets (stratified by knowledge label), so DPO sees
  questions the SFT stage never trained on.
* DPO preference order: correct answer > abstention > hallucinated answer.
  - known:   correct ≻ "I don't know"   (penalises over-abstention)
             correct ≻ own wrong sample   (if one exists)
  - unknown: "I don't know" ≻ own most frequent wrong answer
  - partial: correct ≻ wrong             (sharpens what it half-knows)
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .config import BuildConfig
from .grading import normalize
from .prompts import ABSTAIN_TEXT, build_messages


def most_common(texts: Sequence[str]) -> str | None:
    """Most frequent answer by normalised form; returns the first original phrasing seen."""
    texts = [t for t in texts if normalize(t)]
    if not texts:
        return None
    counts = Counter(normalize(t) for t in texts)
    top = counts.most_common(1)[0][0]
    return next(t for t in texts if normalize(t) == top)


def correct_answer(rec: dict) -> str:
    own = [s["text"] for s in rec["samples"] if s["outcome"] == "correct"]
    if rec["greedy_outcome"] == "correct":
        own.append(rec["greedy"])
    return most_common(own) or rec["aliases"][0]


def wrong_answer(rec: dict) -> str | None:
    wrong = [s["text"] for s in rec["samples"] if s["outcome"] == "incorrect"]
    if rec["greedy_outcome"] == "incorrect":
        wrong.append(rec["greedy"])
    return most_common(wrong)


def _assistant(text: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": text}]


def split_for_dpo(records: Sequence[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified split by knowledge label -> (sft_records, dpo_records)."""
    rng = random.Random(seed)
    sft, dpo = [], []
    for label in ("known", "partial", "unknown"):
        group = [r for r in records if r["label"] == label]
        rng.shuffle(group)
        k = int(round(len(group) * fraction))
        dpo.extend(group[:k])
        sft.extend(group[k:])
    rng.shuffle(sft)
    rng.shuffle(dpo)
    return sft, dpo


def build_sft(records: Sequence[dict], cfg: BuildConfig, seed: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in records:
        label = r["label"]
        if label == "known":
            target = correct_answer(r)
        elif label == "unknown":
            target = ABSTAIN_TEXT
        elif cfg.partial_policy_sft == "answer":
            target = correct_answer(r)
        elif cfg.partial_policy_sft == "abstain":
            target = ABSTAIN_TEXT
        else:
            continue
        rows.append({"id": r["id"], "label": label, "prompt": build_messages(r["question"]),
                     "completion": _assistant(target)})

    if cfg.max_unknown_ratio is not None:
        n_known = sum(x["label"] == "known" for x in rows)
        cap = int(cfg.max_unknown_ratio * n_known)
        unknown = [x for x in rows if x["label"] == "unknown"]
        if len(unknown) > cap:
            rng = random.Random(seed)
            drop = {x["id"] for x in rng.sample(unknown, len(unknown) - cap)}
            rows = [x for x in rows if not (x["label"] == "unknown" and x["id"] in drop)]
    return rows


def build_dpo(records: Sequence[dict], cfg: BuildConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(r: dict, chosen: str, rejected: str, kind: str) -> None:
        if normalize(chosen) == normalize(rejected):
            return
        rows.append({"id": r["id"], "pair_type": kind, "prompt": build_messages(r["question"]),
                     "chosen": _assistant(chosen), "rejected": _assistant(rejected)})

    for r in records:
        label, wrong = r["label"], wrong_answer(r)
        if label == "known":
            good = correct_answer(r)
            if cfg.known_rejects_abstain:
                add(r, good, ABSTAIN_TEXT, "known:answer>abstain")
            if wrong:
                add(r, good, wrong, "known:answer>wrong")
        elif label == "unknown":
            if wrong:
                add(r, ABSTAIN_TEXT, wrong, "unknown:abstain>wrong")
        else:  # partial
            if wrong:
                add(r, correct_answer(r), wrong, "partial:answer>wrong")
    return rows

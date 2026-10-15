"""Knowledge probing: what does the base model actually know?

For each question we take one greedy answer and K sampled answers (plain prompt, no
permission to abstain). The fraction of correct samples, p(correct), estimates whether
the fact is stored in the model:

  known    p(correct) >= known_threshold and the greedy answer is correct
  unknown  p(correct) <= unknown_threshold and the greedy answer is wrong
  partial  everything in between (the model sometimes gets it)

Training targets are built from these labels, so the model is taught to abstain on what
*it* does not know — not on what is hard in general.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

from .config import ProbeConfig
from .data import QAItem
from .grading import grade

log = logging.getLogger(__name__)

Knowledge = Literal["known", "unknown", "partial"]


def label_knowledge(p_correct: float, greedy_correct: bool, cfg: ProbeConfig) -> Knowledge:
    if p_correct >= cfg.known_threshold and greedy_correct:
        return "known"
    if p_correct <= cfg.unknown_threshold and not greedy_correct:
        return "unknown"
    return "partial"


def assemble(item: QAItem, greedy: dict, samples: Sequence[str], cfg: ProbeConfig) -> dict:
    """Pure function: turn raw generations for one item into a probe record."""
    g_out = grade(greedy["text"], item.aliases)
    s = [{"text": t, "outcome": grade(t, item.aliases)} for t in samples]
    p = sum(x["outcome"] == "correct" for x in s) / len(s) if s else float(g_out == "correct")
    return {
        **item.to_dict(),
        "greedy": greedy["text"],
        "greedy_outcome": g_out,
        "greedy_confidence": greedy["confidence"],
        "samples": s,
        "p_correct": p,
        "label": label_knowledge(p, g_out == "correct", cfg),
    }


def run_probe(generator, items: Sequence[QAItem], cfg: ProbeConfig, num_samples: int | None = None) -> list[dict]:
    k = cfg.num_samples if num_samples is None else num_samples
    qs = [it.question for it in items]
    log.info("probing %d questions (greedy + %d samples each)", len(qs), k)
    greedy = generator.greedy(qs)
    samples = generator.sample(qs, k, cfg.temperature, cfg.top_p) if k > 0 else [[] for _ in qs]
    records = [assemble(it, g, s, cfg) for it, g, s in zip(items, greedy, samples)]
    counts = {lab: sum(r["label"] == lab for r in records) for lab in ("known", "partial", "unknown")}
    log.info("knowledge labels: %s", counts)
    return records

"""Evaluation: run every system on every eval set, grade, and compute metrics.

Systems
  base           base model, plain prompt (no permission to abstain)
  base_prompted  base model told it may answer "I don't know"   — the prompt-only baseline
  sft            after LoRA SFT, plain prompt
  sft_dpo        after LoRA SFT + DPO, plain prompt

Plus a *matched-coverage* baseline computed from `base` predictions: the base model
abstains on its least-confident answers until it answers exactly as many questions as
the tuned system. If tuning cannot beat that, a confidence threshold would do the job.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .config import EvalConfig
from .grading import grade
from .metrics import at_coverage, summarize

log = logging.getLogger(__name__)

TRAINED = ("sft", "sft_dpo")


def predict(generator, items: Sequence[dict], abstain_instruction: bool) -> list[dict]:
    gens = generator.greedy([it["question"] for it in items], abstain_instruction=abstain_instruction)
    return [
        {"id": it["id"], "question": it["question"], "aliases": it["aliases"], "prediction": g["text"],
         "confidence": g["confidence"], "outcome": grade(g["text"], it["aliases"])}
        for it, g in zip(items, gens)
    ]


def compute_metrics(
    preds: dict[str, dict[str, list[dict]]],
    knowledge: dict[str, dict[str, str]],
    cfg: EvalConfig,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """preds[dataset][system] -> list of graded predictions; knowledge[dataset][id] -> label."""
    out: dict[str, dict[str, Any]] = {}
    for ds_name, by_system in preds.items():
        labels = knowledge.get(ds_name)
        out[ds_name] = {}
        for system, rows in by_system.items():
            kn = [labels.get(r["id"]) for r in rows] if labels else None
            out[ds_name][system] = summarize(
                [r["outcome"] for r in rows], [r["confidence"] for r in rows], kn,
                penalty=cfg.hallucination_penalty, bins=cfg.ece_bins, n_boot=cfg.bootstrap,
                target_precision=cfg.target_precision, seed=seed,
            )
        base = by_system.get("base")
        if base is None:
            continue
        b_out = [r["outcome"] for r in base]
        b_conf = [r["confidence"] for r in base]
        for system in TRAINED + ("base_prompted",):
            if system in out[ds_name]:
                coverage = 1.0 - out[ds_name][system]["abstention_rate"]
                out[ds_name][system]["matched_base"] = at_coverage(b_out, b_conf, coverage)
    return out

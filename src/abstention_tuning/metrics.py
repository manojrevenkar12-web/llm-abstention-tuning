"""Evaluation metrics: outcome rates, calibration and selective prediction.

Conventions (n = all questions):
  accuracy            correct / n
  hallucination_rate  incorrect (answered and wrong) / n
  abstention_rate     abstain / n
  selective_accuracy  correct / answered            ("precision" of what it does say)
  net_score           accuracy - penalty * hallucination_rate
                      (rewards abstaining over guessing wrong; with penalty 1 a
                       wrong answer costs as much as a right answer earns)

Calibration and risk-coverage use the model's confidence on the answers it gave
(abstentions carry no confidence and are excluded from ECE).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

OUTCOMES = ("correct", "incorrect", "abstain")


def _arr(x: Sequence[Any]) -> np.ndarray:
    return np.asarray(list(x))


def rates(outcomes: Sequence[str]) -> dict[str, float]:
    o = _arr(outcomes)
    n = len(o)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "hallucination_rate": 0.0, "abstention_rate": 0.0,
                "selective_accuracy": 0.0}
    c, i, a = (int((o == k).sum()) for k in OUTCOMES)
    answered = c + i
    return {
        "n": n,
        "accuracy": c / n,
        "hallucination_rate": i / n,
        "abstention_rate": a / n,
        "selective_accuracy": c / answered if answered else 0.0,
    }


# --------------------------------------------------------------------------- calibration


def reliability_bins(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> list[dict[str, float]]:
    """Equal-width confidence bins with mean confidence, accuracy and count per bin."""
    c = _arr(conf).astype(float)
    y = _arr(correct).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (c > lo) & (c <= hi) if b > 0 else (c >= lo) & (c <= hi)
        k = int(mask.sum())
        out.append({
            "lo": float(lo), "hi": float(hi), "count": k,
            "confidence": float(c[mask].mean()) if k else 0.0,
            "accuracy": float(y[mask].mean()) if k else 0.0,
        })
    return out


def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error (equal-width bins, weighted by bin size)."""
    n = len(conf)
    if n == 0:
        return 0.0
    rb = reliability_bins(conf, correct, bins)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in rb))


# --------------------------------------------------------------------------- selective prediction


def _scores(outcomes: Sequence[str], conf: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (order of answered items by confidence desc, correct flags, n_total)."""
    o = _arr(outcomes)
    c = _arr(conf).astype(float)
    answered = np.where(o != "abstain")[0]
    order = answered[np.argsort(-c[answered], kind="stable")]
    return order, (o == "correct"), np.array(len(o))


def risk_coverage(outcomes: Sequence[str], conf: Sequence[float]) -> dict[str, list[float]]:
    """Risk-coverage curve: keep the k most confident answers, k = 1..answered.

    coverage = k / n_total; risk = wrong among kept / k. A model that already abstains
    starts its curve at its own coverage ceiling — further thresholding only lowers it.
    """
    order, correct, n = _scores(outcomes, conf)
    if len(order) == 0 or n == 0:
        return {"coverage": [], "risk": []}
    kept_correct = np.cumsum(correct[order])
    k = np.arange(1, len(order) + 1)
    return {"coverage": (k / int(n)).tolist(), "risk": (1.0 - kept_correct / k).tolist()}


def aurc(outcomes: Sequence[str], conf: Sequence[float]) -> float:
    """Area under the risk-coverage curve over the answered region (lower is better)."""
    rc = risk_coverage(outcomes, conf)
    return float(np.mean(rc["risk"])) if rc["risk"] else 0.0


def at_coverage(outcomes: Sequence[str], conf: Sequence[float], coverage: float) -> dict[str, float]:
    """Abstain on the least confident answers until `coverage` of questions are answered.

    Used for the matched-coverage baseline: "if the base model simply abstained on its
    least confident answers as often as the tuned model abstains, how would it do?"
    """
    order, correct, n = _scores(outcomes, conf)
    n = int(n)
    k = int(round(coverage * n))
    k = max(0, min(k, len(order)))
    kept = order[:k]
    c = int(correct[kept].sum())
    return {
        "coverage": k / n if n else 0.0,
        "accuracy": c / n if n else 0.0,
        "hallucination_rate": (k - c) / n if n else 0.0,
        "selective_accuracy": c / k if k else 0.0,
    }


def coverage_at_precision(outcomes: Sequence[str], conf: Sequence[float], target: float) -> float:
    """Largest coverage at which the kept answers are at least `target` correct (0 if never)."""
    rc = risk_coverage(outcomes, conf)
    best = 0.0
    for cov, risk in zip(rc["coverage"], rc["risk"]):
        if 1.0 - risk >= target - 1e-12:
            best = max(best, cov)
    return float(best)


# --------------------------------------------------------------------------- uncertainty of estimates


def bootstrap_ci(outcomes: Sequence[str], key: str, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for one of the `rates()` keys."""
    o = _arr(outcomes)
    if len(o) == 0 or n_boot <= 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = rates(o[rng.integers(0, len(o), len(o))])[key]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# --------------------------------------------------------------------------- one-call summary


def summarize(
    outcomes: Sequence[str],
    conf: Sequence[float],
    knowledge: Sequence[str | None] | None = None,
    *,
    penalty: float = 1.0,
    bins: int = 10,
    n_boot: int = 1000,
    target_precision: float = 0.9,
    seed: int = 0,
) -> dict[str, Any]:
    o = _arr(outcomes)
    c = _arr(conf).astype(float)
    r = rates(o)
    answered = o != "abstain"
    out: dict[str, Any] = dict(r)
    out["net_score"] = r["accuracy"] - penalty * r["hallucination_rate"]
    out["ece"] = ece(c[answered], (o[answered] == "correct"), bins)
    out["aurc"] = aurc(o, c)
    target_key = f"coverage_at_{round(target_precision * 100)}pct_precision"
    out[target_key] = coverage_at_precision(o, c, target_precision)
    out["ci95"] = {k: bootstrap_ci(o, k, n_boot, seed) for k in
                   ("accuracy", "hallucination_rate", "abstention_rate")}

    if knowledge is not None:
        kn = _arr([k if k is not None else "none" for k in knowledge])
        for label in ("known", "unknown", "partial"):
            m = kn == label
            sub = rates(o[m])
            out[f"{label}_n"] = int(m.sum())
            out[f"{label}_abstention_rate"] = sub["abstention_rate"]
            out[f"{label}_hallucination_rate"] = sub["hallucination_rate"]
            out[f"{label}_accuracy"] = sub["accuracy"]
        # Headline knowledge-aware number: how often it refused a question the base model
        # reliably knows. (`unknown_hallucination_rate` above is its mirror image: how
        # often it still guessed on a question the base model never gets right.)
        out["over_abstention_rate"] = out["known_abstention_rate"]
    return out

"""Release gates over metrics.json — the same idea as a CI quality gate.

A gate compares one metric of one system on one dataset, optionally against a reference
system (or the matched-coverage base). `dataset` may be a name, `in_distribution`, or
`ood` (expands to every other eval set). Gate thresholds are meant to be fixed *before*
looking at results; see configs/gates.yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import yaml
from pydantic import BaseModel

Kind = Literal["relative_reduction", "max_drop", "min_delta", "max", "min"]


class Gate(BaseModel):
    name: str
    system: str
    metric: str
    kind: Kind
    threshold: float
    dataset: str = "in_distribution"
    compare_to: str | None = None  # a system name, or "matched_base"
    severity: Literal["block", "warn"] = "block"


@dataclass
class GateResult:
    name: str
    dataset: str
    value: float | None
    reference: float | None
    threshold: float
    kind: str
    severity: str
    passed: bool
    detail: str


def load_gates(path: str) -> list[Gate]:
    with open(path, encoding="utf-8") as f:
        return [Gate.model_validate(g) for g in yaml.safe_load(f)["gates"]]


def _check(kind: str, value: float, ref: float | None, thr: float) -> tuple[bool, str]:
    if kind == "max":
        return value <= thr, f"{value:.4f} <= {thr}"
    if kind == "min":
        return value >= thr, f"{value:.4f} >= {thr}"
    assert ref is not None
    if kind == "relative_reduction":
        red = (ref - value) / ref if ref > 0 else 0.0
        return red >= thr, f"reduction {red:.1%} (from {ref:.4f} to {value:.4f}) >= {thr:.0%}"
    if kind == "max_drop":
        return ref - value <= thr, f"drop {ref - value:+.4f} (from {ref:.4f} to {value:.4f}) <= {thr}"
    if kind == "min_delta":
        return value - ref >= thr, f"delta {value - ref:+.4f} ({value:.4f} vs {ref:.4f}) >= {thr}"
    raise ValueError(kind)


def evaluate_gates(gates: list[Gate], metrics: dict[str, Any], in_distribution: str) -> list[GateResult]:
    results: list[GateResult] = []
    for g in gates:
        if g.dataset == "in_distribution":
            targets = [in_distribution]
        elif g.dataset == "ood":
            targets = [d for d in metrics if d != in_distribution]
        else:
            targets = [g.dataset]
        for ds in targets:
            m = metrics.get(ds, {})
            sys_m = m.get(g.system)
            if sys_m is None or g.metric not in sys_m:
                results.append(GateResult(g.name, ds, None, None, g.threshold, g.kind, g.severity, False,
                                          f"missing metric {g.system}.{g.metric}"))
                continue
            value = float(sys_m[g.metric])
            ref = None
            if g.compare_to == "matched_base":
                ref = sys_m.get("matched_base", {}).get(g.metric)
            elif g.compare_to:
                ref = m.get(g.compare_to, {}).get(g.metric)
            if g.compare_to and ref is None:
                results.append(GateResult(g.name, ds, value, None, g.threshold, g.kind, g.severity, False,
                                          f"missing reference {g.compare_to}.{g.metric}"))
                continue
            passed, detail = _check(g.kind, value, None if ref is None else float(ref), g.threshold)
            results.append(GateResult(g.name, ds, value, ref, g.threshold, g.kind, g.severity, passed, detail))
    return results


def blocking_failures(results: list[GateResult]) -> list[GateResult]:
    return [r for r in results if r.severity == "block" and not r.passed]


def to_dicts(results: list[GateResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]

"""Markdown report and figures from a finished run. Every number comes from metrics.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import read_jsonl  # noqa: E402
from .metrics import reliability_bins, risk_coverage  # noqa: E402

SYSTEM_LABEL = {"base": "Base", "base_prompted": "Base + abstain prompt", "sft": "SFT", "sft_dpo": "SFT + DPO"}
SYSTEM_COLOR = {"base": "#6b7280", "base_prompted": "#d97706", "sft": "#2563eb", "sft_dpo": "#059669"}
OUTCOME_COLOR = {"correct": "#059669", "abstain": "#9ca3af", "incorrect": "#dc2626"}


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="both", alpha=0.25, linewidth=0.6)


# --------------------------------------------------------------------------- figures


def plot_outcomes(metrics: dict[str, Any], out: Path) -> Path:
    sets = list(metrics)
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.2), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], sets):
        systems = list(metrics[ds])
        y = range(len(systems))
        left = [0.0] * len(systems)
        for outcome, key in (("correct", "accuracy"), ("abstain", "abstention_rate"),
                             ("incorrect", "hallucination_rate")):
            vals = [metrics[ds][s][key] for s in systems]
            ax.barh(list(y), vals, left=left, color=OUTCOME_COLOR[outcome], label=outcome, height=0.62)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_yticks(list(y), [SYSTEM_LABEL.get(s, s) for s in systems])
        ax.set_xlim(0, 1)
        ax.set_title(ds)
        ax.set_xlabel("share of questions")
        _style(ax)
    axes[0][0].invert_yaxis()  # shared y axis: invert once so the first system is on top
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=3, frameon=False)
    fig.tight_layout()
    path = out / "outcomes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_risk_coverage(preds: dict[str, dict[str, list[dict]]], out: Path) -> Path:
    sets = list(preds)
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.4), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], sets):
        for system, rows in preds[ds].items():
            rc = risk_coverage([r["outcome"] for r in rows], [r["confidence"] for r in rows])
            if not rc["coverage"]:
                continue
            ax.plot(rc["coverage"], rc["risk"], color=SYSTEM_COLOR.get(system), lw=1.6,
                    label=SYSTEM_LABEL.get(system, system))
            ax.scatter([rc["coverage"][-1]], [rc["risk"][-1]], color=SYSTEM_COLOR.get(system), s=22, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(ds)
        ax.set_xlabel("coverage (share of questions answered)")
        _style(ax)
    axes[0][0].set_ylabel("selective risk (wrong / answered)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = out / "risk_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reliability(preds_one_set: dict[str, list[dict]], bins: int, title: str, out: Path) -> Path:
    systems = list(preds_one_set)
    fig, axes = plt.subplots(1, len(systems), figsize=(3.2 * len(systems), 3.2), squeeze=False)
    for ax, system in zip(axes[0], systems):
        rows = [r for r in preds_one_set[system] if r["outcome"] != "abstain"]
        rb = reliability_bins([r["confidence"] for r in rows], [r["outcome"] == "correct" for r in rows], bins)
        width = 1.0 / bins
        ax.bar([b["lo"] for b in rb], [b["accuracy"] if b["count"] else 0 for b in rb], width=width,
               align="edge", color=SYSTEM_COLOR.get(system), alpha=0.85, edgecolor="white")
        ax.plot([0, 1], [0, 1], ls="--", color="#111827", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(SYSTEM_LABEL.get(system, system), fontsize=9)
        ax.set_xlabel("confidence")
        _style(ax)
    axes[0][0].set_ylabel("accuracy of answered")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = out / "reliability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training(histories: dict[str, list[dict]], out: Path) -> Path | None:
    panels = []
    if histories.get("sft"):
        panels.append(("SFT loss", [(h["step"], h["loss"]) for h in histories["sft"] if "loss" in h]))
    if histories.get("dpo"):
        panels.append(("DPO loss", [(h["step"], h["loss"]) for h in histories["dpo"] if "loss" in h]))
        acc = [(h["step"], h["rewards/accuracies"]) for h in histories["dpo"] if "rewards/accuracies" in h]
        if acc:
            panels.append(("DPO reward accuracy", acc))
    panels = [p for p in panels if p[1]]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 2.8), squeeze=False)
    for ax, (title, pts) in zip(axes[0], panels):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#2563eb", lw=1.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        _style(ax)
    fig.tight_layout()
    path = out / "training.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- markdown


def _with_ci(m: dict[str, Any], key: str) -> str:
    lo_hi = m.get("ci95", {}).get(key)
    return f"{pct(m[key])} ({pct(lo_hi[0])}–{pct(lo_hi[1])})" if lo_hi else pct(m[key])


def _main_table(ds_metrics: dict[str, Any], target_key: str) -> list[str]:
    has_kn = any("over_abstention_rate" in m for m in ds_metrics.values())
    target_pct = target_key.split("_")[2].replace("pct", "%")
    head = ["System", "Accuracy", "Hallucination", "Abstention", "Selective acc.", "Net score", "ECE", "AURC",
            f"Coverage @ {target_pct} precision"]
    if has_kn:
        head += ["Over-abstention (known)", "Hallucination (unknown)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for system, m in ds_metrics.items():
        row = [SYSTEM_LABEL.get(system, system), _with_ci(m, "accuracy"), _with_ci(m, "hallucination_rate"),
               _with_ci(m, "abstention_rate"), pct(m["selective_accuracy"]), f"{m['net_score']:+.3f}",
               f"{m['ece']:.3f}", f"{m['aurc']:.3f}", pct(m.get(target_key))]
        if has_kn:
            row += [pct(m.get("over_abstention_rate")), pct(m.get("unknown_hallucination_rate"))]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _matched_table(ds_metrics: dict[str, Any]) -> list[str]:
    rows = [(s, m) for s, m in ds_metrics.items() if "matched_base" in m]
    if not rows:
        return []
    lines = ["| System | Coverage | Selective acc. (system) | Selective acc. (base, same coverage) | "
             "Hallucination (system) | Hallucination (base, same coverage) |", "|---|---|---|---|---|---|"]
    for s, m in rows:
        mb = m["matched_base"]
        lines.append(f"| {SYSTEM_LABEL.get(s, s)} | {pct(1 - m['abstention_rate'])} | {pct(m['selective_accuracy'])} "
                     f"| {pct(mb['selective_accuracy'])} | {pct(m['hallucination_rate'])} "
                     f"| {pct(mb['hallucination_rate'])} |")
    return lines


def headline(metrics: dict[str, Any], in_dist: str) -> str:
    m = metrics.get(in_dist, {})
    if "base" not in m or "sft_dpo" not in m:
        return ""
    b, t = m["base"], m["sft_dpo"]
    b_h, t_h = b["hallucination_rate"], t["hallucination_rate"]
    rel = (b_h - t_h) / b_h if b_h else 0.0
    change = f"a {rel:.0%} relative reduction" if rel >= 0 else f"a {-rel:.0%} relative increase"
    return (f"On **{in_dist}** (in-distribution), SFT + DPO changed the hallucination rate from "
            f"{pct(b['hallucination_rate'])} to {pct(t['hallucination_rate'])} ({change}), "
            f"accuracy from {pct(b['accuracy'])} to {pct(t['accuracy'])}, and abstained on "
            f"{pct(t['abstention_rate'])} of questions. Selective accuracy (precision of answers given) went from "
            f"{pct(b['selective_accuracy'])} to {pct(t['selective_accuracy'])}.")


def write_report(run_dir: Path, run_name: str, base_model: str, in_dist: str, ece_bins: int,
                 target_precision: float) -> Path:
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    pred_dir = run_dir / "eval" / "predictions"
    preds: dict[str, dict[str, list[dict]]] = {}
    for ds in metrics:
        preds[ds] = {s: read_jsonl(pred_dir / f"{s}__{ds}.jsonl") for s in metrics[ds]
                     if (pred_dir / f"{s}__{ds}.jsonl").exists()}

    histories = {}
    for stage in ("sft", "dpo"):
        p = run_dir / "models" / f"{stage}_adapter" / "log_history.json"
        if p.exists():
            histories[stage] = json.loads(p.read_text())

    figs = [plot_outcomes(metrics, out), plot_risk_coverage(preds, out)]
    if in_dist in preds:
        figs.append(plot_reliability(preds[in_dist], ece_bins, f"Reliability on {in_dist} (answered only)", out))
    tr = plot_training(histories, out)
    if tr:
        figs.append(tr)

    target_key = f"coverage_at_{int(round(target_precision * 100))}pct_precision"
    md = [f"# Results — {run_name}", "", f"Base model: `{base_model}`. Every number below is generated from "
          "`eval/metrics.json` by `abstention-tuning report`; nothing is hand-edited.", ""]
    h = headline(metrics, in_dist)
    if h:
        md += ["## Headline", "", h, ""]
    md += ["![outcomes](outcomes.png)", ""]
    for ds in metrics:
        tag = " (in-distribution)" if ds == in_dist else " (out-of-distribution)"
        md += [f"## {ds}{tag}", "", *_main_table(metrics[ds], target_key), ""]
        mt = _matched_table(metrics[ds])
        if mt:
            md += ["Matched-coverage baseline — the base model abstaining on its least-confident answers at the "
                   "same rate:", "", *mt, ""]
    md += ["## Risk–coverage", "", "![risk-coverage](risk_coverage.png)", ""]
    if (out / "reliability.png").exists():
        md += ["## Calibration", "", "![reliability](reliability.png)", ""]
    if tr:
        md += ["## Training curves", "", "![training](training.png)", ""]

    stats_p = run_dir / "datasets" / "build_stats.json"
    if stats_p.exists():
        stats = json.loads(stats_p.read_text())
        md += ["## Training data", "", "```json", json.dumps(stats, indent=2), "```", ""]

    gates_p = out / "gates.json"
    if gates_p.exists():
        gates = json.loads(gates_p.read_text())
        md += ["## Release gates", "", "| Gate | Dataset | Result | Severity | Detail |", "|---|---|---|---|---|"]
        for g in gates:
            md.append(f"| {g['name']} | {g['dataset']} | {'PASS' if g['passed'] else 'FAIL'} | {g['severity']} "
                      f"| {g['detail']} |")
        md.append("")

    path = out / "REPORT.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path

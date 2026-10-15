"""Print the README results tables for a committed run, straight from its metrics.json.

    python scripts/results_table.py results/qwen2.5-1.5b

Used so that no number in the README is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LABEL = {"base": "Base", "base_prompted": "Base + \"say I don't know\" prompt", "sft": "SFT", "sft_dpo": "SFT + DPO"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main(run_dir: str) -> None:
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    gates = json.loads((Path(run_dir) / "gates.json").read_text())
    in_dist = next(iter(metrics))

    print(f"**{in_dist} (in-distribution), 1,000 questions**\n")
    print("| System | Accuracy | Hallucination | Abstention | Selective acc. | Over-abstention on known | ECE |")
    print("|---|---|---|---|---|---|---|")
    for s, m in metrics[in_dist].items():
        print(f"| {LABEL.get(s, s)} | {pct(m['accuracy'])} | {pct(m['hallucination_rate'])} | "
              f"{pct(m['abstention_rate'])} | {pct(m['selective_accuracy'])} | "
              f"{pct(m.get('over_abstention_rate', 0))} | {m['ece']:.3f} |")

    print("\n**Out of distribution (SFT + DPO vs base)**\n")
    print("| Dataset | Hallucination (base → tuned) | Accuracy (base → tuned) | Abstention (tuned) |")
    print("|---|---|---|---|")
    for ds, by_sys in metrics.items():
        if ds == in_dist:
            continue
        b, t = by_sys["base"], by_sys["sft_dpo"]
        print(f"| {ds} | {pct(b['hallucination_rate'])} → {pct(t['hallucination_rate'])} | "
              f"{pct(b['accuracy'])} → {pct(t['accuracy'])} | {pct(t['abstention_rate'])} |")

    mb = metrics[in_dist]["sft_dpo"]["matched_base"]
    t = metrics[in_dist]["sft_dpo"]
    coverage = pct(1 - t["abstention_rate"])
    print(f"\n**Matched-coverage check ({in_dist}):** at the same {coverage} coverage, a confidence "
          f"threshold on the base model reaches {pct(mb['selective_accuracy'])} selective accuracy vs "
          f"{pct(t['selective_accuracy'])} for SFT + DPO.")

    print("\n**Release gates:** " + "; ".join(
        f"{'PASS' if g['passed'] else 'FAIL'} ({g['severity']}) {g['name']}"
        + (f" [{g['dataset']}]" if g["dataset"] != in_dist else "") for g in gates))


if __name__ == "__main__":
    main(sys.argv[1])

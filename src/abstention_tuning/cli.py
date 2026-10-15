"""Command line: `abstention-tuning <stage> --config configs/qwen2.5-1.5b.yaml`.

Stages: prepare, probe, build, sft, dpo, evaluate, gates, report, or `all`.

Exit codes: 0 = success, 3 = a blocking release gate failed, anything else = a stage crashed.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .pipeline import (
    STAGES,
    stage_build,
    stage_dpo,
    stage_evaluate,
    stage_gates,
    stage_prepare,
    stage_probe,
    stage_report,
    stage_sft,
)

GATES_FAILED = 3  # distinct from Python's exit code 1 for an uncaught exception


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="abstention-tuning", description=__doc__)
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--gates", default="configs/gates.yaml")
    ap.add_argument("--from-stage", choices=STAGES, help="with `all`: resume from this stage")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    runners = {
        "prepare": lambda: stage_prepare(cfg),
        "probe": lambda: stage_probe(cfg),
        "build": lambda: stage_build(cfg),
        "sft": lambda: stage_sft(cfg),
        "dpo": lambda: stage_dpo(cfg),
        "evaluate": lambda: stage_evaluate(cfg),
        "gates": lambda: stage_gates(cfg, args.gates),
        "report": lambda: stage_report(cfg),
    }

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    if args.stage == "all" and args.from_stage:
        stages = stages[stages.index(args.from_stage):]

    gates_ok = True
    for s in stages:
        logging.getLogger("abstention_tuning").info("=== stage: %s ===", s)
        result = runners[s]()
        if s == "gates":
            gates_ok = bool(result)
    return 0 if gates_ok else GATES_FAILED


if __name__ == "__main__":
    sys.exit(main())

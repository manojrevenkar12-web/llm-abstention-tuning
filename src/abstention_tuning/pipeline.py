"""Pipeline stages. Each stage reads the previous stage's files from the run directory,
so any stage can be re-run on its own (useful when a Colab session disconnects)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import yaml

from .builders import build_dpo, build_sft, split_for_dpo
from .config import RunConfig
from .data import QAItem, disjoint, load_items, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

STAGES = ("prepare", "probe", "build", "sft", "dpo", "evaluate", "gates", "report")


class Paths:
    def __init__(self, cfg: RunConfig):
        r = cfg.run_dir
        self.root = r
        self.data = r / "data"
        self.probe = r / "probe"
        self.datasets = r / "datasets"
        self.models = r / "models"
        self.eval = r / "eval"
        self.report = r / "report"

    def eval_items(self, name: str) -> Path:
        return self.data / f"eval_{name}.jsonl"

    def eval_probe(self, name: str) -> Path:
        return self.probe / f"eval_{name}.jsonl"

    def model(self, stage: str, kind: str) -> Path:
        return self.models / f"{stage}_{kind}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:  # metrics-only installs
        pass


def _generator(cfg: RunConfig, model_path: str):
    from .generation import Generator

    return Generator(model_path, cfg.model.precision, cfg.model.trust_remote_code,
                     cfg.generation.batch_size, cfg.generation.max_new_tokens)


# --------------------------------------------------------------------------- stages


def stage_prepare(cfg: RunConfig) -> None:
    p = Paths(cfg)
    p.root.mkdir(parents=True, exist_ok=True)
    (p.root / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    evals = {spec.name: load_items(spec, cfg.seed) for spec in cfg.data.eval}
    train = disjoint(load_items(cfg.data.train, cfg.seed), evals)
    write_jsonl(p.data / "train.jsonl", (it.to_dict() for it in train))
    for name, items in evals.items():
        write_jsonl(p.eval_items(name), (it.to_dict() for it in items))
    log.info("prepared train=%d, eval=%s", len(train), {k: len(v) for k, v in evals.items()})


def stage_probe(cfg: RunConfig) -> None:
    from .probe import run_probe

    p = Paths(cfg)
    _seed_everything(cfg.seed)
    gen = _generator(cfg, cfg.model.base)
    try:
        train = [QAItem.from_dict(d) for d in read_jsonl(p.data / "train.jsonl")]
        write_jsonl(p.probe / "train.jsonl", run_probe(gen, train, cfg.probe))
        if cfg.eval.probe_samples > 0:
            for spec in cfg.data.eval:
                items = [QAItem.from_dict(d) for d in read_jsonl(p.eval_items(spec.name))]
                recs = run_probe(gen, items, cfg.probe, num_samples=cfg.eval.probe_samples)
                write_jsonl(p.eval_probe(spec.name), recs)
    finally:
        gen.close()


def stage_build(cfg: RunConfig) -> None:
    p = Paths(cfg)
    records = read_jsonl(p.probe / "train.jsonl")
    sft_recs, dpo_recs = split_for_dpo(records, cfg.build.dpo_fraction, cfg.seed)
    sft = build_sft(sft_recs, cfg.build, cfg.seed)
    dpo = build_dpo(dpo_recs, cfg.build)
    write_jsonl(p.datasets / "sft.jsonl", sft)
    write_jsonl(p.datasets / "dpo.jsonl", dpo)

    def count(rows, key):
        out: dict[str, int] = {}
        for r in rows:
            out[r[key]] = out.get(r[key], 0) + 1
        return dict(sorted(out.items()))

    stats = {
        "probed_questions": len(records),
        "probe_labels": count(records, "label"),
        "base_greedy_accuracy_on_train": sum(r["greedy_outcome"] == "correct" for r in records) / max(1, len(records)),
        "sft_examples": len(sft),
        "sft_by_label": count(sft, "label"),
        "dpo_pairs": len(dpo),
        "dpo_by_type": count(dpo, "pair_type"),
    }
    (p.datasets / "build_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("build stats: %s", stats)


def stage_sft(cfg: RunConfig) -> None:
    from .train import train_sft

    p = Paths(cfg)
    _seed_everything(cfg.seed)
    train_sft(cfg.model.base, read_jsonl(p.datasets / "sft.jsonl"), p.model("sft", "adapter"),
              p.model("sft", "merged"), cfg)


def stage_dpo(cfg: RunConfig) -> None:
    from .train import train_dpo

    p = Paths(cfg)
    _seed_everything(cfg.seed)
    train_dpo(str(p.model("sft", "merged")), read_jsonl(p.datasets / "dpo.jsonl"), p.model("dpo", "adapter"),
              p.model("dpo", "merged"), cfg)


def stage_evaluate(cfg: RunConfig) -> None:
    from .evaluate import compute_metrics, predict

    p = Paths(cfg)
    _seed_everything(cfg.seed)
    items = {spec.name: read_jsonl(p.eval_items(spec.name)) for spec in cfg.data.eval}
    model_for = {
        "base": (cfg.model.base, False),
        "base_prompted": (cfg.model.base, True),
        "sft": (str(p.model("sft", "merged")), False),
        "sft_dpo": (str(p.model("dpo", "merged")), False),
    }
    pred_dir = p.eval / "predictions"
    preds: dict[str, dict[str, list[dict]]] = {name: {} for name in items}

    # group systems sharing a checkpoint so each model is loaded once
    by_model: dict[str, list[str]] = {}
    for system in cfg.eval.systems:
        by_model.setdefault(model_for[system][0], []).append(system)
    for model_path, systems in by_model.items():
        gen = _generator(cfg, model_path)
        try:
            for system in systems:
                for name, ds_items in items.items():
                    rows = predict(gen, ds_items, abstain_instruction=model_for[system][1])
                    write_jsonl(pred_dir / f"{system}__{name}.jsonl", rows)
                    preds[name][system] = rows
                    log.info("evaluated %s on %s", system, name)
        finally:
            gen.close()

    # keep system order stable as configured
    preds = {ds: {s: preds[ds][s] for s in cfg.eval.systems} for ds in preds}
    knowledge = {}
    for name in items:
        kp = p.eval_probe(name)
        if kp.exists():
            knowledge[name] = {r["id"]: r["label"] for r in read_jsonl(kp)}
    metrics = compute_metrics(preds, knowledge, cfg.eval, cfg.seed)
    (p.eval / "metrics.json").write_text(json.dumps(metrics, indent=2))


def stage_gates(cfg: RunConfig, gates_path: str) -> bool:
    from .gates import blocking_failures, evaluate_gates, load_gates, to_dicts

    p = Paths(cfg)
    metrics = json.loads((p.eval / "metrics.json").read_text())
    results = evaluate_gates(load_gates(gates_path), metrics, cfg.data.in_distribution)
    p.report.mkdir(parents=True, exist_ok=True)
    (p.report / "gates.json").write_text(json.dumps(to_dicts(results), indent=2))
    for r in results:
        log.info("[%s] %-4s %s (%s): %s", r.severity, "PASS" if r.passed else "FAIL", r.name, r.dataset, r.detail)
    return not blocking_failures(results)


def stage_report(cfg: RunConfig) -> Path:
    from .report import write_report

    path = write_report(cfg.run_dir, cfg.run_name, cfg.model.base, cfg.data.in_distribution, cfg.eval.ece_bins,
                        cfg.eval.target_precision)
    log.info("report -> %s", path)
    return path

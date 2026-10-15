from pathlib import Path

import pytest

from abstention_tuning.config import DatasetSpec, EvalConfig, ProbeConfig, RunConfig, load_config
from abstention_tuning.data import QAItem, disjoint, load_items, map_nq_open, map_popqa, map_triviaqa
from abstention_tuning.evaluate import compute_metrics
from abstention_tuning.gates import Gate, blocking_failures, evaluate_gates
from abstention_tuning.probe import assemble, label_knowledge

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- probe

def test_label_knowledge():
    cfg = ProbeConfig(known_threshold=0.6, unknown_threshold=0.0)
    assert label_knowledge(0.8, True, cfg) == "known"
    assert label_knowledge(0.8, False, cfg) == "partial"
    assert label_knowledge(0.0, False, cfg) == "unknown"
    assert label_knowledge(0.0, True, cfg) == "partial"
    assert label_knowledge(0.3, False, cfg) == "partial"


def test_assemble_record():
    item = QAItem("1", "Capital of France?", ["Paris"], "t")
    r = assemble(item, {"text": "Paris", "confidence": 0.8}, ["Paris", "Lyon", "Paris", "I don't know"], ProbeConfig())
    assert r["greedy_outcome"] == "correct"
    assert r["p_correct"] == 0.5
    assert r["label"] == "partial"
    assert [s["outcome"] for s in r["samples"]] == ["correct", "incorrect", "correct", "abstain"]


# ---------------------------------------------------------------- data

def test_hf_row_mappers():
    t = map_triviaqa({"question": "Q?", "question_id": "tq1",
                      "answer": {"value": "Paris", "aliases": ["Paris", "Paree"], "normalized_aliases": ["paris"]}}, 0)
    assert t.id == "tq1" and t.aliases == ["Paris", "Paree"]
    p = map_popqa({"id": 7, "question": "Q?", "possible_answers": '["a", "b"]'}, 0)
    assert p.aliases == ["a", "b"]
    n = map_nq_open({"question": "who wrote hamlet", "answer": ["Shakespeare"]}, 3)
    assert n.question == "Who wrote hamlet?" and n.id == "nq-3"


def test_jsonl_loader_and_disjoint():
    spec = DatasetSpec(name="fx", source="jsonl", path=str(ROOT / "tests/fixtures/qa_train.jsonl"), limit=5)
    items = load_items(spec, seed=1)
    assert len(items) == 5 and all(it.aliases for it in items)
    assert load_items(spec, seed=1) == items  # deterministic
    kept = disjoint(items, {"e": items[:2]})
    assert len(kept) == 3


# ---------------------------------------------------------------- config

@pytest.mark.parametrize("name", ["qwen2.5-1.5b.yaml", "qwen2.5-1.5b-quick.yaml", "smoke.yaml"])
def test_shipped_configs_validate(name):
    cfg = load_config(ROOT / "configs" / name)
    assert cfg.data.in_distribution in [d.name for d in cfg.data.eval]


def test_in_distribution_must_exist():
    with pytest.raises(ValueError):
        RunConfig.model_validate({"data": {"train": {"name": "a", "source": "triviaqa"},
                                           "eval": [{"name": "b", "source": "popqa"}], "in_distribution": "zzz"}})


# ---------------------------------------------------------------- evaluate + gates

def _preds(outcomes, confs):
    return [{"id": str(i), "outcome": o, "confidence": c} for i, (o, c) in enumerate(zip(outcomes, confs))]


def _metrics():
    base = _preds(["correct", "incorrect", "incorrect", "correct"], [0.9, 0.3, 0.6, 0.8])
    tuned = _preds(["correct", "abstain", "abstain", "correct"], [0.9, 0, 0, 0.8])
    preds = {"ind": {"base": base, "sft_dpo": tuned}, "ood": {"base": base, "sft_dpo": base}}
    knowledge = {"ind": {"0": "known", "1": "unknown", "2": "unknown", "3": "known"}}
    return compute_metrics(preds, knowledge, EvalConfig(bootstrap=10))


def test_compute_metrics_adds_matched_base():
    m = _metrics()
    mb = m["ind"]["sft_dpo"]["matched_base"]
    assert mb["coverage"] == 0.5 and mb["selective_accuracy"] == 1.0
    assert m["ind"]["sft_dpo"]["over_abstention_rate"] == 0.0
    assert "over_abstention_rate" not in m["ood"]["sft_dpo"]


def test_gates():
    m = _metrics()
    gates = [
        Gate(name="halluc", system="sft_dpo", metric="hallucination_rate", kind="relative_reduction",
             compare_to="base", threshold=0.3),
        Gate(name="acc", system="sft_dpo", metric="accuracy", kind="max_drop", compare_to="base", threshold=0.05),
        Gate(name="ood", system="sft_dpo", metric="hallucination_rate", kind="relative_reduction",
             compare_to="base", threshold=0.15, dataset="ood", severity="warn"),
        Gate(name="matched", system="sft_dpo", metric="selective_accuracy", kind="min_delta",
             compare_to="matched_base", threshold=0.0),
        Gate(name="missing", system="sft", metric="accuracy", kind="min", threshold=0.0, severity="warn"),
    ]
    res = {r.name: r for r in evaluate_gates(gates, m, "ind")}
    assert res["halluc"].passed            # 0.5 -> 0.0
    assert res["acc"].passed               # 0.5 -> 0.5
    assert not res["ood"].passed and res["ood"].dataset == "ood"
    assert res["matched"].passed
    assert not res["missing"].passed
    assert blocking_failures(list(res.values())) == []


def test_gate_files_parse():
    from abstention_tuning.gates import load_gates

    assert len(load_gates(str(ROOT / "configs/gates.yaml"))) >= 5

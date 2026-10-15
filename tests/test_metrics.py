import pytest

from abstention_tuning.metrics import (
    at_coverage,
    aurc,
    bootstrap_ci,
    coverage_at_precision,
    ece,
    rates,
    risk_coverage,
    summarize,
)

OUT = ["correct", "incorrect", "abstain", "correct"]


def test_rates():
    r = rates(OUT)
    assert r["n"] == 4
    assert r["accuracy"] == 0.5
    assert r["hallucination_rate"] == 0.25
    assert r["abstention_rate"] == 0.25
    assert r["selective_accuracy"] == pytest.approx(2 / 3)


def test_rates_empty():
    assert rates([])["n"] == 0


def test_ece_zero_when_confidence_equals_accuracy():
    conf = [1.0, 1.0, 0.0, 0.0]
    correct = [True, True, False, False]
    assert ece(conf, correct, bins=10) == pytest.approx(0.0)


def test_ece_overconfident():
    # always 0.9 confident, right half the time -> ECE 0.4
    assert ece([0.9] * 4, [True, False, True, False]) == pytest.approx(0.4)


def test_risk_coverage_orders_by_confidence_and_skips_abstentions():
    out = ["correct", "incorrect", "abstain", "correct"]
    conf = [0.9, 0.2, 0.99, 0.5]
    rc = risk_coverage(out, conf)
    assert rc["coverage"] == pytest.approx([0.25, 0.5, 0.75])
    assert rc["risk"] == pytest.approx([0.0, 0.0, 1 / 3])
    assert aurc(out, conf) == pytest.approx((0 + 0 + 1 / 3) / 3)


def test_at_coverage_matches_threshold_baseline():
    out = ["correct", "incorrect", "correct", "incorrect"]
    conf = [0.9, 0.1, 0.8, 0.7]
    m = at_coverage(out, conf, 0.5)  # keep the two most confident -> both correct
    assert m["coverage"] == 0.5
    assert m["selective_accuracy"] == 1.0
    assert m["hallucination_rate"] == 0.0
    assert m["accuracy"] == 0.5


def test_coverage_at_precision():
    out = ["correct", "correct", "incorrect", "incorrect"]
    conf = [0.9, 0.8, 0.7, 0.6]
    assert coverage_at_precision(out, conf, 1.0) == 0.5
    assert coverage_at_precision(out, conf, 0.6) == 0.75
    assert coverage_at_precision(["incorrect"], [0.9], 0.9) == 0.0


def test_bootstrap_ci_brackets_point_estimate():
    out = ["correct"] * 60 + ["incorrect"] * 40
    lo, hi = bootstrap_ci(out, "accuracy", n_boot=300, seed=1)
    assert lo < 0.6 < hi


def test_summarize_with_knowledge_labels():
    out = ["abstain", "correct", "incorrect", "abstain"]
    conf = [0.0, 0.9, 0.4, 0.0]
    kn = ["known", "known", "unknown", "unknown"]
    s = summarize(out, conf, kn, n_boot=10)
    assert s["over_abstention_rate"] == 0.5          # refused 1 of 2 known
    assert s["unknown_hallucination_rate"] == 0.5    # guessed 1 of 2 unknown
    assert s["net_score"] == pytest.approx(0.25 - 0.25)
    assert "coverage_at_90pct_precision" in s

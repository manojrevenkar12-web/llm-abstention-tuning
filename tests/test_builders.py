from abstention_tuning.builders import (
    build_dpo,
    build_sft,
    correct_answer,
    most_common,
    split_for_dpo,
    wrong_answer,
)
from abstention_tuning.config import BuildConfig
from abstention_tuning.prompts import ABSTAIN_TEXT, PLAIN_SYSTEM


def rec(i, label, greedy, g_out, samples, aliases=("Paris",)):
    return {"id": str(i), "question": f"Q{i}?", "aliases": list(aliases), "source": "t", "label": label,
            "greedy": greedy, "greedy_outcome": g_out, "greedy_confidence": 0.5,
            "samples": [{"text": t, "outcome": o} for t, o in samples], "p_correct": 0.0}


KNOWN = rec(1, "known", "paris", "correct", [("Paris, France", "correct"), ("paris", "correct"), ("Lyon", "incorrect")])
UNKNOWN = rec(2, "unknown", "Lyon", "incorrect", [("Lyon", "incorrect"), ("Nice", "incorrect")])
PARTIAL = rec(3, "partial", "Lyon", "incorrect", [("Paris", "correct"), ("Lyon", "incorrect")])


def test_most_common_uses_normalised_counts_keeps_original_phrasing():
    assert most_common(["Paris", "paris.", "Lyon"]) == "Paris"
    assert most_common(["", "  "]) is None


def test_correct_and_wrong_answers_come_from_the_model():
    assert correct_answer(KNOWN) == "paris"          # its own most frequent correct phrasing
    assert wrong_answer(UNKNOWN) == "Lyon"
    assert wrong_answer(rec(4, "known", "Paris", "correct", [("Paris", "correct")])) is None


def test_sft_targets_and_plain_prompt():
    rows = build_sft([KNOWN, UNKNOWN, PARTIAL], BuildConfig(partial_policy_sft="skip"))
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {"1", "2"}
    assert by_id["1"]["completion"][0]["content"] == "paris"
    assert by_id["2"]["completion"][0]["content"] == ABSTAIN_TEXT
    assert by_id["1"]["prompt"][0]["content"] == PLAIN_SYSTEM  # no abstain instruction during training


def test_sft_partial_policies():
    abstain = build_sft([PARTIAL], BuildConfig(partial_policy_sft="abstain"))
    assert abstain[0]["completion"][0]["content"] == ABSTAIN_TEXT
    assert build_sft([PARTIAL], BuildConfig(partial_policy_sft="answer"))[0]["completion"][0]["content"] == "Paris"


def test_sft_unknown_cap():
    unknowns = [rec(10 + i, "unknown", "x", "incorrect", [("x", "incorrect")]) for i in range(5)]
    rows = build_sft([KNOWN, *unknowns], BuildConfig(max_unknown_ratio=2.0))
    assert sum(r["label"] == "unknown" for r in rows) == 2


def test_dpo_pairs_follow_preference_order():
    rows = build_dpo([KNOWN, UNKNOWN, PARTIAL], BuildConfig())
    types = sorted(r["pair_type"] for r in rows)
    assert types == ["known:answer>abstain", "known:answer>wrong", "partial:answer>wrong", "unknown:abstain>wrong"]
    unk = next(r for r in rows if r["pair_type"].startswith("unknown"))
    assert unk["chosen"][0]["content"] == ABSTAIN_TEXT and unk["rejected"][0]["content"] == "Lyon"
    for r in rows:
        assert r["chosen"][0]["content"] != r["rejected"][0]["content"]


def test_split_is_stratified_and_disjoint():
    recs = [rec(i, lab, "a", "correct", []) for i, lab in enumerate(["known"] * 10 + ["unknown"] * 10)]
    sft, dpo = split_for_dpo(recs, 0.3, seed=0)
    assert not {r["id"] for r in sft} & {r["id"] for r in dpo}
    assert sum(r["label"] == "known" for r in dpo) == 3
    assert sum(r["label"] == "unknown" for r in dpo) == 3

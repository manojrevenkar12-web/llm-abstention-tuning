import pytest

from abstention_tuning.grading import clean_answer, grade, is_abstention, matches_any, normalize


def test_normalize_strips_case_punctuation_articles():
    assert normalize("The  River Nile!") == "river nile"
    assert normalize("Don't") == "dont"


@pytest.mark.parametrize("text", [
    "I don't know.", "I do not know", "i dont know", "I'm not sure.", "I am not certain",
    "Not sure", "No idea", "I cannot answer that.", "Unknown", "", "   ",
])
def test_abstentions_detected(text):
    assert is_abstention(text)


@pytest.mark.parametrize("text", ["Paris", "Mount Everest", "Unknown Pleasures album", "Knowles"])
def test_answers_are_not_abstentions(text):
    assert not is_abstention(text)


def test_alias_match_on_word_boundaries():
    assert matches_any("Paris, France", ["Paris"])
    assert matches_any("the Pacific", ["Pacific Ocean", "Pacific"])
    assert not matches_any("Romeo", ["Rome"])
    assert not matches_any("", ["Rome"])


def test_grade_outcomes():
    assert grade("Paris", ["Paris"]) == "correct"
    assert grade("Lyon", ["Paris"]) == "incorrect"
    assert grade("I don't know.", ["Paris"]) == "abstain"


def test_clean_answer_takes_first_line_and_strips_quotes():
    assert clean_answer('  "Paris"\nIt is the capital.') == "Paris"
    assert clean_answer("\n\nBerlin") == "Berlin"

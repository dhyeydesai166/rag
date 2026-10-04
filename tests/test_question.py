import pytest

from rag.question import clean_question, junk_reason

try:
    from eval_set import CASES
except ImportError:
    from tests.eval_set import CASES


def test_whitespace_and_curly_quotes_are_normalized():
    assert clean_question("  who  gets \u201ccake\u201d?  ") == 'who gets "cake"?'


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("v2", "2.0"),
        ("V2.0", "2.0"),
        ("version 2", "2.0"),
        ("ver. 1", "1.0"),
        ("6:00 a.m.", "6:00 a.m."),
    ],
)
def test_version_forms_are_normalized(raw, expected):
    assert clean_question(raw) == expected


def test_empty_input_is_rejected():
    assert junk_reason(clean_question("   "))
    assert "Please type a question" in junk_reason("")


def test_symbols_and_numbers_only_are_rejected():
    assert junk_reason("???")
    assert junk_reason("12345")


@pytest.mark.parametrize("word", ["hi", "OK", "test"])
def test_single_filler_word_is_rejected(word):
    assert "Ask me about" in junk_reason(clean_question(word))


@pytest.mark.parametrize("text", ["asdfghjkl", "qwrtzp", "aaaaaaa"])
def test_keymash_is_rejected(text):
    assert junk_reason(text)


def test_long_input_is_rejected():
    assert "too long" in junk_reason("a" * 501)


def test_real_questions_pass():
    for case in CASES:
        assert junk_reason(clean_question(case["question"])) is None

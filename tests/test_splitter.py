from rag.chunker import _record
from rag.splitter import (
    estimate_tokens,
    hard_split,
    split_long_chunk,
    split_sentences,
)


def _long_record():
    sentences = [
        f"Rule {number} says employees must follow requirement {number} carefully."
        for number in range(1, 81)
    ]
    text = " ".join(sentences)
    return _record(
        "HR Policy",
        "2.0",
        "hr.docx",
        "9. Long Rule",
        "9. Long Rule",
        "HR Policy|2.0",
        text,
    )


def test_thousand_word_subsection_splits_into_pieces_under_the_cap():
    record = _long_record()
    pieces = split_long_chunk(record)
    assert len(pieces) > 1
    assert all(estimate_tokens(piece["text"]) <= 256 for piece in pieces)
    assert all("9. Long Rule" in piece["embed_text"] for piece in pieces)
    for index in range(len(pieces) - 1):
        left = pieces[index]
        right = pieces[index + 1]
        left_sentences = split_sentences(left["text"])
        right_sentences = split_sentences(right["text"])
        assert left_sentences[-1] == right_sentences[0]
        assert set(left_sentences[:-1]).isdisjoint(right_sentences[1:])
    assert [piece["id"].rsplit("#", 1)[-1] for piece in pieces] == [
        str(number) for number in range(1, len(pieces) + 1)
    ]
    again = split_long_chunk(record)
    assert [(piece["id"], piece["text"]) for piece in pieces] == [
        (piece["id"], piece["text"]) for piece in again
    ]


def test_short_chunks_are_never_split():
    record = _record(
        "HR Policy",
        "2.0",
        "hr.docx",
        "1. Purpose",
        "1. Purpose",
        "HR Policy|2.0",
        "Wait 30 minutes.",
    )
    pieces = split_long_chunk(record)
    assert len(pieces) == 1
    assert "#" not in pieces[0]["id"]


def test_single_oversized_sentence_is_hard_split():
    sentence = " ".join(f"word{number}" for number in range(400))
    parts = hard_split(sentence, 50)
    assert len(parts) > 1
    assert all(estimate_tokens(part) <= 50 for part in parts)
    assert " ".join(parts) == sentence


def test_abbreviations_do_not_end_a_sentence():
    sentences = split_sentences("The gym opens at 6:00 a.m. Employees must arrive.")
    assert sentences == [
        "The gym opens at 6:00 a.m.",
        "Employees must arrive.",
    ]


def test_estimate_tokens_counts_digits_individually():
    assert estimate_tokens("1,000,000") == 9

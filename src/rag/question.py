"""Clean a question and reject input that is not one.

No model is involved: the rules are short and the messages are fixed.
"""

import re
import string

from rag.config import FILLER_WORDS, KEYMASH_CONSONANT_RUN, MAX_QUESTION_CHARS
from rag.messages import (
    FILLER_GREETING,
    NOT_A_QUESTION,
    PLEASE_TYPE_A_QUESTION,
    QUESTION_TOO_LONG,
)

_VERSION = re.compile(r"\b(?:version|ver\.?|v)\s*(\d+)(?:\.(\d+))?\b", re.IGNORECASE)
_LETTERS = re.compile(r"[A-Za-z]")
_VOWEL = re.compile(r"[aeiouy]")
_CONSONANT_RUN = re.compile(rf"[bcdfghjklmnpqrstvwxz]{{{KEYMASH_CONSONANT_RUN},}}")
_REPEAT = re.compile(r"(.)\1{4,}")


class JunkQuestion(Exception):
    """Raised with a friendly message when the input is not a real question."""


def clean_question(raw: str) -> str:
    """Trim, collapse whitespace, straighten curly quotes, and normalize version
    forms ('v2', 'V 2.0', 'version 2' -> '2.0'). No other rewriting."""
    text = (raw or "").strip()
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    text = re.sub(r"\s+", " ", text)

    def replace(match: re.Match) -> str:
        minor = match.group(2) or "0"
        return f"{match.group(1)}.{minor}"

    return _VERSION.sub(replace, text)


def _is_empty(question: str) -> str | None:
    if not question:
        return PLEASE_TYPE_A_QUESTION
    return None


def _is_too_long(question: str) -> str | None:
    if len(question) > MAX_QUESTION_CHARS:
        return QUESTION_TOO_LONG
    return None


def _has_no_letters(question: str) -> str | None:
    if not _LETTERS.search(question):
        return NOT_A_QUESTION
    return None


def _is_filler(question: str) -> str | None:
    """A lone greeting or test word, ignoring punctuation around it ('Hi!', 'ok.')."""
    if question.strip(string.punctuation + " ").lower() in FILLER_WORDS:
        return FILLER_GREETING
    return None


def _is_keymash(question: str) -> str | None:
    if _REPEAT.search(question):
        return NOT_A_QUESTION
    words = re.findall(r"[A-Za-z]+", question)
    long_words = [word.lower() for word in words if len(word) >= 4]
    if long_words and all(
        not _VOWEL.search(word) or _CONSONANT_RUN.search(word) for word in long_words
    ):
        return NOT_A_QUESTION
    return None


def junk_reason(question: str) -> str | None:
    """Return a friendly message if the input is not a real question, else None."""
    for check in (_is_empty, _is_too_long, _has_no_letters, _is_filler, _is_keymash):
        reason = check(question)
        if reason:
            return reason
    return None

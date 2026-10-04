"""Split one section's text so a chunk stays under the token cap.

Splitting never crosses a heading: the caller passes a single section.
"a.m." and "p.m." may end a sentence ("10:00 p.m. Employees"), but their
internal periods are not boundaries, and "No. 5" or "e.g. Towels" stay whole.
"""

import re

from rag.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_SENTENCES,
    SENTENCE_ABBREVIATIONS,
)

TOKEN_PIECES = re.compile(r"\d|[^\W\d_]+|[^\w\s]")
_START = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"“'‘")
# These abbreviations are also real sentence endings in the corpus.
_MAY_END_SENTENCE = {"a.m.", "p.m."}


def estimate_tokens(text: str) -> int:
    """Rough token count: one per word, one per digit, one per punctuation mark.

    Why: Gemma tokenizers split numbers into single digits, so counting digits
    separately keeps the estimate on the high side. The cap is a target;
    Ollama errors (truncate=False) if a chunk ever overflows the real context.
    """
    return len(TOKEN_PIECES.findall(text))


def _abbreviation_at_end(prefix: str) -> str | None:
    lowered = prefix.lower()
    for abbrev in sorted(SENTENCE_ABBREVIATIONS, key=len, reverse=True):
        if lowered.endswith(abbrev.lower()):
            return abbrev.lower()
    return None


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by whitespace and an uppercase letter, digit or
    quote, except after a known abbreviation."""
    sentences = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] not in ".!?":
            index += 1
            continue
        rest = text[index + 1 :]
        gap = re.match(r"\s+", rest)
        if not gap:
            index += 1
            continue
        after = rest[gap.end() :]
        if not after or after[0] not in _START:
            index += 1
            continue
        prefix = text[start : index + 1]
        abbrev = _abbreviation_at_end(prefix)
        if abbrev and abbrev not in _MAY_END_SENTENCE:
            index += 1
            continue
        piece = prefix.strip()
        if piece:
            sentences.append(piece)
        start = index + 1 + gap.end()
        index = start
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def hard_split(sentence: str, max_tokens: int) -> list[str]:
    """Cut a single oversized sentence into word groups under max_tokens.

    Last resort only: a sentence longer than a chunk has no better boundary.
    """
    if estimate_tokens(sentence) <= max_tokens:
        return [sentence]
    words = sentence.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if current and estimate_tokens(trial) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def group_sentences(sentences: list[str], max_tokens: int, overlap: int) -> list[str]:
    """Greedily pack sentences into pieces; each new piece starts with the last
    `overlap` sentences of the previous piece."""
    pieces: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        joined = " ".join([*current, sentence])
        if current and estimate_tokens(joined) > max_tokens:
            pieces.append(current)
            carried = current[-overlap:] if overlap else []
            if estimate_tokens(" ".join([*carried, sentence])) > max_tokens:
                carried = []
            current = [*carried, sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(current)
    return [" ".join(piece) for piece in pieces]


def split_long_chunk(record: dict) -> list[dict]:
    """Return [record] if it fits, else pieces of the same section under the cap.

    Pieces keep heading_path, policy and version; each gets an ordinal (#1, #2)
    in its id and the heading in its embed_text.
    """
    from rag.chunker import make_piece

    if estimate_tokens(record["text"]) <= CHUNK_MAX_TOKENS:
        return [record]
    sentences = [
        part
        for sentence in split_sentences(record["text"])
        for part in hard_split(sentence, CHUNK_MAX_TOKENS)
    ]
    texts = group_sentences(sentences, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_SENTENCES)
    pieces = []
    for ordinal, text in enumerate(texts, start=1):
        pieces.append(make_piece(record, text, ordinal))
    return pieces

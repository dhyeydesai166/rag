"""BM25 over the same words the question uses, after stemming.

Negations stay in the index. Policy rules turn on them ("No Rollover").
"""

import math
import re

import snowballstemmer

from rag.config import BM25_B, BM25_K1, LEXICAL_TOP_K

STEMMER = snowballstemmer.stemmer("english")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "could",
        "would",
        "may",
        "might",
        "must",
        "shall",
        "do",
        "does",
        "did",
        "doing",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "where",
        "why",
        "as",
    }
)


def analyze(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords, stem.

    The same function is used for chunks and queries, so 'allocated' in a
    question matches 'allocation' in a heading.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    kept = [word for word in words if word not in STOPWORDS]
    return STEMMER.stemWords(kept)


def bm25_scores(
    query_terms: list[str],
    documents: list[list[str]],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    """Okapi BM25 on already-analyzed terms."""
    if not documents:
        return []
    lengths = [len(doc) for doc in documents]
    average = sum(lengths) / len(lengths) if lengths else 0
    counts = []
    document_frequency = {term: 0 for term in set(query_terms)}
    for doc in documents:
        count: dict[str, int] = {}
        for term in doc:
            count[term] = count.get(term, 0) + 1
        counts.append(count)
        for term in document_frequency:
            if term in count:
                document_frequency[term] += 1
    total = len(documents)
    scores = []
    for length, count in zip(lengths, counts, strict=True):
        if average == 0:
            scores.append(0.0)
            continue
        score = 0.0
        for term in query_terms:
            frequency = count.get(term, 0)
            if frequency == 0:
                continue
            found = document_frequency[term]
            idf = math.log(1 + (total - found + 0.5) / (found + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / average)
            score += idf * (frequency * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def bm25_ranked(
    search_text: str, chunks: list[dict], top_k: int = LEXICAL_TOP_K
) -> list[str]:
    """Chunk ids with BM25 score > 0, best first, ties broken by chunk id.

    Why only > 0: a chunk sharing no term with the question is not lexical
    evidence and must not earn a rank (or RRF credit) from its list position.
    """
    if not chunks:
        return []
    query_terms = analyze(search_text)
    documents = [analyze(chunk.get("embed_text") or chunk["text"]) for chunk in chunks]
    scores = bm25_scores(query_terms, documents)
    ranked = sorted(
        range(len(chunks)),
        key=lambda index: (-scores[index], chunks[index]["id"]),
    )
    return [chunks[index]["id"] for index in ranked if scores[index] > 0][:top_k]

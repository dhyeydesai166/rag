"""Choose lookup or compare by nearest example questions.

A lookup on the latest version is still a useful answer when the question
is ambiguous. A comparison that the user did not ask for is not, so compare
has to win by a margin.
"""

import math
import sys

from rag.config import ROUTE_COMPARE_MARGIN, ROUTE_MIN_SIMILARITY
from rag.route_examples import COMPARE_EXAMPLES, LOOKUP_EXAMPLES

_CACHE: dict[int, tuple[list[list[float]], list[list[float]]]] = {}


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def embed_examples(embedder) -> tuple[list[list[float]], list[list[float]]]:
    """Embed both example lists once per process with the 'similarity' task."""
    key = id(embedder)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    lookup = embedder.embed(list(LOOKUP_EXAMPLES), task="similarity")
    compare = embedder.embed(list(COMPARE_EXAMPLES), task="similarity")
    _CACHE[key] = (lookup, compare)
    return _CACHE[key]


def choose_route(
    best_lookup: float,
    best_compare: float,
    margin: float = ROUTE_COMPARE_MARGIN,
    min_similarity: float = ROUTE_MIN_SIMILARITY,
) -> str:
    """'compare' only for a clear match that is clearly closer to compare.

    Why the floor: a greeting or an off-topic question is far from every
    example, and which weak match is slightly closer says nothing.
    """
    if max(best_lookup, best_compare) < min_similarity:
        return "lookup"
    if best_compare >= best_lookup + margin:
        return "compare"
    return "lookup"


def classify_route(
    question_vector: list[float],
    lookup_vectors: list[list[float]],
    compare_vectors: list[list[float]],
    margin: float = ROUTE_COMPARE_MARGIN,
    min_similarity: float = ROUTE_MIN_SIMILARITY,
) -> str:
    """'compare' only if the question is clearly closer to a compare example.

    Why lean to lookup: a lookup on the latest version is still a correct answer
    for an ambiguous question; a spurious comparison is confusing.
    """
    best_lookup, best_compare = route_scores(
        question_vector, lookup_vectors, compare_vectors
    )
    return choose_route(best_lookup, best_compare, margin, min_similarity)


def route_scores(
    question_vector: list[float],
    lookup_vectors: list[list[float]],
    compare_vectors: list[list[float]],
) -> tuple[float, float]:
    best_lookup = max(cosine(question_vector, vector) for vector in lookup_vectors)
    best_compare = max(cosine(question_vector, vector) for vector in compare_vectors)
    return best_lookup, best_compare


def main(argv=None) -> int:
    """Print the route for each eval question. Needs a running Ollama server."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["report"]:
        print("usage: python -m rag.route report")
        return 2
    from pathlib import Path

    from adapter.embedding_adapter import EmbeddingAdapter

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from evals.cases import CASES
    from rag.question import clean_question

    embedder = EmbeddingAdapter()
    lookup_vectors, compare_vectors = embed_examples(embedder)
    lowest_score = None
    lowest_id = ""
    problems = 0
    for case in CASES:
        cleaned = clean_question(case["question"])
        vector = embedder.embed([cleaned], task="similarity")[0]
        best_lookup, best_compare = route_scores(
            vector, lookup_vectors, compare_vectors
        )
        chosen = classify_route(vector, lookup_vectors, compare_vectors)
        expected = case.get("route", "")
        best = max(best_lookup, best_compare)
        if lowest_score is None or best < lowest_score:
            lowest_score = best
            lowest_id = case["id"]
        if chosen != expected or best < ROUTE_MIN_SIMILARITY:
            problems += 1
        print(
            f"chosen={chosen} expected={expected} "
            f"best_lookup={best_lookup:.3f} best_compare={best_compare:.3f} "
            f"question={case['question']}"
        )
    print(
        f"lowest best similarity: {lowest_score:.3f} ({lowest_id}); "
        f"ROUTE_MIN_SIMILARITY={ROUTE_MIN_SIMILARITY:.2f}"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

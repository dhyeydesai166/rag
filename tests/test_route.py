import pytest

from rag.route import choose_route, classify_route, cosine


def test_cosine_handles_empty_and_mismatched_vectors():
    assert cosine([], [1.0]) == 0.0
    assert cosine([1.0], [1.0, 0.0]) == 0.0
    assert cosine([0.0], [0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_question_near_a_compare_example_is_compare():
    assert classify_route([0.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]]) == "compare"


def test_question_near_a_lookup_example_is_lookup():
    assert classify_route([1.0, 0.0], [[1.0, 0.0]], [[0.0, 1.0]]) == "lookup"


def test_tie_goes_to_lookup():
    assert classify_route([1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]]) == "lookup"


def test_weak_match_goes_to_lookup_even_when_compare_is_closer():
    assert choose_route(0.392, 0.416) == "lookup"


def test_clear_compare_above_the_floor_is_compare():
    assert choose_route(0.581, 0.690) == "compare"


def test_the_floor_is_inclusive():
    assert choose_route(0.30, 0.50, min_similarity=0.50) == "compare"


def test_near_tie_above_the_floor_is_lookup():
    assert choose_route(0.600, 0.610) == "lookup"


def test_a_new_embedder_does_not_reuse_a_freed_embedders_examples():
    import gc

    from rag.route import embed_examples

    class Counter:
        def __init__(self, marker: float) -> None:
            self.marker = marker
            self.calls = 0

        def embed(self, texts, task):
            self.calls += 1
            return [[float(self.marker)] for _ in texts]

    first = Counter(1)
    embed_examples(first)
    del first
    gc.collect()
    second = Counter(2)
    embed_examples(second)
    del second
    gc.collect()
    third = Counter(3)
    embed_examples(third)
    assert third.calls == 2


def test_margin_is_respected():
    lookup = [[1.0, 0.0]]
    compare = [[0.98, 0.2]]
    question = [1.0, 0.0]
    assert classify_route(question, lookup, compare, margin=0.5) == "lookup"


def test_compare_on_single_version_policy_falls_back_to_lookup(tmp_path, rag_logs):
    from test_retrieve import FakeEmbedder, FakeReranker, record, store

    from rag.retrieve import retrieve

    rows = [record("Health & Wellness Policy", "1.0", "1. Purpose", "rest")]
    embedder = FakeEmbedder([0.0, 1.0], compare=True)
    found = retrieve(
        "What changed in the Health & Wellness Policy?",
        embedder,
        store(tmp_path / "chroma", rows, [[1.0, 0.0]]),
        FakeReranker(),
        embedder.examples(),
    )
    assert found["kind"] == "lookup"
    assert "reason=single version" in rag_logs.text


@pytest.mark.ollama
def test_every_eval_question_gets_its_expected_route():
    from adapter.embedding_adapter import EmbeddingAdapter
    from evals.cases import CASES
    from rag.question import clean_question
    from rag.route import embed_examples

    labeled = [case for case in CASES if case.get("route")]
    embedder = EmbeddingAdapter()
    lookup_vectors, compare_vectors = embed_examples(embedder)
    misses = []
    for case in labeled:
        vector = embedder.embed([clean_question(case["question"])], task="similarity")[
            0
        ]
        chosen = classify_route(vector, lookup_vectors, compare_vectors)
        if chosen != case["route"]:
            misses.append(case["question"])
    assert misses == []


@pytest.mark.ollama
def test_every_eval_question_clears_the_route_floor():
    from adapter.embedding_adapter import EmbeddingAdapter
    from evals.cases import CASES
    from rag.config import ROUTE_MIN_SIMILARITY
    from rag.question import clean_question
    from rag.route import embed_examples, route_scores

    embedder = EmbeddingAdapter()
    lookup_vectors, compare_vectors = embed_examples(embedder)
    below = []
    for case in CASES:
        vector = embedder.embed([clean_question(case["question"])], task="similarity")[
            0
        ]
        best_lookup, best_compare = route_scores(
            vector, lookup_vectors, compare_vectors
        )
        if max(best_lookup, best_compare) < ROUTE_MIN_SIMILARITY:
            below.append(case["id"])
    assert below == []

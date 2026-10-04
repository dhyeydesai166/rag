from rag.lexical import STOPWORDS, analyze, bm25_ranked, bm25_scores


def test_bm25_scores_keyword_overlap():
    documents = [analyze(text) for text in ["birthday cake", "vacation days", "???"]]
    scores = bm25_scores(analyze("cake"), documents)
    assert scores[0] > scores[1]
    assert scores[1] == 0
    assert bm25_scores(analyze("cake"), []) == []
    assert bm25_scores(analyze("cake"), [analyze("???")]) == [0.0]


def test_chunks_without_any_query_term_are_excluded():
    chunks = [
        {"id": "cake", "embed_text": "birthday cake"},
        {"id": "vacation", "embed_text": "vacation days"},
    ]
    assert bm25_ranked("cake", chunks) == ["cake"]


def test_stemming_matches_word_forms():
    chunks = [{"id": "amount", "embed_text": "Allocation Amount"}]
    assert bm25_ranked("allocated", chunks) == ["amount"]


def test_stopwords_are_ignored_but_negations_are_kept():
    assert "the" not in analyze("the cake")
    assert "cake" in analyze("the cake")
    for word in ("no", "not", "never", "without"):
        assert word not in STOPWORDS
        assert analyze(word) == [word]


def test_heading_words_are_searchable():
    chunks = [
        {
            "id": "a",
            "text": "employees receive some",
            "embed_text": "Allocation Amount\nemployees receive some",
        }
    ]
    assert bm25_ranked("allocation", chunks) == ["a"]


def test_same_analyzer_for_query_and_chunks():
    text = "No Rollover of tokens"
    assert analyze(text) == analyze(text)

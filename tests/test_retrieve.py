import sys
from types import SimpleNamespace

from adapter.database_adapter import DatabaseAdapter
from rag.logutil import disable_question_log, enable_question_log, stage
from rag.manifest import IndexMissing
from rag.retrieve import main, retrieve
from rag.route_examples import COMPARE_EXAMPLES, LOOKUP_EXAMPLES


def record(policy, version, heading, text):
    return {
        "id": f"{policy}|{version}|{heading}",
        "text": text,
        "policy": policy,
        "version": version,
        "section": heading,
        "heading_path": heading,
        "parent_id": f"{policy}|{version}",
        "source": "policy.docx",
        "word_count": len(text.split()),
        "embed_text": f"{policy} {version}\n{heading}\n{text}",
        "text_sha256": "text",
        "embed_sha256": f"{policy}|{version}|{heading}",
        "ordinal": None,
    }


def store(path, rows, vectors):
    database = DatabaseAdapter(path, "testbuild", create=True)
    database.upsert(rows, vectors)
    return database


class FakeEmbedder:
    def __init__(self, vector, compare=False):
        self.vector = list(vector)
        self.compare = compare
        self.tasks = []

    def embed(self, texts, task):
        self.tasks.append(task)
        vectors = []
        for text in texts:
            if text in COMPARE_EXAMPLES and self.compare:
                vectors.append([0.0, 1.0])
            elif text in LOOKUP_EXAMPLES:
                vectors.append([1.0, 0.0])
            elif self.compare:
                vectors.append([0.0, 1.0])
            else:
                vectors.append(self.vector)
        return vectors

    def examples(self):
        from rag.route import embed_examples

        return embed_examples(self)


class FakeReranker:
    def __init__(self, reverse=False, fail=False):
        self.reverse = reverse
        self.fail = fail
        self.documents = []

    def rank(self, question, documents, top_n):
        self.documents.append(list(documents))
        if self.fail:
            from adapter.rerank_adapter import RerankUnavailable

            raise RerankUnavailable("down")
        order = list(range(len(documents)))
        if self.reverse:
            order.reverse()
        return order[:top_n]


def ask(tmp_path, rows, vectors, question, compare=False, reverse=False, fail=False):
    embedder = FakeEmbedder([1.0, 0.0], compare=compare)
    reranker = FakeReranker(reverse=reverse, fail=fail)
    found = retrieve(
        question,
        embedder,
        store(tmp_path / "chroma", rows, vectors),
        reranker,
        embedder.examples(),
    )
    return found, reranker.documents


def test_lookup_drops_older_versions(tmp_path):
    rows = [
        record("HR Policy", "2.0", "3. Leave", "vacation days"),
        record("HR Policy", "1.0", "3. Leave", "birthday cake"),
    ]
    found, documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0]],
        "who gets cake?",
    )
    assert found["kind"] == "lookup"
    assert found["hits"][0]["text"] == "vacation days"
    assert "birthday cake" not in documents[0][0]


def test_lookup_keeps_a_named_version(tmp_path):
    rows = [
        record("HR Policy", "2.0", "3. Leave", "vacation days"),
        record("HR Policy", "1.0", "3. Leave", "birthday cake"),
    ]
    found, documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0]],
        "What did HR Policy 1.0 say about cake?",
    )
    assert found["hits"][0]["version"] == "1.0"
    assert "birthday cake" in documents[0][0]


def test_lookup_can_name_a_policy_without_a_version(tmp_path):
    rows = [
        record("HR Policy", "2.0", "3. Leave", "vacation days"),
        record("HR Policy", "1.0", "3. Leave", "birthday cake"),
        record("Preparedness Policy", "2.0", "1. Purpose", "drill"),
    ]
    found, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]],
        "What does the HR Policy say about leave?",
    )
    assert [hit["text"] for hit in found["hits"]] == ["vacation days"]


def test_compare_pairs_current_and_previous(tmp_path):
    rows = [
        record("HR Policy", "2.0", "3. Leave", "no dessert"),
        record("HR Policy", "1.0", "3. Leave", "cake on friday"),
        record("HR Policy", "2.0", "8. Added", "new clause"),
        record("HR Policy", "1.0", "9. Only Old", "removed clause"),
    ]
    found, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0], [0.2, 0.8], [0.8, 0.2]],
        "What changed in the HR Policy?",
        compare=True,
    )
    by_heading = {hit["heading_path"]: hit for hit in found["hits"]}
    leave = by_heading["3. Leave"]
    assert leave["current"]["text"] == "no dessert"
    assert leave["previous"]["text"] == "cake on friday"
    assert by_heading["8. Added"]["previous"] is None
    assert by_heading["9. Only Old"]["current"] is None


def test_compare_with_one_version_has_no_previous_side(tmp_path, rag_logs):
    rows = [record("Health & Wellness Policy", "1.0", "1. Purpose", "rest")]
    found, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0]],
        "What changed in the Health & Wellness Policy?",
        compare=True,
    )
    assert found["kind"] == "lookup"
    assert found["hits"][0]["text"] == "rest"
    assert "reason=single version" in rag_logs.text


def test_unknown_compare_policy_falls_back_to_lookup(tmp_path, rag_logs):
    rows = [record("HR Policy", "2.0", "3. Leave", "birthday cake")]
    found, documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0]],
        "What changed between the versions?",
        compare=True,
    )
    assert "reason=no multi-version policy" in rag_logs.text
    assert found["kind"] == "lookup"
    assert "birthday cake" in documents[0][0]


def test_empty_collection_skips_the_answer_call(tmp_path, rag_logs):
    embedder = FakeEmbedder([1.0, 0.0])
    reranker = FakeReranker()
    found = retrieve(
        "who gets cake?",
        embedder,
        DatabaseAdapter(tmp_path / "chroma", "testbuild", create=True),
        reranker,
        embedder.examples(),
    )
    assert found["kind"] == "lookup"
    assert found["hits"] == []
    assert reranker.documents == []
    assert "hits=0" in rag_logs.text


def test_reranker_order_reaches_the_answer(tmp_path):
    rows = [
        record("HR Policy", "2.0", "1. Purpose", "vacation days"),
        record("HR Policy", "2.0", "3. Leave", "birthday cake"),
    ]
    found, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.2, 0.9]],
        "What does the HR Policy cover?",
        reverse=True,
    )
    assert [hit["text"] for hit in found["hits"]] == ["birthday cake", "vacation days"]


def test_reranker_failure_falls_back_to_fused_order_with_a_warning(tmp_path, rag_logs):
    rows = [
        record("HR Policy", "2.0", "1. Purpose", "vacation days"),
        record("HR Policy", "2.0", "3. Leave", "birthday cake"),
    ]
    found, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0]],
        "What does the HR Policy say?",
        fail=True,
    )
    assert found["hits"]
    assert "using fused order" in rag_logs.text


def test_no_reranker_uses_fused_order(tmp_path, rag_logs):
    rows = [record("HR Policy", "2.0", "1. Purpose", "vacation days")]
    embedder = FakeEmbedder([1.0, 0.0])
    found = retrieve(
        "What does the HR Policy say?",
        embedder,
        store(tmp_path / "chroma", rows, [[1.0, 0.0]]),
        None,
        embedder.examples(),
    )
    assert found["hits"][0]["text"] == "vacation days"
    assert "no COHERE_API_KEY" in rag_logs.text


def test_dense_and_lexical_use_the_same_filter():
    class FakeDatabase:
        def __init__(self):
            self.calls = []
            self.chunks = [
                record("HR Policy", "2.0", "3. Leave", "vacation days"),
            ]

        def chunks_where(self, where):
            self.calls.append(("chunks", where))
            if where is None:
                return self.chunks
            return self.chunks

        def dense_search(self, vector, where, k):
            self.calls.append(("dense", where))
            return [self.chunks[0]["id"]]

    embedder = FakeEmbedder([1.0, 0.0])
    database = FakeDatabase()
    retrieve(
        "What does the HR Policy say about leave?",
        embedder,
        database,
        FakeReranker(),
        embedder.examples(),
    )
    dense = [where for kind, where in database.calls if kind == "dense"]
    lexical = [
        where
        for kind, where in database.calls
        if kind == "chunks" and where is not None
    ]
    assert dense
    assert dense[0] == lexical[0]


def test_main_reports_a_missing_index(monkeypatch, capsys):
    monkeypatch.setattr("rag.retrieve._check_models", lambda: None)

    def missing(path):
        raise IndexMissing("index x not found; run python -m rag.ingest --rebuild")

    monkeypatch.setattr("rag.retrieve.open_active_index", missing)
    assert main(["who gets cake?"]) == 1
    assert "run python -m rag.ingest --rebuild" in capsys.readouterr().out


def test_junk_question_never_calls_a_model(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("rag.retrieve._check_models", lambda: called.append("model"))
    assert main(["hi"]) == 2
    assert called == []
    assert "Ask me about" in capsys.readouterr().out


def test_main_prints_the_answer(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr("rag.retrieve._check_models", lambda: None)
    monkeypatch.setattr("rag.retrieve.open_active_index", lambda path: path)
    monkeypatch.setattr("rag.retrieve.EmbeddingAdapter", lambda: "embedder")
    monkeypatch.setattr("rag.retrieve.make_reranker", lambda: "reranker")
    monkeypatch.setattr("rag.retrieve.embed_examples", lambda embedder: ([], []))
    monkeypatch.setattr("rag.retrieve.GenerationAdapter", lambda: "generator")

    def fake_retrieve(question, embedder, database, reranker, examples=None):
        seen["question"] = question
        seen["database"] = database
        seen["reranker"] = reranker
        return {
            "kind": "lookup",
            "hits": [{"text": "cake"}],
            "route": {},
            "filters": None,
        }

    monkeypatch.setattr(sys, "argv", ["retrieve.py", "who gets cake?"])
    monkeypatch.setattr("rag.retrieve.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "rag.retrieve.generate",
        lambda question, result, model: SimpleNamespace(
            text="cake" if model == "generator" else ""
        ),
    )
    assert main() == 0
    assert capsys.readouterr().out.strip() == "cake"
    assert seen["question"] == "who gets cake?"
    assert seen["database"] == "chroma"
    assert seen["reranker"] == "reranker"


def test_main_uses_the_given_database(monkeypatch):
    seen = {}
    monkeypatch.setattr("rag.retrieve._check_models", lambda: None)
    monkeypatch.setattr("rag.retrieve.open_active_index", lambda path: path)
    monkeypatch.setattr("rag.retrieve.EmbeddingAdapter", lambda: None)
    monkeypatch.setattr("rag.retrieve.make_reranker", lambda: None)
    monkeypatch.setattr("rag.retrieve.embed_examples", lambda embedder: ([], []))
    monkeypatch.setattr("rag.retrieve.GenerationAdapter", lambda: None)

    def fake_retrieve(question, embedder, database, reranker, examples=None):
        seen["database"] = database
        return {"kind": "lookup", "hits": [], "route": {}, "filters": None}

    monkeypatch.setattr("rag.retrieve.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "rag.retrieve.generate", lambda *args: SimpleNamespace(text="ok")
    )
    assert main(["question", "other"]) == 0
    assert seen["database"] == "other"


def test_trace_enables_stage_logs(monkeypatch):
    seen = {}

    def fake_main(argv=None, trace=False):
        seen["trace"] = trace
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("rag.trace.ask", fake_main)
    from rag.trace import main as trace_main

    assert trace_main(["who gets cake?"]) == 0
    assert seen == {"trace": True, "argv": ["who gets cake?"]}


def test_stage_logs_latency_only_for_a_question(rag_logs):
    with stage("hybrid"):
        pass
    assert "hybrid latency=" not in rag_logs.text
    enable_question_log()
    try:
        with stage("hybrid"):
            pass
    finally:
        disable_question_log()
    assert "hybrid latency=" in rag_logs.text

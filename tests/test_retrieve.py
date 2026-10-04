import sys

from adapter.database_adapter import DatabaseAdapter
from rag.logutil import disable_question_log, enable_question_log, stage
from rag.retrieve import apply_rerank, bm25_scores, cosine, fuse, main, retrieve


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
    database = DatabaseAdapter(path, "testbuild")
    database.upsert(rows, vectors)
    return database


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.tasks = []

    def embed(self, texts, task):
        self.tasks.append(task)
        return [self.vector]


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


class FakeReranker:
    def __init__(self, reverse=False, partial=False):
        self.reverse = reverse
        self.partial = partial
        self.documents = []

    def rerank(self, question, documents):
        self.documents.append(list(documents))
        if self.partial:
            return documents[:1]
        if self.reverse:
            return list(reversed(documents))
        return list(documents)


def ask(tmp_path, rows, vectors, replies, reverse=False, partial=False):
    model = FakeModel(replies)
    reranker = FakeReranker(reverse=reverse, partial=partial)
    found = retrieve(
        "cake",
        FakeEmbedder([1.0, 0.0]),
        model,
        store(tmp_path / "chroma", rows, vectors),
        reranker,
    )
    return found, model.prompts, reranker.documents


def test_bm25_scores_keyword_overlap():
    scores = bm25_scores("cake", ["birthday cake", "vacation days", "???"])
    assert scores[0] > scores[1]
    assert scores[1] == 0
    assert bm25_scores("cake", []) == []
    assert bm25_scores("cake", ["???"]) == [0.0]


def test_cosine_handles_empty_and_mismatched_vectors():
    assert cosine([], [1.0]) == 0.0
    assert cosine([1.0], [1.0, 0.0]) == 0.0
    assert cosine([0.0], [0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_fuse_breaks_ties_by_id():
    rows = [{"id": "b", "text": "cake"}, {"id": "a", "text": "cake"}]
    fused = fuse(rows, [1.0, 1.0], [1.0, 1.0])
    assert [hit["id"] for hit in fused] == ["a", "b"]


def test_keyword_hit_outranks_the_closer_distractor():
    rows = [
        {"id": "close", "text": "vacation days"},
        {"id": "mid", "text": "office hours"},
        {"id": "other", "text": "parking"},
        {"id": "cake", "text": "birthday cake"},
    ]
    keyword = bm25_scores("cake", [row["text"] for row in rows])
    fused = fuse(rows, [0.4, 0.3, 0.2, 0.35], keyword)
    assert fused[0]["id"] == "cake"


def test_lookup_drops_older_versions(tmp_path):
    rows = [
        record("HR Policy", "2.0", "3. Leave", "vacation days"),
        record("HR Policy", "1.0", "3. Leave", "birthday cake"),
    ]
    text, prompts, documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0]],
        ['{"kind":"lookup","policy":"","version":""}'],
    )
    assert text["kind"] == "lookup"
    assert text["hits"][0]["text"] == "vacation days"
    assert "birthday cake" not in documents[0][0]
    assert len(prompts) == 1
    assert FakeEmbedder([1.0, 0.0]).embed(["cake"], "query") == [[1.0, 0.0]]


def test_lookup_keeps_a_named_version(tmp_path):
    rows = [
        record("HR Policy", "2.0", "3. Leave", "vacation days"),
        record("HR Policy", "1.0", "3. Leave", "birthday cake"),
    ]
    found, _prompts, documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0]],
        ['{"kind":"lookup","policy":"HR Policy","version":"1.0"}'],
    )
    assert documents[0] == ["birthday cake"]
    assert found["hits"][0]["version"] == "1.0"


def test_lookup_can_name_a_policy_without_a_version(tmp_path):
    rows = [
        record("HR Policy", "2.0", "3. Leave", "vacation days"),
        record("HR Policy", "1.0", "3. Leave", "birthday cake"),
        record("Preparedness Policy", "2.0", "1. Purpose", "drill"),
    ]
    _text, _prompts, documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]],
        ['{"kind":"lookup","policy":"HR Policy","version":""}'],
    )
    assert documents[0] == ["vacation days"]


def test_compare_pairs_current_and_previous(tmp_path):
    rows = [
        record("HR Policy", "2.0", "3. Leave", "no dessert"),
        record("HR Policy", "1.0", "3. Leave", "cake on friday"),
        record("HR Policy", "2.0", "8. Added", "new clause"),
        record("HR Policy", "1.0", "9. Only Old", "removed clause"),
    ]
    found, _prompts, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0], [0.2, 0.8], [0.8, 0.2]],
        ['{"kind":"compare","policy":"HR Policy","version":""}'],
    )
    by_heading = {hit["heading_path"]: hit for hit in found["hits"]}
    leave = by_heading["3. Leave"]
    assert leave["current"]["text"] == "no dessert"
    assert leave["previous"]["text"] == "cake on friday"
    assert by_heading["8. Added"]["current"]["text"] == "new clause"
    assert by_heading["8. Added"]["previous"] is None
    assert by_heading["9. Only Old"]["previous"]["text"] == "removed clause"
    assert by_heading["9. Only Old"]["current"] is None


def test_compare_with_one_version_has_no_previous_side(tmp_path):
    rows = [record("Health & Wellness Policy", "1.0", "1. Purpose", "rest")]
    found, _prompts, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0]],
        ['{"kind":"compare","policy":"Health & Wellness Policy","version":""}'],
    )
    assert found["kind"] == "compare"
    assert found["hits"][0]["current"]["text"] == "rest"
    assert found["hits"][0]["previous"] is None


def test_unknown_compare_policy_falls_back_to_lookup(tmp_path, rag_logs):
    rows = [record("HR Policy", "2.0", "3. Leave", "birthday cake")]
    found, _prompts, documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0]],
        ['{"kind":"compare","policy":"","version":""}'],
    )
    assert "kind=lookup reason=unknown policy" in rag_logs.text
    assert found["kind"] == "lookup"
    assert documents[0] == ["birthday cake"]


def test_empty_collection_skips_the_answer_call(tmp_path, rag_logs):
    model = FakeModel(['{"kind":"lookup","policy":"","version":""}'])
    reranker = FakeReranker()
    text = retrieve(
        "cake",
        FakeEmbedder([1.0, 0.0]),
        model,
        DatabaseAdapter(tmp_path / "chroma", "testbuild"),
        reranker,
    )
    assert text["kind"] == "lookup"
    assert text["hits"] == []
    assert len(model.prompts) == 1
    assert reranker.documents == []
    assert "hits=0" in rag_logs.text


def test_reranker_order_reaches_the_answer(tmp_path):
    rows = [
        record("HR Policy", "2.0", "1. Purpose", "vacation days"),
        record("HR Policy", "2.0", "3. Leave", "birthday cake"),
    ]
    found, _prompts, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0]],
        ['{"kind":"lookup","policy":"HR Policy","version":"2.0"}'],
        reverse=True,
    )
    assert [hit["text"] for hit in found["hits"]] == ["birthday cake", "vacation days"]


def test_partial_rerank_keeps_the_remaining_chunks(tmp_path):
    rows = [
        record("HR Policy", "2.0", "1. Purpose", "vacation days"),
        record("HR Policy", "2.0", "3. Leave", "birthday cake"),
    ]
    found, _prompts, _documents = ask(
        tmp_path,
        rows,
        [[1.0, 0.0], [0.0, 1.0]],
        ['{"kind":"lookup","policy":"HR Policy","version":"2.0"}'],
        partial=True,
    )
    assert {hit["text"] for hit in found["hits"]} == {"vacation days", "birthday cake"}


def test_duplicate_rerank_text_stays_paired():
    items = [{"id": "a"}, {"id": "b"}]
    ordered = apply_rerank(
        "cake", items, ["same", "same"], FakeReranker(reverse=True), 3
    )
    assert [item["id"] for item in ordered] == ["a", "b"]


def test_unknown_rerank_text_is_ignored():
    class Drop:
        def rerank(self, question, documents):
            return ["missing"]

    assert apply_rerank("cake", [{"id": "a"}], ["cake"], Drop(), 3) == [{"id": "a"}]


def test_main_prints_the_answer(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr("rag.retrieve.check_model_pin", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag.retrieve.active_build_id", lambda path: "testbuild")

    def fake_retrieve(question, embedder, model, database, reranker, n=3):
        seen["question"] = question
        seen["database"] = database
        seen["model"] = model
        seen["reranker"] = reranker
        return {"kind": "lookup", "hits": [{"text": "cake"}]}

    def fake_adapter(model=None):
        return model or "generator"

    monkeypatch.setattr(sys, "argv", ["retrieve.py", "who gets cake?"])
    monkeypatch.setattr("rag.retrieve.EmbeddingAdapter", lambda: "embedder")
    monkeypatch.setattr("rag.retrieve.GenerationAdapter", fake_adapter)
    monkeypatch.setattr("rag.retrieve.DatabaseAdapter", lambda path, build_id: path)
    monkeypatch.setattr("rag.retrieve.RerankerAdapter", lambda: "reranker")
    monkeypatch.setattr("rag.retrieve.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "rag.retrieve.generate",
        lambda question, kind, hits, model: "cake" if model == "generator" else "",
    )
    assert main() == 0
    assert capsys.readouterr().out.strip() == "cake"
    assert main(trace=True) == 0
    traced = capsys.readouterr().out.splitlines()
    assert traced[:-1] == ["cake"]
    assert traced[-1].startswith("latency: ")
    assert traced[-1].endswith("s")
    assert seen["question"] == "who gets cake?"
    assert seen["database"] == "chroma"
    assert seen["model"] == "gemma3:4b"
    assert seen["reranker"] == "reranker"


def test_main_uses_the_given_database(monkeypatch):
    seen = {}
    monkeypatch.setattr("rag.retrieve.check_model_pin", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag.retrieve.active_build_id", lambda path: "testbuild")

    def fake_retrieve(question, embedder, model, database, reranker, n=3):
        seen["database"] = database
        return {"kind": "lookup", "hits": []}

    monkeypatch.setattr("rag.retrieve.EmbeddingAdapter", lambda: None)
    monkeypatch.setattr("rag.retrieve.GenerationAdapter", lambda model=None: None)
    monkeypatch.setattr("rag.retrieve.DatabaseAdapter", lambda path, build_id: path)
    monkeypatch.setattr("rag.retrieve.RerankerAdapter", lambda: None)
    monkeypatch.setattr("rag.retrieve.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.retrieve.generate", lambda *args: "ok")
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

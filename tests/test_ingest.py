from pathlib import Path

import pytest

from adapter.database_adapter import open_active_index
from rag.ingest import check_staged_index, ingest, main
from rag.manifest import active_build_id, load_manifest, save_manifest

DOCS = Path(__file__).resolve().parents[1] / "docs"


class FakeEmbedder:
    def __init__(self):
        self.tasks = []
        self.texts = []

    def embed(self, texts, task):
        self.tasks.append(task)
        self.texts.extend(texts)
        return [[float(len(self.texts) + index), 1.0] for index, _ in enumerate(texts)]


def _loaded(name, policy, version, heading, text):
    return {
        "policy": policy,
        "version": version,
        "source": name,
        "lines": [],
        "blocks": [{"level": 1, "heading": heading, "text": text}],
    }


def _write_docs(directory, names):
    directory.mkdir()
    for name, body in names:
        (directory / name).write_bytes(body)


def test_ingest_stores_pdf_and_docx_versions(tmp_path, rag_logs):
    embedder = FakeEmbedder()
    db_path = tmp_path / "chroma"
    report = ingest(DOCS, embedder, db_path)
    assert report.embedded > 0
    database = open_active_index(db_path)
    stored = database.collection.get(include=["metadatas", "documents"])
    sources = {meta["source"] for meta in stored["metadatas"]}
    versions: dict[str, set[str]] = {}
    for meta in stored["metadatas"]:
        assert meta["policy"]
        assert meta["version"]
        assert meta["parent_id"]
        assert meta["text_sha256"]
        assert meta["embed_sha256"]
        assert meta["embed_text"]
        versions.setdefault(meta["policy"], set()).add(meta["version"])

    assert any(name.endswith(".pdf") for name in sources)
    assert any(name.endswith(".docx") for name in sources)
    assert versions["HR Policy"] == {"1.0", "2.0"}
    assert versions["Preparedness Policy"] == {"1.0", "2.0"}
    assert versions["Time & Usage Policy"] == {"1.0", "2.0"}
    assert versions["Health & Wellness Policy"] == {"1.0"}
    assert embedder.tasks
    assert set(embedder.tasks) == {"document"}
    assert all(not doc.startswith("HR Policy") for doc in stored["documents"])

    again = ingest(DOCS, FakeEmbedder(), db_path)
    assert again.embedded == 0
    assert again.skipped == sorted(
        path.name for path in DOCS.iterdir() if path.suffix.lower() in {".pdf", ".docx"}
    )
    assert "skipped=7" in rag_logs.text


def test_missing_manifest_tells_you_to_ingest(tmp_path):
    with pytest.raises(RuntimeError, match="run python -m rag.ingest first"):
        active_build_id(tmp_path)


def test_empty_directory_errors(tmp_path):
    with pytest.raises(ValueError, match="no policy files"):
        ingest(tmp_path, FakeEmbedder(), tmp_path / "chroma")


def test_validation_failure_stores_nothing(tmp_path, rag_logs, monkeypatch):
    def bad_chunk(blocks, policy, version, source):
        return [
            {
                "id": "bad",
                "text": "text",
                "policy": "HR Policy",
                "section": "1. Purpose",
                "heading_path": "1. Purpose",
                "parent_id": "HR Policy|2.0",
                "source": source,
                "embed_text": "HR Policy 2.0\n1. Purpose\ntext",
                "text_sha256": "abc",
                "embed_sha256": "def",
                "word_count": 1,
                "ordinal": None,
            }
        ]

    monkeypatch.setattr("rag.ingest.chunk", bad_chunk)
    db_path = tmp_path / "chroma"
    error = ingest(DOCS, FakeEmbedder(), db_path)
    failures = [
        entry for entry in rag_logs.records if "missing field: version" in entry.message
    ]
    assert error == "missing field: version"
    assert len(failures) == 2
    assert not (db_path / "index_manifest.json").exists()


def test_main_defaults_to_docs_and_chroma(monkeypatch):
    seen = {}

    def fake_ingest(directory, embedder, db_path, read_file=None, rebuild=False):
        seen["directory"] = directory
        seen["db_path"] = db_path
        seen["rebuild"] = rebuild

    monkeypatch.setattr("rag.ingest.check_model_pin", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag.ingest.EmbeddingAdapter", FakeEmbedder)
    monkeypatch.setattr("rag.ingest.ingest", fake_ingest)
    assert main([]) == 0
    assert seen["directory"] == "docs"
    assert seen["db_path"] == "chroma"
    assert seen["rebuild"] is False


def test_main_returns_the_validation_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("rag.ingest.check_model_pin", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag.ingest.EmbeddingAdapter", FakeEmbedder)
    monkeypatch.setattr(
        "rag.ingest.ingest", lambda *args, **kwargs: "missing field: version"
    )
    assert main(["docs", str(tmp_path / "chroma")]) == 1
    assert capsys.readouterr().out.strip() == "missing field: version"


def test_unchanged_files_are_skipped(tmp_path):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"a"), ("b.docx", b"b")])
    files = {
        "a.pdf": _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text."),
        "b.docx": _loaded("b.docx", "HR Policy", "2.0", "1. Purpose", "Beta text."),
    }
    db_path = tmp_path / "chroma"
    first = ingest(
        docs, FakeEmbedder(), db_path, read_file=lambda path: files[path.name]
    )
    assert set(first.updated) == {"a.pdf", "b.docx"}
    second = ingest(
        docs, FakeEmbedder(), db_path, read_file=lambda path: files[path.name]
    )
    assert second.skipped == ["a.pdf", "b.docx"]
    assert second.embedded == 0


def test_changed_file_replaces_its_chunks(tmp_path):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"one"), ("b.docx", b"two")])
    files = {
        "a.pdf": _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text."),
        "b.docx": _loaded("b.docx", "HR Policy", "2.0", "2. Scope", "Beta text."),
    }
    db_path = tmp_path / "chroma"

    def read_file(path):
        return files[path.name]

    ingest(docs, FakeEmbedder(), db_path, read_file=read_file)
    (docs / "a.pdf").write_bytes(b"one-edited")
    files["a.pdf"] = _loaded("a.pdf", "HR Policy", "1.0", "1. Revised", "Alpha text.")
    ingest(docs, FakeEmbedder(), db_path, read_file=read_file)
    database = open_active_index(db_path)
    ids = set(database.collection.get()["ids"])
    assert "HR Policy|1.0|1. Purpose" not in ids
    assert "HR Policy|1.0|1. Revised" in ids
    assert "HR Policy|2.0|2. Scope" in ids


def test_unchanged_text_reuses_its_vector(tmp_path):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"v1")])

    def loaded(text):
        return {
            "policy": "HR Policy",
            "version": "1.0",
            "source": "a.pdf",
            "lines": [],
            "blocks": [
                {"level": 1, "heading": "1. Purpose", "text": "Keep this."},
                {"level": 1, "heading": "2. Scope", "text": text},
            ],
        }

    state = {"body": loaded("First scope.")}
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: state["body"])
    (docs / "a.pdf").write_bytes(b"v2")
    state["body"] = loaded("Second scope.")
    embedder = FakeEmbedder()
    report = ingest(docs, embedder, db_path, read_file=lambda path: state["body"])
    assert report.embedded == 1
    assert report.reused == 1
    assert len(embedder.texts) == 1
    assert "Second scope." in embedder.texts[0]


def test_deleted_file_removes_its_chunks(tmp_path):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"a"), ("b.docx", b"b")])
    files = {
        "a.pdf": _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text."),
        "b.docx": _loaded("b.docx", "HR Policy", "2.0", "2. Scope", "Beta text."),
    }
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: files[path.name])
    (docs / "b.docx").unlink()
    report = ingest(
        docs, FakeEmbedder(), db_path, read_file=lambda path: files[path.name]
    )
    assert report.removed == ["b.docx"]
    database = open_active_index(db_path)
    ids = database.collection.get()["ids"]
    assert ids == ["HR Policy|1.0|1. Purpose"]


def test_crash_before_manifest_save_is_repaired_on_next_run(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"v1")])
    state = {"body": _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text.")}
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: state["body"])
    (docs / "a.pdf").write_bytes(b"v2")
    state["body"] = _loaded("a.pdf", "HR Policy", "1.0", "1. Revised", "Alpha text.")
    real_save = save_manifest
    calls = {"n": 0}

    def flaky(path, manifest):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash")
        real_save(path, manifest)

    monkeypatch.setattr("rag.ingest.save_manifest", flaky)
    with pytest.raises(RuntimeError, match="crash"):
        ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: state["body"])
    monkeypatch.setattr("rag.ingest.save_manifest", real_save)
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: state["body"])
    manifest = load_manifest(db_path)
    assert manifest["files"]["a.pdf"]["chunk_ids"] == ["HR Policy|1.0|1. Revised"]
    database = open_active_index(db_path)
    assert database.collection.get()["ids"] == ["HR Policy|1.0|1. Revised"]


def test_changing_chunk_params_triggers_a_rebuild_and_swap(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"a")])
    loaded = _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text.")
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: loaded)
    first = active_build_id(db_path)
    monkeypatch.setattr("rag.config.CHUNK_MAX_TOKENS", 8)
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: loaded)
    second = active_build_id(db_path)
    assert second != first
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )
    names = {collection.name for collection in client.list_collections()}
    active = load_manifest(db_path)["active_collection"]
    assert active in names
    assert len(names) == 1
    assert active.startswith(f"policies__{second}__")
    assert load_manifest(db_path)["active_build_id"] == second


def test_rebuild_flag_forces_a_rebuild(tmp_path):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"a")])
    loaded = _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text.")
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: loaded)
    embedder = FakeEmbedder()
    report = ingest(
        docs, embedder, db_path, read_file=lambda path: loaded, rebuild=True
    )
    assert report.embedded == 1
    assert embedder.texts


def _collection_names(db_path):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )
    return {collection.name for collection in client.list_collections()}


def test_corrupt_file_during_rebuild_keeps_the_old_index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    sources = [
        DOCS / "Doofenshmirtz Evil Inc - HR Policy v1.0.pdf",
        DOCS / "Doofenshmirtz Evil Inc - Health Policy v1.0.pdf",
    ]
    for source in sources:
        (docs / source.name).write_bytes(source.read_bytes())
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path)
    before_count = open_active_index(db_path).count()
    before = load_manifest(db_path)
    (docs / "zz-corrupt.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="empty extract"):
        ingest(docs, FakeEmbedder(), db_path, rebuild=True)
    assert open_active_index(db_path).count() == before_count
    assert load_manifest(db_path) == before
    assert _collection_names(db_path) == {before["active_collection"]}


def test_validation_failure_during_rebuild_keeps_the_old_index(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"a")])
    loaded = _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text.")
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: loaded)
    before_count = open_active_index(db_path).count()

    def bad_chunk(blocks, policy, version, source):
        return [
            {
                "id": "HR Policy|1.0|1. Purpose",
                "text": "Alpha text.",
                "policy": policy,
                "section": "1. Purpose",
                "heading_path": "1. Purpose",
                "parent_id": "HR Policy|1.0",
                "source": source,
                "ordinal": None,
            }
        ]

    monkeypatch.setattr("rag.ingest.chunk", bad_chunk)
    error = ingest(
        docs, FakeEmbedder(), db_path, read_file=lambda path: loaded, rebuild=True
    )
    assert error == "missing field: version"
    assert open_active_index(db_path).count() == before_count


def test_rebuild_switches_to_a_new_collection_and_drops_the_old_one(tmp_path):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"a")])
    loaded = _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text.")
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: loaded)
    first = load_manifest(db_path)["active_collection"]
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: loaded, rebuild=True)
    second = load_manifest(db_path)["active_collection"]
    build_id = load_manifest(db_path)["active_build_id"]
    assert second != first
    assert second.startswith(f"policies__{build_id}__")
    assert _collection_names(db_path) == {second}


def test_two_rebuilds_in_a_row_get_different_collection_names(tmp_path):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"a")])
    loaded = _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text.")
    db_path = tmp_path / "chroma"
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: loaded, rebuild=True)
    first = load_manifest(db_path)["active_collection"]
    ingest(docs, FakeEmbedder(), db_path, read_file=lambda path: loaded, rebuild=True)
    second = load_manifest(db_path)["active_collection"]
    assert first != second


def test_file_without_chunks_fails_the_rebuild(tmp_path):
    docs = tmp_path / "docs"
    _write_docs(docs, [("a.pdf", b"a"), ("b.docx", b"b")])
    files = {
        "a.pdf": _loaded("a.pdf", "HR Policy", "1.0", "1. Purpose", "Alpha text."),
        "b.docx": {
            "policy": "HR Policy",
            "version": "2.0",
            "source": "b.docx",
            "lines": [],
            "blocks": [],
        },
    }
    db_path = tmp_path / "chroma"
    error = ingest(
        docs, FakeEmbedder(), db_path, read_file=lambda path: files[path.name]
    )
    assert error == "no chunks from: b.docx"
    assert not (db_path / "index_manifest.json").exists()


def test_staged_index_missing_a_chunk_is_rejected():
    class FakeDatabase:
        def all_ids(self):
            return ["a"]

    files = {"a.pdf": {"chunk_ids": ["a", "b"]}}
    with pytest.raises(ValueError, match="expected 2"):
        check_staged_index(FakeDatabase(), files)

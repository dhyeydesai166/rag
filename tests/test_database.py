from adapter.database_adapter import DatabaseAdapter, where_for

BUILD = "testbuild"


def _record(
    ordinal=None, source="hr.pdf", record_id="HR Policy|1.0|1. Purpose"
) -> dict:
    text = "Purpose body"
    return {
        "id": record_id,
        "text": text,
        "policy": "HR Policy",
        "version": "1.0",
        "section": "1. Purpose",
        "heading_path": "1. Purpose",
        "parent_id": "HR Policy|1.0",
        "source": source,
        "embed_text": f"HR Policy 1.0\n1. Purpose\n{text}",
        "text_sha256": "text-sha",
        "embed_sha256": f"embed-{record_id}",
        "word_count": 2,
        "ordinal": ordinal,
    }


def test_upsert_is_idempotent_and_keeps_metadata(tmp_path):
    database = DatabaseAdapter(tmp_path / "chroma", BUILD)
    records = [_record()]
    vectors = [[0.1, 0.2]]
    database.upsert(records, vectors)
    database.upsert(records, vectors)
    assert database.collection.count() == 1
    stored = database.collection.get(include=["metadatas", "documents"])
    meta = stored["metadatas"][0]
    assert meta["source"] == "hr.pdf"
    assert meta["version"] == "1.0"
    assert meta["section"] == "1. Purpose"
    assert meta["embed_sha256"] == "embed-HR Policy|1.0|1. Purpose"
    assert meta["ordinal"] == 0
    assert stored["documents"][0] == "Purpose body"


def test_chunks_where_returns_text_and_metadata_without_vectors(tmp_path):
    database = DatabaseAdapter(tmp_path / "chroma", BUILD)
    database.upsert([_record()], [[0.25, 0.75]])
    stored = database.chunks_where(None)
    assert stored[0]["text"] == "Purpose body"
    assert stored[0]["policy"] == "HR Policy"
    assert stored[0]["word_count"] == 2
    assert "vector" not in stored[0]
    assert stored[0]["ordinal"] is None
    vectors = database.vectors_by_embed_sha(["embed-HR Policy|1.0|1. Purpose"])
    assert vectors["embed-HR Policy|1.0|1. Purpose"] == [0.25, 0.75]


def test_chunks_where_on_an_empty_collection(tmp_path):
    assert DatabaseAdapter(tmp_path / "chroma", BUILD).chunks_where(None) == []


def test_delete_ids_removes_only_those_chunks(tmp_path):
    database = DatabaseAdapter(tmp_path / "chroma", BUILD)
    first = _record(record_id="HR Policy|1.0|1. Purpose")
    second = _record(record_id="HR Policy|1.0|2. Scope", source="hr.pdf")
    second["heading_path"] = "2. Scope"
    second["embed_sha256"] = "embed-scope"
    database.upsert([first, second], [[0.1, 0.2], [0.3, 0.4]])
    database.delete_ids([first["id"]])
    assert database.collection.get()["ids"] == [second["id"]]
    database.delete_ids([])
    assert database.collection.count() == 1


def test_ids_for_source(tmp_path):
    database = DatabaseAdapter(tmp_path / "chroma", BUILD)
    keep = _record(source="hr.pdf")
    other = _record(record_id="Health Policy|1.0|1. Purpose", source="health.pdf")
    database.upsert([keep, other], [[0.1, 0.2], [0.2, 0.3]])
    assert database.ids_for_source("hr.pdf") == [keep["id"]]


def test_vectors_by_embed_sha_returns_stored_vectors(tmp_path):
    database = DatabaseAdapter(tmp_path / "chroma", BUILD)
    record = _record()
    database.upsert([record], [[0.25, 0.75]])
    found = database.vectors_by_embed_sha([record["embed_sha256"], "missing"])
    assert found == {record["embed_sha256"]: [0.25, 0.75]}
    assert database.vectors_by_embed_sha([]) == {}


def test_chunks_where_filters_by_policy_and_version(tmp_path):
    database = DatabaseAdapter(tmp_path / "chroma", BUILD)
    current = _record()
    older = _record(record_id="HR Policy|2.0|1. Purpose")
    older["version"] = "2.0"
    database.upsert([current, older], [[0.1, 0.0], [0.0, 1.0]])
    chunks = database.chunks_where(where_for([("HR Policy", "1.0")]))
    assert [chunk["id"] for chunk in chunks] == [current["id"]]
    assert "vector" not in chunks[0]


def test_dense_search_respects_the_filter(tmp_path):
    database = DatabaseAdapter(tmp_path / "chroma", BUILD)
    close = _record()
    far = _record(record_id="HR Policy|2.0|1. Purpose")
    far["version"] = "2.0"
    far["embed_sha256"] = "other"
    database.upsert([close, far], [[1.0, 0.0], [0.0, 1.0]])
    ids = database.dense_search([1.0, 0.0], where_for([("HR Policy", "2.0")]), 5)
    assert ids == [far["id"]]


def test_where_for_one_and_many_pairs():
    assert where_for([]) is None
    assert where_for([("HR Policy", "2.0")]) == {
        "$and": [{"policy": "HR Policy"}, {"version": "2.0"}]
    }
    many = where_for([("HR Policy", "1.0"), ("HR Policy", "2.0")])
    assert "$or" in many
    assert len(many["$or"]) == 2

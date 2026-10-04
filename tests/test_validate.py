from rag.validate import validate


def _record(**overrides):
    base = {
        "id": "HR Policy|1.0|1. Purpose",
        "text": "Purpose text",
        "policy": "HR Policy",
        "version": "1.0",
        "section": "1. Purpose",
        "heading_path": "1. Purpose",
        "parent_id": "HR Policy|1.0",
        "source": "hr.pdf",
        "embed_text": "HR Policy v1.0\n1. Purpose\nPurpose text",
        "word_count": 2,
        "embed": True,
    }
    base.update(overrides)
    return base


def test_valid_record_passes(rag_logs):
    assert validate(_record()) is None
    assert "pass" in rag_logs.text


def test_missing_field_fails_after_two_attempts(rag_logs):
    error = validate(_record(version=""))
    assert error == "missing field: version"
    failures = [r for r in rag_logs.records if "missing field: version" in r.message]
    assert len(failures) == 2
    assert "attempt=2" in failures[1].message

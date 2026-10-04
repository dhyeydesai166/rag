from pathlib import Path

import pytest

from rag.reader import (
    normalize_version,
    policy_from_filename,
    read,
    version_key,
)

DOCS = Path(__file__).resolve().parents[1] / "docs"
HR_PDF = DOCS / "Doofenshmirtz Evil Inc - HR Policy v1.0.pdf"
HR_DOCX = DOCS / "Doofenshmirtz Evil Inc - HR Policy v2.0.docx"


def test_policy_and_version_from_filename():
    assert policy_from_filename(HR_PDF) == ("HR Policy", "1.0")
    assert policy_from_filename(HR_DOCX) == ("HR Policy", "2.0")
    time_path = DOCS / "Doofenshmirtz Evil Inc - Time and Usage Policy v1.0.pdf"
    assert policy_from_filename(time_path) == ("Time and Usage Policy", "1.0")


def test_title_line_wins_over_filename(tmp_path, monkeypatch):
    path = tmp_path / "Doofenshmirtz Evil Inc - Time and Usage Policy v2.0.docx"
    path.write_bytes(b"not a real docx")
    monkeypatch.setattr(
        "rag.reader._load_text",
        lambda _path: (
            "Company\nTime & Usage Policy — Version 2.0\n1. Purpose. Be on time."
        ),
    )
    loaded = read(path)
    assert loaded["policy"] == "Time & Usage Policy"
    assert loaded["version"] == "2.0"


def test_filename_is_the_fallback(tmp_path, monkeypatch):
    path = tmp_path / "Notes - HR Policy v1.0.pdf"
    path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "rag.reader._load_text",
        lambda _path: "1. Purpose. This document explains the rules.",
    )
    loaded = read(path)
    assert (loaded["policy"], loaded["version"]) == ("HR Policy", "1.0")


def test_version_mismatch_between_title_and_filename_is_an_error(tmp_path, monkeypatch):
    path = tmp_path / "Doofenshmirtz Evil Inc - HR Policy v1.0.pdf"
    path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "rag.reader._load_text",
        lambda _path: "HR Policy — Version 2.0\n1. Purpose. Rules.",
    )
    with pytest.raises(ValueError, match="version mismatch"):
        read(path)


def test_unparseable_file_gives_a_clear_error(tmp_path, monkeypatch):
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "rag.reader._load_text", lambda _path: "No title here.\nJust text."
    )
    with pytest.raises(ValueError, match="cannot find policy name and version"):
        read(path)


def test_normalize_version():
    assert normalize_version("2") == "2.0"
    assert normalize_version("2.0") == "2.0"
    assert normalize_version("2.0.1") == "2.0.1"


def test_version_key_sorts_numerically():
    assert sorted(["10.0", "2.0", "9.0"], key=version_key) == ["2.0", "9.0", "10.0"]


def test_read_pdf_and_docx_return_lines_and_metadata():
    pdf = read(HR_PDF)
    docx = read(HR_DOCX)
    assert pdf["policy"] == "HR Policy"
    assert pdf["version"] == "1.0"
    assert pdf["source"] == HR_PDF.name
    assert any("1. Purpose" in line for line in pdf["lines"])
    assert "joke" in " ".join(pdf["lines"]).lower()
    assert docx["version"] == "2.0"
    assert docx["source"].endswith(".docx")
    assert len(docx["lines"]) > 5
    purpose = next(block for block in pdf["blocks"] if block["heading"] == "1. Purpose")
    child = next(
        block for block in pdf["blocks"] if block["heading"] == "3.1 Requirement"
    )
    assert purpose["level"] == 1
    assert child["level"] == 2
    assert purpose["text"]


def test_unsupported_suffix_is_rejected(tmp_path):
    bad = tmp_path / "note.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError, match="unsupported suffix"):
        read(bad)


def test_empty_extract_is_rejected(tmp_path, monkeypatch):
    empty = tmp_path / "Doofenshmirtz Evil Inc - HR Policy v1.0.pdf"
    empty.write_bytes(b"%PDF-1.4 empty")

    def fake_load(_path):
        return ""

    monkeypatch.setattr("rag.reader._load_text", fake_load)
    with pytest.raises(ValueError, match="empty extract"):
        read(empty)

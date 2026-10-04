import json
from pathlib import Path

from rag.chunker import chunks_from_directory

ROOT = Path(__file__).resolve().parents[1]


def test_chunks_json_matches_the_current_chunker():
    stored = json.loads((ROOT / "chunks.json").read_text(encoding="utf-8"))
    fresh = chunks_from_directory(ROOT / "docs")
    assert fresh == stored

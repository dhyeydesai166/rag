"""Turn heading blocks into chunk records.

Level-3 headings (N.N.N) stay under their level-1 section, the same way
level-2 headings do. A section that has subsections keeps its intro text as
its own chunk only when that intro is non-empty. parent_id groups the chunks
of one section; it is not required to be the id of a stored record.
"""

import hashlib
import json
import sys
from pathlib import Path

from rag.logutil import log
from rag.reader import SUPPORTED, read
from rag.splitter import split_long_chunk


def make_id(
    policy: str, version: str, heading_path: str, ordinal: int | None = None
) -> str:
    """'policy|version|heading_path', plus '#n' for pieces of a split chunk.

    Why: the id is built only from where the text lives, so re-ingesting the
    same document always gives the same ids (upserts replace, not duplicate).
    """
    base = f"{policy}|{version}|{heading_path}"
    return base if ordinal is None else f"{base}#{ordinal}"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_text_for(policy: str, version: str, heading_path: str, text: str) -> str:
    return f"{policy} {version}\n{heading_path}\n{text}"


def make_piece(record: dict, text: str, ordinal: int) -> dict:
    """Copy a chunk record onto one piece of a split section."""
    policy = record["policy"]
    version = record["version"]
    heading_path = record["heading_path"]
    embed_text = embed_text_for(policy, version, heading_path, text)
    return {
        **record,
        "id": make_id(policy, version, heading_path, ordinal),
        "text": text,
        "ordinal": ordinal,
        "word_count": len(text.split()),
        "embed_text": embed_text,
        "text_sha256": sha256_hex(text),
        "embed_sha256": sha256_hex(embed_text),
    }


def _record(
    policy: str,
    version: str,
    source: str,
    section: str,
    heading_path: str,
    parent_id: str,
    text: str,
) -> dict:
    embed_text = embed_text_for(policy, version, heading_path, text)
    return {
        "id": make_id(policy, version, heading_path),
        "text": text,
        "policy": policy,
        "version": version,
        "section": section,
        "heading_path": heading_path,
        "parent_id": parent_id,
        "source": source,
        "word_count": len(text.split()),
        "ordinal": None,
        "embed_text": embed_text,
        "text_sha256": sha256_hex(text),
        "embed_sha256": sha256_hex(embed_text),
    }


def chunk(blocks: list[dict], policy: str, version: str, source: str) -> list[dict]:
    document_id = f"{policy}|{version}"
    sections: list[dict] = []
    current = None
    for block in blocks:
        if block["level"] == 1:
            current = {
                "heading": block["heading"],
                "text": block["text"],
                "children": [],
            }
            sections.append(current)
        elif current is not None:
            current["children"].append(block)

    records = []
    for section in sections:
        section_heading = section["heading"]
        if section["children"]:
            if section["text"].strip():
                records.extend(
                    split_long_chunk(
                        _record(
                            policy,
                            version,
                            source,
                            section_heading,
                            section_heading,
                            document_id,
                            section["text"],
                        )
                    )
                )
            for child in section["children"]:
                heading_path = f"{section_heading} > {child['heading']}"
                records.extend(
                    split_long_chunk(
                        _record(
                            policy,
                            version,
                            source,
                            section_heading,
                            heading_path,
                            make_id(policy, version, section_heading),
                            child["text"],
                        )
                    )
                )
        else:
            records.extend(
                split_long_chunk(
                    _record(
                        policy,
                        version,
                        source,
                        section_heading,
                        section_heading,
                        document_id,
                        section["text"],
                    )
                )
            )

    log("chunker", f"source={source} sections={len(sections)} chunks={len(records)}")
    return records


def chunks_from_directory(directory: Path) -> list[dict]:
    records = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        loaded = read(path)
        records.extend(
            chunk(loaded["blocks"], loaded["policy"], loaded["version"], path.name)
        )
    return records


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    directory = Path(argv[0] if argv else "docs")
    output = Path(argv[1] if len(argv) > 1 else "chunks.json")
    records = chunks_from_directory(directory)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# TODO: Switch from structural to hierarchical chunking as the corpus grows.
# parent_id is the foundation for that later change.

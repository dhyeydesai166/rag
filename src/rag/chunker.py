from pathlib import Path

from rag.logutil import log
from rag.reader import policy_and_version


def make_id(policy: str, version: str, heading_path: str) -> str:
    return f"{policy}|{version}|{heading_path}"


def _record(
    policy: str,
    version: str,
    source: str,
    section: str,
    heading_path: str,
    parent_id: str,
    text: str,
    embed: bool,
) -> dict:
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
        "embed": embed,
        "embed_text": f"{policy} v{version}\n{heading_path}\n{text}",
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
            records.append(
                _record(
                    policy,
                    version,
                    source,
                    section_heading,
                    section_heading,
                    document_id,
                    section["text"],
                    False,
                )
            )
            for child in section["children"]:
                heading_path = f"{section_heading} > {child['heading']}"
                records.append(
                    _record(
                        policy,
                        version,
                        source,
                        section_heading,
                        heading_path,
                        make_id(policy, version, section_heading),
                        child["text"],
                        True,
                    )
                )
        else:
            records.append(
                _record(
                    policy,
                    version,
                    source,
                    section_heading,
                    section_heading,
                    document_id,
                    section["text"],
                    True,
                )
            )

    children = [record for record in records if record["embed"]]
    parent_ids = sorted({record["parent_id"] for record in children})
    log(
        "chunker",
        f"source={source} sections={len(sections)} children={len(children)} "
        f"parents={','.join(parent_ids)}",
    )
    return records


# TODO: Switch from structural to hierarchical chunking as the corpus grows.
# The parent_id field is the foundation for that later change.


def chunk_path(path: Path | str, blocks: list[dict]) -> list[dict]:
    path = Path(path)
    policy, version = policy_and_version(path)
    return chunk(blocks, policy, version, path.name)

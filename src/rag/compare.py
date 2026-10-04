"""Pair sections across two versions of one policy by normalized title."""

import re

from rag.filters import Filters
from rag.reader import version_key
from rag.section_map import apply_renames

SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
TITLE_SUFFIX = re.compile(
    r"\s+[—–-]\s+(?:Updated|New in Version \d+(?:\.\d+)*)$",
    re.IGNORECASE,
)


def normalize_title(heading_path: str) -> str:
    """'6. Token Allocation > 6.1 Allocation Amount'
    -> 'token allocation > allocation amount'.

    Strips section numbers and the 'Updated' / 'New in Version N' suffixes,
    because those change between versions while the section stays the same.
    """
    parts = []
    for part in heading_path.split(" > "):
        stripped = TITLE_SUFFIX.sub("", SECTION_NUMBER.sub("", part)).strip().lower()
        parts.append(stripped)
    return " > ".join(parts)


def compare_versions(filters: Filters, versions: tuple[str, ...]) -> tuple[str, str]:
    """Older version first. Two named versions win; else previous vs latest."""
    ordered = tuple(sorted(versions, key=version_key))
    named = [version for version in filters.versions if version in ordered]
    if len(named) >= 2:
        return tuple(sorted(named[:2], key=version_key))
    if len(named) == 1:
        chosen = named[0]
        if chosen == ordered[-1]:
            return ordered[-2], ordered[-1]
        return chosen, ordered[-1]
    return ordered[-2], ordered[-1]


def side(row: dict) -> dict:
    return {
        "id": row["id"],
        "text": row["text"],
        "version": row["version"],
        "heading_path": row["heading_path"],
        "section": row["section"],
        "source": row["source"],
    }


def _pair(policy: str, title: str, old: dict | None, new: dict | None) -> dict:
    heading = ""
    if new is not None:
        heading = new["heading_path"]
    elif old is not None:
        heading = old["heading_path"]
    return {
        "policy": policy,
        "title": title,
        "heading_path": heading,
        "score": 0.0,
        "current": None if new is None else side(new),
        "previous": None if old is None else side(old),
    }


def pair_by_title(
    old_chunks: list[dict], new_chunks: list[dict], renames: dict[str, str]
) -> list[dict]:
    """Pair every section of the old version with its counterpart in the new one.

    Unmatched old sections are 'removed' (current=None); unmatched new ones
    are 'added' (previous=None). `renames` maps old normalized titles to new
    ones for sections whose title changed.
    """
    old_by: dict[tuple, dict] = {}
    for chunk in old_chunks:
        title = apply_renames(normalize_title(chunk["heading_path"]), renames)
        key = (title, chunk.get("ordinal") or 0)
        if key in old_by:
            other = old_by[key]["heading_path"]
            raise ValueError(f"{other} and {chunk['heading_path']} both map to {title}")
        old_by[key] = chunk
    new_by: dict[tuple, dict] = {}
    for chunk in new_chunks:
        title = normalize_title(chunk["heading_path"])
        key = (title, chunk.get("ordinal") or 0)
        new_by[key] = chunk
    policy = ""
    if old_chunks:
        policy = old_chunks[0]["policy"]
    elif new_chunks:
        policy = new_chunks[0]["policy"]
    pairs = []
    seen = set()
    for key, old in old_by.items():
        seen.add(key)
        pairs.append(_pair(policy, key[0], old, new_by.get(key)))
    for key, new in new_by.items():
        if key in seen:
            continue
        pairs.append(_pair(policy, key[0], None, new))
    return pairs


def pair_text(pair: dict) -> str:
    current = _side_text("current", pair["current"])
    previous = _side_text("previous", pair["previous"])
    return f"{pair['heading_path']}\n{current}\n{previous}"


def _side_text(label: str, item) -> str:
    if item is None:
        return label
    return f"{label} {item['version']}\n{item['text']}"

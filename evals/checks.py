"""Code checks for retrieval and answers.

Misses are things the answer left out. Inventions are things it said that
the cited passage does not support. They are counted separately on purpose.
"""

import re
from dataclasses import dataclass

from rag.models import Answer

# Word stems, so 'reduced' and 'a reduction' both count. 'version N' counts
# because a passage that mentions another version is describing a change.
CHANGE_WORDS = re.compile(
    r"\b(chang\w*|increas\w*|reduc\w*|decreas\w*|remov\w*|added|addition\w*|"
    r"new|newly|no longer|previous\w*|now|used to|instead of|replac\w*|"
    r"version \d+)\b",
    re.IGNORECASE,
)


def numbers_in(text: str, versions: frozenset[str] = frozenset()) -> set[str]:
    """Digit groups with thousands separators removed: '1,000,000' -> '1000000'.

    Known version labels ('2.0') are left out: naming a version is not a number
    the passage has to contain.
    """
    found = {
        match.replace(",", "") for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    }
    return found - versions


def known_versions(retrieved: dict[str, dict]) -> frozenset[str]:
    """Versions of the passages the answer model was shown."""
    return frozenset(
        passage["version"] for passage in retrieved.values() if passage.get("version")
    )


def has_change_language(text: str, own_version: str = "") -> bool:
    """Change words, ignoring a mention of the cited passage's own version.

    Why: 'In version 2.0 the limit is 45 minutes' citing a 2.0 passage says where
    the rule lives; it does not describe a change.
    """
    if own_version:
        own = rf"\bversion\s+{re.escape(own_version)}\b"
        text = re.sub(own, "", text, flags=re.IGNORECASE)
    return bool(CHANGE_WORDS.search(text))


def retrieval_hit(gold: list[str], retrieved_ids: set[str]) -> bool:
    """True when every gold id was retrieved. An empty gold list is a hit."""
    return set(gold) <= retrieved_ids


def retrieved_ids(kind: str, hits: list[dict]) -> list[str]:
    """Chunk ids in the final hits. Compare counts both sides of each pair."""
    if kind != "compare":
        return [hit["id"] for hit in hits]
    found = []
    for hit in hits:
        for side_name in ("previous", "current"):
            side = hit.get(side_name)
            if side and side.get("id"):
                found.append(side["id"])
    return found


def passages_by_id(kind: str, hits: list[dict]) -> dict[str, dict]:
    """Cited-text lookup for the chunks the answer model was shown."""
    if kind != "compare":
        return {hit["id"]: hit for hit in hits}
    found = {}
    for hit in hits:
        for side_name in ("previous", "current"):
            side = hit.get(side_name)
            if side and side.get("id"):
                found[side["id"]] = side
    return found


@dataclass
class CheckResult:
    misses: list[str]
    inventions: list[str]


def check_answer(
    answer: Answer, retrieved: dict[str, dict], route: str, case: dict
) -> CheckResult:
    """Score one answer against its case with plain string and number checks."""
    misses: list[str] = []
    inventions: list[str] = []
    versions = known_versions(retrieved)
    for claim in answer.claims:
        chunk = retrieved.get(claim.chunk_id)
        if chunk is None:
            inventions.append(f"cites a chunk that was not retrieved: {claim.chunk_id}")
            continue
        extra = numbers_in(claim.text, versions) - numbers_in(chunk["text"], versions)
        for number in sorted(extra):
            inventions.append(f"number {number} is not in {claim.chunk_id}")
        for fact in case["stale_facts"]:
            # A current document may quote the old value ("a reduction from
            # 1,000,000"); it is only a stale fact if the cited chunk does not say it.
            in_claim = fact.lower() in claim.text.lower()
            in_chunk = fact.lower() in chunk["text"].lower()
            if in_claim and not in_chunk:
                inventions.append(f"stale fact {fact!r}")
        if (
            route == "lookup"
            and has_change_language(claim.text, chunk.get("version", ""))
            and not has_change_language(chunk["text"])
        ):
            inventions.append(f"describes a change the source does not: {claim.text!r}")
    if answer.status != case["expect_status"]:
        bucket = inventions if answer.status == "answered" else misses
        bucket.append(f"status {answer.status}, expected {case['expect_status']}")
    cited = {claim.chunk_id for claim in answer.claims}
    if case["gold_chunks"] and not cited & set(case["gold_chunks"]):
        misses.append("no gold chunk cited")
    answer_text = " ".join(claim.text for claim in answer.claims).lower()
    for options in case["required_facts"]:
        if not any(option.lower() in answer_text for option in options):
            misses.append(f"missing fact {options[0]!r}")
    return CheckResult(misses, inventions)

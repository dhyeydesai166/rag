"""Code checks for retrieval and answers.

Misses are things the answer left out. Inventions are things it said that
the cited passage does not support. They are counted separately on purpose.
"""

import re
from dataclasses import dataclass

from rag.generate import same_claim
from rag.models import Answer

# Word stems, so 'reduced' and 'a reduction' both count. 'version N' counts
# because a passage that mentions another version is describing a change.
CHANGE_WORDS = re.compile(
    r"\b(chang\w*|increas\w*|reduc\w*|decreas\w*|remov\w*|added|addition\w*|"
    r"new|newly|no longer|previous\w*|now|used to|instead of|replac\w*|"
    r"version \d+)\b",
    re.IGNORECASE,
)
# A claim says something changed. '\b' keeps 'unchanged' from matching 'chang'.
COMPARE_CHANGE_WORDS = re.compile(
    r"\b(chang\w*|increas\w*|decreas\w*|reduc\w*|rais\w*|lower\w*|added|"
    r"remov\w*|replac\w*|extend\w*|shorten\w*|no longer)\b",
    re.IGNORECASE,
)
# Wording that says there was no change, so 'has not changed' is not a change claim.
NO_CHANGE_WORDS = re.compile(
    r"\b(unchanged|not chang\w*|no change\w*|did not change|remain\w*|same|"
    r"both versions)\b",
    re.IGNORECASE,
)
# Change words about an amount; only these are judged by comparing numbers.
AMOUNT_CHANGE_WORDS = re.compile(
    r"\b(increas\w*|decreas\w*|reduc\w*|rais\w*|lower\w*|extend\w*|shorten\w*)\b",
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


def counterparts_by_id(kind: str, hits: list[dict]) -> dict[str, dict | None]:
    """Map each comparison passage id to the other version in its pair."""
    if kind != "compare":
        return {}
    found: dict[str, dict | None] = {}
    for pair in hits:
        old, new = pair.get("previous"), pair.get("current")
        if old:
            found[old["id"]] = new
        if new:
            found[new["id"]] = old
    return found


def asserts_change(text: str) -> bool:
    return bool(COMPARE_CHANGE_WORDS.search(text)) and not NO_CHANGE_WORDS.search(text)


def normalized(text: str) -> str:
    """Lowercase words and digits only, so punctuation and spacing do not count."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def same_in_both_versions(
    claim_text: str, cited: dict, other: dict | None, versions: frozenset[str]
) -> str | None:
    """Why a change claim is contradicted by its two passages, or None.

    Why: a model can say 'increased' about a rule both versions state the same
    way; prompt wording did not stop it, so the eval must count it.
    """
    if other is None or not asserts_change(claim_text):
        return None
    if normalized(cited["text"]) == normalized(other["text"]):
        return "both versions have the same text"
    if AMOUNT_CHANGE_WORDS.search(claim_text):
        numbers = numbers_in(cited["text"], versions)
        if numbers and numbers == numbers_in(other["text"], versions):
            return "both versions give the same numbers"
    return None


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
    answer: Answer,
    retrieved: dict[str, dict],
    route: str,
    case: dict,
    counterparts: dict[str, dict | None] | None = None,
    display_lines: list[str] | None = None,
    reranked: bool = True,
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
        if route == "compare" and claim.chunk_id in (counterparts or {}):
            reason = same_in_both_versions(
                claim.text, chunk, counterparts[claim.chunk_id], versions
            )
            if reason:
                misses.append(f"claims a change, but {reason}: {claim.text!r}")
    if answer.status != case["expect_status"]:
        bucket = inventions if answer.status == "answered" else misses
        bucket.append(f"status {answer.status}, expected {case['expect_status']}")
    cited = {claim.chunk_id for claim in answer.claims}
    # Without Cohere the fridge question's top 3 are HR chunks, so this case
    # cannot cite the shelter section. Score it only when Cohere ranked.
    relax = bool(case.get("needs_rerank")) and not reranked
    if not relax and case["gold_chunks"] and not cited & set(case["gold_chunks"]):
        misses.append("no gold chunk cited")
    if not relax and case.get("cited_only_gold"):
        for chunk_id in sorted(cited - set(case["gold_chunks"])):
            misses.append(f"cited a chunk outside gold: {chunk_id}")
    if case.get("no_repeated_claims"):
        texts = [claim.text for claim in answer.claims]
        for index, text in enumerate(texts):
            if any(same_claim(text, earlier) for earlier in texts[:index]):
                misses.append(f"repeated claim: {text!r}")
    shown = " ".join(
        [*(claim.text for claim in answer.claims), *(display_lines or [])]
    ).lower()
    if not relax:
        for options in case["required_facts"]:
            if not any(option.lower() in shown for option in options):
                misses.append(f"missing fact {options[0]!r}")
    return CheckResult(misses, inventions)

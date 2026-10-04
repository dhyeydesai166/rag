"""Pull policy and version filters out of a question with plain matching.

An alias that is also an ordinary word ("health", "time") is not listed,
because those words appear inside the rules themselves.
"""

import re
from dataclasses import dataclass

from rag.config import POLICY_ALIASES, VERSION_PATTERN
from rag.logutil import log
from rag.reader import version_key

_VERSION = re.compile(VERSION_PATTERN)


@dataclass(frozen=True)
class Filters:
    policy: str | None
    versions: tuple[str, ...]
    search_text: str


def find_policy(
    question: str, catalog: dict[str, tuple[str, ...]]
) -> tuple[str | None, list[tuple[int, int]]]:
    """Match POLICY_ALIASES as whole words, longest alias first.

    Returns (policy, spans). If aliases of two different policies match,
    return (None, []) and log it: searching everything is safer than guessing.
    """
    lowered = question.lower()
    found: list[tuple[str, int, int, int]] = []
    for policy, aliases in POLICY_ALIASES.items():
        if policy not in catalog:
            continue
        for alias in sorted(aliases, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(alias)}\b", lowered):
                found.append((policy, match.start(), match.end(), len(alias)))
    if not found:
        return None, []
    policies = {item[0] for item in found}
    if len(policies) > 1:
        log("filters", "two policies named; searching all")
        return None, []
    spans = [(start, end) for _policy, start, end, _length in found]
    return found[0][0], _merge_spans(spans)


def find_versions(
    question: str, known: set[str]
) -> tuple[list[str], list[tuple[int, int]]]:
    """Find N.N numbers that are real versions of the chosen policy (or of any policy).

    Why the catalog check: '1.5 hours' must not become a version filter.
    """
    versions: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in _VERSION.finditer(question):
        version = match.group(0)
        if version not in known:
            continue
        if version not in versions:
            versions.append(version)
        spans.append(match.span())
    return versions, spans


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Cut the matched spans and tidy leftover spaces and punctuation."""
    kept = []
    cursor = 0
    for start, end in _merge_spans(spans):
        kept.append(text[cursor:start])
        cursor = end
    kept.append(text[cursor:])
    cleaned = " ".join("".join(kept).split())
    cleaned = re.sub(r"\s+([?.!,;:])", r"\1", cleaned)
    cleaned = re.sub(r"^[\s?.!,;:]+", "", cleaned)
    return cleaned.strip()


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def extract_filters(question: str, catalog: dict[str, tuple[str, ...]]) -> Filters:
    """Policy, versions and search text from the cleaned question, by plain matching."""
    policy, policy_spans = find_policy(question, catalog)
    if policy:
        known = set(catalog[policy])
    else:
        known = {version for versions in catalog.values() for version in versions}
    versions, version_spans = find_versions(question, known)
    search_text = remove_spans(question, policy_spans + version_spans)
    return Filters(policy, tuple(versions), search_text)


def lookup_targets(
    filters: Filters, catalog: dict[str, tuple[str, ...]]
) -> list[tuple[str, str]]:
    """(policy, version) pairs a lookup is allowed to search."""
    if filters.policy and filters.versions:
        return [(filters.policy, filters.versions[0])]
    if filters.policy:
        return [(filters.policy, catalog[filters.policy][-1])]
    if filters.versions:
        version = filters.versions[0]
        return [
            (name, version) for name, versions in catalog.items() if version in versions
        ]
    return [(name, versions[-1]) for name, versions in catalog.items()]


def catalog_from_chunks(chunks: list[dict]) -> dict[str, tuple[str, ...]]:
    policies: dict[str, set[str]] = {}
    for chunk in chunks:
        policies.setdefault(chunk["policy"], set()).add(chunk["version"])
    return {
        name: tuple(sorted(versions, key=version_key))
        for name, versions in sorted(policies.items())
    }

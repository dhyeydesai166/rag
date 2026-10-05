"""Turn retrieved passages into a cited answer. The model only reads and judges."""

import hashlib
import html
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from rag.compare import SECTION_NUMBER, TITLE_SUFFIX
from rag.config import COMPARE_PROMPT, LOOKUP_PROMPT
from rag.logutil import log, stage, warn
from rag.messages import NOT_IN_SOURCES_MESSAGE
from rag.models import Answer

PROMPTS = Path(__file__).parent / "prompts"

# Near-duplicate lookup sentences only. Overlap is both sentences together, so
# a shorter sentence that only adds "only on Fridays" is not swallowed.
# Comparisons and conflicts do not use this.
REPEAT_WORD_OVERLAP = 0.8
_NEGATION_WORDS = frozenset({"no", "not", "cannot", "never"})
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
}
_CONTENT_STOP = frozenset(
    "a an the of to for and or in on at by with from that this is are was be "
    "been it its their they who when should must may will can into upon than "
    "any no not".split()
)


def _numbers(text: str) -> frozenset[str]:
    """Digit groups, and number words, as the same values.

    '1,000,000' and 'five' both become number tokens, so 'five days' and
    'six days' are not the same claim.
    """
    digits = {
        match.replace(",", "") for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    }
    words = set(re.findall(r"[a-z]+", text.lower()))
    digits.update(_NUMBER_WORDS[word] for word in words if word in _NUMBER_WORDS)
    return frozenset(digits)


def _has_negation(text: str) -> bool:
    lowered = text.lower()
    if "n't" in lowered or "n’t" in lowered:
        return True
    words = set(re.findall(r"[a-z0-9]+", lowered))
    return bool(words & _NEGATION_WORDS)


def same_claim(left: str, right: str) -> bool:
    """True when two sentences state the same rule.

    Why: several passages often say one rule, and the model writes a sentence
    for each. Different numbers are different facts, whether written as digits
    or as words. A negation ('not', 'no', 'cannot', 'never', or n't) is not the
    same claim as its opposite. Overlap uses both sentences together, so a
    short sentence is not dropped just because a longer one contains its words.
    Display of a comparison or a conflict does not use this.
    """
    if _numbers(left) != _numbers(right):
        return False
    if _has_negation(left) != _has_negation(right):
        return False
    left_words = _content_words(left)
    right_words = _content_words(right)
    union = left_words | right_words
    if not union:
        return True
    return len(left_words & right_words) / len(union) >= REPEAT_WORD_OVERLAP


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _CONTENT_STOP
    }


def load_prompt(name: str) -> str:
    """Read a prompt file shipped with the package (src/rag/prompts/<name>.md)."""
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def prompt_sha256(name: str) -> str:
    """Hash recorded in provenance, so an accidental in-place edit is visible."""
    return hashlib.sha256(load_prompt(name).encode()).hexdigest()


def source_tag(chunk: dict) -> str:
    """Wrap one passage so its text cannot be read as instructions.

    Attribute values and text are escaped so passage text cannot close the tag.
    """
    attrs = " ".join(
        f'{name}="{html.escape(str(value), quote=True)}"'
        for name, value in (
            ("id", chunk["id"]),
            ("policy", chunk["policy"]),
            ("version", chunk["version"]),
            ("section", chunk["heading_path"]),
        )
    )
    body = html.escape(chunk["text"], quote=False)
    return f"<source {attrs}>\n{body}\n</source>"


def describe_route(route: dict) -> str:
    """'lookup on HR Policy 2.0' or 'compare Time & Usage Policy 1.0 -> 2.0'."""
    if route.get("kind") == "compare":
        old, new = route["versions"]
        return f"compare {route['policy']} {old} -> {new}"
    targets = route.get("targets") or []
    if not targets:
        return "lookup"
    named = ", ".join(f"{policy} {version}" for policy, version in targets)
    return f"lookup on {named}"


def _side_tag(side: dict | None, policy: str, version: str) -> str:
    if side is None:
        escaped = html.escape(version, quote=True)
        return f'<missing version="{escaped}"/>'
    chunk = dict(side)
    chunk["policy"] = policy
    return source_tag(chunk)


def pair_tag(pair: dict, old_version: str, new_version: str) -> str:
    """Older passage, then newer, inside one pair. A missing side stays visible."""
    section = html.escape(pair["title"], quote=True)
    older = _side_tag(pair["previous"], pair["policy"], old_version)
    newer = _side_tag(pair["current"], pair["policy"], new_version)
    return f'<pair section="{section}">\n{older}\n{newer}\n</pair>'


def build_user_message(question: str, route: dict, hits: list[dict]) -> str:
    """Original question, the code-decided route, and tagged passages."""
    if route.get("kind") == "compare":
        old, new = route["versions"]
        passages = "\n\n".join(pair_tag(hit, old, new) for hit in hits)
    else:
        passages = "\n\n".join(source_tag(hit) for hit in hits)
    return f"Question: {question}\n\nRoute: {describe_route(route)}\n\n{passages}"


def answer_schema(allowed_ids: list[str]) -> dict:
    """JSON schema for Answer, with chunk_id limited to the ids actually sent.

    Why the enum: Ollama constrains decoding to the schema, so the model cannot
    invent an id; the eval's citation check then guards against regressions.
    """
    schema = Answer.model_json_schema()
    schema["$defs"]["Claim"]["properties"]["chunk_id"]["enum"] = list(allowed_ids)
    return schema


def sources_by_id(route: dict, hits: list[dict]) -> dict[str, dict]:
    """Every passage id the model was shown, for citation rendering."""
    found = {}
    if route.get("kind") == "compare":
        for pair in hits:
            for side in (pair.get("previous"), pair.get("current")):
                if side:
                    found[side["id"]] = {**side, "policy": pair["policy"]}
        return found
    for hit in hits:
        found[hit["id"]] = hit
    return found


def parse_answer(raw: str) -> Answer:
    """Validate the model's JSON. Invalid JSON becomes status 'not_in_sources'
    with a logged warning; raw model text is never shown to the user."""
    try:
        return Answer.model_validate_json(raw)
    except (ValidationError, ValueError):
        warn("generate", "model returned invalid JSON; showing not_in_sources")
        return Answer(status="not_in_sources", claims=[])


def _clean_heading_part(part: str) -> str:
    return TITLE_SUFFIX.sub("", SECTION_NUMBER.sub("", part)).strip()


def section_label(heading_path: str) -> str:
    """'4. Foosball Time > 4.2 Dispute Resolution'
    -> 'Foosball Time > Dispute Resolution'.

    Keeps the parent so 'Daily Allowance' is not both video games and foosball.
    Strips a trailing 'Updated' or 'New in Version N' suffix from each part,
    the same way title pairing does.
    """
    parts = []
    for part in heading_path.split(" > "):
        cleaned = _clean_heading_part(part)
        if cleaned:
            parts.append(cleaned)
    return " > ".join(parts)


def missing_side_lines(route: dict, hits: list[dict], cited_ids: set[str]) -> list[str]:
    """Name a one-sided section when the answer cites the side that exists.

    Why: title pairing already knows a section was removed or added. The
    printed answer should say so for a pair the model cited, without depending
    on the model's wording. An uncited pair is not named.
    """
    if route.get("kind") != "compare":
        return []
    versions = route.get("versions") or ()
    if len(versions) != 2:
        return []
    new = versions[1]
    lines = []
    for pair in hits:
        label = section_label(pair.get("heading_path") or "")
        if not label:
            continue
        previous = pair.get("previous")
        current = pair.get("current")
        if previous and not current and previous.get("id") in cited_ids:
            lines.append(f"Removed in version {new}: {label}")
        elif current and not previous and current.get("id") in cited_ids:
            lines.append(f"Added in version {new}: {label}")
    return lines


def _plain(text: str) -> str:
    """Lowercase words and digits, so case and punctuation do not count."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _has_phrase(haystack: str, phrase: str) -> bool:
    return f" {phrase} " in f" {haystack} "


def _side_note_parts(note: str) -> tuple[str, str] | None:
    """'Removed in version 2.0: Parent > Leaf' -> the phrase and the leaf."""
    lowered = note.strip().lower()
    for kind in ("removed in version ", "added in version "):
        if not lowered.startswith(kind):
            continue
        version, sep, label = note.strip()[len(kind) :].partition(":")
        leaf = label.strip().split(" > ")[-1]
        if not sep or not version.strip() or not leaf:
            return None
        return f"{kind}{version.strip()}", leaf
    return None


def _note_already_said(note: str, sentences: list[str]) -> bool:
    """True when a sentence already states this added or removed line.

    The printed label includes the parent, and the model often names only the
    leaf. 'Removed in version 2.0: Dispute Resolution.' already says the note
    'Removed in version 2.0: Foosball Time > Dispute Resolution'.
    """
    plains = [_plain(sentence) for sentence in sentences]
    if _plain(note) in plains:
        return True
    parts = _side_note_parts(note)
    if parts is None:
        return False
    phrase, leaf = _plain(parts[0]), _plain(parts[1])
    if not phrase or not leaf:
        return False
    return any(
        _has_phrase(plain, phrase) and _has_phrase(plain, leaf) for plain in plains
    )


def _with_period(text: str) -> str:
    """End a printed sentence with one period.

    A trailing comma, semicolon, or colon is removed first, so 'score,' becomes
    'score.' rather than 'score,.'.
    """
    stripped = text.strip()
    if not stripped or stripped[-1] in ".?!":
        return stripped
    stripped = stripped.rstrip(",;:").rstrip()
    if not stripped or stripped[-1] in ".?!":
        return stripped
    return f"{stripped}."


def render(
    answer: Answer,
    sources: dict[str, dict],
    kind: str | None = None,
    notes: list[str] | None = None,
) -> str:
    """Plain sentences, a blank line, then only the sources those sentences cite.

    A later lookup sentence is dropped only when it nearly repeats an earlier
    one and uses the same numbers. Comparisons and conflicts keep every
    sentence. Added and removed lines are printed on their own lines.
    """
    if answer.status == "not_in_sources":
        return NOT_IN_SOURCES_MESSAGE
    skip_dedupe = kind == "compare" or answer.status == "conflicting"
    cited: list[dict] = []
    seen: set[str] = set()
    sentences = []
    for claim in answer.claims:
        source = sources.get(claim.chunk_id)
        if source is None:
            warn("generate", f"dropping claim with unknown chunk_id={claim.chunk_id}")
            continue
        text = claim.text.strip()
        if not skip_dedupe and any(same_claim(text, earlier) for earlier in sentences):
            continue
        sentences.append(text)
        if claim.chunk_id not in seen:
            seen.add(claim.chunk_id)
            cited.append(source)
    kept_notes = [
        note for note in (notes or []) if not _note_already_said(note, sentences)
    ]
    if not sentences and not kept_notes:
        return NOT_IN_SOURCES_MESSAGE
    if answer.status == "conflicting":
        sentences.insert(0, "The sources disagree.")
    listing = ["Sources"]
    for index, source in enumerate(cited, start=1):
        policy = source["policy"]
        version = source["version"]
        heading = source["heading_path"]
        listing.append(f"[{index}] {policy} {version}, {heading}")
    paragraph = " ".join(_with_period(sentence) for sentence in sentences)
    note_block = "\n".join(_with_period(note) for note in kept_notes)
    body = "\n".join(part for part in (paragraph, note_block) if part)
    return body + "\n\n" + "\n".join(listing)


@dataclass
class GeneratedAnswer:
    answer: Answer
    text: str
    prompt_name: str


def generate(question: str, result: dict, model) -> GeneratedAnswer:
    """Pick the prompt for the route, call the model with the schema, render."""
    route = result.get("route") or {"kind": result["kind"]}
    hits = result["hits"]
    prompt_name = COMPARE_PROMPT if route.get("kind") == "compare" else LOOKUP_PROMPT
    if not hits:
        empty = Answer(status="not_in_sources", claims=[])
        return GeneratedAnswer(empty, NOT_IN_SOURCES_MESSAGE, prompt_name)
    sources = sources_by_id(route, hits)
    user = build_user_message(question, route, hits)
    schema = answer_schema(list(sources))
    with stage("generate"):
        raw = model.generate(user, system=load_prompt(prompt_name), schema=schema)
    answer = parse_answer(raw)
    log("generate", f"kind={route.get('kind')} hits={len(hits)} status={answer.status}")
    cited_ids = {claim.chunk_id for claim in answer.claims}
    notes = missing_side_lines(route, hits, cited_ids)
    text = render(answer, sources, kind=route.get("kind"), notes=notes)
    return GeneratedAnswer(answer, text, prompt_name)

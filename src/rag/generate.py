"""Turn retrieved passages into a cited answer. The model only reads and judges."""

import hashlib
import html
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from rag.config import COMPARE_PROMPT, LOOKUP_PROMPT
from rag.logutil import log, stage, warn
from rag.messages import NOT_IN_SOURCES_MESSAGE
from rag.models import Answer

PROMPTS = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt file shipped with the package (src/rag/prompts/<name>.txt)."""
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


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


def render(answer: Answer, sources: dict[str, dict]) -> str:
    """Plain sentences, a blank line, then only the sources those sentences cite."""
    if answer.status == "not_in_sources":
        return NOT_IN_SOURCES_MESSAGE
    cited: list[dict] = []
    seen: set[str] = set()
    sentences = []
    for claim in answer.claims:
        source = sources.get(claim.chunk_id)
        if source is None:
            warn("generate", f"dropping claim with unknown chunk_id={claim.chunk_id}")
            continue
        if claim.chunk_id not in seen:
            seen.add(claim.chunk_id)
            cited.append(source)
        sentences.append(claim.text.strip())
    if not sentences:
        return NOT_IN_SOURCES_MESSAGE
    if answer.status == "conflicting":
        sentences.insert(0, "The sources disagree.")
    listing = ["Sources"]
    for index, source in enumerate(cited, start=1):
        policy = source["policy"]
        version = source["version"]
        heading = source["heading_path"]
        listing.append(f"[{index}] {policy} {version}, {heading}")
    return " ".join(sentences) + "\n\n" + "\n".join(listing)


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
    return GeneratedAnswer(answer, render(answer, sources), prompt_name)

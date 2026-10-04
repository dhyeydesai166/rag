from rag.config import GENERATE_MODEL
from rag.logutil import log, stage

EMPTY = "No matching policy text."
SYSTEM = (
    "You answer questions about Doofenshmirtz Evil Inc policies.\n"
    "Use only the policy passages in the user message.\n"
    "If the passages do not contain the answer, say so.\n"
    "For a comparison, describe what changed between the current and "
    "previous text of each section.\n"
    "Write the answer in as few sentences as possible to cover understanding. "
    "Include the specific rule from the passages so the answer can stand on "
    "its own. Do not include policy names, versions, headings, or citations.\n"
    "Do not use outside knowledge."
)


def generation_model() -> str:
    return GENERATE_MODEL


def side_text(label: str, item) -> str:
    if item is None:
        return label
    return f"{label} {item['version']}\n{item['text']}"


def chunk_block(hit: dict) -> str:
    return f"{hit['policy']} {hit['version']} {hit['heading_path']}\n{hit['text']}"


def pair_block(pair: dict) -> str:
    current = side_text("current", pair["current"])
    previous = side_text("previous", pair["previous"])
    return f"{pair['policy']} {pair['heading_path']}\n{current}\n{previous}"


def citations(kind: str, hits: list) -> str:
    lines = []
    for hit in hits:
        if kind == "compare":
            if hit.get("current"):
                lines.append(
                    f"{hit['policy']} {hit['current']['version']}, "
                    f"{hit['heading_path']}"
                )
            if hit.get("previous"):
                lines.append(
                    f"{hit['policy']} {hit['previous']['version']}, "
                    f"{hit['heading_path']}"
                )
        else:
            lines.append(f"{hit['policy']} {hit['version']}, {hit['heading_path']}")
    return "\n".join(lines)


def generate(question: str, kind: str, hits: list, model) -> str:
    if not hits:
        return EMPTY
    blocks = pair_block if kind == "compare" else chunk_block
    body = "\n\n".join(blocks(hit) for hit in hits)
    with stage("generate"):
        text = model.generate(f"Question: {question}\n\n{body}", system=SYSTEM)
    log("generate", f"kind={kind} hits={len(hits)}")
    return f"{text.strip()}\n\n{citations(kind, hits)}"


# TODO: Add LLM as judge before entering enterprise

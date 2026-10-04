"""Read a policy file into heading blocks.

Policy name and version come from the document title line. The filename is
only a fallback, because filenames drift ("Time and Usage" vs "Time & Usage").
"""

import re
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from rag.config import TITLE_SEARCH_LINES
from rag.logutil import log

SUPPORTED = {".pdf", ".docx"}
SECTION = re.compile(r"^(\d+)\.\s+(.*)$")
SUBSECTION = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+(.*)$")
TITLE_LINE = re.compile(
    r"^\s*(?P<policy>.+?\bPolicy)\s*[—–-]\s*Version\s+(?P<version>\d+(?:\.\d+)*)\s*$"
)
FILENAME = re.compile(r"^(?:.+? - )?(?P<policy>.+?)\s+v(?P<version>\d+(?:\.\d+)*)$")


def normalize_version(raw: str) -> str:
    """'2' -> '2.0', '2.0' -> '2.0', '2.0.1' -> '2.0.1'."""
    parts = raw.split(".")
    if len(parts) == 1:
        return f"{parts[0]}.0"
    return raw


def version_key(version: str) -> tuple[int, ...]:
    """Sort key so '10.0' sorts after '9.0' (a string sort would not)."""
    return tuple(int(part) for part in version.split("."))


def policy_from_title(lines: list[str]) -> tuple[str, str] | None:
    """Find 'X Policy — Version N.N' near the top of the document.

    Why: the document's own title is the source of truth; filenames drift
    ("Time and Usage" vs "Time & Usage").
    """
    for line in lines[:TITLE_SEARCH_LINES]:
        match = TITLE_LINE.match(line)
        if match:
            return match["policy"].strip(), normalize_version(match["version"])
    return None


def policy_from_filename(path: Path) -> tuple[str, str] | None:
    """Fallback for documents without a title line."""
    match = FILENAME.match(Path(path).stem)
    if not match:
        return None
    return match["policy"].strip(), normalize_version(match["version"])


def _load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"unsupported suffix: {suffix}")
    documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    return "\n".join((doc.text or "") for doc in documents).strip()


def blocks_from_lines(lines: list[str]) -> list[dict]:
    blocks: list[dict] = []
    for raw in lines:
        line = " ".join(raw.split())
        if not line:
            continue
        match = SUBSECTION.match(line) or SECTION.match(line)
        if match:
            number, rest = match.group(1), match.group(2)
            if ". " in rest:
                title, body = rest.split(". ", 1)
            else:
                title, body = rest, ""
            separator = " " if "." in number else ". "
            blocks.append(
                {
                    "level": number.count(".") + 1,
                    "heading": f"{number}{separator}{title}",
                    "text": body,
                }
            )
        elif blocks:
            blocks[-1]["text"] = f"{blocks[-1]['text']} {line}".strip()
    return blocks


def _policy_and_version(path: Path, lines: list[str]) -> tuple[str, str]:
    nonempty = [line for line in lines if line.strip()]
    title = policy_from_title(nonempty)
    named = policy_from_filename(path)
    if title and named and title[1] != named[1]:
        raise ValueError(
            f"version mismatch in {path.name}: "
            f"title says {title[1]}, filename says {named[1]}"
        )
    if title and named and title[0] != named[0]:
        log("reader", f"file={path.name} title={title[0]!r} filename={named[0]!r}")
    chosen = title or named
    if chosen is None:
        raise ValueError(
            f"cannot find policy name and version in {path.name}: "
            "add a title line 'X Policy — Version N.N'"
        )
    return chosen


def read(path: Path | str) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"unsupported suffix: {suffix}")
    text = _load_text(path)
    if not text.strip():
        raise ValueError(f"empty extract: {path.name}")
    lines = text.splitlines()
    policy, version = _policy_and_version(path, lines)
    blocks = blocks_from_lines(lines)
    fmt = path.suffix.lower().lstrip(".")
    log(
        "reader",
        f"file={path.name} format={fmt} policy={policy} version={version} "
        f"lines={len(lines)} blocks={len(blocks)}",
    )
    return {
        "policy": policy,
        "version": version,
        "source": path.name,
        "lines": lines,
        "blocks": blocks,
    }

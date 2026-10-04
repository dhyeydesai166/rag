"""Facts needed to reproduce an eval result."""

import json
import os
import platform
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, datetime

from rag.config import (
    CHUNK_MAX_TOKENS,
    COMPARE_PROMPT,
    EMBED_MODEL,
    FUSED_TOP_K,
    GENERATE_MODEL,
    GENERATION_SEED,
    GENERATION_TEMPERATURE,
    LOOKUP_PROMPT,
    OLLAMA_HOST,
    RERANK_MODEL,
    RERANK_TOP_N,
    RRF_K,
)
from rag.generate import prompt_sha256
from rag.model_pins import _hex_digest, installed_digests


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_sha() -> str:
    """HEAD, with '-dirty' when the worktree has uncommitted changes."""
    try:
        sha = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    if status:
        return f"{sha}-dirty"
    return sha


def digest_of(client, model: str) -> str:
    """First 12 hex chars of the installed model, or '' when it is missing."""
    return _hex_digest(installed_digests(client).get(model, ""))[:12]


def ollama_server_version() -> str:
    """Version string from the local Ollama server, or 'unknown'."""
    url = OLLAMA_HOST.rstrip("/") + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode())
        return str(payload.get("version") or "unknown")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return "unknown"


def ci_run() -> dict:
    """Which CI run produced a result, so a downloaded artifact can be traced back.

    Empty outside CI.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not run_id:
        return {}
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    return {
        "event": os.environ.get("GITHUB_EVENT_NAME", ""),
        "run_url": f"{server}/{repository}/actions/runs/{run_id}",
    }


def provenance(client, build: dict) -> dict:
    """Facts needed to reproduce a result: code, models, prompts, index, date."""
    return {
        "git_sha": git_sha(),
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "embed_model": EMBED_MODEL,
        "embed_digest": digest_of(client, EMBED_MODEL),
        "generate_model": GENERATE_MODEL,
        "generate_digest": digest_of(client, GENERATE_MODEL),
        "rerank_model": RERANK_MODEL,
        "prompts": {
            name: prompt_sha256(name) for name in (LOOKUP_PROMPT, COMPARE_PROMPT)
        },
        "build_id": build.get("build_id", ""),
        "collection": build.get("collection", ""),
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "rrf_k": RRF_K,
        "fused_top_k": FUSED_TOP_K,
        "rerank_top_n": RERANK_TOP_N,
        "temperature": GENERATION_TEMPERATURE,
        "seed": GENERATION_SEED,
        "ollama_version": ollama_server_version(),
        "python": platform.python_version(),
        "ci": ci_run(),
    }

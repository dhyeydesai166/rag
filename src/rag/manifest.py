"""Record which files and which build are active.

Chroma has no multi-statement transactions. The manifest records a file's
hash only after that file's chunks are fully written, and it is replaced
atomically so a crash leaves a state the next run can repair.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from rag import config


def index_params() -> dict:
    """Everything that changes the vectors: model, digest, dim, chunking settings."""
    return {
        "embed_model": config.EMBED_MODEL,
        "embed_digest": config.EMBED_MODEL_DIGEST,
        "embed_dim": config.EMBED_DIM,
        "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
        "chunk_overlap_sentences": config.CHUNK_OVERLAP_SENTENCES,
        "chunker_version": config.CHUNKER_VERSION,
    }


def compute_build_id(params: dict) -> str:
    """First 12 hex chars of sha256 over the params as sorted JSON.

    Why: the same settings always give the same build id, and any change to
    them gives a new one, which is exactly when a full rebuild is needed.
    """
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_path(db_path: Path) -> Path:
    return Path(db_path) / "index_manifest.json"


def empty_manifest() -> dict:
    return {"active_build_id": "", "builds": {}, "files": {}}


def load_manifest(db_path: Path) -> dict:
    path = _manifest_path(db_path)
    if not path.is_file():
        return empty_manifest()
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(db_path: Path, manifest: dict) -> None:
    root = Path(db_path)
    root.mkdir(parents=True, exist_ok=True)
    target = _manifest_path(root)
    temporary = root / "index_manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


class IndexMissing(RuntimeError):
    """The index the manifest names is missing or empty; ingest must run first.

    A RuntimeError, so code that catches the old 'run ingest first' error still works.
    """


def active_build_id(db_path: Path) -> str:
    build_id = load_manifest(db_path).get("active_build_id") or ""
    if not build_id:
        raise IndexMissing("run python -m rag.ingest first")
    return build_id


def active_collection_name(manifest: dict) -> str:
    """Collection the manifest points at. Older manifests only record a build id."""
    if manifest.get("active_collection"):
        return manifest["active_collection"]
    build_id = manifest.get("active_build_id") or ""
    if not build_id:
        raise IndexMissing("run python -m rag.ingest first")
    return f"{config.COLLECTION_PREFIX}__{build_id}"


def staging_collection_name(build_id: str, stamp: str) -> str:
    """A fresh name per rebuild, so the active collection is never written to."""
    return f"{config.COLLECTION_PREFIX}__{build_id}__{stamp}"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    """UTC stamp with microseconds so two rebuilds in one second differ."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")

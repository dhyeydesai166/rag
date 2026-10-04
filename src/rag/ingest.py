"""Bring the Chroma index in line with the files in a directory.

Unchanged files are skipped. A changed file has its chunks replaced.
A removed file has its chunks deleted. A change in model or chunking
settings rebuilds a new collection, then swaps the manifest over to it.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import ollama

from adapter.database_adapter import DatabaseAdapter
from adapter.embedding_adapter import EmbeddingAdapter
from rag.chunker import chunk
from rag.config import EMBED_MODEL, EMBED_MODEL_DIGEST, OLLAMA_HOST
from rag.logutil import log
from rag.manifest import (
    compute_build_id,
    file_sha256,
    index_params,
    load_manifest,
    save_manifest,
    utc_now,
)
from rag.model_pins import check_model_pin
from rag.reader import SUPPORTED, read
from rag.validate import validate

# TODO: Add support for .md filetype. The reader already takes policy and
# version from the title line, so a markdown reader can plug in later.


@dataclass
class IngestReport:
    skipped: list[str]
    updated: list[str]
    removed: list[str]
    embedded: int
    reused: int


def list_policy_files(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"missing directory: {directory}")
    files = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in SUPPORTED
    )
    if not files:
        raise ValueError(f"no policy files: {directory}")
    return files


def reuse_or_embed(records, database, embedder) -> tuple[list, int, int]:
    """Reuse a stored vector when embed_sha256 matches; embed only the rest."""
    stored = database.vectors_by_embed_sha(
        [record["embed_sha256"] for record in records]
    )
    missing = [record for record in records if record["embed_sha256"] not in stored]
    fresh = {}
    if missing:
        ids = [record["id"] for record in missing]
        try:
            vectors = embedder.embed(
                [record["embed_text"] for record in missing], task="document"
            )
        except ollama.ResponseError as error:
            raise ollama.ResponseError(
                f"chunk too long for {EMBED_MODEL}: {ids}"
            ) from error
        for record, vector in zip(missing, vectors, strict=True):
            fresh[record["embed_sha256"]] = vector
    ordered = []
    for record in records:
        digest = record["embed_sha256"]
        ordered.append(fresh[digest] if digest in fresh else stored[digest])
    return ordered, len(missing), len(records) - len(missing)


def replace_file_chunks(database, embedder, records, old_ids) -> tuple[int, int]:
    vectors, embedded, reused = reuse_or_embed(records, database, embedder)
    database.upsert(records, vectors)
    stale = sorted(set(old_ids) - {record["id"] for record in records})
    database.delete_ids(stale)
    return embedded, reused


def _validated(path: Path, read_file) -> list[dict] | str:
    loaded = read_file(path)
    records = chunk(loaded["blocks"], loaded["policy"], loaded["version"], path.name)
    for record in records:
        error = validate(record)
        if error:
            log("ingest", "failed")
            return error
    return records


def _summary(report: IngestReport) -> IngestReport:
    log(
        "ingest",
        f"skipped={len(report.skipped)} updated={len(report.updated)} "
        f"removed={len(report.removed)} embedded={report.embedded} "
        f"reused={report.reused}",
    )
    return report


def rebuild_index(directory, embedder, db_path, params, build_id, read_file):
    """Build a complete new collection, then point the manifest at it.

    Why: vectors from different models or chunking settings must never be mixed
    in one index. Queries keep using the old build until the swap.
    """
    db_path = Path(db_path)
    previous = load_manifest(db_path)
    old_ids = [key for key in previous.get("builds", {}) if key != build_id]
    database = DatabaseAdapter(db_path, build_id)
    database.drop()
    database = DatabaseAdapter(db_path, build_id)
    report = IngestReport([], [], [], 0, 0)
    file_entries = {}
    for path in list_policy_files(directory):
        records = _validated(path, read_file)
        if isinstance(records, str):
            database.drop()
            return records
        embedded, reused = replace_file_chunks(database, embedder, records, [])
        file_entries[path.name] = {
            "sha256": file_sha256(path),
            "chunk_ids": [record["id"] for record in records],
        }
        report.updated.append(path.name)
        report.embedded += embedded
        report.reused += reused
    builds = dict(previous.get("builds", {}))
    builds[build_id] = {**params, "created_at": utc_now()}
    manifest = {
        "active_build_id": build_id,
        "builds": builds,
        "files": file_entries,
    }
    save_manifest(db_path, manifest)
    for old_id in old_ids:
        DatabaseAdapter(db_path, old_id).drop()
    manifest["builds"] = {build_id: builds[build_id]}
    save_manifest(db_path, manifest)
    return _summary(report)


def ingest(directory, embedder, db_path, read_file=read, rebuild: bool = False):
    """Bring the index in line with the files in `directory`.

    Unchanged files are skipped; changed files have their chunks replaced;
    removed files have their chunks deleted. A change in model or chunking
    settings triggers a full rebuild into a new collection, then a swap.
    """
    directory = Path(directory)
    db_path = Path(db_path)
    files = list_policy_files(directory)
    params = index_params()
    build_id = compute_build_id(params)
    manifest = load_manifest(db_path)
    if rebuild or build_id != manifest.get("active_build_id"):
        return rebuild_index(directory, embedder, db_path, params, build_id, read_file)

    database = DatabaseAdapter(db_path, build_id)
    report = IngestReport([], [], [], 0, 0)
    seen = set()
    for path in files:
        seen.add(path.name)
        digest = file_sha256(path)
        previous = manifest["files"].get(path.name)
        if previous and previous.get("sha256") == digest:
            report.skipped.append(path.name)
            continue
        records = _validated(path, read_file)
        if isinstance(records, str):
            return records
        old_ids = previous.get("chunk_ids", []) if previous else []
        embedded, reused = replace_file_chunks(database, embedder, records, old_ids)
        manifest["files"][path.name] = {
            "sha256": digest,
            "chunk_ids": [record["id"] for record in records],
        }
        save_manifest(db_path, manifest)
        report.updated.append(path.name)
        report.embedded += embedded
        report.reused += reused

    for name in sorted(set(manifest["files"]) - seen):
        database.delete_ids(manifest["files"][name].get("chunk_ids", []))
        del manifest["files"][name]
        save_manifest(db_path, manifest)
        report.removed.append(name)
    return _summary(report)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rebuild = False
    if "--rebuild" in argv:
        argv.remove("--rebuild")
        rebuild = True
    directory = argv[0] if argv else "docs"
    db_path = argv[1] if len(argv) > 1 else "chroma"
    client = ollama.Client(host=OLLAMA_HOST)
    check_model_pin(client, EMBED_MODEL, EMBED_MODEL_DIGEST)
    result = ingest(
        directory,
        embedder=EmbeddingAdapter(),
        db_path=db_path,
        rebuild=rebuild,
    )
    if isinstance(result, str):
        print(result)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

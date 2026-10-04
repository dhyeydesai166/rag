import sys
from pathlib import Path

import ollama

from adapter.database_adapter import DatabaseAdapter
from adapter.embedding_adapter import EmbeddingAdapter
from rag.chunker import chunk_path
from rag.config import EMBED_MODEL, EMBED_MODEL_DIGEST, OLLAMA_HOST
from rag.logutil import log
from rag.model_pins import check_model_pin
from rag.reader import read
from rag.validate import validate

SUFFIXES = {".pdf", ".docx"}


# TODO: Add support for .md filetype


def ingest(directory, embedder, database, read_file=read, chunk_file=chunk_path):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"missing directory: {directory}")
    files = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in SUFFIXES
    )
    if not files:
        raise ValueError(f"no policy files: {directory}")
    log("ingest", f"directory={directory} files={len(files)}")
    pending = []
    for path in files:
        loaded = read_file(path)
        records = chunk_file(path, loaded["blocks"])
        for record in records:
            if not record.get("embed"):
                continue
            error = validate(record)
            if error:
                log("ingest", "failed")
                return error
            pending.append(record)
    if pending:
        vectors = embedder.embed(
            [record["embed_text"] for record in pending], task="document"
        )
        database.upsert(pending, vectors)
    log("ingest", "finished")
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    directory = argv[0] if argv else "docs"
    db_path = argv[1] if len(argv) > 1 else "chroma"
    client = ollama.Client(host=OLLAMA_HOST)
    check_model_pin(client, EMBED_MODEL, EMBED_MODEL_DIGEST)
    error = ingest(
        directory, embedder=EmbeddingAdapter(), database=DatabaseAdapter(db_path)
    )
    if error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

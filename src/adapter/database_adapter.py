"""Chroma store for one collection.

Reads open the collection the manifest names and never create an empty one.
A rebuild writes a new collection, then the manifest switches to it.
Metadata cannot be None, so an unsplit chunk stores ordinal 0.
"""

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from rag.config import COLLECTION_PREFIX
from rag.logutil import log
from rag.manifest import IndexMissing, active_collection_name, load_manifest


def where_for(pairs: list[tuple[str, str]]) -> dict | None:
    """Chroma filter for a list of (policy, version) pairs.

    Chroma's $or/$and need at least two items, so a single pair is a plain $and.
    """
    clauses = [
        {"$and": [{"policy": policy}, {"version": version}]}
        for policy, version in pairs
    ]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


class DatabaseAdapter:
    def __init__(self, path, name: str, create: bool = False):
        """Open an existing collection, or create a new one when `create` is True.

        Why two modes: reads must never create an empty index by accident, and
        a rebuild must never write into a collection that already exists.
        """
        self.path = str(path)
        self.name = name
        self.client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(anonymized_telemetry=False),
        )
        if create:
            self.collection = self.client.create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection = self._existing(name)

    def _existing(self, name: str):
        try:
            return self.client.get_collection(name=name)
        except NotFoundError as error:
            raise IndexMissing(
                f"index {name} not found; run python -m rag.ingest --rebuild"
            ) from error

    def count(self) -> int:
        return self.collection.count()

    def all_ids(self) -> list[str]:
        return list(self.collection.get(include=[])["ids"])

    def upsert(self, records: list[dict], vectors: list[list[float]]) -> None:
        if not records:
            return
        self.collection.upsert(
            ids=[record["id"] for record in records],
            documents=[record["text"] for record in records],
            embeddings=vectors,
            metadatas=[self._metadata(record) for record in records],
        )
        log("database", f"path={self.path} upserts={len(records)}")

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=list(ids))

    def ids_for_source(self, source: str) -> list[str]:
        stored = self.collection.get(where={"source": source}, include=[])
        return list(stored["ids"])

    def vectors_by_embed_sha(self, shas: list[str]) -> dict[str, list[float]]:
        """Stored vectors keyed by embed_sha256, for reuse."""
        if not shas:
            return {}
        where = (
            {"embed_sha256": shas[0]}
            if len(shas) == 1
            else {"embed_sha256": {"$in": list(shas)}}
        )
        stored = self.collection.get(where=where, include=["embeddings", "metadatas"])
        found = {}
        embeddings = stored.get("embeddings")
        metadatas = stored.get("metadatas") or []
        for index, meta in enumerate(metadatas):
            vector = None if embeddings is None else embeddings[index]
            if vector is None or not meta:
                continue
            found[meta["embed_sha256"]] = [float(value) for value in vector]
        return found

    def chunks_where(self, where: dict | None) -> list[dict]:
        """Chunks without vectors, matching a metadata filter."""
        kwargs = {"include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        stored = self.collection.get(**kwargs)
        return self._chunks(stored)

    def dense_search(
        self, vector: list[float], where: dict | None, k: int
    ) -> list[str]:
        """Chunk ids ordered by cosine similarity, filtered in Chroma."""
        count = self.collection.count()
        if k <= 0 or count == 0:
            return []
        kwargs = {
            "query_embeddings": [vector],
            "n_results": min(k, count),
        }
        if where:
            kwargs["where"] = where
        found = self.collection.query(**kwargs)
        ids = found.get("ids") or []
        return list(ids[0]) if ids else []

    def drop(self) -> None:
        names = [collection.name for collection in self.client.list_collections()]
        if self.name in names:
            self.client.delete_collection(self.name)

    def _chunks(self, stored: dict) -> list[dict]:
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        return [
            self._one(record_id, documents[index], metadatas[index])
            for index, record_id in enumerate(stored["ids"])
        ]

    def _one(self, record_id: str, document: str, meta: dict | None) -> dict:
        meta = meta or {}
        ordinal = meta.get("ordinal", 0)
        return {
            "id": record_id,
            "text": document,
            "policy": meta.get("policy", ""),
            "version": meta.get("version", ""),
            "section": meta.get("section", ""),
            "heading_path": meta.get("heading_path", ""),
            "parent_id": meta.get("parent_id", ""),
            "source": meta.get("source", ""),
            "word_count": meta.get("word_count", 0),
            "embed_text": meta.get("embed_text", ""),
            "text_sha256": meta.get("text_sha256", ""),
            "embed_sha256": meta.get("embed_sha256", ""),
            "ordinal": None if not ordinal else int(ordinal),
        }

    def _metadata(self, record: dict) -> dict:
        return {
            "policy": record["policy"],
            "version": record["version"],
            "section": record["section"],
            "heading_path": record["heading_path"],
            "parent_id": record["parent_id"],
            "source": record["source"],
            "word_count": record["word_count"],
            "embed_text": record.get("embed_text", ""),
            "text_sha256": record.get("text_sha256", ""),
            "embed_sha256": record.get("embed_sha256", ""),
            "ordinal": int(record.get("ordinal") or 0),
        }


def open_active_index(db_path) -> DatabaseAdapter:
    """Open the collection the manifest names; fail loudly if it is missing or empty.

    Why: an empty index makes every answer 'not in the documents', which hides
    a broken setup behind a plausible reply.
    """
    name = active_collection_name(load_manifest(db_path))
    database = DatabaseAdapter(db_path, name)
    if database.count() == 0:
        raise IndexMissing(f"index {name} is empty; run python -m rag.ingest --rebuild")
    return database


def drop_collections_except(path, keep: str) -> list[str]:
    """Drop every policies__ collection except `keep`; return the dropped names.

    Why: old builds and leftovers from an interrupted rebuild are removed only
    after the manifest already points at `keep`.
    """
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    prefix = f"{COLLECTION_PREFIX}__"
    names = sorted(collection.name for collection in client.list_collections())
    dropped = [name for name in names if name.startswith(prefix) and name != keep]
    for name in dropped:
        client.delete_collection(name)
    return dropped

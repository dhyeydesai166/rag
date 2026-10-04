"""Chroma store for one embedding build.

Each build has its own collection so vectors from different models or
chunking settings are never mixed. Metadata cannot be None, so an unsplit
chunk stores ordinal 0.
"""

import chromadb
from chromadb.config import Settings

from rag.config import COLLECTION_PREFIX
from rag.logutil import log


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
    def __init__(self, path, build_id: str):
        self.path = str(path)
        self.build_id = build_id
        self.name = f"{COLLECTION_PREFIX}__{build_id}"
        self.client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},
        )

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

    def rows(self) -> list[dict]:
        stored = self.collection.get(include=["documents", "metadatas", "embeddings"])
        embeddings = stored.get("embeddings")
        records = []
        for index, record_id in enumerate(stored["ids"]):
            chunk = self._one(
                record_id,
                stored["documents"][index],
                stored["metadatas"][index],
            )
            vector = None if embeddings is None else embeddings[index]
            chunk["vector"] = (
                [] if vector is None else [float(value) for value in vector]
            )
            records.append(chunk)
        log("database", f"path={self.path} rows={len(records)}")
        return records

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

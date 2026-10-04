import chromadb
from chromadb.config import Settings

from rag.config import COLLECTION_NAME
from rag.logutil import log


class DatabaseAdapter:
    def __init__(self, path, name: str = COLLECTION_NAME):
        self.path = str(path)
        self.client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, records: list[dict], vectors: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[record["id"] for record in records],
            documents=[record["text"] for record in records],
            embeddings=vectors,
            metadatas=[
                {
                    "policy": record["policy"],
                    "version": record["version"],
                    "section": record["section"],
                    "heading_path": record["heading_path"],
                    "parent_id": record["parent_id"],
                    "source": record["source"],
                    "word_count": record["word_count"],
                }
                for record in records
            ],
        )
        log("database", f"path={self.path} upserts={len(records)}")

    def rows(self) -> list[dict]:
        stored = self.collection.get(include=["documents", "metadatas", "embeddings"])
        embeddings = stored.get("embeddings")
        records = []
        for index, record_id in enumerate(stored["ids"]):
            meta = stored["metadatas"][index] or {}
            vector = None if embeddings is None else embeddings[index]
            records.append(
                {
                    "id": record_id,
                    "text": stored["documents"][index],
                    "policy": meta.get("policy", ""),
                    "version": meta.get("version", ""),
                    "section": meta.get("section", ""),
                    "heading_path": meta.get("heading_path", ""),
                    "parent_id": meta.get("parent_id", ""),
                    "source": meta.get("source", ""),
                    "word_count": meta.get("word_count", 0),
                    "vector": []
                    if vector is None
                    else [float(value) for value in vector],
                }
            )
        log("database", f"path={self.path} rows={len(records)}")
        return records

import ollama

from rag.config import EMBED_BATCH_SIZE, EMBED_DIM, EMBED_MODEL, OLLAMA_HOST
from rag.logutil import log

PREFIX = {
    "document": "title: none | text: ",
    "query": "task: search result | query: ",
    "similarity": "task: sentence similarity | query: ",
}


class EmbeddingAdapter:
    def __init__(self, client=None, model: str | None = None, host: str | None = None):
        self.model = model or EMBED_MODEL
        self.client = client or ollama.Client(host=host or OLLAMA_HOST)

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        if task not in PREFIX:
            raise ValueError(task)
        if not texts:
            return []
        result = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            result.extend(self._embed_batch(batch, task))
        return result

    def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        payloads = [PREFIX[task] + text for text in texts]
        response = self.client.embed(model=self.model, input=payloads, truncate=False)
        vectors = (
            response["embeddings"]
            if isinstance(response, dict)
            else response.embeddings
        )
        result = [[float(value) for value in vector] for vector in vectors]
        for vector in result:
            if len(vector) != EMBED_DIM:
                raise ValueError(f"expected {EMBED_DIM} dims, got {len(vector)}")
        dimension = len(result[0]) if result else 0
        log("embedder", f"task={task} texts={len(texts)} dimension={dimension}")
        return result

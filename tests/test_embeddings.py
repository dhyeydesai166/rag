import pytest

from adapter.embedding_adapter import EmbeddingAdapter
from rag.config import EMBED_DIM


class FakeClient:
    def __init__(self, width=EMBED_DIM):
        self.calls = []
        self.width = width

    def embed(self, model, input):
        self.calls.append({"model": model, "input": input})
        return {
            "embeddings": [
                [float(index)] + [1.0] * (self.width - 1)
                for index, _ in enumerate(input)
            ]
        }


def test_document_and_query_use_task_prefixes_and_config_model():
    client = FakeClient()
    embedder = EmbeddingAdapter(client=client, model="embeddinggemma:300m")
    docs = embedder.embed(["alpha"], task="document")
    queries = embedder.embed(["beta"], task="query")
    assert len(docs[0]) == EMBED_DIM
    assert docs[0][0] == 0.0
    assert len(queries[0]) == EMBED_DIM
    assert client.calls[0]["model"] == "embeddinggemma:300m"
    assert client.calls[0]["input"][0].startswith("title: none | text: ")
    assert client.calls[1]["input"][0].startswith("task: search result | query: ")


def test_empty_list_returns_empty():
    client = FakeClient()
    assert EmbeddingAdapter(client=client).embed([], task="document") == []
    assert client.calls == []


def test_bad_task_raises():
    with pytest.raises(ValueError):
        EmbeddingAdapter(client=FakeClient()).embed(["x"], task="other")


def test_wrong_dimension_is_rejected():
    with pytest.raises(ValueError, match="expected 768 dims, got 2"):
        EmbeddingAdapter(client=FakeClient(width=2)).embed(["x"], task="document")

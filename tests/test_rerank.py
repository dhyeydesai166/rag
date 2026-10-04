import json

import pytest

from adapter.rerank_adapter import RerankerAdapter, post_rerank


def test_adapter_returns_ranked_documents(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_RERANK_MODEL", raising=False)

    def post(api_key, model, question, documents):
        assert api_key == "secret"
        assert model == "rerank-v3.5"
        assert question == "cake"
        return list(reversed(documents))

    ranked = RerankerAdapter(post=post, api_key="secret").rerank("cake", ["a", "b"])
    assert ranked == ["b", "a"]


def test_adapter_skips_an_empty_document_list(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "secret")
    assert RerankerAdapter(post=lambda *args: ["nope"]).rerank("cake", []) == []


def test_adapter_fails_when_the_key_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        RerankerAdapter(env_path=tmp_path / "missing.env")


def test_adapter_reads_the_key_from_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('COHERE_API_KEY="from-file"\n# comment\nignored\n')
    seen = {}

    def post(api_key, model, question, documents):
        seen["api_key"] = api_key
        return documents

    RerankerAdapter(post=post, env_path=env_file).rerank("cake", ["a"])
    assert seen["api_key"] == "from-file"


def test_adapter_uses_the_configured_model(monkeypatch):
    monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-test")
    seen = {}

    def post(api_key, model, question, documents):
        seen["model"] = model
        return documents

    RerankerAdapter(post=post, api_key="secret").rerank("cake", ["a"])
    assert seen["model"] == "rerank-test"


def test_post_rerank_orders_by_relevance(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.1},
                        {"index": 1, "relevance_score": 0.9},
                    ]
                }
            ).encode()

    def urlopen(request):
        assert request.full_url == "https://api.cohere.com/v2/rerank"
        assert request.get_header("Authorization") == "Bearer secret"
        return Response()

    monkeypatch.setattr("adapter.rerank_adapter.urlopen", urlopen)
    assert post_rerank("secret", "rerank-v3.5", "cake", ["vacation", "cake"]) == [
        "cake",
        "vacation",
    ]

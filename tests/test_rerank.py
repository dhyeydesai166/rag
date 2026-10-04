import json
import urllib.error

import pytest

from adapter.rerank_adapter import (
    RerankerAdapter,
    RerankUnavailable,
    make_reranker,
    post_rerank,
)


def test_adapter_returns_ranked_indices(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_RERANK_MODEL", raising=False)

    def post(api_key, model, question, documents, top_n, timeout):
        assert api_key == "secret"
        assert model == "rerank-v3.5"
        assert question == "cake"
        assert top_n == 2
        return [1, 0]

    ranked = RerankerAdapter(post=post, api_key="secret").rank("cake", ["a", "b"], 2)
    assert ranked == [1, 0]


def test_adapter_skips_an_empty_document_list(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "secret")
    assert (
        RerankerAdapter(post=lambda *args: [0], api_key="secret").rank("cake", [], 3)
        == []
    )


def test_adapter_fails_when_the_key_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        RerankerAdapter(env_path=tmp_path / "missing.env")


def test_missing_key_gives_no_reranker(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert make_reranker(env_path=tmp_path / "missing.env") is None


def test_adapter_reads_the_key_from_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('COHERE_API_KEY="from-file"\n# comment\nignored\n')
    seen = {}

    def post(api_key, model, question, documents, top_n, timeout):
        seen["api_key"] = api_key
        return [0]

    RerankerAdapter(post=post, env_path=env_file).rank("cake", ["a"], 1)
    assert seen["api_key"] == "from-file"


def test_adapter_uses_the_configured_model(monkeypatch):
    monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-test")
    seen = {}

    def post(api_key, model, question, documents, top_n, timeout):
        seen["model"] = model
        return [0]

    RerankerAdapter(post=post, api_key="secret").rank("cake", ["a"], 1)
    assert seen["model"] == "rerank-test"


def test_post_rerank_returns_indices_and_sends_top_n(monkeypatch):
    seen = {}

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

    def urlopen(request, timeout):
        seen["timeout"] = timeout
        body = json.loads(request.data.decode())
        seen["top_n"] = body["top_n"]
        assert request.full_url == "https://api.cohere.com/v2/rerank"
        assert request.get_header("Authorization") == "Bearer secret"
        return Response()

    monkeypatch.setattr("adapter.rerank_adapter.urlopen", urlopen)
    assert post_rerank(
        "secret", "rerank-v3.5", "cake", ["vacation", "cake"], 1, 10
    ) == [
        1,
        0,
    ]
    assert seen == {"timeout": 10, "top_n": 1}


def test_timeout_is_passed_to_urlopen(monkeypatch):
    test_post_rerank_returns_indices_and_sends_top_n(monkeypatch)


def test_one_retry_then_success(monkeypatch):
    monkeypatch.setattr("adapter.rerank_adapter.time.sleep", lambda _seconds: None)
    calls = {"n": 0}

    def post(api_key, model, question, documents, top_n, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("slow")
        return [0]

    order = RerankerAdapter(post=post, api_key="secret").rank("cake", ["a"], 1)
    assert order == [0]
    assert calls["n"] == 2


def test_two_failures_raise_rerank_unavailable(monkeypatch):
    monkeypatch.setattr("adapter.rerank_adapter.time.sleep", lambda _seconds: None)

    def post(*args):
        raise TimeoutError("slow")

    with pytest.raises(RerankUnavailable):
        RerankerAdapter(post=post, api_key="secret").rank("cake", ["a"], 1)


def test_auth_error_is_not_retried():
    calls = {"n": 0}

    def post(*args):
        calls["n"] += 1
        raise urllib.error.HTTPError("url", 401, "no", hdrs=None, fp=None)

    with pytest.raises(RerankUnavailable):
        RerankerAdapter(post=post, api_key="secret").rank("cake", ["a"], 1)
    assert calls["n"] == 1

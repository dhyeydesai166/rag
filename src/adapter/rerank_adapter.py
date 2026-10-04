"""Cohere rerank. A missing key or an outage is reported by the caller, not raised here
unless rank() itself is used and the service fails.
"""

import json
import time
import urllib.error
from urllib.request import Request, urlopen

from rag.config import (
    RERANK_MODEL,
    RERANK_RETRIES,
    RERANK_RETRY_DELAY_SECONDS,
    RERANK_TIMEOUT_SECONDS,
    env_value,
)
from rag.logutil import log


class RerankUnavailable(Exception):
    """Cohere could not rerank. The caller keeps the fused order."""


def cohere_api_key(path=".env") -> str:
    key = env_value("COHERE_API_KEY", path=path)
    if not key:
        raise ValueError("COHERE_API_KEY is missing")
    return key


def cohere_model_name(path=".env") -> str:
    return env_value("COHERE_RERANK_MODEL", RERANK_MODEL, path)


def post_rerank(
    api_key: str,
    model: str,
    question: str,
    documents: list[str],
    top_n: int,
    timeout: float,
) -> list[int]:
    """Indices in relevance order. Cohere v2 returns an index per result."""
    body = json.dumps(
        {
            "model": model,
            "query": question,
            "documents": documents,
            "top_n": top_n,
        }
    ).encode()
    request = Request(
        "https://api.cohere.com/v2/rerank",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    ranked = sorted(
        payload["results"],
        key=lambda item: item["relevance_score"],
        reverse=True,
    )
    return [item["index"] for item in ranked]


def make_reranker(env_path=".env") -> "RerankerAdapter | None":
    """None when COHERE_API_KEY is missing, so retrieval can fall back."""
    if not env_value("COHERE_API_KEY", path=env_path):
        return None
    return RerankerAdapter(env_path=env_path)


class RerankerAdapter:
    def __init__(
        self,
        post=None,
        api_key: str | None = None,
        model: str | None = None,
        env_path=".env",
        timeout: float = RERANK_TIMEOUT_SECONDS,
    ):
        self.api_key = cohere_api_key(env_path) if api_key is None else api_key
        self.model = model or cohere_model_name(env_path)
        self.post = post or post_rerank
        self.timeout = timeout

    def rank(self, question: str, documents: list[str], top_n: int) -> list[int]:
        """Retry once on timeout, connection errors, HTTP 429 or 5xx.

        Other HTTP 4xx errors (bad key, bad request) fail immediately.
        """
        if not documents:
            return []
        last_error: Exception | None = None
        for attempt in range(RERANK_RETRIES + 1):
            try:
                order = self.post(
                    self.api_key,
                    self.model,
                    question,
                    documents,
                    top_n,
                    self.timeout,
                )
                log("rerank", f"documents={len(documents)} ranked={len(order)}")
                return order
            except urllib.error.HTTPError as error:
                if error.code not in _retryable_status(error.code):
                    raise RerankUnavailable(f"HTTP {error.code}") from error
                last_error = error
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
            if attempt < RERANK_RETRIES:
                time.sleep(RERANK_RETRY_DELAY_SECONDS)
        raise RerankUnavailable(str(last_error))


def _retryable_status(code: int) -> set[int]:
    if code == 429 or code >= 500:
        return {code}
    return set()

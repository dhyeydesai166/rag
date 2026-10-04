"""Cohere rerank. A missing key or an outage is reported by the caller, not raised here
unless rank() itself is used and the service fails.
"""

import http.client
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

# Network failures: timeouts, refused, reset or dropped connections, truncated
# bodies. URLError, TimeoutError and ConnectionResetError are all OSError;
# IncompleteRead and RemoteDisconnected are http.client errors.
TRANSPORT_ERRORS = (OSError, http.client.HTTPException)
# A response we cannot read: not JSON (JSONDecodeError is a ValueError), or
# missing 'results' / 'index' / 'relevance_score', or the wrong shape.
BAD_RESPONSE_ERRORS = (ValueError, KeyError, TypeError)


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
        """Retry once on network errors, HTTP 429 or 5xx; anything else fails at once.

        Every failure becomes RerankUnavailable, so the caller can keep the fused order.
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
            except urllib.error.HTTPError as error:  # before OSError: it is one
                if not _is_retryable_status(error.code):
                    raise RerankUnavailable(f"HTTP {error.code}") from error
                last_error = error
            except TRANSPORT_ERRORS as error:
                last_error = error
            except BAD_RESPONSE_ERRORS as error:
                raise RerankUnavailable(f"unreadable response: {error!r}") from error
            else:
                log("rerank", f"documents={len(documents)} ranked={len(order)}")
                return order
            if attempt < RERANK_RETRIES:
                time.sleep(RERANK_RETRY_DELAY_SECONDS)
        raise RerankUnavailable(f"{type(last_error).__name__}: {last_error}")


def _is_retryable_status(code: int) -> bool:
    """429 (rate limit) and 5xx (server trouble) can pass; other 4xx will not."""
    return code == 429 or code >= 500

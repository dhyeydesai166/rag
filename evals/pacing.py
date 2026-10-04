"""Pacing between eval questions so a trial Cohere key is not rate-limited."""

import time

from rag.config import EVAL_PAUSE_SECONDS


def pause_between_questions(reranker, sleep=time.sleep) -> None:
    """Wait EVAL_PAUSE_SECONDS, but only when Cohere is in use.

    Why: without a key there are no rerank calls, so waiting only slows the run.
    """
    if reranker is not None:
        sleep(EVAL_PAUSE_SECONDS)

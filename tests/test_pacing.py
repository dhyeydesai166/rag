from evals.pacing import pause_between_questions
from rag.config import EVAL_PAUSE_SECONDS


def test_no_pause_without_a_cohere_key():
    calls = []
    pause_between_questions(None, sleep=calls.append)
    assert calls == []


def test_pause_uses_the_configured_seconds():
    calls = []
    pause_between_questions(object(), sleep=calls.append)
    assert calls == [EVAL_PAUSE_SECONDS]

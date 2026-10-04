import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("rag")
_question = False


def configure_logging() -> None:
    logger.setLevel(logging.INFO)
    # Third-party libraries and basicConfig attach root handlers. Propagation
    # would print every rag line a second time.
    logger.propagate = False
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.name = "console"
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


def silence_console() -> None:
    configure_logging()
    for handler in logger.handlers:
        if handler.name == "console":
            handler.setLevel(logging.WARNING)


def enable_question_log() -> None:
    global _question
    _question = True
    configure_logging()
    for handler in logger.handlers:
        if handler.name == "console":
            handler.setLevel(logging.INFO)


def disable_question_log() -> None:
    global _question
    _question = False
    for handler in logger.handlers:
        if handler.name == "console":
            handler.setLevel(logging.WARNING)


def log(step: str, message: str) -> None:
    configure_logging()
    logger.info("%s %s", step, message)


def warn(step: str, message: str) -> None:
    """Warning that stays visible when the console is silenced.

    Used when a service fails and the pipeline continues with a fallback.
    """
    configure_logging()
    logger.warning("%s %s", step, message)


@contextmanager
def stage(step: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        if _question:
            elapsed = time.perf_counter() - started
            log(step, f"latency={elapsed:.3f}s")

import logging

from rag.logutil import log


def test_each_log_line_is_printed_once(capsys):
    rag_logger = logging.getLogger("rag")
    saved = list(rag_logger.handlers)
    rag_logger.handlers.clear()
    root = logging.getLogger()
    received = []

    class RootHandler(logging.Handler):
        def emit(self, record):
            received.append(record)

    handler = RootHandler()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.INFO)
    try:
        log("x", "hello-once")
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)
        rag_logger.handlers.clear()
        rag_logger.handlers.extend(saved)
    assert received == []
    assert capsys.readouterr().err.count("hello-once") == 1

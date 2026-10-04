import logging

import pytest


@pytest.fixture
def rag_logs(caplog):
    """Attach caplog to the rag logger, which does not propagate to the root."""
    logger = logging.getLogger("rag")
    logger.addHandler(caplog.handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO, logger="rag")
    yield caplog
    logger.removeHandler(caplog.handler)
    logger.setLevel(previous)

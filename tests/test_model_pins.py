import pytest

from rag.model_pins import check_model_pin, installed_digests


class FakeClient:
    def __init__(self, models):
        self.models = models

    def list(self):
        return {"models": self.models}


def test_installed_digests_reads_object_models():
    class Model:
        model = "gemma3:4b"
        digest = "sha256:a2af6cc3eb7f"

    class Response:
        models = [Model()]

    class Client:
        def list(self):
            return Response()

    assert installed_digests(Client()) == {"gemma3:4b": "sha256:a2af6cc3eb7f"}


def test_installed_digests_reads_name_and_digest():
    client = FakeClient(
        [{"model": "embeddinggemma:300m", "digest": "sha256:85462619ee72abcd"}]
    )
    assert installed_digests(client) == {
        "embeddinggemma:300m": "sha256:85462619ee72abcd"
    }


def test_matching_digest_passes():
    client = FakeClient(
        [{"model": "embeddinggemma:300m", "digest": "sha256:85462619ee72abcd"}]
    )
    check_model_pin(client, "embeddinggemma:300m", "85462619ee72")


def test_mismatched_digest_raises():
    client = FakeClient(
        [{"model": "embeddinggemma:300m", "digest": "sha256:ffffffffffffabcd"}]
    )
    with pytest.raises(RuntimeError, match="expected 85462619ee72"):
        check_model_pin(client, "embeddinggemma:300m", "85462619ee72")


def test_missing_model_raises():
    client = FakeClient([])
    with pytest.raises(RuntimeError, match="missing"):
        check_model_pin(client, "embeddinggemma:300m", "85462619ee72")

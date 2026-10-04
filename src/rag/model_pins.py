"""Check that the Ollama models on the server are the ones this build pinned."""


def _as_mapping(item) -> dict:
    if isinstance(item, dict):
        return item
    return {
        "model": getattr(item, "model", None) or getattr(item, "name", ""),
        "digest": getattr(item, "digest", "") or "",
    }


def _hex_digest(digest: str) -> str:
    text = (digest or "").strip()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    return text


def installed_digests(client) -> dict[str, str]:
    """Map 'name:tag' -> digest for every model on the Ollama server (client.list())."""
    response = client.list()
    models = response["models"] if isinstance(response, dict) else response.models
    mapping = {}
    for item in models:
        fields = _as_mapping(item)
        name = fields.get("model") or fields.get("name") or ""
        if name:
            mapping[name] = fields.get("digest") or ""
    return mapping


def check_model_pin(client, model: str, expected_digest: str) -> None:
    """Stop with a clear message if the installed model is not the pinned one.

    Why: a silently different model changes every vector and every answer.
    Digests from the API are full sha256 hashes; `ollama list` shows the first
    12 hex characters, which is what config.py stores.
    """
    digest = _hex_digest(installed_digests(client).get(model, ""))
    if not digest.startswith(expected_digest):
        shown = digest[:12] or "missing"
        raise RuntimeError(
            f"{model} has digest {shown}, expected {expected_digest}. "
            f"Run `ollama pull {model}` or update config.py on purpose."
        )

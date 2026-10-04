import os
from pathlib import Path

# Exact tag instead of :latest so a new upload cannot silently change vectors.
EMBED_MODEL = "embeddinggemma:300m"
# First 12 hex chars of the manifest digest (`ollama list` ID column).
# embeddinggemma:300m and :latest currently share this digest.
EMBED_MODEL_DIGEST = "85462619ee72"
# embeddinggemma returns 768-dim vectors; a mismatch means a different model.
EMBED_DIM = 768

# Small model used only by the legacy LLM router. Removed when routing moves to code.
ROUTE_MODEL = "gemma3:4b"
# Default answer model; CI overrides with the smaller gemma3:4b for speed.
GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "gemma3:12b")
# Allowed answer models and their digests (ID column of `ollama list`).
GENERATE_MODEL_DIGESTS = {
    "gemma3:12b": "f4031aab637d",
    "gemma3:4b": "a2af6cc3eb7f",
}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CHROMA_PATH = "chroma"
COLLECTION_NAME = "policies"

# RRF constant from Cormack et al., 2009. Large enough that rank 1 vs rank 2
# is not a cliff, so one retriever cannot dominate.
RRF_K = 60
# Shortlist handed to the reranker. Same size as the previous FUSE_N.
FUSED_TOP_K = 20
# Passages the answer model sees: enough for a rule plus its exception.
RERANK_TOP_N = 3


def read_env(path=".env") -> dict[str, str]:
    values = {}
    file = Path(path)
    if not file.is_file():
        return values
    for raw in file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def env_value(name: str, default: str = "", path=".env") -> str:
    return os.environ.get(name) or read_env(path).get(name, default)

import os
from pathlib import Path

EMBED_MODEL = "embeddinggemma:latest"
ROUTE_MODEL = "gemma3:4b"
GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "gemma3:12b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CHROMA_PATH = "chroma"
COLLECTION_NAME = "policies"


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

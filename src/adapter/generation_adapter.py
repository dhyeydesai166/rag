import ollama

from rag.config import GENERATE_MODEL, OLLAMA_HOST
from rag.logutil import log


class GenerationAdapter:
    def __init__(self, client=None, model: str | None = None, host: str | None = None):
        self.model = model or GENERATE_MODEL
        self.client = client or ollama.Client(host=host or OLLAMA_HOST)

    def generate(self, prompt: str, system: str | None = None) -> str:
        kwargs = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            kwargs["system"] = system
        response = self.client.generate(**kwargs)
        text = response["response"] if isinstance(response, dict) else response.response
        log("generate", f"model={self.model} chars={len(text)}")
        return text

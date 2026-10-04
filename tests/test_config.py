from rag import config


def test_pinned_models_and_chroma_paths():
    assert config.EMBED_MODEL == "embeddinggemma:300m"
    assert config.EMBED_MODEL_DIGEST == "85462619ee72"
    assert config.EMBED_DIM == 768
    assert config.ROUTE_MODEL == "gemma3:4b"
    assert config.GENERATE_MODEL == "gemma3:12b"
    assert config.GENERATE_MODEL_DIGESTS["gemma3:4b"] == "a2af6cc3eb7f"
    assert config.GENERATE_MODEL_DIGESTS["gemma3:12b"] == "f4031aab637d"
    assert config.OLLAMA_HOST == "http://127.0.0.1:11434"
    assert config.CHROMA_PATH == "chroma"
    assert config.COLLECTION_NAME == "policies"
    assert config.RRF_K == 60
    assert config.FUSED_TOP_K == 20
    assert config.RERANK_TOP_N == 3

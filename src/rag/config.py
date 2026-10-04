import os
from pathlib import Path

# Exact tag instead of :latest so a new upload cannot silently change vectors.
EMBED_MODEL = "embeddinggemma:300m"
# First 12 hex chars of the manifest digest (`ollama list` ID column).
# embeddinggemma:300m and :latest currently share this digest.
EMBED_MODEL_DIGEST = "85462619ee72"
# embeddinggemma returns 768-dim vectors; a mismatch means a different model.
EMBED_DIM = 768

# Default answer model; CI overrides with the smaller gemma3:4b for speed.
GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "gemma3:12b")
# Allowed answer models and their digests (ID column of `ollama list`).
GENERATE_MODEL_DIGESTS = {
    "gemma3:12b": "f4031aab637d",
    "gemma3:4b": "a2af6cc3eb7f",
}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CHROMA_PATH = "chroma"
# Collections are policies__<build_id>__<utc stamp>: one per rebuild, so a rebuild
# never writes into the collection queries are using. The manifest names the active one.
COLLECTION_PREFIX = "policies"

# The title is the second line in every current document; 5 leaves room for a
# header line without scanning body text.
TITLE_SEARCH_LINES = 5

# One subject per vector: a 256-token chunk is a few short paragraphs, small
# enough that its vector reflects one rule. It is also far below embeddinggemma's
# 2048-token context, so the heading prefix never pushes a chunk into truncation.
CHUNK_MAX_TOKENS = 256
# Repeat one sentence between neighbouring pieces of the SAME section so a rule
# that straddles the cut is readable in both. Never overlap across headings.
CHUNK_OVERLAP_SENTENCES = 1
# Bump when chunking rules change; part of build_id, so a change forces a rebuild.
CHUNKER_VERSION = 2
# Abbreviations whose period does not end a sentence (seen in the corpus: "a.m.").
SENTENCE_ABBREVIATIONS = ("a.m.", "p.m.", "e.g.", "i.e.", "etc.", "vs.", "No.")

# Keeps one embedding request small enough to retry and to name in an error.
# The corpus is about 100 chunks today.
EMBED_BATCH_SIZE = 64

# Long enough for any real policy question; longer inputs are usually pasted documents.
MAX_QUESTION_CHARS = 500
# Single-word inputs that are greetings or tests, not questions.
FILLER_WORDS = frozenset(
    {"hi", "hello", "hey", "ok", "okay", "test", "thanks", "yo", "help"}
)
# 'asdfghjkl' has 8 consonants in a row; real English words rarely exceed 5.
KEYMASH_CONSONANT_RUN = 6

# Ways people name each policy. Bare "health" or "time" also appear inside
# rules, so they are not aliases.
POLICY_ALIASES = {
    "HR Policy": ("hr policy", "hr", "human resources"),
    "Health & Wellness Policy": (
        "health & wellness policy",
        "health and wellness policy",
        "health & wellness",
        "health and wellness",
        "health policy",
        "wellness policy",
    ),
    "Preparedness Policy": ("preparedness policy", "preparedness"),
    "Time & Usage Policy": (
        "time & usage policy",
        "time and usage policy",
        "time & usage",
        "time and usage",
        "usage policy",
    ),
}
# A version is digits with at least one dot, after clean_question.
VERSION_PATTERN = r"\b\d+\.\d+(?:\.\d+)*\b"

# Compare must beat lookup by this cosine margin. Similarities are only
# compared within one question. `python -m rag.route report` routes every eval
# question correctly with 0.02.
ROUTE_COMPARE_MARGIN = 0.02
# A question whose best cosine similarity to every example question is below this
# is not clearly a lookup or a compare, so it takes the safe route (lookup).
# Measured with embeddinggemma:300m: all 21 eval questions score >= 0.564;
# off-topic inputs that routed to compare scored 0.416-0.483. 0.50 sits between.
# Re-check with `python -m rag.route report` after changing examples or the model.
ROUTE_MIN_SIMILARITY = 0.50

# Standard Okapi BM25 defaults; the corpus is too small to tune them.
BM25_K1 = 1.5
BM25_B = 0.75
# Each retriever contributes up to 20 candidates. A policy version has about
# 15-26 chunks today, so this covers everything now and still bounds work later.
DENSE_TOP_K = 20
LEXICAL_TOP_K = 20

# RRF constant from Cormack et al., 2009. Large enough that rank 1 vs rank 2
# is not a cliff, so one retriever cannot dominate.
RRF_K = 60
# Shortlist handed to the reranker: enough to recover from a weak first stage,
# small enough for one Cohere call.
FUSED_TOP_K = 20

# Cohere model; recorded in provenance.
RERANK_MODEL = "rerank-v3.5"
# Passages the answer model sees: enough for a rule plus its exception.
RERANK_TOP_N = 3
# Bump the suffix (lookup_v2) instead of editing a prompt in place, so every
# eval result names the exact prompt it used.
LOOKUP_PROMPT = "lookup_v2"
COMPARE_PROMPT = "compare_v2"
# Temperature 0 and a fixed seed: the same question and passages give the same
# answer, so eval runs are comparable and regressions are visible.
GENERATION_TEMPERATURE = 0
GENERATION_SEED = 42
# Temperature 0 should make runs identical; three runs reveal nondeterminism
# from hardware or server versions at modest cost.
ANSWER_EVAL_RUNS = 3

# A rerank call normally returns in well under a second; 10s means something is wrong.
RERANK_TIMEOUT_SECONDS = 10
# One retry covers a transient blip without making a real outage slow.
RERANK_RETRIES = 1
RERANK_RETRY_DELAY_SECONDS = 1.0
# Cohere trial keys allow 10 rerank calls a minute. An eval makes one call per
# question, so 60 / 10 = 6 seconds between questions stays under the limit.
# Applied only when a Cohere key is set. Set 0 for a production key.
EVAL_PAUSE_SECONDS = float(os.environ.get("EVAL_PAUSE_SECONDS", "6"))


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

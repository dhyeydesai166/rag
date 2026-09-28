import math
import re
import sys
import time

from adpater.database_adapter import DatabaseAdapter
from adpater.embedding_adapter import EmbeddingAdapter
from adpater.generation_adapter import GenerationAdapter
from adpater.rerank_adapter import RerankerAdapter
from rag.config import ROUTE_MODEL
from rag.generate import generate
from rag.logutil import (
    disable_question_log,
    enable_question_log,
    log,
    silence_console,
    stage,
)
from rag.router import route

RRF = 60
FUSE_N = 20
TOP_N = 3
ALIASES = {
    "Time and Usage Policy": "Time & Usage Policy",
    "Health Policy": "Health & Wellness Policy",
}


def canonicalize(name: str) -> str:
    return ALIASES.get(name, name)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def catalog(rows: list[dict]) -> dict:
    policies = {}
    for row in rows:
        policies.setdefault(row["policy"], set()).add(row["version"])
    return {
        name: tuple(sorted(versions, key=version_key))
        for name, versions in policies.items()
    }


def version_key(version: str) -> tuple:
    return tuple(int(part) for part in version.split("."))


def fold(rows: list[dict]) -> list[dict]:
    seen = set()
    folded = []
    for row in rows:
        row = {**row, "policy": canonicalize(row["policy"])}
        key = (row["policy"], row["version"], row["heading_path"])
        if key in seen:
            continue
        seen.add(key)
        folded.append(row)
    return folded


def lookup_pairs(policies: dict, policy: str, version: str) -> list[tuple]:
    if policy and version:
        return [(policy, version)]
    if policy:
        return [(policy, policies[policy][-1])]
    return [(name, versions[-1]) for name, versions in policies.items()]


def candidates(rows: list[dict], pairs: list[tuple]) -> list[dict]:
    allowed = set(pairs)
    return [row for row in rows if (row["policy"], row["version"]) in allowed]


def cosine(left: list[float], right: list[float]) -> float:
    # TODO: switch this from simple cosine to hnsw as corpus grows, foundations baked into code but not implemented
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def cosine_scores(query: list[float], rows: list[dict]) -> list[float]:
    return [cosine(query, row["vector"]) for row in rows]


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75):
    query_terms = tokens(query)
    docs = [tokens(document) for document in documents]
    lengths = [len(doc) for doc in docs]
    average = sum(lengths) / len(lengths) if lengths else 0
    counts = []
    document_frequency = {term: 0 for term in set(query_terms)}
    for doc in docs:
        count = {}
        for term in doc:
            count[term] = count.get(term, 0) + 1
        counts.append(count)
        for term in document_frequency:
            if term in count:
                document_frequency[term] += 1
    total = len(docs)
    scores = []
    for length, count in zip(lengths, counts, strict=True):
        if average == 0:
            scores.append(0.0)
            continue
        score = 0.0
        for term in query_terms:
            frequency = count.get(term, 0)
            if frequency == 0:
                continue
            found = document_frequency[term]
            idf = math.log(1 + (total - found + 0.5) / (found + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / average)
            score += idf * (frequency * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def ranks(rows: list[dict], scores: list[float]) -> list[int]:
    order = sorted(
        range(len(rows)),
        key=lambda index: (-scores[index], rows[index]["id"]),
    )
    places = [0] * len(rows)
    for place, index in enumerate(order, start=1):
        places[index] = place
    return places


def fuse(rows, semantic, keyword, fuse_n=FUSE_N):
    semantic_rank = ranks(rows, semantic)
    keyword_rank = ranks(rows, keyword)
    hits = []
    for index, row in enumerate(rows):
        score = 1 / (RRF + semantic_rank[index]) + 1 / (RRF + keyword_rank[index])
        hits.append({**row, "score": score})
    hits.sort(key=lambda hit: (-hit["score"], hit["id"]))
    return hits[:fuse_n]


def hybrid(question, query, rows):
    if not rows:
        return []
    return fuse(
        rows,
        cosine_scores(query, rows),
        bm25_scores(question, [row["text"] for row in rows]),
    )


def side(row: dict) -> dict:
    return {
        "id": row["id"],
        "text": row["text"],
        "version": row["version"],
        "section": row["section"],
        "parent_id": row["parent_id"],
        "source": row["source"],
    }


def pair_hits(latest_hits, previous_hits):
    previous_by_path = {hit["heading_path"]: hit for hit in previous_hits}
    pairs = []
    seen = set()
    for hit in latest_hits:
        path = hit["heading_path"]
        seen.add(path)
        previous = previous_by_path.get(path)
        pairs.append(
            {
                "policy": hit["policy"],
                "heading_path": path,
                "score": hit["score"],
                "current": side(hit),
                "previous": None if previous is None else side(previous),
            }
        )
    for hit in previous_hits:
        if hit["heading_path"] in seen:
            continue
        pairs.append(
            {
                "policy": hit["policy"],
                "heading_path": hit["heading_path"],
                "score": hit["score"],
                "current": None,
                "previous": side(hit),
            }
        )
    pairs.sort(key=lambda pair: (-pair["score"], pair["heading_path"]))
    return pairs[:FUSE_N]


def side_text(label: str, item) -> str:
    if item is None:
        return label
    return f"{label} {item['version']}\n{item['text']}"


def pair_text(pair: dict) -> str:
    current = side_text("current", pair["current"])
    previous = side_text("previous", pair["previous"])
    return f"{pair['heading_path']}\n{current}\n{previous}"


def apply_rerank(question, items, texts, reranker, n):
    if not items:
        return []
    by_text = {}
    for item, text in zip(items, texts, strict=True):
        by_text.setdefault(text, []).append(item)
    ordered = []
    for text in reranker.rerank(question, texts):
        bucket = by_text.get(text)
        if not bucket:
            continue
        ordered.append(bucket.pop(0))
    for bucket in by_text.values():
        ordered.extend(bucket)
    return ordered[:n]


def top_ids(hits) -> str:
    ids = []
    for hit in hits:
        if "current" in hit:
            ids.append(hit["heading_path"])
        else:
            ids.append(hit["id"])
    return ",".join(ids)


def retrieve(question, embedder, model, database, reranker, n=TOP_N) -> dict:
    with stage("database"):
        rows = fold(database.rows())
    policies = catalog(rows)
    with stage("router"):
        decision = route(question, model, policies)
    with stage("embedder"):
        vector = embedder.embed([question], task="query")[0]
    kind = decision["kind"]
    policy = decision["policy"]
    if kind == "compare" and policy not in policies:
        log("router", "kind=lookup reason=unknown policy")
        kind = "lookup"
        policy = ""
        decision = {"kind": "lookup", "policy": "", "version": ""}
    if kind == "compare":
        hits = compare(question, vector, rows, policies, policy, reranker, n)
    else:
        hits = lookup(question, vector, rows, policies, decision, reranker, n)
    log("retrieve", f"hits={len(hits)} top={top_ids(hits)}")
    return {"kind": kind, "hits": hits}


def lookup(question, vector, rows, policies, decision, reranker, n):
    chosen = candidates(
        rows, lookup_pairs(policies, decision["policy"], decision["version"])
    )
    log("retrieve", f"candidates={len(chosen)}")
    with stage("hybrid"):
        fused = hybrid(question, vector, chosen)
    with stage("rerank"):
        return apply_rerank(
            question, fused, [row["text"] for row in fused], reranker, n
        )


def compare(question, vector, rows, policies, policy, reranker, n):
    versions = policies[policy]
    latest = versions[-1]
    previous = versions[-2] if len(versions) > 1 else None
    # TODO: This always compares only to one version before, even on explicit request it diffs only to previous version, fix it
    current_rows = candidates(rows, [(policy, latest)])
    previous_rows = [] if previous is None else candidates(rows, [(policy, previous)])
    log("retrieve", f"candidates={len(current_rows) + len(previous_rows)}")
    with stage("hybrid"):
        pairs = pair_hits(
            hybrid(question, vector, current_rows),
            hybrid(question, vector, previous_rows),
        )
    with stage("rerank"):
        return apply_rerank(
            question, pairs, [pair_text(pair) for pair in pairs], reranker, n
        )


def main(argv=None, trace: bool = False) -> int:
    started = time.perf_counter()
    argv = list(sys.argv[1:] if argv is None else argv)
    question = argv[0] if argv else ""
    db_path = argv[1] if len(argv) > 1 else "chroma"
    if trace:
        enable_question_log()
    else:
        silence_console()
    try:
        router = GenerationAdapter(model=ROUTE_MODEL)
        answerer = GenerationAdapter()
        found = retrieve(
            question,
            embedder=EmbeddingAdapter(),
            model=router,
            database=DatabaseAdapter(db_path),
            reranker=RerankerAdapter(),
        )
        text = generate(question, found["kind"], found["hits"], answerer)
        print(text)
        if trace:
            elapsed = time.perf_counter() - started
            print(f"latency: {elapsed:.3f}s")
    finally:
        if trace:
            disable_question_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover

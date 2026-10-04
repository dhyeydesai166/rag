"""Clean, filter, route, search, fuse, and rerank a policy question.

Routing and filters are code. The answer model is the only model that
writes text, and it is called from main after this module returns.
"""

import sys
import time

import ollama

from adapter.database_adapter import DatabaseAdapter, where_for
from adapter.embedding_adapter import EmbeddingAdapter
from adapter.generation_adapter import GenerationAdapter
from adapter.rerank_adapter import RerankUnavailable, make_reranker
from rag.compare import compare_versions, pair_by_title, pair_text
from rag.config import (
    DENSE_TOP_K,
    EMBED_MODEL,
    EMBED_MODEL_DIGEST,
    FUSED_TOP_K,
    GENERATE_MODEL,
    GENERATE_MODEL_DIGESTS,
    OLLAMA_HOST,
    RERANK_TOP_N,
)
from rag.filters import catalog_from_chunks, extract_filters, lookup_targets
from rag.fusion import rrf_fuse
from rag.generate import generate
from rag.lexical import bm25_ranked
from rag.logutil import (
    disable_question_log,
    enable_question_log,
    log,
    silence_console,
    stage,
    warn,
)
from rag.manifest import active_build_id
from rag.model_pins import check_model_pin
from rag.question import JunkQuestion, clean_question, junk_reason
from rag.route import classify_route, embed_examples, route_scores
from rag.section_map import SECTION_RENAMES


def rerank_with_fallback(question, items, texts, reranker, keep: int = RERANK_TOP_N):
    """Reorder the fused shortlist with Cohere; on any failure keep the fused order.

    Returns (items, used_reranker). Why the flag: the retrieval eval treats
    the final top 3 as a gate only when Cohere actually ranked them.

    Why the fallback: the reranker improves precision but is an external
    service; an outage or a missing key must cost quality, not availability.
    """
    if not items:
        return [], False
    if reranker is None:
        warn("rerank", "no COHERE_API_KEY; using fused order")
        return items[:keep], False
    try:
        order = reranker.rank(question, texts, top_n=keep)
    except RerankUnavailable as error:
        warn("rerank", f"Cohere unavailable ({error}); using fused order")
        return items[:keep], False
    kept = [items[index] for index in order if 0 <= index < len(items)][:keep]
    return kept, True


def _search(search_text, vector, chunks, where, database):
    with stage("dense"):
        dense_ids = database.dense_search(vector, where, DENSE_TOP_K)
    with stage("lexical"):
        lexical_ids = bm25_ranked(search_text, chunks)
    with stage("fuse"):
        return rrf_fuse(dense_ids, lexical_ids)


def _policy_from_dense(vector, database, catalog) -> str | None:
    pairs = [
        (name, version)
        for name, versions in catalog.items()
        if len(versions) >= 2
        for version in versions
    ]
    if not pairs:
        return None
    ids = database.dense_search(vector, where_for(pairs), 1)
    if not ids:
        return None
    chunks = database.chunks_where(where_for(pairs))
    by_id = {chunk["id"]: chunk for chunk in chunks}
    chunk = by_id.get(ids[0])
    if chunk is None:
        return None
    return chunk["policy"]


def _lookup(filters, catalog, search_vector, database, question, reranker):
    targets = lookup_targets(filters, catalog)
    where = where_for(targets)
    chunks = database.chunks_where(where)
    fused = _search(filters.search_text, search_vector, chunks, where, database)
    by_id = {chunk["id"]: chunk for chunk in chunks}
    items = [by_id[hit.chunk_id] for hit in fused if hit.chunk_id in by_id]
    texts = [item.get("embed_text") or item["text"] for item in items]
    hits, used = rerank_with_fallback(question, items, texts, reranker)
    return hits, [item["id"] for item in items], used


def _compare(filters, policy, catalog, search_vector, database, question, reranker):
    versions = catalog[policy]
    old, new = compare_versions(filters, versions)
    old_chunks = database.chunks_where(where_for([(policy, old)]))
    new_chunks = database.chunks_where(where_for([(policy, new)]))
    renames = SECTION_RENAMES.get((policy, old, new), {})
    pairs = pair_by_title(old_chunks, new_chunks, renames)
    old_hits = {
        hit.chunk_id: hit
        for hit in _search(
            filters.search_text,
            search_vector,
            old_chunks,
            where_for([(policy, old)]),
            database,
        )
    }
    new_hits = {
        hit.chunk_id: hit
        for hit in _search(
            filters.search_text,
            search_vector,
            new_chunks,
            where_for([(policy, new)]),
            database,
        )
    }
    scored = []
    for pair in pairs:
        ranks = []
        scores = []
        for side_name, table in (("previous", old_hits), ("current", new_hits)):
            side = pair[side_name]
            if side and side["id"] in table:
                hit = table[side["id"]]
                scores.append(hit.score)
                ranks.append(hit.best_rank)
        pair["score"] = max(scores) if scores else 0.0
        best_rank = min(ranks) if ranks else len(old_chunks) + len(new_chunks) + 1
        scored.append((pair, best_rank))
    scored.sort(key=lambda item: (-item[0]["score"], item[1], item[0]["title"]))
    shortlist = [pair for pair, _rank in scored[:FUSED_TOP_K]]
    texts = [pair_text(pair) for pair in shortlist]
    hits, used = rerank_with_fallback(question, shortlist, texts, reranker)
    fused_ids = []
    for pair in shortlist:
        for side_name in ("previous", "current"):
            side = pair.get(side_name)
            if side:
                fused_ids.append(side["id"])
    return hits, fused_ids, used


def retrieve(question: str, embedder, database, reranker, examples=None) -> dict:
    """Clean, filter, route, search, fuse, rerank.

    Returns {'kind', 'route', 'hits', 'filters'}; raises JunkQuestion for junk input.
    """
    with stage("clean"):
        cleaned = clean_question(question)
        reason = junk_reason(cleaned)
        if reason:
            raise JunkQuestion(reason)
    chunks = database.chunks_where(None)
    catalog = catalog_from_chunks(chunks)
    with stage("filters"):
        filters = extract_filters(cleaned, catalog)
    if examples is None:
        examples = embed_examples(embedder)
    lookup_vectors, compare_vectors = examples
    with stage("route"):
        question_vector = embedder.embed([cleaned], task="similarity")[0]
        best_lookup, best_compare = route_scores(
            question_vector, lookup_vectors, compare_vectors
        )
        kind = classify_route(question_vector, lookup_vectors, compare_vectors)
        log(
            "route",
            f"kind={kind} best_lookup={best_lookup:.3f} "
            f"best_compare={best_compare:.3f}",
        )
    search_vector = embedder.embed([filters.search_text or cleaned], task="query")[0]
    policy = filters.policy
    if kind == "compare":
        if policy and len(catalog.get(policy, ())) < 2:
            kind = "lookup"
            log("route", "kind=lookup reason=single version")
        elif not policy:
            policy = _policy_from_dense(search_vector, database, catalog)
            if not policy:
                kind = "lookup"
                log("route", "kind=lookup reason=no multi-version policy")
            else:
                log("route", f"kind=compare policy={policy} reason=top dense hit")
    if kind == "compare" and policy:
        hits, fused_ids, reranked = _compare(
            filters, policy, catalog, search_vector, database, cleaned, reranker
        )
        route = {
            "kind": "compare",
            "policy": policy,
            "versions": compare_versions(filters, catalog[policy]),
        }
    else:
        hits, fused_ids, reranked = _lookup(
            filters, catalog, search_vector, database, cleaned, reranker
        )
        route = {"kind": "lookup", "targets": lookup_targets(filters, catalog)}
        kind = "lookup"
    log(
        "retrieve",
        f"policy={filters.policy} versions={','.join(filters.versions) or '-'} "
        f"kind={kind} search={filters.search_text!r} hits={len(hits)}",
    )
    return {
        "kind": kind,
        "route": route,
        "hits": hits,
        "filters": filters,
        "fused_ids": fused_ids,
        "reranked": reranked,
    }


def _check_models() -> None:
    client = ollama.Client(host=OLLAMA_HOST)
    check_model_pin(client, EMBED_MODEL, EMBED_MODEL_DIGEST)
    answer_digest = GENERATE_MODEL_DIGESTS.get(GENERATE_MODEL)
    if not answer_digest:
        raise RuntimeError(
            f"{GENERATE_MODEL} has no pinned digest. "
            "Add it to GENERATE_MODEL_DIGESTS in config.py."
        )
    check_model_pin(client, GENERATE_MODEL, answer_digest)


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
        cleaned = clean_question(question)
        reason = junk_reason(cleaned)
        if reason:
            print(reason)
            return 2
        _check_models()
        embedder = EmbeddingAdapter()
        database = DatabaseAdapter(db_path, active_build_id(db_path))
        found = retrieve(
            question,
            embedder=embedder,
            database=database,
            reranker=make_reranker(),
            examples=embed_examples(embedder),
        )
        generated = generate(question, found, GenerationAdapter())
        print(generated.text)
        if trace:
            elapsed = time.perf_counter() - started
            print(f"latency: {elapsed:.3f}s")
    except JunkQuestion as error:
        print(str(error))
        return 2
    finally:
        if trace:
            disable_question_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

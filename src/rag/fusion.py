"""Reciprocal Rank Fusion of the dense list and the lexical list."""

from dataclasses import dataclass

from rag.config import FUSED_TOP_K, RRF_K


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    score: float
    best_rank: int
    dense_rank: int | None


def fusion_order(hit: FusedHit, not_in_dense: int) -> tuple:
    """Higher score, then better single rank, then better dense rank, then id."""
    dense = hit.dense_rank if hit.dense_rank is not None else not_in_dense
    return (-hit.score, hit.best_rank, dense, hit.chunk_id)


def rrf_fuse(
    dense_ids: list[str],
    lexical_ids: list[str],
    k: int = RRF_K,
    top_k: int = FUSED_TOP_K,
) -> list[FusedHit]:
    """Sum 1/(k + rank) over the lists a chunk appears in.

    Ties: the chunk with the better single rank wins (strong evidence in one
    list beats middling evidence in both), then the better dense rank (dense
    is the more general signal), then chunk id only to make the order total.
    """
    dense_rank = {cid: rank for rank, cid in enumerate(dense_ids, start=1)}
    lexical_rank = {cid: rank for rank, cid in enumerate(lexical_ids, start=1)}
    hits = []
    for cid in dense_rank.keys() | lexical_rank.keys():
        ranks = [
            rank
            for rank in (dense_rank.get(cid), lexical_rank.get(cid))
            if rank is not None
        ]
        hits.append(
            FusedHit(
                cid,
                sum(1 / (k + rank) for rank in ranks),
                min(ranks),
                dense_rank.get(cid),
            )
        )
    not_in_dense = len(dense_ids) + 1
    hits.sort(key=lambda hit: fusion_order(hit, not_in_dense))
    return hits[:top_k]

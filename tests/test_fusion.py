from rag.fusion import FusedHit, fusion_order, rrf_fuse
from rag.lexical import bm25_ranked


def test_rrf_score_is_the_sum_of_reciprocal_ranks():
    fused = rrf_fuse(["a", "b"], ["b", "a"])
    scores = {hit.chunk_id: hit.score for hit in fused}
    assert scores["a"] == (1 / 61) + (1 / 62)
    assert scores["b"] == (1 / 62) + (1 / 61)


def test_chunk_in_only_one_list_still_scores():
    fused = rrf_fuse(["only-dense"], [])
    assert fused[0].chunk_id == "only-dense"
    assert fused[0].score == 1 / 61


def test_dense_rank_breaks_a_tie_between_one_list_hits():
    fused = rrf_fuse(["z-dense"], ["a-lex"])
    assert [hit.chunk_id for hit in fused] == ["z-dense", "a-lex"]


def test_better_single_rank_wins_a_score_tie():
    stronger = FusedHit("z-id", 0.5, best_rank=1, dense_rank=2)
    weaker = FusedHit("a-id", 0.5, best_rank=2, dense_rank=1)
    assert fusion_order(stronger, 9) < fusion_order(weaker, 9)


def test_tie_break_does_not_depend_on_alphabetical_id():
    fused = rrf_fuse(["z-dense", "a-other"], ["a-lex"])
    assert fused[0].chunk_id == "z-dense"


def test_keyword_match_lifts_a_chunk_above_a_slightly_closer_distractor():
    chunks = [
        {"id": "a-vacation", "embed_text": "vacation days"},
        {"id": "b-hours", "embed_text": "office hours"},
        {"id": "z-cake", "embed_text": "birthday cake"},
    ]
    dense_ids = ["a-vacation", "z-cake", "b-hours"]
    lexical_ids = bm25_ranked("cake", chunks)
    assert lexical_ids == ["z-cake"]
    fused = rrf_fuse(dense_ids, lexical_ids)
    scores = {hit.chunk_id: hit.score for hit in fused}
    assert scores["z-cake"] > scores["a-vacation"]
    assert fused[0].chunk_id == "z-cake"

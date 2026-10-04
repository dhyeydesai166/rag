from evals.checks import (
    check_answer,
    has_change_language,
    numbers_in,
    retrieval_hit,
)
from evals.run_answers import totals
from evals.run_retrieval import passes_gate
from rag.models import Answer, Claim


def _case(**overrides):
    base = {
        "expect_status": "answered",
        "gold_chunks": ["gold"],
        "required_facts": [["500,000"]],
        "stale_facts": ["1,000,000"],
    }
    base.update(overrides)
    return base


def _answer(text, chunk_id="gold", status="answered"):
    return Answer(status=status, claims=[Claim(text=text, chunk_id=chunk_id)])


def test_numbers_drop_thousands_separators():
    assert numbers_in("Issued 1,000,000 tokens, then 500,000.") == {"1000000", "500000"}


def test_retrieval_hit_requires_every_gold_id():
    assert retrieval_hit(["a", "b"], {"a", "b", "c"})
    assert not retrieval_hit(["a"], set())
    assert retrieval_hit([], set())


def test_unknown_chunk_is_an_invention():
    result = check_answer(
        _answer("Employees get 500,000 tokens.", chunk_id="missing"),
        {"gold": {"text": "Employees get 500,000 tokens."}},
        "lookup",
        _case(),
    )
    assert any("not retrieved" in item for item in result.inventions)


def test_number_not_in_the_cited_chunk_is_an_invention():
    result = check_answer(
        _answer("Employees get 500,000 tokens."),
        {"gold": {"text": "Employees get some tokens."}},
        "lookup",
        _case(required_facts=[["tokens"]], stale_facts=[]),
    )
    assert any("number 500000" in item for item in result.inventions)


def test_stale_fact_is_allowed_when_the_cited_chunk_says_it():
    chunk = "A reduction from the 1,000,000 issued under Version 1.0 to 500,000."
    result = check_answer(
        _answer("Employees now receive 500,000, a reduction from 1,000,000."),
        {"gold": {"text": chunk}},
        "lookup",
        _case(),
    )
    assert not any(item.startswith("stale fact") for item in result.inventions)


def test_stale_fact_from_another_version_is_an_invention():
    result = check_answer(
        _answer("Employees receive 1,000,000 tokens."),
        {"gold": {"text": "Employees receive 500,000 tokens."}},
        "lookup",
        _case(required_facts=[["1,000,000"]], stale_facts=["1,000,000"]),
    )
    assert any("stale fact" in item for item in result.inventions)


def test_lookup_change_language_must_be_in_the_source():
    assert has_change_language("a reduction from the old amount")
    result = check_answer(
        _answer("The allowance was reduced."),
        {"gold": {"text": "Employees receive 500,000 tokens."}},
        "lookup",
        _case(required_facts=[["reduced"]], stale_facts=[]),
    )
    assert any("describes a change" in item for item in result.inventions)
    allowed = check_answer(
        _answer("The allowance was reduced."),
        {"gold": {"text": "This is a reduction from the previous amount."}},
        "lookup",
        _case(required_facts=[["reduced"]], stale_facts=[]),
    )
    assert not any("describes a change" in item for item in allowed.inventions)


def test_version_label_is_not_an_invented_number():
    passage = {
        "text": "Video game time remains unchanged at up to 45 minutes per workday, "
        "taken in increments of no fewer than 15 minutes.",
        "version": "2.0",
    }
    result = check_answer(
        _answer("in version 2.0 the limit is 45 minutes"),
        {"gold": passage},
        "lookup",
        _case(required_facts=[], stale_facts=[]),
    )
    assert result.inventions == []


def test_numbers_in_skips_known_versions():
    assert numbers_in("in version 2.0 the limit is 45", frozenset({"2.0"})) == {"45"}


def test_other_numbers_are_still_checked_next_to_a_version():
    passage = {
        "text": "Video game time remains unchanged at up to 45 minutes per workday.",
        "version": "2.0",
    }
    result = check_answer(
        _answer("in version 2.0 the limit is 60 minutes"),
        {"gold": passage},
        "lookup",
        _case(required_facts=[], stale_facts=[]),
    )
    assert result.inventions == ["number 60 is not in gold"]


def test_an_unknown_version_is_still_an_invented_number():
    passage = {
        "text": "Video game time remains unchanged at up to 45 minutes per workday.",
        "version": "2.0",
    }
    result = check_answer(
        _answer("in version 3.0 the limit is 45 minutes"),
        {"gold": passage},
        "lookup",
        _case(required_facts=[], stale_facts=[]),
    )
    assert any("number 3.0" in item for item in result.inventions)


def test_naming_another_version_in_a_lookup_is_still_change_language():
    passage = {
        "text": "Video game time remains unchanged at up to 45 minutes per workday.",
        "version": "2.0",
    }
    result = check_answer(
        _answer("Unlike version 1.0, the limit is 45 minutes."),
        {"gold": passage},
        "lookup",
        _case(required_facts=[], stale_facts=[]),
    )
    assert any("describes a change" in item for item in result.inventions)


def test_compare_allows_change_language():
    result = check_answer(
        _answer("The allowance was reduced to 500,000."),
        {"gold": {"text": "Employees receive 500,000 tokens."}},
        "compare",
        _case(stale_facts=[]),
    )
    assert not any("describes a change" in item for item in result.inventions)


def test_missing_gold_citation_and_required_fact_are_misses():
    result = check_answer(
        _answer("Employees receive tokens.", chunk_id="other"),
        {
            "other": {"text": "Employees receive tokens."},
            "gold": {"text": "Employees receive 500,000 tokens."},
        },
        "lookup",
        _case(),
    )
    assert "no gold chunk cited" in result.misses
    assert any(item.startswith("missing fact") for item in result.misses)


def test_status_mismatch_is_an_invention_when_the_model_answers():
    result = check_answer(
        _answer("There is a limit."),
        {"gold": {"text": "There is a limit."}},
        "lookup",
        _case(
            expect_status="not_in_sources",
            gold_chunks=[],
            required_facts=[],
            stale_facts=[],
        ),
    )
    assert any(item.startswith("status answered") for item in result.inventions)


def test_status_mismatch_is_a_miss_when_the_model_declines():
    result = check_answer(
        Answer(status="not_in_sources", claims=[]),
        {},
        "lookup",
        _case(),
    )
    assert any(item.startswith("status not_in_sources") for item in result.misses)


def test_retrieval_gate_ignores_top3_when_rerank_fell_back():
    row = {
        "route_ok": True,
        "gold_chunks": ["a"],
        "gold_in_fused": True,
        "gold_in_top": False,
        "reranked": False,
    }
    assert passes_gate(row)
    assert not passes_gate({**row, "reranked": True})
    assert not passes_gate({**row, "gold_in_fused": False})
    assert not passes_gate({**row, "route_ok": False})
    assert passes_gate({**row, "gold_chunks": []})


def test_totals_count_misses_and_inventions_separately():
    summary = totals(
        [
            {"misses": ["m"], "inventions": [], "route_ok": True},
            {"misses": [], "inventions": ["i"], "route_ok": False},
            {"misses": [], "inventions": [], "route_ok": True},
        ]
    )
    assert summary == {
        "misses": 1,
        "inventions": 1,
        "clean_cases": 1,
        "route_ok": 2,
    }

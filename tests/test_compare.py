import pytest

from rag.compare import compare_versions, normalize_title, pair_by_title
from rag.filters import Filters
from rag.section_map import apply_renames


def _chunk(policy, version, heading, text="body"):
    return {
        "id": f"{policy}|{version}|{heading}",
        "text": text,
        "policy": policy,
        "version": version,
        "section": heading.split(" > ")[0],
        "heading_path": heading,
        "source": "policy.docx",
        "ordinal": None,
    }


@pytest.mark.parametrize(
    ("heading", "title"),
    [
        (
            "4. Nuclear Apocalypse Protocol — Updated > 4.2 Hazmat Suit Eligibility",
            "nuclear apocalypse protocol > hazmat suit eligibility",
        ),
        (
            "7. AI Apocalypse Protocol — New in Version 2.0 > 7.1 Immediate Actions",
            "ai apocalypse protocol > immediate actions",
        ),
        (
            "8. Token Depletion — Consequences",
            "token depletion — consequences",
        ),
    ],
)
def test_normalize_title_strips_numbers_and_suffixes(heading, title):
    assert normalize_title(heading) == title


def test_renumbered_sections_pair_by_title():
    old = [
        _chunk(
            "Time & Usage Policy",
            "1.0",
            "5. Token Allocation > 5.1 Allocation Amount",
        )
    ]
    new = [
        _chunk(
            "Time & Usage Policy",
            "2.0",
            "6. Token Allocation > 6.1 Allocation Amount",
        )
    ]
    pairs = pair_by_title(old, new, {})
    assert len(pairs) == 1
    assert pairs[0]["previous"]["version"] == "1.0"
    assert pairs[0]["current"]["version"] == "2.0"


def test_renamed_sections_pair_through_the_section_map():
    from rag.section_map import SECTION_RENAMES

    renames = SECTION_RENAMES[("Preparedness Policy", "1.0", "2.0")]
    old = [
        _chunk(
            "Preparedness Policy",
            "1.0",
            "4. Nuclear Apocalypse Protocol > 4.2 All-Clear Timing",
        )
    ]
    new = [
        _chunk(
            "Preparedness Policy",
            "2.0",
            "4. Nuclear Apocalypse Protocol — Updated > 4.3 Duration of Sheltering",
        )
    ]
    pairs = pair_by_title(old, new, renames)
    assert pairs[0]["previous"]["id"].endswith("All-Clear Timing")
    assert pairs[0]["current"]["id"].endswith("Duration of Sheltering")


def test_parent_rename_also_renames_children():
    renames = {"foosball time": "foosball time and the winner-takes-tokens rule"}
    title = apply_renames("foosball time > daily allowance", renames)
    assert title == "foosball time and the winner-takes-tokens rule > daily allowance"


def test_removed_section_has_no_current_side():
    old = [
        _chunk(
            "Time & Usage Policy",
            "1.0",
            "4. Foosball Time > 4.2 Dispute Resolution",
            "winner keeps the table",
        )
    ]
    pairs = pair_by_title(old, [], {})
    assert pairs[0]["current"] is None
    assert pairs[0]["previous"]["text"] == "winner keeps the table"


def test_added_section_has_no_previous_side():
    new = [_chunk("HR Policy", "2.0", "8. Added", "new clause")]
    pairs = pair_by_title([], new, {})
    assert pairs[0]["previous"] is None
    assert pairs[0]["current"]["text"] == "new clause"


def test_two_old_sections_on_one_new_section_is_rejected():
    old = [
        _chunk("HR Policy", "1.0", "1. Alpha"),
        _chunk("HR Policy", "1.0", "2. Beta"),
    ]
    with pytest.raises(ValueError, match="both map"):
        pair_by_title(old, [], {"alpha": "same", "beta": "same"})


def test_compare_versions_selection():
    versions = ("1.0", "2.0")
    assert compare_versions(Filters(None, (), "q"), versions) == ("1.0", "2.0")
    assert compare_versions(Filters(None, ("2.0",), "q"), versions) == ("1.0", "2.0")
    assert compare_versions(Filters(None, ("1.0",), "q"), versions) == ("1.0", "2.0")
    assert compare_versions(Filters(None, ("1.0", "2.0"), "q"), versions) == (
        "1.0",
        "2.0",
    )

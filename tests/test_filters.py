import json
from pathlib import Path

from rag.config import POLICY_ALIASES
from rag.filters import extract_filters, lookup_targets

CATALOG = {
    "HR Policy": ("1.0", "2.0"),
    "Health & Wellness Policy": ("1.0",),
    "Preparedness Policy": ("1.0", "2.0"),
    "Time & Usage Policy": ("1.0", "2.0"),
}


def test_policy_alias_is_found_and_removed_from_search_text():
    filters = extract_filters(
        "Under the HR Policy, how long must an employee wait?",
        CATALOG,
    )
    assert filters.policy == "HR Policy"
    assert "HR Policy" not in filters.search_text
    assert "how long" in filters.search_text


def test_time_and_usage_spellings_map_to_one_policy():
    first = extract_filters("What does the Time and Usage Policy say?", CATALOG)
    second = extract_filters("What does the Time & Usage Policy say?", CATALOG)
    assert first.policy == second.policy == "Time & Usage Policy"


def test_health_inside_rule_text_is_not_a_policy():
    filters = extract_filters(
        "what are the health consequences of the spoonful rule?", CATALOG
    )
    assert filters.policy is None


def test_two_different_policies_mean_no_policy_filter():
    filters = extract_filters("Compare HR Policy and Preparedness Policy", CATALOG)
    assert filters.policy is None


def test_version_is_extracted_only_if_it_exists():
    named = extract_filters("What did HR Policy 1.0 say about leave?", CATALOG)
    assert named.versions == ("1.0",)
    hours = extract_filters("Is the break 1.5 hours?", CATALOG)
    assert hours.versions == ()


def test_no_version_means_latest():
    filters = extract_filters("Who gets cake under the HR Policy?", CATALOG)
    assert lookup_targets(filters, CATALOG) == [("HR Policy", "2.0")]


def test_version_without_policy_targets_every_policy_with_that_version():
    filters = extract_filters("What did version 1.0 say about leave?", CATALOG)
    assert filters.policy is None
    assert filters.versions == ("1.0",)
    assert lookup_targets(filters, CATALOG) == [
        ("HR Policy", "1.0"),
        ("Health & Wellness Policy", "1.0"),
        ("Preparedness Policy", "1.0"),
        ("Time & Usage Policy", "1.0"),
    ]


def test_every_alias_key_is_a_real_policy_name():
    root = Path(__file__).resolve().parents[1]
    stored = {row["policy"] for row in json.loads((root / "chunks.json").read_text())}
    assert set(POLICY_ALIASES) <= stored

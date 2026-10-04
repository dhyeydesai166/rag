"""Sections whose title changed between versions, so title matching alone
cannot pair them. Keys: (policy, old_version, new_version). Values: old
normalized title -> new normalized title. Add an entry when a document renames
a section; the compare eval cases catch a missing entry.

Left unmapped on purpose (they are real additions or removals):
- Preparedness 2.0 earthquake, AI apocalypse, foosball leaderboard, and
  hazmat suit eligibility are new.
- Time & Usage 1.0 dispute resolution is removed. Skincare stations is
  removed; mandatory skincare is mapped onto the 2.0 leaf that kept the rule.
- Time & Usage 2.0 foosball monitoring is new.
- HR 2.0 cake-sharing default and weekend abandonment are new. The 1.0
  refrigerator leaf maps to the ownership subsection.
"""

SECTION_RENAMES = {
    ("Preparedness Policy", "1.0", "2.0"): {
        "nuclear apocalypse protocol > shelter position": (
            "nuclear apocalypse protocol > shelter location"
        ),
        "nuclear apocalypse protocol > all-clear timing": (
            "nuclear apocalypse protocol > duration of sheltering"
        ),
        "general preparedness expectations": (
            "general preparedness expectations and restocking"
        ),
        "general preparedness expectations > centralized supplies": (
            "general preparedness expectations and restocking > centralized restocking"
        ),
        "zombie apocalypse protocol > improvised weapons": (
            "zombie apocalypse protocol > weapon eligibility"
        ),
    },
    ("Time & Usage Policy", "1.0", "2.0"): {
        "foosball time": "foosball time and the winner-takes-tokens rule",
        "token depletion — consequences > mandatory skincare": (
            "token depletion — consequences"
        ),
    },
    ("HR Policy", "1.0", "2.0"): {
        "shared refrigerator policy": "shared refrigerator policy > ownership",
    },
}


def apply_renames(title: str, renames: dict[str, str]) -> str:
    """Rename an old normalized title: an exact match first, else the longest
    matching parent prefix ('foosball time > daily allowance' ->
    'foosball time and the winner-takes-tokens rule > daily allowance')."""
    if title in renames:
        return renames[title]
    best = ""
    for old, _new in renames.items():
        prefix = old + " > "
        if title.startswith(prefix) and len(old) > len(best):
            best = old
    if not best:
        return title
    return renames[best] + title[len(best) :]

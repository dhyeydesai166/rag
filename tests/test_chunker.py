from rag.chunker import chunk, make_id, sha256_hex


def test_intro_text_under_a_section_with_subsections_becomes_its_own_chunk():
    blocks = [
        {"level": 1, "heading": "3. Video Game Time", "text": "Play is allowed."},
        {"level": 2, "heading": "3.1 Daily Allowance", "text": "Forty-five minutes."},
    ]
    records = chunk(blocks, "Time & Usage Policy", "2.0", "time.docx")
    intro = next(
        record for record in records if record["heading_path"] == "3. Video Game Time"
    )
    assert intro["text"] == "Play is allowed."
    assert intro["id"] == "Time & Usage Policy|2.0|3. Video Game Time"
    assert "embed" not in intro


def test_section_without_intro_text_produces_no_empty_chunk():
    blocks = [
        {"level": 1, "heading": "3. Email Tone Requirement", "text": ""},
        {
            "level": 2,
            "heading": "3.1 Requirement",
            "text": "Every email starts with a joke.",
        },
    ]
    records = chunk(blocks, "HR Policy", "2.0", "HR Policy v2.0.docx")
    assert [record["heading_path"] for record in records] == [
        "3. Email Tone Requirement > 3.1 Requirement"
    ]
    assert all(record["text"].strip() for record in records)


def test_subsection_chunk_carries_heading_path_policy_and_version():
    blocks = [
        {"level": 1, "heading": "3. Email Tone Requirement", "text": ""},
        {
            "level": 2,
            "heading": "3.1 Requirement",
            "text": "Every email starts with a joke.",
        },
        {
            "level": 1,
            "heading": "6. Boss Error Grace Period",
            "text": "Wait 30 minutes.",
        },
    ]
    records = chunk(blocks, "HR Policy", "2.0", "HR Policy v2.0.docx")
    child = records[0]
    assert child["heading_path"] == "3. Email Tone Requirement > 3.1 Requirement"
    assert child["section"] == "3. Email Tone Requirement"
    assert child["policy"] == "HR Policy"
    assert child["version"] == "2.0"
    assert child["parent_id"] == "HR Policy|2.0|3. Email Tone Requirement"
    assert child["embed_text"] == (
        "HR Policy 2.0\n"
        "3. Email Tone Requirement > 3.1 Requirement\n"
        "Every email starts with a joke."
    )
    leaf = records[1]
    assert leaf["heading_path"] == "6. Boss Error Grace Period"
    assert leaf["parent_id"] == "HR Policy|2.0"
    assert leaf["text"] == "Wait 30 minutes."
    assert leaf["embed_text"] == (
        "HR Policy 2.0\n6. Boss Error Grace Period\nWait 30 minutes."
    )


def test_the_same_policy_version_and_heading_path_always_make_the_same_id():
    first = make_id("HR Policy", "2.0", "3. Email Tone Requirement > 3.1 Requirement")
    second = make_id("HR Policy", "2.0", "3. Email Tone Requirement > 3.1 Requirement")
    assert first == second
    assert first == "HR Policy|2.0|3. Email Tone Requirement > 3.1 Requirement"


def test_split_pieces_get_ordinal_suffixes():
    assert make_id("HR Policy", "2.0", "1. Purpose", 2) == "HR Policy|2.0|1. Purpose#2"


def test_text_hash_changes_only_when_text_changes():
    blocks = [{"level": 1, "heading": "1. Purpose", "text": "Original rule."}]
    first = chunk(blocks, "HR Policy", "1.0", "hr.pdf")[0]
    same = chunk(blocks, "HR Policy", "1.0", "hr.pdf")[0]
    changed = chunk(
        [{"level": 1, "heading": "1. Purpose", "text": "Edited rule."}],
        "HR Policy",
        "1.0",
        "hr.pdf",
    )[0]
    assert first["text_sha256"] == same["text_sha256"]
    assert first["text_sha256"] == sha256_hex("Original rule.")
    assert changed["text_sha256"] != first["text_sha256"]


def test_embed_hash_changes_when_heading_is_renumbered():
    text = "Employees receive tokens."
    first = chunk(
        [{"level": 1, "heading": "5.1 Allocation Amount", "text": text}],
        "P",
        "1.0",
        "a.pdf",
    )[0]
    second = chunk(
        [{"level": 1, "heading": "6.1 Allocation Amount", "text": text}],
        "P",
        "1.0",
        "a.pdf",
    )[0]
    assert first["text_sha256"] == second["text_sha256"]
    assert first["embed_sha256"] != second["embed_sha256"]

"""Retrieval and answer scores for the fixed eval set."""


def expected_keys(kind: str, chunk_ids: list[str]) -> set[str]:
    keys = set()
    for chunk_id in chunk_ids:
        _policy, version, heading = chunk_id.split("|", 2)
        if kind == "compare":
            keys.add(f"{version}|{heading}")
        else:
            keys.add(chunk_id)
    return keys


def retrieved_keys(kind: str, hits: list[dict]) -> set[str]:
    keys = set()
    for hit in hits:
        if kind == "compare":
            for side in ("current", "previous"):
                item = hit.get(side)
                if item:
                    keys.add(f"{item['version']}|{hit['heading_path']}")
        else:
            keys.add(hit["id"])
    return keys


def retrieval_recall(expected: set[str], retrieved: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & retrieved) / len(expected)


def _has_phrase(folded: str, phrase: str | list[str] | tuple[str, ...]) -> bool:
    options = phrase if isinstance(phrase, (list, tuple)) else [phrase]
    return any(option.lower() in folded for option in options)


def answer_ok(text: str, must_contain: list, must_not_contain: list[str]) -> bool:
    folded = text.lower()
    has_required = all(_has_phrase(folded, phrase) for phrase in must_contain)
    avoids_banned = all(phrase.lower() not in folded for phrase in must_not_contain)
    return has_required and avoids_banned


def aggregate(rows: list[dict]) -> dict[str, float]:
    count = len(rows)
    if count == 0:
        return {"recall": 0.0, "accuracy": 0.0}
    return {
        "recall": sum(row["recall"] for row in rows) / count,
        "accuracy": sum(row["answer_ok"] for row in rows) / count,
    }


def format_report(rows: list[dict], totals: dict[str, float]) -> str:
    lines = [
        "",
        f"{'recall':>8} {'answer':>8} {'latency':>10}  question",
    ]
    for row in rows:
        mark = "yes" if row["answer_ok"] else "no"
        lines.append(
            f"{row['recall']:8.3f} {mark:>8} {row['latency']:9.3f}s  {row['question']}"
        )
    lines.append(f"recall={totals['recall']:.3f}  accuracy={totals['accuracy']:.3f}")
    return "\n".join(lines)

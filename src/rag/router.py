import json

from rag.config import ROUTE_MODEL
from rag.logutil import log

PROMPT = (
    "Reply with one JSON object only: "
    '{{"kind":"lookup"|"compare","policy":"","version":""}}\n'
    "Use compare only when the question asks what changed, what's new, a "
    "diff, or how one version of a policy differs from another.\n"
    "A question that names an old version without asking for a change is "
    "lookup.\n"
    "policy must be one of the catalog names or empty. version is the named "
    "version for lookup, or empty.\n"
    "\n"
    "Catalog:\n"
    "{catalog}\n"
    "\n"
    "Examples:\n"
    "what changed in HR Policy -> "
    '{{"kind":"compare","policy":"HR Policy","version":""}}\n'
    "what's new in preparedness -> "
    '{{"kind":"compare","policy":"Preparedness Policy","version":""}}\n'
    "diff the time policy -> "
    '{{"kind":"compare","policy":"Time & Usage Policy","version":""}}\n'
    "how did v2 differ for HR Policy -> "
    '{{"kind":"compare","policy":"HR Policy","version":""}}\n'
    "what did HR Policy 1.0 say about leave -> "
    '{{"kind":"lookup","policy":"HR Policy","version":"1.0"}}\n'
    "\n"
    "Question:\n"
    "{question}\n"
)


def routing_model() -> str:
    return ROUTE_MODEL


def route(question: str, model, policies: dict) -> dict:
    catalog = "\n".join(
        f"{name}: {', '.join(versions)}" for name, versions in sorted(policies.items())
    )
    decision = parse_route(
        model.generate(PROMPT.format(catalog=catalog, question=question)),
        policies,
    )
    log(
        "router",
        f"kind={decision['kind']} policy={decision['policy']} "
        f"version={decision['version']}",
    )
    return decision


def parse_route(raw, policies: dict) -> dict:
    fallback = {"kind": "lookup", "policy": "", "version": ""}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback
    if not isinstance(data, dict):
        return fallback
    kind = data.get("kind")
    policy = data.get("policy") or ""
    version = data.get("version") or ""
    if kind not in {"lookup", "compare"}:
        return fallback
    if policy and policy not in policies:
        return fallback
    versions = policies.get(policy, ())
    if version and version not in versions:
        return fallback
    return {"kind": kind, "policy": policy, "version": version}

from rag.provenance import ci_run, git_sha, provenance


def test_git_sha_adds_dirty_when_the_worktree_is_not_clean(monkeypatch):
    def fake_git(*args):
        if args == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("status", "--porcelain"):
            return " M src/rag/generate.py"
        raise AssertionError(args)

    monkeypatch.setattr("rag.provenance._git", fake_git)
    assert git_sha() == "abc123-dirty"


def test_git_sha_is_clean_when_status_is_empty(monkeypatch):
    monkeypatch.setattr(
        "rag.provenance._git",
        lambda *args: "abc123" if args[0] == "rev-parse" else "",
    )
    assert git_sha() == "abc123"


def test_provenance_records_prompts_models_and_the_build(monkeypatch):
    monkeypatch.setattr("rag.provenance.git_sha", lambda: "abc123")
    monkeypatch.setattr("rag.provenance.digest_of", lambda client, model: "digest")
    monkeypatch.setattr("rag.provenance.ollama_server_version", lambda: "0.0.0")

    class Client:
        def list(self):
            return {"models": []}

    record = provenance(Client(), {"build_id": "build1"})
    assert record["git_sha"] == "abc123"
    assert record["build_id"] == "build1"
    assert record["collection"] == ""
    assert record["embed_digest"] == "digest"
    assert record["temperature"] == 0
    assert record["seed"] == 42
    assert set(record["prompts"]) == {"lookup_v5", "compare_v2"}
    assert record["ollama_version"] == "0.0.0"


def test_ci_run_is_empty_outside_ci(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    assert ci_run() == {}


def test_ci_run_links_to_the_actions_run(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/rag")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.delenv("GITHUB_SERVER_URL", raising=False)
    assert ci_run() == {
        "event": "schedule",
        "run_url": "https://github.com/owner/rag/actions/runs/42",
    }

from types import SimpleNamespace

from adapter.generation_adapter import GenerationAdapter
from rag.generate import EMPTY, SYSTEM, generate, generation_model


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, model, prompt, stream):
        self.calls.append({"model": model, "prompt": prompt, "stream": stream})
        return self.response


def test_generation_model_is_pinned_gemma3_12b():
    assert generation_model() == "gemma3:12b"


def test_generation_adapter_reads_a_dict_response():
    client = FakeClient({"response": "cake"})
    text = GenerationAdapter(client=client, model="gemma3:12b").generate("who")
    assert text == "cake"
    assert client.calls[0]["model"] == "gemma3:12b"
    assert client.calls[0]["prompt"] == "who"
    assert client.calls[0]["stream"] is False


def test_generation_adapter_reads_an_object_response():
    client = FakeClient(SimpleNamespace(response="leave"))
    assert GenerationAdapter(client=client).generate("q") == "leave"


def test_generation_adapter_forwards_the_system_prompt():
    client = FakeClient({"response": "cited"})

    def generate(model, prompt, stream, system=None):
        client.calls.append(
            {"model": model, "prompt": prompt, "stream": stream, "system": system}
        )
        return client.response

    client.generate = generate
    text = GenerationAdapter(client=client).generate("Question: cake", system="rules")
    assert text == "cited"
    assert client.calls[0]["system"] == "rules"


class RecordingModel:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, system=None):
        self.calls.append({"prompt": prompt, "system": system})
        return "answer"


def test_generate_sends_lookup_passages_with_the_system_prompt():
    model = RecordingModel()
    text = generate(
        "who gets cake?",
        "lookup",
        [
            {
                "policy": "HR Policy",
                "version": "2.0",
                "heading_path": "3. Leave",
                "text": "Employees receive cake on Friday.",
            }
        ],
        model,
    )
    assert text.startswith("answer\n\n")
    assert "HR Policy 2.0, 3. Leave" in text
    assert model.calls[0]["system"] == SYSTEM
    assert "Question: who gets cake?" in model.calls[0]["prompt"]
    assert "HR Policy 2.0 3. Leave" in model.calls[0]["prompt"]
    assert "Employees receive cake on Friday." in model.calls[0]["prompt"]


def test_generate_sends_compare_pairs_including_a_missing_side():
    model = RecordingModel()
    text = generate(
        "what changed?",
        "compare",
        [
            {
                "policy": "HR Policy",
                "heading_path": "8. Added",
                "current": {"version": "2.0", "text": "new clause"},
                "previous": None,
            },
            {
                "policy": "HR Policy",
                "heading_path": "3. Leave",
                "current": {"version": "2.0", "text": "no dessert"},
                "previous": {"version": "1.0", "text": "cake"},
            },
        ],
        model,
    )
    assert text.startswith("answer\n\n")
    assert "HR Policy 2.0, 8. Added" in text
    assert "HR Policy 1.0, 8. Added" not in text
    assert "HR Policy 2.0, 3. Leave" in text
    assert "HR Policy 1.0, 3. Leave" in text
    prompt = model.calls[0]["prompt"]
    assert "HR Policy 3. Leave" in prompt
    assert "current 2.0" in prompt
    assert "no dessert" in prompt
    assert prompt.rstrip().endswith("cake")


def test_generate_skips_the_model_when_there_are_no_hits():
    model = RecordingModel()
    assert generate("cake", "lookup", [], model) == EMPTY
    assert model.calls == []


def test_one_model_records_the_route_prompt_and_the_answer_prompt():
    from rag.router import route

    model = RecordingModel()
    route("what changed?", model, {"HR Policy": ("1.0", "2.0")})
    generate(
        "what changed?",
        "lookup",
        [
            {
                "policy": "HR Policy",
                "version": "2.0",
                "heading_path": "3. Leave",
                "text": "cake",
            }
        ],
        model,
    )
    assert model.calls[0]["system"] is None
    assert "Question:" in model.calls[0]["prompt"]
    assert model.calls[1]["system"] == SYSTEM
    assert "Question: what changed?" in model.calls[1]["prompt"]

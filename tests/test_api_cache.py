"""Mock provider and tokenizer; enforce zero external calls in unit tests."""

import json

import pytest

from signalbench.core import config
from signalbench.models.api_llm import APIModel, BudgetExceeded


class Encoder:
    def encode(self, text):
        return list(text.encode())


class ZeroExamples:
    def get(self, text):
        return []


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "signalbench.models.api_llm.tiktoken.encoding_for_model", lambda _: Encoder()
    )
    cfg = config("configs/default.yaml")
    cfg["paths"]["cache"] = str(tmp_path)
    return APIModel(cfg, ZeroExamples())


def test_cache_prevents_repeat_calls(api, monkeypatch):
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-placeholder")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "positive"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1},
            }

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("signalbench.models.api_llm.httpx.post", post)
    first = api.predict(["Revenue rose strongly."])
    second = api.predict(["Revenue rose strongly."])
    assert first == second
    assert len(calls) == 1
    assert first[0]["score"] == 1
    assert "probabilities" not in first[0]
    assert api.estimate(["Revenue rose strongly."]) == 0
    assert api.db.execute("SELECT SUM(cost) FROM responses").fetchone()[0] == pytest.approx(3.6e-6)


def test_budget_guard_before_network(api, monkeypatch):
    api.cfg["budget"] = 0
    monkeypatch.setattr(
        "signalbench.models.api_llm.httpx.post",
        lambda *a, **kw: pytest.fail("Budget violation called API"),
    )
    with pytest.raises(BudgetExceeded):
        api.predict(["An event"])
    assert api.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_crash_reservation_is_not_retried(api):
    from signalbench.core import digest

    body = api.request("uncertain")
    api.db.execute(
        "INSERT INTO responses VALUES (?,?,NULL,?,NULL)",
        (digest(body), json.dumps(body), api.upper_cost(body)),
    )
    api.db.commit()
    with pytest.raises(RuntimeError, match="reconcile"):
        api.predict(["uncertain"])


def test_cache_key_changes_with_prompt_and_model(api):
    from signalbench.core import digest

    original = digest(api.request("text"))
    api.cfg = dict(api.cfg, model="different-model")
    assert original != digest(api.request("text"))

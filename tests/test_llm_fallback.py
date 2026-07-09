"""LLM fallback: model routing, cost accounting, parsing, degradation."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.llm_fallback import LLMFallback, LLMResult, estimate_cost


def make_settings(**overrides) -> Settings:
    s = Settings()
    s.openai_api_key = overrides.pop("openai_api_key", "test-key")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_openai_response(payload: dict, prompt_tokens=500, completion_tokens=100):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.chat.completions.create.return_value = make_openai_response(
        {"events": [{"name": "Fair A", "date": "March 14, 2025",
                     "time": "6:00 pm", "location": "Sacramento"}]}
    )
    return client


def test_routing_simple_vs_complex_model():
    fb = LLMFallback(make_settings(llm_complexity_node_threshold=40))
    assert fb.pick_model(10) == "gpt-4o-mini"
    assert fb.pick_model(41) == "gpt-4o"


def test_regroup_returns_parsed_events(mock_client, node_factory):
    fb = LLMFallback(make_settings(), client=mock_client)
    nodes = node_factory([{"text_context": "March 14, 2025", "contains_date": 1},
                          {"text_context": "Sacramento Fair"}])
    result = fb.regroup_nodes(nodes)

    assert result.error is None
    assert result.events == [{"name": "Fair A", "date": "March 14, 2025",
                              "time": "6:00 pm", "location": "Sacramento"}]
    assert result.model == "gpt-4o-mini"
    # cost computed from the API-reported usage, not the estimate
    assert result.cost_usd == pytest.approx(estimate_cost("gpt-4o-mini", 500, 100))
    assert result.cost_usd > 0


def test_cost_cap_refuses_call(mock_client, node_factory):
    fb = LLMFallback(make_settings(llm_max_cost_per_request=0.0), client=mock_client)
    result = fb.regroup_nodes(node_factory([{"text_context": "x" * 50}]))
    assert result.error == "cost_cap_exceeded"
    mock_client.chat.completions.create.assert_not_called()


def test_disabled_without_api_key(node_factory):
    fb = LLMFallback(make_settings(openai_api_key=""))
    assert not fb.enabled
    result = fb.regroup_nodes(node_factory([{"text_context": "x"}]))
    assert result.error == "llm_disabled"


def test_api_error_degrades_gracefully(node_factory):
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("rate limited")
    fb = LLMFallback(make_settings(), client=client)
    result = fb.regroup_nodes(node_factory([{"text_context": "some event text"}]))
    assert result.error is not None
    assert result.events == []


def test_malformed_json_reported_as_parse_error(node_factory):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    fb = LLMFallback(make_settings(), client=client)
    result = fb.regroup_nodes(node_factory([{"text_context": "some event text"}]))
    assert result.error and result.error.startswith("parse_error")
    # tokens were still spent and must be accounted for
    assert result.cost_usd > 0


def test_pricing_table():
    assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)

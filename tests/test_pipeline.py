"""Pipeline integration: confidence gating and LLM escalation path."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.pipeline import (
    EventExtractionPipeline,
    adaptive_top_k,
    assemble_event,
    cluster_confidence,
)


def make_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def mock_llm_client(events: list[dict]) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"events": events}))
        )],
        usage=SimpleNamespace(prompt_tokens=800, completion_tokens=150),
    )
    return client


def test_adaptive_top_k():
    assert adaptive_top_k(50) == 40      # small page, floored by 4*events
    assert adaptive_top_k(150) == 50
    assert adaptive_top_k(500) == 80


def test_assemble_event_extracts_fields(node_factory):
    nodes = node_factory([
        {"text_context": "Sacramento College Fair", "tag": "H2", "text_length": 23},
        {"text_context": "March 14, 2025", "contains_date": 1},
        {"text_context": "6:00 pm", "contains_time": 1},
        {"text_context": "Convention Center", "text_word_location": 1},
    ])
    ev = assemble_event(nodes)
    assert ev == {
        "name": "Sacramento College Fair",
        "date": "March 14, 2025",
        "time": "6:00 pm",
        "location": "Convention Center",
    }


def test_confidence_penalizes_merged_events(node_factory):
    clean = node_factory([
        {"text_context": "Fair", "tag": "H2"},
        {"text_context": "March 14, 2025", "contains_date": 1},
    ])
    clean["calibrated_score"] = 0.9
    merged = node_factory(
        [{"text_context": f"March {i}, 2025", "contains_date": 1} for i in range(1, 5)]
    )
    merged["calibrated_score"] = 0.9

    ev_clean = assemble_event(clean)
    ev_merged = assemble_event(merged)
    assert cluster_confidence(clean, ev_clean) > cluster_confidence(merged, ev_merged)


def test_llm_fallback_recovers_merged_events(sample_html):
    llm_events = [
        {"name": "Sacramento College Fair", "date": "March 14, 2025",
         "time": "6:00 pm - 8:30 pm", "location": "Sacramento Convention Center"},
        {"name": "Fresno College Fair", "date": "April 2, 2025",
         "time": "5:30 pm - 8:00 pm", "location": "Fresno Fairgrounds Hall"},
    ]
    settings = make_settings(openai_api_key="test-key")
    pipeline = EventExtractionPipeline(settings, llm_client=mock_llm_client(llm_events))

    result = pipeline.extract(sample_html)

    assert result.extraction_method in ("llm_fallback", "hybrid")
    assert result.llm_cost_usd > 0
    dates = {ev["date"] for ev in result.events}
    assert {"March 14, 2025", "April 2, 2025"} <= dates
    assert any(ev["extraction_method"] == "llm_fallback" for ev in result.events)


def test_llm_disabled_falls_back_to_ml(sample_html):
    settings = make_settings(openai_api_key="")
    pipeline = EventExtractionPipeline(settings)

    result = pipeline.extract(sample_html)

    assert result.extraction_method == "ml"
    assert result.llm_cost_usd == 0.0
    assert all(ev["extraction_method"] == "ml" for ev in result.events)


def test_use_llm_fallback_false_never_calls_llm(sample_html):
    client = mock_llm_client([])
    settings = make_settings(openai_api_key="test-key")
    pipeline = EventExtractionPipeline(settings, llm_client=client)

    result = pipeline.extract(sample_html, use_llm_fallback=False)

    client.chat.completions.create.assert_not_called()
    assert result.extraction_method == "ml"

"""API contract tests: /health, /extract, /metrics, monitoring side-effects."""
import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate monitoring in a temp file; models/ must exist (trained artifacts).
    monkeypatch.setenv("MONITORING_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(main_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["llm_enabled"] is False


def test_extract_returns_events(client, sample_html):
    r = client.post("/extract", json={"html": sample_html, "site": "test-site"})
    assert r.status_code == 200
    body = r.json()

    assert body["n_nodes"] > 0
    assert body["extraction_method"] in ("ml", "llm_fallback", "hybrid")
    for ev in body["events"]:
        assert set(ev) == {"name", "date", "time", "location",
                           "confidence_score", "extraction_method"}
        assert 0.0 <= ev["confidence_score"] <= 1.0

    # Without an API key, nothing may be attributed to the LLM.
    assert all(ev["extraction_method"] == "ml" for ev in body["events"])
    assert body["llm_cost_usd"] == 0.0


def test_extract_finds_event_fields(client, sample_html):
    # Without an LLM key the merged low-confidence cluster is returned as-is
    # (the documented failure mode the fallback exists for) — but its date
    # field must still be populated from the page.
    r = client.post("/extract", json={"html": sample_html})
    events = r.json()["events"]
    assert events, "at least one event expected"
    assert "March 14, 2025" in json.dumps(events)


def test_extract_rejects_empty_html(client):
    assert client.post("/extract", json={"html": ""}).status_code == 422


def test_metrics_aggregates_requests(client, sample_html):
    client.post("/extract", json={"html": sample_html, "site": "site-a"})
    client.post("/extract", json={"html": sample_html, "site": "site-a"})

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_requests"] == 2
    assert body["requests_by_site"] == {"site-a": 2}
    assert body["avg_latency_ms"] > 0
    assert "ml" in body["method_breakdown"]

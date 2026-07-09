"""Pydantic request/response contracts for the API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ExtractionMethod = Literal["ml", "llm_fallback", "hybrid"]


class ExtractRequest(BaseModel):
    html: str = Field(..., min_length=1, description="Raw HTML of the page")
    site: Optional[str] = Field(
        default=None, description="Optional source identifier used for monitoring"
    )
    use_llm_fallback: bool = Field(
        default=True,
        description="Allow escalation of low-confidence clusters to the LLM",
    )


class Event(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    extraction_method: ExtractionMethod


class ExtractResponse(BaseModel):
    events: List[Event]
    site: Optional[str]
    n_nodes: int = Field(..., description="DOM nodes extracted from the HTML")
    n_candidates: int = Field(..., description="Nodes retained after top-K selection")
    extraction_method: ExtractionMethod = Field(
        ..., description="'hybrid' when both ML and LLM produced events"
    )
    llm_cost_usd: float = 0.0
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    llm_enabled: bool
    version: str


class MetricsResponse(BaseModel):
    total_requests: int
    total_events_extracted: int
    total_llm_cost_usd: float
    avg_latency_ms: float
    avg_nodes_per_request: float
    method_breakdown: dict
    requests_by_site: dict

"""
FastAPI service exposing the event-extraction pipeline.

Endpoints:
- POST /extract  — raw HTML in, structured events out
- GET  /health   — liveness + artifact status (used by Docker/Render checks)
- GET  /metrics  — aggregated monitoring stats
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.config import get_settings
from app.monitoring import RequestLogger, RequestRecord, now_ms
from app.pipeline import EventExtractionPipeline
from app.schemas import (
    Event,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    MetricsResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Load artifacts once at startup, not per request.
    settings = get_settings()
    app_.state.pipeline = EventExtractionPipeline(settings)
    app_.state.request_logger = RequestLogger(settings.monitoring_path)
    if not app_.state.pipeline.ready:
        logger.warning("Pipeline started WITHOUT model artifacts — /extract will 503")
    yield


app = FastAPI(
    title="Event Extraction API",
    description="ML + LLM-fallback extraction of structured events from raw HTML",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline: EventExtractionPipeline = request.app.state.pipeline
    return HealthResponse(
        status="ok" if pipeline.ready else "degraded",
        model_loaded=pipeline.ready,
        llm_enabled=pipeline.llm.enabled,
        version=API_VERSION,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    return MetricsResponse(**request.app.state.request_logger.aggregate())


@app.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest, request: Request) -> ExtractResponse:
    pipeline: EventExtractionPipeline = request.app.state.pipeline
    req_logger: RequestLogger = request.app.state.request_logger

    if not pipeline.ready:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    start = now_ms()
    try:
        result = pipeline.extract(payload.html, use_llm_fallback=payload.use_llm_fallback)
    except Exception as exc:
        latency = now_ms() - start
        req_logger.log(RequestRecord(
            timestamp=time.time(), site=payload.site, n_nodes=0, n_candidates=0,
            n_events=0, extraction_method="ml", llm_model=None, llm_cost_usd=0.0,
            latency_ms=round(latency, 2), status="error", error=str(exc),
        ))
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail="Extraction failed") from exc

    latency = now_ms() - start
    req_logger.log(RequestRecord(
        timestamp=time.time(),
        site=payload.site,
        n_nodes=result.n_nodes,
        n_candidates=result.n_candidates,
        n_events=len(result.events),
        extraction_method=result.extraction_method,
        llm_model=result.llm_model,
        llm_cost_usd=round(result.llm_cost_usd, 6),
        latency_ms=round(latency, 2),
    ))

    return ExtractResponse(
        events=[Event(**ev) for ev in result.events],
        site=payload.site,
        n_nodes=result.n_nodes,
        n_candidates=result.n_candidates,
        extraction_method=result.extraction_method,
        llm_cost_usd=round(result.llm_cost_usd, 6),
        latency_ms=round(latency, 2),
    )

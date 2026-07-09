"""
LLM fallback layer for cluster repair.

When the ML clustering is not trusted (low calibrated confidence, or a
structurally suspect cluster), the candidate nodes are sent to an LLM that
regroups them into events and extracts the fields directly.

Cost-aware orchestration:
- few candidate nodes  → ``gpt-4o-mini`` (cheap, sufficient for simple pages)
- many candidate nodes → ``gpt-4o`` (dense listings are exactly where the
  heuristic clustering fails, and where the small model also degrades)
- a per-request USD cap refuses escalation rather than overspending.

The client is injectable so tests can mock the OpenAI API entirely.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

from app.config import Settings

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output) — update alongside provider pricing.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

SYSTEM_PROMPT = """\
You are an expert at reconstructing structured events from fragments of web pages.
You receive DOM text nodes (in page rendering order) extracted from a single page
that lists one or more events (college fairs, conferences, meetings...).

Group the nodes into distinct events and extract for each event:
- name: the event's title
- date: the date as written on the page (do not invent or reformat)
- time: the time or time range as written
- location: venue/city as written

Rules:
- Only use text present in the nodes. Never invent values; use null when absent.
- Nodes that are navigation, headers or unrelated text must be ignored.
- Adjacent nodes usually belong to the same event; a new date usually starts a new event.

Return ONLY a JSON object: {"events": [{"name": ..., "date": ..., "time": ..., "location": ...}]}
"""


@dataclass
class LLMResult:
    events: List[dict] = field(default_factory=list)
    model: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICING.get(model, PRICING["gpt-4o"])
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _estimate_input_tokens(prompt: str) -> int:
    # ~4 chars/token is the standard rough estimate for English text;
    # only used for the pre-call budget check, real usage comes from the API.
    return len(prompt) // 4 + len(SYSTEM_PROMPT) // 4


def build_user_prompt(nodes: pd.DataFrame) -> str:
    lines = []
    for _, row in nodes.sort_values("rendering_order").iterrows():
        text = str(row.get("text_context", "")).strip()
        if text:
            lines.append(f"[{int(row['rendering_order'])}] <{row.get('tag', '?')}> {text}")
    return "Text nodes in rendering order:\n" + "\n".join(lines)


class LLMFallback:
    def __init__(self, settings: Settings, client: Any = None) -> None:
        """``client`` allows dependency injection of a mock in tests."""
        self.settings = settings
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None or self.settings.llm_enabled

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # imported lazily: optional dependency path
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def pick_model(self, n_nodes: int) -> str:
        if n_nodes > self.settings.llm_complexity_node_threshold:
            return self.settings.llm_complex_model
        return self.settings.llm_simple_model

    def regroup_nodes(self, nodes: pd.DataFrame) -> LLMResult:
        """Send candidate nodes to the LLM, return reconstructed events."""
        if not self.enabled:
            return LLMResult(error="llm_disabled")
        if nodes.empty:
            return LLMResult(error="no_nodes")

        model = self.pick_model(len(nodes))
        user_prompt = build_user_prompt(nodes)

        # Pre-call budget check: refuse rather than overspend. Output tokens
        # are unknown up-front; assume a generous 1000.
        projected = estimate_cost(model, _estimate_input_tokens(user_prompt), 1000)
        if projected > self.settings.llm_max_cost_per_request:
            logger.warning(
                "LLM fallback skipped: projected cost $%.4f exceeds cap $%.4f",
                projected, self.settings.llm_max_cost_per_request,
            )
            return LLMResult(error="cost_cap_exceeded", model=model)

        try:
            response = self._get_client().chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # network/auth/rate-limit: degrade, don't crash
            logger.exception("LLM fallback call failed")
            return LLMResult(error=f"api_error: {exc}", model=model)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = estimate_cost(model, input_tokens, output_tokens)

        try:
            payload = json.loads(response.choices[0].message.content)
            events = payload.get("events", [])
            if not isinstance(events, list):
                raise ValueError("'events' is not a list")
        except (json.JSONDecodeError, ValueError, AttributeError, IndexError) as exc:
            return LLMResult(
                error=f"parse_error: {exc}", model=model,
                cost_usd=cost, input_tokens=input_tokens, output_tokens=output_tokens,
            )

        clean_events = []
        for ev in events:
            if isinstance(ev, dict):
                clean_events.append({
                    "name": ev.get("name"),
                    "date": ev.get("date"),
                    "time": ev.get("time"),
                    "location": ev.get("location"),
                })

        return LLMResult(
            events=clean_events, model=model, cost_usd=cost,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )

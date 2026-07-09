"""
Central configuration, driven by environment variables.

A plain dataclass is used instead of pydantic-settings to keep the
dependency surface minimal — every value has a sane default so the
service boots with zero configuration (LLM fallback simply disables
itself when no API key is present).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- artifacts -------------------------------------------------------
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_DIR", REPO_ROOT / "models"))
    )

    # --- LLM fallback ----------------------------------------------------
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    # Below this confidence the cluster is escalated to the LLM. 0.40 is
    # the LOSO-benchmark knee (scripts/tune_threshold.py): 26% of ML
    # failures caught for 4% waste and 11% of events escalated.
    llm_confidence_threshold: float = field(
        default_factory=lambda: _env_float("LLM_CONFIDENCE_THRESHOLD", 0.40)
    )
    # Above this many candidate nodes the "complex" model is used.
    llm_complexity_node_threshold: int = field(
        default_factory=lambda: _env_int("LLM_COMPLEXITY_NODE_THRESHOLD", 40)
    )
    llm_simple_model: str = field(
        default_factory=lambda: os.getenv("LLM_SIMPLE_MODEL", "gpt-4o-mini")
    )
    llm_complex_model: str = field(
        default_factory=lambda: os.getenv("LLM_COMPLEX_MODEL", "gpt-4o")
    )
    # Hard cap on LLM spend per request (USD) — refuses escalation past it.
    llm_max_cost_per_request: float = field(
        default_factory=lambda: _env_float("LLM_MAX_COST_PER_REQUEST", 0.05)
    )

    # --- monitoring ------------------------------------------------------
    monitoring_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MONITORING_PATH", REPO_ROOT / "data" / "monitoring" / "requests.jsonl")
        )
    )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    """Fresh read of the environment — cheap, and keeps tests hermetic."""
    return Settings()

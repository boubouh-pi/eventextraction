"""
End-to-end inference pipeline: raw HTML → structured events.

Stages (mirrors the validated training pipeline):
  1. parse HTML into DOM nodes with the training feature schema
  2. score every node with the production LightGBM model
  3. calibrate scores (Platt) into probabilities
  4. adaptive top-K selection + noise removal
  5. DOM clustering with the 4 validated post-processing steps
  6. assemble each cluster into an event + confidence score
  7. escalate low-confidence / suspect clusters to the LLM fallback

The fallback decision is per-cluster, not per-page: a page where 9 events
cluster cleanly and 1 is suspect only pays LLM cost for the suspect one.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

import app  # noqa: F401  (installs the src/ path shim)
from app.calibration import PlattCalibrator
from app.clustering import cluster_candidates, remove_obvious_noise
from app.config import Settings, get_settings
from app.html_parser import parse_html_to_nodes
from app.llm_fallback import LLMFallback
from dateclf.features import add_dom_neighbor_features

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    events: List[dict] = field(default_factory=list)
    n_nodes: int = 0
    n_candidates: int = 0
    extraction_method: str = "ml"
    llm_cost_usd: float = 0.0
    llm_model: Optional[str] = None


def adaptive_top_k(n_nodes: int, n_events_estimated: int = 10) -> int:
    """K grows with page size; always ≥4 nodes per estimated event."""
    if n_nodes < 100:
        k = 30
    elif n_nodes < 200:
        k = 50
    else:
        k = 80
    return max(k, min(n_events_estimated * 4, 100))


def _first_text(nodes: pd.DataFrame, mask: pd.Series) -> Optional[str]:
    sel = nodes[mask]
    if sel.empty:
        return None
    return str(sel.sort_values("rendering_order").iloc[0]["text_context"]).strip()


def assemble_event(cluster_nodes: pd.DataFrame) -> dict:
    """
    Heuristic field extraction from one cluster, using the same pattern
    features the model was trained on.
    """
    nodes = cluster_nodes.sort_values("rendering_order")

    date = _first_text(nodes, nodes["contains_date"].astype(bool))
    time = _first_text(nodes, nodes["contains_time"].astype(bool))

    loc_mask = nodes.get("text_word_location", pd.Series(0, index=nodes.index)).astype(bool)
    location = _first_text(nodes, loc_mask)

    # Name: prefer a heading tag, else the longest mostly-letters node that
    # is not the date/time/location text itself.
    heading = nodes[nodes["tag"].astype(str).str.lower().isin(["h1", "h2", "h3", "h4"])]
    name = None
    if not heading.empty:
        name = str(heading.iloc[0]["text_context"]).strip()
    else:
        used = {date, time, location}
        candidates = nodes[
            (~nodes["text_context"].isin(used))
            & (nodes["letter_ratio"] > 0.5)
            & (nodes["text_length"].between(10, 120))
        ]
        if not candidates.empty:
            name = str(candidates.sort_values("text_length", ascending=False)
                       .iloc[0]["text_context"]).strip()

    return {"name": name, "date": date, "time": time, "location": location}


def cluster_confidence(cluster_nodes: pd.DataFrame, event: dict) -> float:
    """
    Confidence that this cluster is one correctly-grouped event.

    Two regimes, because the failure modes differ:
    - periodic clusters: the grouping was validated structurally (regular
      marker, temporal sanity check), so the base is high and independent
      of node scores — benchmarked periodic segments are right ~90% of the
      time even when the detector scored their nodes low.
    - gap-chain clusters: base = mean calibrated node probability, plus
      penalties for missing fields (the gap chain has no structural proof).

    Shared penalties target merged events (≥3 date markers) and
    mega-clusters — the documented dense-site failure modes.
    """
    method = ""
    if "cluster_method" in cluster_nodes.columns and len(cluster_nodes):
        method = str(cluster_nodes["cluster_method"].iloc[0])

    if method == "periodic":
        base = 0.85
        penalty = 1.0
    else:
        base = float(cluster_nodes["calibrated_score"].mean())
        penalty = 1.0
        if event["date"] is None:    # an event without any date is suspicious
            penalty *= 0.7
        if event["name"] is None:
            penalty *= 0.85

    n_dates = int(cluster_nodes["contains_date"].sum())
    if n_dates >= 3:          # several dates in one cluster → probably merged events
        penalty *= 0.5
    if len(cluster_nodes) > 15:  # mega-cluster → clustering likely failed
        penalty *= 0.6

    return float(np.clip(base * penalty, 0.0, 1.0))


class EventExtractionPipeline:
    def __init__(self, settings: Optional[Settings] = None, llm_client=None) -> None:
        self.settings = settings or get_settings()
        self.model: Optional[lgb.Booster] = None
        self.calibrator = PlattCalibrator()
        self.feature_names: List[str] = []
        self.llm = LLMFallback(self.settings, client=llm_client)
        self._load_artifacts()

    # --- artifacts -------------------------------------------------------
    def _load_artifacts(self) -> None:
        model_dir = self.settings.model_dir
        model_path = model_dir / "model.txt"
        calib_path = model_dir / "calibrator.joblib"
        feats_path = model_dir / "features.json"

        if model_path.exists():
            self.model = lgb.Booster(model_file=str(model_path))
            self.feature_names = self.model.feature_name()
            logger.info("Loaded model with %d features", len(self.feature_names))
        else:
            logger.warning("No model at %s — run scripts/train_production.py", model_path)

        if calib_path.exists():
            self.calibrator = PlattCalibrator.load(calib_path)
        if feats_path.exists():
            self.feature_names = json.loads(feats_path.read_text(encoding="utf-8"))

    @property
    def ready(self) -> bool:
        return self.model is not None

    # --- scoring ---------------------------------------------------------
    def score_nodes(self, nodes: pd.DataFrame) -> pd.DataFrame:
        nodes = add_dom_neighbor_features(nodes)

        X = pd.DataFrame(index=nodes.index)
        for feat in self.feature_names:
            if feat in nodes.columns:
                X[feat] = nodes[feat]
            else:
                X[feat] = 0  # feature absent from live parse → neutral value
        for c in ("tag", "parent_tag"):
            if c in X.columns:
                X[c] = X[c].astype("category")

        nodes = nodes.copy()
        nodes["score"] = self.model.predict(X)
        nodes["calibrated_score"] = self.calibrator.transform(nodes["score"].to_numpy())
        return nodes

    # --- main entrypoint --------------------------------------------------
    def extract(self, html: str, use_llm_fallback: bool = True) -> ExtractionResult:
        if not self.ready:
            raise RuntimeError("Model artifacts not loaded — run scripts/train_production.py")

        nodes = parse_html_to_nodes(html)
        result = ExtractionResult(n_nodes=len(nodes))
        if nodes.empty:
            return result

        scored = self.score_nodes(nodes)

        k = adaptive_top_k(len(scored))
        top_k = scored.nlargest(min(k, len(scored)), "calibrated_score").copy()
        top_k_clean = remove_obvious_noise(top_k)
        result.n_candidates = len(top_k_clean)
        if top_k_clean.empty:
            return result

        clustered = cluster_candidates(top_k_clean, scored)

        ml_events: List[dict] = []
        suspect_nodes: List[pd.DataFrame] = []

        for cluster_id, cluster_nodes in clustered.groupby("pred_cluster"):
            event = assemble_event(cluster_nodes)
            confidence = cluster_confidence(cluster_nodes, event)

            # Empty clusters (no field at all) are noise, not events.
            if all(event[f] is None for f in ("name", "date", "time", "location")):
                continue

            if confidence < self.settings.llm_confidence_threshold and use_llm_fallback:
                suspect_nodes.append(cluster_nodes)
            else:
                event["confidence_score"] = round(confidence, 4)
                event["extraction_method"] = "ml"
                ml_events.append(event)

        llm_events: List[dict] = []
        if suspect_nodes and use_llm_fallback and self.llm.enabled:
            # One LLM call for all suspect clusters together: cheaper than
            # per-cluster calls, and the LLM re-decides the grouping anyway.
            batch = pd.concat(suspect_nodes, ignore_index=True)
            llm_result = self.llm.regroup_nodes(batch)
            result.llm_cost_usd = llm_result.cost_usd
            result.llm_model = llm_result.model or None

            if llm_result.error is None:
                for ev in llm_result.events:
                    ev["confidence_score"] = 0.75  # LLM output: good but unverified
                    ev["extraction_method"] = "llm_fallback"
                    llm_events.append(ev)
            else:
                logger.warning("LLM fallback failed (%s) — keeping ML clusters", llm_result.error)
                for cluster_nodes in suspect_nodes:
                    event = assemble_event(cluster_nodes)
                    event["confidence_score"] = round(cluster_confidence(cluster_nodes, event), 4)
                    event["extraction_method"] = "ml"
                    ml_events.append(event)
        elif suspect_nodes:
            # Fallback unavailable/disabled: return low-confidence ML events.
            for cluster_nodes in suspect_nodes:
                event = assemble_event(cluster_nodes)
                event["confidence_score"] = round(cluster_confidence(cluster_nodes, event), 4)
                event["extraction_method"] = "ml"
                ml_events.append(event)

        result.events = ml_events + llm_events
        if ml_events and llm_events:
            result.extraction_method = "hybrid"
        elif llm_events:
            result.extraction_method = "llm_fallback"
        else:
            result.extraction_method = "ml"
        return result

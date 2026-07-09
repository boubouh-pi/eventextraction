"""
Before/after evaluation of the LLM fallback on the LOSO benchmark.

Replays the 16 LOSO folds with the *inference* pipeline (no label leakage:
clustering expansion uses pattern features, exactly like production), then:

1. BEFORE — strict perfect-reconstitution rate with ML clustering only.
2. Gate    — which clusters the confidence gate would escalate.
3. AFTER  — if OPENAI_API_KEY is set, escalated clusters are actually sent
   to the LLM; a ground-truth event counts as recovered when one LLM event
   contains its date text and (when expected) its time text.

Without an API key the script still reports BEFORE + escalation coverage,
so the README numbers are always reproducible.

Run from the repo root:  python scripts/eval_fallback.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from app.calibration import PlattCalibrator  # noqa: E402
from app.clustering import cluster_candidates, remove_obvious_noise  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm_fallback import LLMFallback  # noqa: E402
from app.pipeline import adaptive_top_k, assemble_event, cluster_confidence  # noqa: E402
from dateclf.data import (  # noqa: E402
    add_is_event_member_label,
    build_feature_matrix_for_member,
    load_config,
    load_raw_merged,
)
from dateclf.features import add_dom_neighbor_features  # noqa: E402
from dateclf.split import loso_folds_event  # noqa: E402
from dateclf.train import extract_and_validate_fields  # noqa: E402
from train_production import FEATURES_TO_KEEP, train_booster  # noqa: E402


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _field_ok(gt_event: pd.DataFrame, cluster_nodes: pd.DataFrame, word: str) -> bool:
    gt_f = gt_event[gt_event["label"].str.contains(word, case=False, na=False)]
    if gt_f.empty:
        return True
    ext = cluster_nodes[
        cluster_nodes["label"].str.contains(word, case=False, na=False)
    ]
    gt_texts = set(gt_f["text_context"].str.strip().str.lower())
    ext_texts = set(ext["text_context"].str.strip().str.lower())
    return len(gt_texts & ext_texts) >= max(1, len(gt_texts) * 0.33)


def _per_event_outcomes(clustered: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """event_id → strict per-event perfection (same 4 conditions as the
    aggregate metric in dateclf.train.extract_and_validate_fields)."""
    outcomes = {}
    for event_id in gt["event_id"].dropna().unique():
        gt_event = gt[gt["event_id"] == event_id]
        detected = clustered[clustered["row_id"].isin(gt_event["row_id"])]
        if detected.empty:
            outcomes[event_id] = False
            continue
        clusters = detected["pred_cluster"].unique()
        if len(clusters) != 1:
            outcomes[event_id] = False
            continue
        cluster_nodes = clustered[clustered["pred_cluster"] == clusters[0]]
        annotated = gt[gt["row_id"].isin(cluster_nodes["row_id"])]
        others = annotated["event_id"].dropna().unique()
        if len(others) != 1 or others[0] != event_id:
            outcomes[event_id] = False
            continue
        outcomes[event_id] = all(
            _field_ok(gt_event, cluster_nodes, w)
            for w in ("Date", "Time", "Location")
        )
    return outcomes


def llm_recovers_event(gt_event: pd.DataFrame, llm_events: list[dict]) -> bool:
    """An LLM event recovers a GT event if it carries its date (and time)."""
    gt_dates = [_norm(t) for t in gt_event.loc[
        gt_event["label"].str.contains("Date", case=False, na=False), "text_context"]]
    gt_times = [_norm(t) for t in gt_event.loc[
        gt_event["label"].str.contains("Time", case=False, na=False), "text_context"]]

    for ev in llm_events:
        blob = _norm(" ".join(str(v) for v in ev.values() if v))
        date_ok = not gt_dates or any(d in blob or blob.find(d[:12]) >= 0 for d in gt_dates)
        time_ok = not gt_times or any(t in blob for t in gt_times)
        if date_ok and time_ok and (gt_dates or gt_times):
            return True
    return False


def main() -> None:
    settings = get_settings()
    llm = LLMFallback(settings)
    threshold = settings.llm_confidence_threshold

    cfg = load_config(REPO_ROOT / "config.yaml")
    cfg["data"]["raw_dir"] = str(REPO_ROOT / "data" / "raw")

    df = load_raw_merged(cfg)
    df = add_dom_neighbor_features(df)
    df = add_is_event_member_label(df, cfg)
    X, y, _ = build_feature_matrix_for_member(df)
    features = [f for f in FEATURES_TO_KEEP if f in X.columns]
    X = X[features].copy()
    cat_features = [c for c in ("tag", "parent_tag") if c in X.columns]

    calibrator = PlattCalibrator.load(settings.model_dir / "calibrator.joblib")

    print(f"LLM fallback: {'ENABLED' if llm.enabled else 'disabled (no OPENAI_API_KEY)'}")
    print(f"Confidence threshold: {threshold}\n")

    rows = []
    event_rows = []  # per-event confidence vs ML outcome, for threshold sweeps
    total_cost = 0.0

    for i, fold in enumerate(loso_folds_event(df, y_col="is_event_member"), start=1):
        booster = train_booster(cfg, X.loc[fold.train_idx], y.loc[fold.train_idx],
                                seed_offset=i, cat_features=cat_features)

        keep_cols = ["text_context", "label", "event_id", "rendering_order",
                     "parent_index", "depth", "contains_date", "contains_time",
                     "text_length", "letter_ratio", "tag", "text_word_location"]
        df_test = df.loc[fold.test_idx, [c for c in keep_cols if c in df.columns]].copy()
        df_test["score"] = booster.predict(X.loc[fold.test_idx])
        df_test["calibrated_score"] = calibrator.transform(df_test["score"].to_numpy())
        df_test["row_id"] = df_test.index

        k = adaptive_top_k(len(df_test), df_test["event_id"].nunique())
        top_k = df_test.nlargest(min(k, len(df_test)), "score").copy()
        top_k_clean = remove_obvious_noise(top_k)
        if len(top_k_clean) < 2:
            continue

        clustered = cluster_candidates(top_k_clean, df_test)
        gt = df_test[df_test["event_id"].notna()].copy()

        before = extract_and_validate_fields(clustered, gt)

        # --- confidence gate ---------------------------------------------
        suspect_frames = []
        cluster_conf: dict = {}
        for cluster_id, cluster_nodes in clustered.groupby("pred_cluster"):
            event = assemble_event(cluster_nodes)
            if all(event[f] is None for f in ("name", "date", "time", "location")):
                continue
            conf = cluster_confidence(cluster_nodes, event)
            cluster_conf[cluster_id] = conf
            if conf < threshold:
                suspect_frames.append(cluster_nodes)

        # per-event record: ML outcome + confidence of its cluster(s),
        # consumed by scripts/tune_threshold.py to sweep gate thresholds
        per_event = _per_event_outcomes(clustered, gt)
        for event_id, ml_perfect in per_event.items():
            ev_nodes = clustered[clustered["row_id"].isin(
                gt.loc[gt["event_id"] == event_id, "row_id"]
            )]
            confs = [cluster_conf[c] for c in ev_nodes["pred_cluster"].unique()
                     if c in cluster_conf]
            event_rows.append({
                "site": fold.holdout_site,
                "event_id": event_id,
                "ml_perfect": ml_perfect,
                "confidence": min(confs) if confs else 0.0,
            })

        # GT events touching a suspect cluster
        suspect_ids = (
            pd.concat(suspect_frames)["row_id"].unique() if suspect_frames else []
        )
        gt_suspect_events = gt.loc[gt["row_id"].isin(suspect_ids), "event_id"].dropna().unique()

        recovered = 0
        fold_cost = 0.0
        if suspect_frames and llm.enabled:
            batch = pd.concat(suspect_frames, ignore_index=True)
            res = llm.regroup_nodes(batch)
            fold_cost = res.cost_usd
            total_cost += fold_cost
            if res.error is None:
                for event_id in gt_suspect_events:
                    gt_event = gt[gt["event_id"] == event_id]
                    if llm_recovers_event(gt_event, res.events):
                        recovered += 1

        n_events = before["n_events"]
        before_rate = before["perfect_reconstitution_rate"]
        # AFTER = events already perfect via ML + suspect events the LLM recovered
        after_rate = (
            (before_rate * n_events + recovered) / n_events if n_events else 0.0
        )

        rows.append({
            "site": fold.holdout_site,
            "n_events": n_events,
            "before_perfect": round(before_rate, 3),
            "n_escalated_events": len(gt_suspect_events),
            "llm_recovered": recovered,
            "after_perfect": round(min(after_rate, 1.0), 3),
            "llm_cost_usd": round(fold_cost, 5),
        })
        print(f"fold {i:2d} {fold.holdout_site:45s} "
              f"before={before_rate:5.1%} escalated={len(gt_suspect_events):2d} "
              f"recovered={recovered:2d}")

    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    print(f"\nMean perfect reconstitution BEFORE: {out['before_perfect'].mean():.1%}")
    if llm.enabled:
        print(f"Mean perfect reconstitution AFTER:  {out['after_perfect'].mean():.1%}")
        print(f"Total LLM cost for the benchmark:   ${total_cost:.4f}")
    else:
        share = out["n_escalated_events"].sum() / max(out["n_events"].sum(), 1)
        print(f"Events the gate would escalate:     {share:.1%} "
              "(set OPENAI_API_KEY to measure recovery)")

    out_path = REPO_ROOT / "runs" / "fallback_eval.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)
    pd.DataFrame(event_rows).to_csv(REPO_ROOT / "runs" / "event_confidences.csv",
                                    index=False)
    print(f"\nSaved {out_path} and runs/event_confidences.csv")


if __name__ == "__main__":
    main()

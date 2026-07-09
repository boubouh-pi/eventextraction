"""
Train the production artifacts consumed by the API.

Produces in ``models/``:
- model.txt                — LightGBM booster trained on ALL 16 sites
- calibrator.joblib        — Platt calibrator fitted on OUT-OF-FOLD scores
- features.json            — ordered feature list the model expects
- feature_importance.csv   — gain importance of the production model

Why OOF calibration: the production model's scores on its own training
sites are optimistically high. Calibrating on LOSO out-of-fold scores
means the mapping score→probability reflects performance on *unseen*
sites, which is exactly the production condition (every incoming page
is a site the model never saw).

Run from the repo root:  python scripts/train_production.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from app.calibration import PlattCalibrator  # noqa: E402
from dateclf.data import (  # noqa: E402
    add_is_event_member_label,
    build_feature_matrix_for_member,
    load_config,
    load_raw_merged,
)
from dateclf.features import add_dom_neighbor_features  # noqa: E402
from dateclf.sampling import undersample_negatives  # noqa: E402
from dateclf.split import loso_folds_event  # noqa: E402

# Same feature subset as the validated pipeline (train.py).
FEATURES_TO_KEEP = [
    "depth", "tag", "parent_tag", "num_siblings",
    "text_length", "word_count",
    "letter_ratio", "digit_ratio", "whitespace_ratio",
    "contains_date", "contains_time",
    "starts_with_digit", "ends_with_digit",
    "has_class", "has_id",
    "attr_has_word_date", "attr_has_word_time", "attr_has_word_location",
    "text_has_word_date", "text_word_time", "text_word_location",
    "prev_contains_date", "next_contains_date",
    "prev_contains_time", "next_contains_time",
    "same_parent_as_prev", "same_parent_as_next",
]


def lgb_params(cfg: dict, scale_pos_weight: float) -> dict:
    m = cfg.get("model", {})
    return {
        "objective": "binary",
        "learning_rate": m.get("learning_rate", 0.05),
        "num_leaves": m.get("num_leaves", 63),
        "min_data_in_leaf": m.get("min_data_in_leaf", 30),
        "feature_fraction": m.get("feature_fraction", 0.9),
        "bagging_fraction": m.get("bagging_fraction", 0.8),
        "bagging_freq": m.get("bagging_freq", 1),
        "lambda_l2": m.get("lambda_l2", 1.0),
        "metric": "None",
        "verbosity": -1,
        "seed": cfg.get("seed", 42),
        "scale_pos_weight": scale_pos_weight,
    }


def train_booster(cfg: dict, X: pd.DataFrame, y: pd.Series, seed_offset: int,
                  cat_features: list[str]) -> lgb.Booster:
    X_s, y_s = undersample_negatives(
        X, y,
        seed=cfg["seed"] + seed_offset,
        keep_neg_ratio=cfg.get("sampling", {}).get("keep_neg_ratio", 0.15),
    )
    spw = int((y_s == 0).sum()) / max(int(y_s.sum()), 1)
    dtrain = lgb.Dataset(
        X_s, label=y_s,
        categorical_feature=cat_features or "auto",
        free_raw_data=False,
    )
    return lgb.train(
        lgb_params(cfg, spw), dtrain,
        num_boost_round=cfg.get("model", {}).get("num_boost_round", 400),
    )


def main() -> None:
    cfg = load_config(REPO_ROOT / "config.yaml")
    cfg["data"]["raw_dir"] = str(REPO_ROOT / "data" / "raw")

    df = load_raw_merged(cfg)
    df = add_dom_neighbor_features(df)
    df = add_is_event_member_label(df, cfg)
    X, y, _ = build_feature_matrix_for_member(df)

    features = [f for f in FEATURES_TO_KEEP if f in X.columns]
    X = X[features].copy()
    cat_features = [c for c in ("tag", "parent_tag") if c in X.columns]
    print(f"Training with {len(features)} features on {len(X)} nodes "
          f"({int(y.sum())} positives)")

    # ---- 1. Out-of-fold scores for calibration (LOSO) -------------------
    oof_scores = pd.Series(np.nan, index=X.index)
    folds = list(loso_folds_event(df, y_col="is_event_member"))
    for i, fold in enumerate(folds, start=1):
        booster = train_booster(cfg, X.loc[fold.train_idx], y.loc[fold.train_idx],
                                seed_offset=i, cat_features=cat_features)
        oof_scores.loc[fold.test_idx] = booster.predict(X.loc[fold.test_idx])
        print(f"  fold {i}/{len(folds)} ({fold.holdout_site}) scored")

    valid = oof_scores.notna()
    calibrator = PlattCalibrator().fit(
        oof_scores[valid].to_numpy(), y[valid].to_numpy()
    )

    # ---- 2. Production model on ALL data --------------------------------
    booster = train_booster(cfg, X, y, seed_offset=0, cat_features=cat_features)

    fi = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    # ---- 3. Save artifacts ----------------------------------------------
    out = REPO_ROOT / "models"
    out.mkdir(exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    calibrator.save(out / "calibrator.joblib")
    (out / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    fi.to_csv(out / "feature_importance.csv", index=False)

    print(f"\nArtifacts written to {out}")
    print("\nTop-10 features by gain:")
    print(fi.head(10).to_string(index=False))

    # Quick calibration sanity check.
    calibrated = calibrator.transform(oof_scores[valid].to_numpy())
    for lo, hi in [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        band = (calibrated >= lo) & (calibrated < hi if hi < 1 else calibrated <= hi)
        if band.sum():
            actual = y[valid][band].mean()
            print(f"  calibrated [{lo:.1f},{hi:.1f}): predicted~{calibrated[band].mean():.2f} "
                  f"actual={actual:.2f} (n={int(band.sum())})")


if __name__ == "__main__":
    main()

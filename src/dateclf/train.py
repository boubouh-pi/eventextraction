from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from dateclf.data import (
    load_config,
    load_raw_merged,
    add_is_field_labels,
    build_feature_matrix_for_field,
)
from dateclf.split import loso_folds, Fold
from dateclf.sampling import undersample_other
from dateclf.metrics import precision_recall_at_k_per_page


@dataclass
class FoldResult:
    site: str
    metrics_at_k: Dict[int, Dict[str, float]]


def train_field_loso(
    config_path: str = "config.yaml",
    target_name: str = "is_Date",
) -> Dict[str, Any]:
    """
    Generic LOSO training for any binary target:
      target_name in {"is_Date", "is_Time", "is_Location", "is_Name"}
    """
    cfg = load_config(config_path)

    ks: List[int] = cfg["eval"]["ks"]
    page_col: str = cfg.get("eval", {}).get("page_col", "text_context")

    # LightGBM config from YAML (with defaults)
    lgbm_cfg = dict(cfg.get("lgbm", {}))
    num_boost_round = int(lgbm_cfg.pop("num_boost_round", 400))

    df = load_raw_merged(cfg)
    df = add_is_field_labels(df, cfg)  # strict mapping + warning if unmapped labels

    X, y, groups = build_feature_matrix_for_field(df, target_name)

    # Categorical features for LightGBM
    cat_features = [c for c in ["tag", "parent_tag"] if c in X.columns]
    for c in cat_features:
        X[c] = X[c].astype("category")

    folds: List[Fold] = list(loso_folds(df, site_col="site_id", target_col=target_name))

    all_fold_results: List[FoldResult] = []
    fi_all: List[pd.DataFrame] = []

    for i, fold in enumerate(folds, 1):
        train_idx = fold.train_idx
        test_idx = fold.test_idx

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        # ✅ FIX 1: scale_pos_weight computed on FULL train (before undersampling)
        n_pos_full = int(y_train.sum())
        n_neg_full = int((y_train == 0).sum())
        spw = (n_neg_full / max(n_pos_full, 1))

        # Undersample negatives in TRAIN only
        X_train_s, y_train_s = undersample_other(
            X_train, y_train, seed=cfg["seed"] + i, keep_neg_ratio=0.15
        )

        params = {
            "objective": "binary",
            "metric": "None",
            "verbosity": -1,
            "seed": cfg["seed"],
            "scale_pos_weight": spw,
            **lgbm_cfg,  # ✅ FIX 3: configurable hyperparams
        }

        dtrain = lgb.Dataset(
            X_train_s,
            label=y_train_s,
            categorical_feature=cat_features if cat_features else "auto",
            free_raw_data=False,
        )

        model = lgb.train(params, dtrain, num_boost_round=num_boost_round)

        # Predict probabilities on test
        test_scores = model.predict(X_test)

        # ✅ FIX 2: page proxy column configurable
        if page_col not in df.columns:
            raise ValueError(f"eval.page_col='{page_col}' not found in df columns.")
        df_test = df.loc[test_idx, [page_col, target_name]].copy()
        df_test = df_test.rename(columns={page_col: "page_id"})
        df_test["score"] = test_scores

        metrics_at_k = precision_recall_at_k_per_page(
            df_test,
            score_col="score",
            y_col=target_name,
            page_col="page_id",
            ks=ks,
            ignore_pages_with_no_positives_for_recall=True,
            ignore_pages_with_no_positives_for_precision=False,
        )

        # Feature importance (gain)
        fi = pd.DataFrame({
            "feature": model.feature_name(),
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False)
        fi_all.append(fi.assign(site=fold.holdout_site))

        all_fold_results.append(FoldResult(site=fold.holdout_site, metrics_at_k=metrics_at_k))

        # ✅ FIX 4: grouped output per fold (cleaner)
        print(f"\nFold {i}/{len(folds)} holdout={fold.holdout_site} target={target_name}")
        for k in ks:
            m = metrics_at_k[k]
            print(
                f"  P@{k}={m['precision']:.3f}  R@{k}={m['recall']:.3f}  "
                f"(pages P:{m['num_pages_precision']} R:{m['num_pages_recall']})"
            )
        print("  Top-10 features (gain):")
        print(fi.head(10).to_string(index=False))

        # Save artifacts
        out_root = Path("runs") / target_name
        out_dir = out_root / fold.holdout_site
        out_dir.mkdir(parents=True, exist_ok=True)

        model.save_model(str(out_dir / "model.txt"))
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {"holdout_site": fold.holdout_site, "target": target_name, "metrics_at_k": metrics_at_k},
                indent=2
            ),
            encoding="utf-8",
        )
        df_test.to_parquet(out_dir / "preds.parquet", index=False)
        fi.to_csv(out_dir / "feature_importance.csv", index=False)

    # Aggregate feature importance across folds
    fi_cat = pd.concat(fi_all, ignore_index=True)
    fi_mean = (
        fi_cat.groupby("feature")[["importance_gain", "importance_split"]]
        .mean()
        .sort_values("importance_gain", ascending=False)
        .reset_index()
    )
    out_root = Path("runs") / target_name
    out_root.mkdir(parents=True, exist_ok=True)
    fi_mean.to_csv(out_root / "feature_importance_mean.csv", index=False)

    print("\n=== Mean Feature Importance across folds (Top 15, gain) ===")
    print(fi_mean.head(15).to_string(index=False))

    # Macro mean over folds
    agg: Dict[int, Dict[str, float]] = {}
    for k in ks:
        ps = [fr.metrics_at_k[k]["precision"] for fr in all_fold_results]
        rs = [fr.metrics_at_k[k]["recall"] for fr in all_fold_results]
        agg[k] = {
            "precision_mean": float(np.mean(ps)),
            "recall_mean": float(np.mean(rs)),
        }

    summary = {
        "target": target_name,
        "num_folds": len(all_fold_results),
        "folds": [{"site": fr.site, "metrics_at_k": fr.metrics_at_k} for fr in all_fold_results],
        "macro_mean": agg,
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# Backward-compatible wrapper (optional)
def train_date_loso(config_path: str = "config.yaml") -> Dict[str, Any]:
    return train_field_loso(config_path=config_path, target_name="is_Date")

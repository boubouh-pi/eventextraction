from __future__ import annotations

import argparse
from typing import List

from dateclf.data import (
    load_config,
    load_raw_merged,
    add_is_field_labels,
    build_feature_matrix_for_field,
)
from dateclf.split import loso_folds
from dateclf.train import train_field_loso


ALL_TARGETS: List[str] = ["is_Date", "is_Time", "is_Location", "is_Name"]


def cmd_data_check(config_path: str) -> int:
    cfg = load_config(config_path)
    df = load_raw_merged(cfg)

    print("OK: merged CSVs")
    print("rows:", len(df))
    print("sites:", df["site_id"].nunique())
    print("columns:", len(df.columns))
    print("first cols:", df.columns.tolist()[:20])

    label_col = cfg["data"]["label_col"]
    if label_col in df.columns:
        vc = df[label_col].astype(str).value_counts().head(20)
        print("\nTop label values:")
        print(vc.to_string())

    # ✅ Raw NaN check (useful early diagnostic)
    nan_total = int(df.isna().sum().sum())
    print(f"\nNaN total in raw df: {nan_total}")
    if nan_total > 0:
        nan_by_col = df.isna().sum()
        nan_cols = nan_by_col[nan_by_col > 0].sort_values(ascending=False)
        print("NaN columns (top):")
        print(nan_cols.head(10).to_string())

    # Strict label mapping -> atomic targets (with warning for unmapped labels)
    df = add_is_field_labels(df, cfg)

    print("\nTarget stats:")
    for t in ALL_TARGETS:
        if t in df.columns:
            pos = int(df[t].sum())
            rate = float(df[t].mean())
            print(f"  {t}: positives={pos} rate={rate:.4f}")

    return 0


def cmd_feature_check(config_path: str, target: str) -> int:
    cfg = load_config(config_path)
    df = load_raw_merged(cfg)
    df = add_is_field_labels(df, cfg)

    X, y, groups = build_feature_matrix_for_field(df, target)

    print(f"OK: feature matrix built ({target})")
    print("X shape:", X.shape)
    print("y positives:", int(y.sum()), "rate:", round(float(y.mean()), 4))
    print("num sites (groups):", groups.nunique())

    print("\nDtypes count:")
    print(X.dtypes.value_counts().to_string())

    obj_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print("\nString/Object columns:", obj_cols)

    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if len(nan_cols) == 0:
        print("\nNaN check: OK (no missing values)")
    else:
        print("\nNaN columns (top):")
        print(nan_cols.head(20).to_string())
        worst = nan_cols.head(10)
        print("\nNaN rate (top):")
        print(((worst / len(X)).round(4)).to_string())

    return 0


def cmd_fold_check(config_path: str, target: str) -> int:
    cfg = load_config(config_path)
    df = load_raw_merged(cfg)
    df = add_is_field_labels(df, cfg)

    folds = list(loso_folds(df, site_col="site_id", target_col=target))
    print(f"Eligible LOSO folds for {target}: {len(folds)}")

    # ✅ Fold diagnostic table
    print(f"\n{'Holdout':<35} {'Train rows':<12} {'Test rows':<12} {'Test positives':<14}")
    for f in folds:
        n_train = len(f.train_idx)
        n_test = len(f.test_idx)
        n_pos = int(df.loc[f.test_idx, target].sum())
        print(f"{f.holdout_site:<35} {n_train:<12} {n_test:<12} {n_pos:<14}")

    return 0


def cmd_train_field(config_path: str, target: str) -> int:
    train_field_loso(config_path=config_path, target_name=target)
    return 0


def cmd_train_all(config_path: str) -> int:
    for target in ALL_TARGETS:
        print(f"\n{'='*60}\nTraining {target}\n{'='*60}")
        ret = cmd_train_field(config_path, target)
        if ret != 0:
            return ret
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dateclf")
    p.add_argument("--config", default="config.yaml")

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("data-check", help="Merge CSVs + label distribution + raw NaN check + target stats")

    p_feat = sub.add_parser("feature-check", help="Build X/y/groups for a target and run sanity checks")
    p_feat.add_argument("--target", required=True, choices=ALL_TARGETS)

    p_fold = sub.add_parser("fold-check", help="List eligible LOSO folds + fold sizes + #positives in test")
    p_fold.add_argument("--target", required=True, choices=ALL_TARGETS)

    p_train = sub.add_parser("train-field", help="Train a single field with LOSO and save artifacts")
    p_train.add_argument("--target", required=True, choices=ALL_TARGETS)

    sub.add_parser("train-all", help="Train all fields sequentially (Date/Time/Location/Name)")

    # Backward-compatible commands (documented)
    sub.add_parser("train-date", help="[deprecated] alias for train-field --target is_Date")
    sub.add_parser("fold-check-date", help="[deprecated] alias for fold-check --target is_Date")

    args = p.parse_args(argv)

    if args.command == "data-check":
        return cmd_data_check(args.config)

    if args.command == "feature-check":
        return cmd_feature_check(args.config, args.target)

    if args.command == "fold-check":
        return cmd_fold_check(args.config, args.target)

    if args.command == "train-field":
        return cmd_train_field(args.config, args.target)

    if args.command == "train-all":
        return cmd_train_all(args.config)

    # Back-compat
    if args.command == "train-date":
        return cmd_train_field(args.config, "is_Date")

    if args.command == "fold-check-date":
        return cmd_fold_check(args.config, "is_Date")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

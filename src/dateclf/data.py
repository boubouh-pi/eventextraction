from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings

import pandas as pd
import yaml


# -------------------------
# Config + data loading
# -------------------------

def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_site_id_from_filename(path: Path) -> str:
    return path.stem


def load_raw_merged(config: Dict[str, Any]) -> pd.DataFrame:
    dcfg = config["data"]
    raw_dir = Path(dcfg["raw_dir"])
    files = sorted(raw_dir.glob(dcfg.get("file_glob", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir.resolve()}")

    sep = dcfg.get("sep", None)
    if not sep or sep == "TO_FILL":
        raise ValueError("Please set data.sep in config.yaml (your CSV separator).")

    encoding = dcfg.get("encoding", "utf-8")
    site_source = dcfg.get("site_id_source", "filename")

    dfs: List[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp, sep=sep, encoding=encoding, engine="python")

        if site_source == "filename":
            df["site_id"] = _infer_site_id_from_filename(fp)
        elif site_source == "column":
            if "site_id" not in df.columns:
                raise ValueError(f"'site_id' column missing in {fp.name}")
            df["site_id"] = df["site_id"].astype(str)
        else:
            raise ValueError("site_id_source must be 'filename' or 'column'")

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["site_id"] = out["site_id"].astype(str)
    return out


# -------------------------
# Date-specific (legacy compatibility)
# -------------------------
import warnings
def add_is_date_label(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    warnings.warn(
        "add_is_date_label() is deprecated. Use add_is_field_labels() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    """
    Create binary target is_Date.
    Rule: is_Date = 1 iff label contains 'date' (case-insensitive).
    """
    label_col = config["data"]["label_col"]
    if label_col not in df.columns:
        raise ValueError(
            f"label_col='{label_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    labels = df[label_col].astype(str).fillna("").str.lower()

    df = df.copy()
    df["is_Date"] = labels.str.contains("date").astype(int)
    return df


def build_feature_matrix_for_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X, y, groups for Date classifier.
    - X: features only (no leakage columns)
    - y: binary target is_Date
    - groups: site_id (for LOSO splitting, not a feature)
    """
    exclude_cols = {
        "label",
        "is_Date",
        "site_id",
        "event_id",
        "attributes",
        "text_context",
        "link",
    }

    cols_to_drop = [c for c in exclude_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    if "is_Date" not in df.columns:
        raise ValueError("is_Date missing. Call add_is_date_label() first.")
    y = df["is_Date"].astype(int)

    if "site_id" not in df.columns:
        raise ValueError("site_id missing. Ensure loader adds site_id.")
    groups = df["site_id"].astype(str)

    return X, y, groups


# =========================
# Generic targets + features (SAFE)
# =========================

ALL_TARGET_COLS = ["is_Date", "is_Time", "is_Location", "is_Name"]

# Columns never allowed in X (leakage / too free-form)
BASE_EXCLUDE_COLS = frozenset([
    "label", "site_id", "event_id", "attributes", "text_context", "link",
])


# Pattern features per field (avoid cross-field shortcuts)
FIELD_PATTERN_FEATURES: Dict[str, List[str]] = {
    "is_Date": ["contains_date"],
    "is_Time": ["contains_time"],
    "is_Location": [],   # add contains_location if you create it later
    "is_Name": [],
}

# Explicit mapping for labels (equality strict, auditable)
LABEL_TO_FIELDS: Dict[str, List[str]] = {
    "date": ["is_Date"],
    "time": ["is_Time"],
    "location": ["is_Location"],
    "name": ["is_Name"],
    "other": [],
    "description": [],

    # composites (add any others you observe in your corpus)
    "datetime": ["is_Date", "is_Time"],
    "starttime": ["is_Time"],
    "endtime": ["is_Time"],
    "startendtime": ["is_Time"],
    "timelocation": ["is_Time", "is_Location"],
    "namelocation": ["is_Name", "is_Location"],
    "namelink": ["is_Name"],

    # if you want:
    # "description": [],
}


def add_is_field_labels(df: pd.DataFrame, config: Dict[str, Any], label_col: str | None = None) -> pd.DataFrame:
    """
    Create all atomic targets from the original label column using strict equality + mapping.
    This avoids substring false positives (e.g., 'lifetime' -> 'time').

    - label column is configurable via config["data"]["label_col"] by default.
    - unknown labels trigger a warning (auditability).
    """
    if label_col is None:
        label_col = config["data"]["label_col"]

    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found in df columns.")

    df = df.copy()

    # init targets to 0
    for t in ALL_TARGET_COLS:
        df[t] = 0

    labels = df[label_col].astype(str).fillna("").str.lower()

    for raw_label, fields in LABEL_TO_FIELDS.items():
        mask = labels == raw_label
        if mask.any():
            for field in fields:
                df.loc[mask, field] = 1

    # audit: unknown labels
    known = set(LABEL_TO_FIELDS.keys())
    observed = set(labels.unique())
    unknown = observed - known - {"", "nan", "none"}

    if unknown:
        warnings.warn(f"Unmapped labels detected (please review LABEL_TO_FIELDS): {sorted(unknown)}")

    return df


def get_feature_cols_for_field(df: pd.DataFrame, target_name: str) -> List[str]:
    """
    Build safe feature list:
    - excludes leakage cols + ALL target cols
    - keeps only the pattern feature(s) for this target_name
    """
    if target_name not in ALL_TARGET_COLS:
        raise ValueError(f"Unknown target_name={target_name}")

    exclude = set(BASE_EXCLUDE_COLS) | set(ALL_TARGET_COLS)

    candidate = [c for c in df.columns if c not in exclude]

    # Remove pattern features not belonging to this field
    all_pattern_feats = set(sum(FIELD_PATTERN_FEATURES.values(), []))
    allowed = set(FIELD_PATTERN_FEATURES.get(target_name, []))

    out: List[str] = []
    for c in candidate:
        if c in all_pattern_feats and c not in allowed:
            continue
        out.append(c)

    return out


def build_feature_matrix_for_field(df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return X, y, groups for any target.
    groups = site_id (for LOSO), NOT in X.
    """
    if target_name not in ALL_TARGET_COLS:
        raise ValueError(f"Unknown target_name={target_name}")
    if "site_id" not in df.columns:
        raise ValueError("site_id missing (needed for LOSO).")
    if target_name not in df.columns:
        raise ValueError(f"{target_name} missing. Call add_is_field_labels() first.")

    feature_cols = get_feature_cols_for_field(df, target_name)

    X = df[feature_cols].copy()
    y = df[target_name].astype(int).copy()
    groups = df["site_id"].astype(str).copy()

    # Hard anti-leakage assertions
    for t in ALL_TARGET_COLS:
        if t in X.columns:
            raise AssertionError(f"Leakage: {t} present in X")
    for c in ["label", "site_id", "event_id"]:
        if c in X.columns:
            raise AssertionError(f"Leakage: {c} present in X")

    return X, y, groups

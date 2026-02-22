from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List
import pandas as pd


@dataclass(frozen=True)
class Fold:
    holdout_site: str
    train_idx: pd.Index
    test_idx: pd.Index


def loso_folds(df: pd.DataFrame, site_col: str = "site_id", target_col: str = "") -> Iterator[Fold]:
    """
    Generic Leave-One-Site-Out folds.

    - Requires target_col explicitly (e.g. "is_Date", "is_Time").
    - Only yields folds where the holdout site has at least 1 positive for target_col.
    - Assumes df index is reset (0..n-1); raises if not.
    """
    if not target_col:
        raise ValueError("target_col must be provided explicitly (e.g., 'is_Date', 'is_Time').")

    if site_col not in df.columns:
        raise ValueError(f"{site_col} not found in DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in DataFrame.")

    # Defensive: avoid weird indices after filtering
    if not (len(df) > 0 and df.index.is_monotonic_increasing and df.index[0] == 0):
        raise ValueError("df index must be reset before calling loso_folds(). Use df.reset_index(drop=True).")

    pos_by_site = df.groupby(site_col)[target_col].sum()
    eligible_sites: List[str] = sorted(pos_by_site[pos_by_site > 0].index.astype(str).tolist())

    for s in eligible_sites:
        test_mask = df[site_col].astype(str) == str(s)
        yield Fold(
            holdout_site=str(s),
            train_idx=df.index[~test_mask],
            test_idx=df.index[test_mask],
        )


# Backward-compatible wrapper (optional)
def loso_folds_date(df: pd.DataFrame, site_col: str = "site_id") -> Iterator[Fold]:
    return loso_folds(df, site_col=site_col, target_col="is_Date")

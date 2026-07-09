"""
Sweep the LLM-fallback confidence threshold offline (zero LLM cost).

Consumes runs/event_confidences.csv (written by eval_fallback.py): one row
per ground-truth event with its cluster confidence and whether the ML
pipeline reconstituted it perfectly.

For each threshold t, an event escalates when its confidence < t:
- gate recall  = escalated ML-failures / all ML-failures  (what the LLM
  gets a chance to fix — the benefit)
- waste        = escalated ML-successes / all ML-successes (already-perfect
  events sent to the LLM — pure cost)

The recommended threshold maximises (gate_recall − escalated_share): the
benefit is proportional to recoverable failures, the LLM bill to escalated
events. Some failures are structurally confident (contradictory labelings)
and unreachable at any threshold below 0.85 — chasing them means
escalating everything.

Run:  python scripts/tune_threshold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = REPO_ROOT / "runs" / "event_confidences.csv"
    if not path.exists():
        sys.exit("Run scripts/eval_fallback.py first (writes event_confidences.csv)")

    df = pd.read_csv(path)
    failures = df[~df["ml_perfect"]]
    successes = df[df["ml_perfect"]]
    print(f"{len(df)} events: {len(successes)} ML-perfect, {len(failures)} failures\n")

    rows = []
    for t in np.arange(0.05, 1.0, 0.05):
        esc_fail = (failures["confidence"] < t).mean() if len(failures) else 0.0
        esc_succ = (successes["confidence"] < t).mean() if len(successes) else 0.0
        share = (df["confidence"] < t).mean()
        rows.append({
            "threshold": round(t, 2),
            "gate_recall": round(esc_fail, 3),
            "waste": round(esc_succ, 3),
            "escalated_share": round(share, 3),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    table["net"] = table["gate_recall"] - table["escalated_share"]
    rec = table.loc[table["net"].idxmax()]
    print(f"\nRecommended threshold: {rec['threshold']} "
          f"(gate recall {rec['gate_recall']:.0%}, waste {rec['waste']:.0%}, "
          f"escalates {rec['escalated_share']:.0%} of events)")
    table.to_csv(REPO_ROOT / "runs" / "threshold_sweep.csv", index=False)


if __name__ == "__main__":
    main()

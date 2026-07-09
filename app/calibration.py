"""
Platt scaling for LightGBM scores.

LightGBM trained with ``scale_pos_weight`` + undersampling outputs scores
whose *ranking* is good but whose absolute values are not probabilities
(the class prior was distorted twice). The LLM-fallback trigger compares
scores to a threshold, so they must be calibrated first.

Platt scaling fits a 1-D logistic regression ``P(y=1 | s) = sigmoid(a*s + b)``
on **out-of-fold** scores (LOSO folds), so the calibrator never sees scores
produced by a model that saw the same site — mirroring production conditions
where every incoming site is unseen.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self) -> None:
        self._lr = LogisticRegression(solver="lbfgs")
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        y_true = np.asarray(y_true, dtype=int)
        self._lr.fit(raw_scores, y_true)
        self.fitted = True
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        if not self.fitted:
            # Identity fallback keeps the pipeline usable pre-calibration.
            return np.clip(raw_scores.ravel(), 0.0, 1.0)
        return self._lr.predict_proba(raw_scores)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PlattCalibrator":
        return joblib.load(path)

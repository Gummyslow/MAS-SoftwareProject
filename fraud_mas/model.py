"""Ensemble model training, inference, and threshold logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from fraud_mas.config import FRAUD_THRESHOLD_HIGH, FRAUD_THRESHOLD_LOW, XGB_PARAMS
from fraud_mas.data_io import load_model, save_model
from fraud_mas.features import FEATURE_COLS

AGENT_SCORE_COLS = ["behav_score", "geo_score", "nlp_score", "net_score"]

ALL_FEATURES = FEATURE_COLS + AGENT_SCORE_COLS


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def train(df: pd.DataFrame, label_col: str = "label") -> XGBClassifier:
    feat_cols = _available(df, ALL_FEATURES)
    X = df[feat_cols].fillna(0)
    y = df[label_col]

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    save_model(model)
    return model


def predict_proba(df: pd.DataFrame, model: XGBClassifier | None = None) -> np.ndarray:
    if model is None:
        model = load_model()
    feat_cols = _available(df, ALL_FEATURES)
    X = df[feat_cols].fillna(0)
    return model.predict_proba(X)[:, 1]


def apply_thresholds(scores: np.ndarray) -> list[str]:
    """
    Returns 'fraud', 'legit', or 'review' for each score.
    'review' rows are sent to the LLM orchestrator.
    """
    labels = []
    for s in scores:
        if s >= FRAUD_THRESHOLD_HIGH:
            labels.append("fraud")
        elif s <= FRAUD_THRESHOLD_LOW:
            labels.append("legit")
        else:
            labels.append("review")
    return labels


def evaluate(df: pd.DataFrame, scores: np.ndarray, label_col: str = "label") -> dict:
    y_true = df[label_col].values
    y_pred = (scores >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, scores), 4),
        "f1":      round(f1_score(y_true, y_pred), 4),
    }

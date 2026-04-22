"""Ensemble model training, inference, and threshold logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from fraud_mas.config import FEATURE_COLS_PATH, FRAUD_THRESHOLD_HIGH, FRAUD_THRESHOLD_LOW, XGB_PARAMS
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

    import pickle
    FEATURE_COLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(feat_cols, f)

    return model


def _load_train_feature_cols() -> list[str] | None:
    import pickle
    if FEATURE_COLS_PATH.exists() and FEATURE_COLS_PATH.stat().st_size > 0:
        with open(FEATURE_COLS_PATH, "rb") as f:
            return pickle.load(f)
    return None


_RULE_WEIGHTS = {
    "card_test_score":     0.20,
    "amount_zscore":       0.08,
    "behav_score":         0.22,
    "geo_score":           0.15,
    "nlp_score":           0.12,
    "net_score":           0.13,
    "is_night":            0.04,
    "new_recipient":       0.06,
}


def rule_based_score(df: pd.DataFrame) -> np.ndarray:
    """Weighted rule-based ensemble — used when no trained model is available (Level 1)."""
    scores = np.zeros(len(df))
    total_weight = 0.0
    for col, w in _RULE_WEIGHTS.items():
        if col in df.columns:
            scores += df[col].fillna(0).clip(0, 1).values * w
            total_weight += w
    if total_weight > 0:
        scores /= total_weight
    return scores.clip(0, 1)


def predict_proba(df: pd.DataFrame, model: XGBClassifier | None = None) -> np.ndarray:
    if model is None:
        model = load_model()

    if model is None:
        # No trained model — fall back to rule-based weighted ensemble (Level 1)
        return rule_based_score(df)

    # Determine the exact feature columns the model expects, in order
    train_cols = _load_train_feature_cols()
    if train_cols is None:
        booster_names = model.get_booster().feature_names
        if booster_names:
            train_cols = list(booster_names)

    if train_cols is not None:
        df = df.copy()
        for col in train_cols:
            if col not in df.columns:
                df[col] = 0.0
        X = df[train_cols].fillna(0)
    else:
        X = df[_available(df, ALL_FEATURES)].fillna(0)

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

"""Agent 1 – Statistical feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fraud_mas.config import AMOUNT_LOG_SHIFT, TOP_N_COUNTRIES, TOP_N_MERCHANTS


def engineer_features(df: pd.DataFrame, encoders: dict | None = None, fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Build a feature matrix from raw transaction data.

    Returns
    -------
    features : pd.DataFrame
    encoders : dict  (label encoders fitted on training data)
    """
    df = df.copy()
    encoders = encoders or {}

    # --- Amount features ---
    df["amount_log"] = np.log1p(df["amount"] + AMOUNT_LOG_SHIFT)
    df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9)

    # --- Time features ---
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"])
        df["hour"]      = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]   = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # --- User-level aggregates (within dataset) ---
    user_stats = df.groupby("user_id")["amount"].agg(
        user_tx_count="count",
        user_mean_amount="mean",
        user_std_amount="std",
    ).reset_index()
    df = df.merge(user_stats, on="user_id", how="left")
    df["user_std_amount"] = df["user_std_amount"].fillna(0)
    df["amount_vs_user_mean"] = df["amount"] / (df["user_mean_amount"] + 1e-9)

    # --- Categorical encoding ---
    cat_cols = ["merchant", "country", "device_type", "payment_method"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        if col == "merchant":
            top = df[col].value_counts().nlargest(TOP_N_MERCHANTS).index
        elif col == "country":
            top = df[col].value_counts().nlargest(TOP_N_COUNTRIES).index
        else:
            top = df[col].value_counts().index

        df[col] = df[col].where(df[col].isin(top), other="__other__")

        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                df[col + "_enc"] = 0
            else:
                known = set(le.classes_)
                df[col] = df[col].where(df[col].isin(known), other="__other__")
                if "__other__" not in known:
                    df[col + "_enc"] = 0
                else:
                    df[col + "_enc"] = le.transform(df[col].astype(str))

    return df, encoders


FEATURE_COLS = [
    "amount_log", "amount_zscore", "amount_vs_user_mean",
    "hour", "day_of_week", "is_weekend", "is_night",
    "user_tx_count", "user_mean_amount", "user_std_amount",
    "merchant_enc", "country_enc", "device_type_enc", "payment_method_enc",
]


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].fillna(0)

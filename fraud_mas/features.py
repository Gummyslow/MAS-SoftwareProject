"""Agent 1 – Statistical feature engineering (challenge column names)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fraud_mas.config import AMOUNT_LOG_SHIFT, TOP_N_COUNTRIES, TOP_N_MERCHANTS


def _rolling_tx_count(df: pd.DataFrame, sender_col: str, hours: int) -> pd.Series:
    """Vectorised rolling transaction count per sender over the past `hours` hours."""
    df = df.sort_values([sender_col, "timestamp"])
    ts = pd.to_datetime(df["timestamp"])
    result = pd.Series(0, index=df.index)
    cutoff = pd.Timedelta(hours=hours)
    for sid, grp in df.groupby(sender_col):
        t = ts.loc[grp.index]
        counts = []
        for i, (idx, ti) in enumerate(zip(grp.index, t)):
            counts.append(int((t.iloc[:i] >= ti - cutoff).sum()))
        result.loc[grp.index] = counts
    return result


def engineer_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = encoders or {}

    # ── identify the sender column ──────────────────────────────────────────
    sender_col    = "sender_id"    if "sender_id"    in df.columns else None
    recipient_col = "recipient_id" if "recipient_id" in df.columns else None

    # ── amount features ─────────────────────────────────────────────────────
    df["amount_log"] = np.log1p(df["amount"] + AMOUNT_LOG_SHIFT)

    if sender_col:
        sender_stats = df.groupby(sender_col)["amount"].agg(
            sender_mean="mean",
            sender_std="std",
            sender_median="median",
            sender_max="max",
            sender_count="count",
        ).fillna(0).reset_index()
        df = df.merge(sender_stats, on=sender_col, how="left")
        df["sender_std"]   = df["sender_std"].fillna(0)
        df["amount_zscore"] = (df["amount"] - df["sender_mean"]) / (df["sender_std"] + 1e-6)
        df["amount_vs_user_mean"] = df["amount"] / (df["sender_mean"] + 1e-9)
        df["is_personal_record"]  = (df["amount"] == df["sender_max"]).astype(int)
        # alias for model compat
        df["user_tx_count"]    = df["sender_count"]
        df["user_mean_amount"] = df["sender_mean"]
        df["user_std_amount"]  = df["sender_std"]
    else:
        df["amount_zscore"]       = 0.0
        df["amount_vs_user_mean"] = 1.0
        df["is_personal_record"]  = 0
        df["user_tx_count"]       = 1
        df["user_mean_amount"]    = df["amount"]
        df["user_std_amount"]     = 0.0

    if "balance_after" in df.columns:
        df["balance_ratio"] = df["amount"] / (df["balance_after"].abs() + 1e-6)
    else:
        df["balance_ratio"] = 0.0

    # ── temporal features ───────────────────────────────────────────────────
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"])
        df["hour"]        = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["month"]       = dt.dt.month
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]    = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
        df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    else:
        for c in ["hour", "day_of_week", "month", "is_weekend", "is_night", "hour_sin", "hour_cos"]:
            df[c] = 0

    # ── velocity features ───────────────────────────────────────────────────
    if sender_col and "timestamp" in df.columns:
        df["tx_count_1h"]  = _rolling_tx_count(df, sender_col, 1)
        df["tx_count_24h"] = _rolling_tx_count(df, sender_col, 24)
    else:
        df["tx_count_1h"]  = 0
        df["tx_count_24h"] = 0

    # ── new recipient flag ──────────────────────────────────────────────────
    if sender_col and recipient_col:
        seen: set = set()
        flags = []
        for _, row in df.sort_values("timestamp").iterrows() if "timestamp" in df.columns else df.iterrows():
            pair = (row[sender_col], row[recipient_col])
            flags.append(0 if pair in seen else 1)
            seen.add(pair)
        df["new_recipient"] = flags
    else:
        df["new_recipient"] = 0

    # ── card-test score ─────────────────────────────────────────────────────
    if sender_col and "timestamp" in df.columns:
        df = df.sort_values([sender_col, "timestamp"])
        df["_prev_amount"] = df.groupby(sender_col)["amount"].shift(1)
        df["_prev_ts"]     = pd.to_datetime(df.groupby(sender_col)["timestamp"].shift(1).values)
        df["_time_min"]    = (pd.to_datetime(df["timestamp"]) - df["_prev_ts"]).dt.total_seconds() / 60
        df["_time_min"]    = df["_time_min"].fillna(9999)

        def _card_test(row):
            if pd.isna(row["_prev_amount"]):
                return 0.0
            ratio   = row["amount"] / (row["_prev_amount"] + 1e-6)
            time_ok = 0 < row["_time_min"] < 60
            if row["_prev_amount"] < 10 and ratio > 50 and time_ok:
                return 1.0
            if row["_prev_amount"] < 10 and ratio > 20 and time_ok:
                return 0.7
            if ratio > 10 and time_ok:
                return 0.3
            return 0.0

        df["card_test_score"] = df.apply(_card_test, axis=1)
        df.drop(columns=["_prev_amount", "_prev_ts", "_time_min"], inplace=True)
    else:
        df["card_test_score"] = 0.0

    # ── categorical encoding ────────────────────────────────────────────────
    cat_cols = {
        "transaction_type": "tx_type_enc",
        "payment_method":   "payment_method_enc",
        "merchant":         "merchant_enc",
        "country":          "country_enc",
        "device_type":      "device_type_enc",
    }
    for col, out_col in cat_cols.items():
        if col not in df.columns:
            df[out_col] = 0
            continue
        if fit:
            le = LabelEncoder()
            df[out_col] = le.fit_transform(df[col].fillna("unknown").astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                df[out_col] = 0
            else:
                known = set(le.classes_)
                mapped = df[col].fillna("unknown").astype(str).where(
                    df[col].fillna("unknown").astype(str).isin(known), other="unknown"
                )
                if "unknown" not in known:
                    df[out_col] = 0
                else:
                    df[out_col] = le.transform(mapped)

    return df, encoders


FEATURE_COLS = [
    "amount_log", "amount_zscore", "balance_ratio", "amount_vs_user_mean",
    "is_personal_record",
    "hour", "hour_sin", "hour_cos", "day_of_week", "month",
    "is_weekend", "is_night",
    "tx_count_1h", "tx_count_24h",
    "new_recipient",
    "card_test_score",
    "user_tx_count", "user_mean_amount", "user_std_amount",
    "tx_type_enc", "payment_method_enc",
]


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].fillna(0)

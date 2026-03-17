"""Agent 2 – Behavioral pattern analysis."""

import pandas as pd
import numpy as np


def compute_behavioral_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds behavioral risk columns to df (in-place copy).

    Signals
    -------
    behav_velocity_1h   : # transactions by same user in the last hour
    behav_velocity_24h  : # transactions by same user in the last 24 hours
    behav_new_merchant  : 1 if this user has never used this merchant before
    behav_large_jump    : 1 if amount > 3× user's historical mean
    behav_score         : composite [0, 1]
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("_ts")

        vel_1h, vel_24h = [], []
        for idx, row in df.iterrows():
            user_mask = df["user_id"] == row["user_id"]
            past_1h  = ((row["_ts"] - df.loc[user_mask, "_ts"]).dt.total_seconds().between(0, 3600))
            past_24h = ((row["_ts"] - df.loc[user_mask, "_ts"]).dt.total_seconds().between(0, 86400))
            vel_1h.append(int(past_1h.sum()) - 1)   # exclude self
            vel_24h.append(int(past_24h.sum()) - 1)

        df["behav_velocity_1h"]  = [max(0, v) for v in vel_1h]
        df["behav_velocity_24h"] = [max(0, v) for v in vel_24h]
        df.drop(columns=["_ts"], inplace=True)
    else:
        df["behav_velocity_1h"]  = 0
        df["behav_velocity_24h"] = 0

    # New-merchant signal
    if "merchant" in df.columns:
        seen: dict[str, set] = {}
        new_merchant_flags = []
        for _, row in df.iterrows():
            uid, mer = row["user_id"], row["merchant"]
            if uid not in seen:
                seen[uid] = set()
            new_merchant_flags.append(int(mer not in seen[uid]))
            seen[uid].add(mer)
        df["behav_new_merchant"] = new_merchant_flags
    else:
        df["behav_new_merchant"] = 0

    # Large jump signal
    user_mean = df.groupby("user_id")["amount"].transform("mean")
    df["behav_large_jump"] = (df["amount"] > 3 * user_mean).astype(int)

    # Composite score
    df["behav_score"] = (
        0.30 * np.clip(df["behav_velocity_1h"] / 5, 0, 1) +
        0.20 * np.clip(df["behav_velocity_24h"] / 20, 0, 1) +
        0.25 * df["behav_new_merchant"] +
        0.25 * df["behav_large_jump"]
    )

    return df

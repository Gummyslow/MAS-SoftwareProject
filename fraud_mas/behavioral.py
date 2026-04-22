"""Agent 2 – Behavioral pattern analysis (vectorised, challenge column names)."""

import pandas as pd
import numpy as np


def compute_behavioral_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signals
    -------
    behav_velocity_1h   : tx count by same sender in last hour
    behav_velocity_24h  : tx count by same sender in last 24h
    behav_new_recipient : 1 if sender never sent to this recipient before
    behav_large_jump    : 1 if amount > 3× sender historical mean
    burst_score         : card-test + large-jump composite
    behav_score         : composite [0, 1]
    """
    df = df.copy()

    sender_col    = "sender_id"    if "sender_id"    in df.columns else None
    recipient_col = "recipient_id" if "recipient_id" in df.columns else None

    # ── velocity (vectorised rolling) ───────────────────────────────────────
    if sender_col and "timestamp" in df.columns:
        df = df.sort_values([sender_col, "timestamp"]).copy()
        ts = pd.to_datetime(df["timestamp"])

        vel_1h, vel_24h = [], []
        for sid, grp in df.groupby(sender_col):
            t = ts.loc[grp.index]
            for i, ti in enumerate(t):
                vel_1h.append(int((t.iloc[:i] >= ti - pd.Timedelta(hours=1)).sum()))
                vel_24h.append(int((t.iloc[:i] >= ti - pd.Timedelta(hours=24)).sum()))

        df["behav_velocity_1h"]  = vel_1h
        df["behav_velocity_24h"] = vel_24h
    else:
        df["behav_velocity_1h"]  = 0
        df["behav_velocity_24h"] = 0

    # ── new-recipient flag ───────────────────────────────────────────────────
    if sender_col and recipient_col:
        seen: set = set()
        flags = []
        iter_df = df.sort_values("timestamp") if "timestamp" in df.columns else df
        for _, row in iter_df.iterrows():
            pair = (row[sender_col], row[recipient_col])
            flags.append(int(pair not in seen))
            seen.add(pair)
        # align back to original index order
        new_recip = pd.Series(flags, index=iter_df.index)
        df["behav_new_recipient"] = new_recip.reindex(df.index).fillna(0).astype(int)
    else:
        df["behav_new_recipient"] = 0

    # ── large-jump ───────────────────────────────────────────────────────────
    if sender_col:
        sender_mean = df.groupby(sender_col)["amount"].transform("mean")
        df["behav_large_jump"] = (df["amount"] > 3 * sender_mean).astype(int)
    else:
        df["behav_large_jump"] = 0

    # ── burst score (many tx in 1h) ──────────────────────────────────────────
    df["burst_score"] = np.clip(df["behav_velocity_1h"] / 5, 0, 1)

    # ── composite score ──────────────────────────────────────────────────────
    df["behav_score"] = (
        0.30 * np.clip(df["behav_velocity_1h"]  / 5,  0, 1) +
        0.15 * np.clip(df["behav_velocity_24h"] / 20, 0, 1) +
        0.30 * df["behav_new_recipient"] +
        0.25 * df["behav_large_jump"]
    ).clip(0, 1)

    return df

"""Agent 5 – Graph/network fraud-ring detection (challenge column names)."""

import pandas as pd
import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


def _build_graph(df: pd.DataFrame) -> "nx.DiGraph":
    """Directed graph: sender_id → recipient_id, weighted by tx count."""
    G = nx.DiGraph()
    sender_col    = "sender_id"    if "sender_id"    in df.columns else None
    recipient_col = "recipient_id" if "recipient_id" in df.columns else None

    if sender_col and recipient_col:
        for _, row in df.iterrows():
            s, r = row[sender_col], row[recipient_col]
            if G.has_edge(s, r):
                G[s][r]["weight"] += 1
            else:
                G.add_edge(s, r, weight=1)

    # Also add shared-attribute edges (device/ip from users.json if present)
    for field in ["device_id", "ip_address", "email", "phone"]:
        if field not in df.columns or sender_col is None:
            continue
        groups = df.groupby(field)[sender_col].apply(set)
        for users in groups:
            users = list(users)
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    if not G.has_edge(users[i], users[j]):
                        G.add_edge(users[i], users[j], weight=1)
                    else:
                        G[users[i]][users[j]]["weight"] += 1

    return G


def compute_network_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signals
    -------
    net_shared_device    : # others sharing same device_id (0 if not in data)
    net_shared_ip        : # others sharing same ip_address (0 if not in data)
    net_degree           : out-degree in transaction graph (# unique recipients)
    net_new_node_score   : 1 if sender has very few prior transactions
    net_community_risk   : fraction of sender's recipients who also receive from many senders
    net_score            : composite [0, 1]
    """
    df = df.copy()
    sender_col    = "sender_id"    if "sender_id"    in df.columns else None
    recipient_col = "recipient_id" if "recipient_id" in df.columns else None

    # ── shared device/ip ─────────────────────────────────────────────────────
    for field, out_col in [("device_id", "net_shared_device"), ("ip_address", "net_shared_ip")]:
        if field in df.columns and sender_col:
            counts = df.groupby(field)[sender_col].transform("nunique") - 1
            df[out_col] = counts.clip(lower=0)
        else:
            df[out_col] = 0

    # ── transaction graph metrics ─────────────────────────────────────────────
    if sender_col and recipient_col:
        # out-degree: unique recipients per sender
        out_deg = df.groupby(sender_col)[recipient_col].nunique()
        df["net_degree"] = df[sender_col].map(out_deg).fillna(0).astype(int)

        # in-degree of recipient: how many distinct senders send to same recipient
        in_deg = df.groupby(recipient_col)[sender_col].nunique()
        df["_recip_in_deg"] = df[recipient_col].map(in_deg).fillna(0)

        # community risk: mean in-degree of all recipients of this sender
        df["net_community_risk"] = df.groupby(sender_col)["_recip_in_deg"].transform("mean")
        df["net_community_risk"] = np.clip(df["net_community_risk"] / 10, 0, 1)
        df.drop(columns=["_recip_in_deg"], inplace=True)

        # new-node score: senders with very few total transactions
        sender_tx_count = df.groupby(sender_col)[recipient_col].transform("count")
        df["net_new_node_score"] = (sender_tx_count <= 2).astype(float)
    else:
        df["net_degree"]          = 0
        df["net_community_risk"]  = 0.0
        df["net_new_node_score"]  = 0.0

    df["net_score"] = (
        0.20 * np.clip(df["net_shared_device"] / 5, 0, 1) +
        0.20 * np.clip(df["net_shared_ip"] / 5,     0, 1) +
        0.20 * df["net_community_risk"] +
        0.20 * np.clip(df["net_degree"] / 10,        0, 1) +
        0.20 * df["net_new_node_score"]
    ).clip(0, 1)

    return df

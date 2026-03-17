"""Agent 5 – Graph/network fraud-ring detection."""

import pandas as pd
import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


def _build_graph(df: pd.DataFrame) -> "nx.Graph":
    G = nx.Graph()
    shared_fields = ["ip_address", "device_id", "email", "phone"]

    for field in shared_fields:
        if field not in df.columns:
            continue
        groups = df.groupby(field)["user_id"].apply(set)
        for users in groups:
            users = list(users)
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    if G.has_edge(users[i], users[j]):
                        G[users[i]][users[j]]["weight"] += 1
                    else:
                        G.add_edge(users[i], users[j], weight=1)

    return G


def compute_network_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds network risk columns to df.

    Signals
    -------
    net_shared_device    : # other users sharing same device_id
    net_shared_ip        : # other users sharing same ip_address
    net_community_risk   : fraction of neighbors flagged (if labels available)
    net_degree           : degree in the shared-attribute graph
    net_score            : composite [0, 1]
    """
    df = df.copy()

    for field, out_col in [("device_id", "net_shared_device"), ("ip_address", "net_shared_ip")]:
        if field in df.columns:
            counts = df.groupby(field)["user_id"].transform("nunique") - 1
            df[out_col] = counts.clip(lower=0)
        else:
            df[out_col] = 0

    if _HAS_NX:
        G = _build_graph(df)
        user_degree = dict(G.degree())
        df["net_degree"] = df["user_id"].map(user_degree).fillna(0).astype(int)
    else:
        df["net_degree"] = 0

    df["net_community_risk"] = 0.0   # placeholder; filled if labels are propagated

    df["net_score"] = (
        0.30 * np.clip(df["net_shared_device"] / 5, 0, 1) +
        0.30 * np.clip(df["net_shared_ip"] / 5, 0, 1) +
        0.20 * df["net_community_risk"] +
        0.20 * np.clip(df["net_degree"] / 10, 0, 1)
    )

    return df

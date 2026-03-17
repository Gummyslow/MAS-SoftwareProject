"""Agent 3 – Geolocation anomaly detection."""

import math
import pandas as pd


_HIGH_RISK_COUNTRIES = {
    "NG", "RU", "CN", "KP", "IR", "VE", "BY", "UA", "RO", "PK",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def compute_geo_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds geo risk columns to df.

    Signals
    -------
    geo_high_risk_country : 1 if country ISO2 in high-risk set
    geo_distance_km       : km from user's most common location (requires lat/lon)
    geo_impossible_travel : 1 if distance / time implies >900 km/h between consecutive txs
    geo_score             : composite [0, 1]
    """
    df = df.copy()

    # High-risk country
    if "country" in df.columns:
        df["geo_high_risk_country"] = df["country"].str.upper().isin(_HIGH_RISK_COUNTRIES).astype(int)
    else:
        df["geo_high_risk_country"] = 0

    has_coords = "latitude" in df.columns and "longitude" in df.columns

    if has_coords:
        # Distance from user's median home location
        user_home = df.groupby("user_id")[["latitude", "longitude"]].median().rename(
            columns={"latitude": "home_lat", "longitude": "home_lon"}
        )
        df = df.merge(user_home, on="user_id", how="left")
        df["geo_distance_km"] = df.apply(
            lambda r: _haversine_km(r["home_lat"], r["home_lon"], r["latitude"], r["longitude"]),
            axis=1,
        )

        # Impossible travel between consecutive transactions
        if "timestamp" in df.columns:
            df["_ts"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(["user_id", "_ts"])
            df["_prev_lat"] = df.groupby("user_id")["latitude"].shift(1)
            df["_prev_lon"] = df.groupby("user_id")["longitude"].shift(1)
            df["_prev_ts"]  = df.groupby("user_id")["_ts"].shift(1)

            def impossible_travel(row):
                if pd.isna(row["_prev_lat"]):
                    return 0
                hours = (row["_ts"] - row["_prev_ts"]).total_seconds() / 3600
                if hours < 0.01:
                    return 1
                dist = _haversine_km(row["_prev_lat"], row["_prev_lon"], row["latitude"], row["longitude"])
                return int(dist / hours > 900)

            df["geo_impossible_travel"] = df.apply(impossible_travel, axis=1)
            df.drop(columns=["_ts", "_prev_lat", "_prev_lon", "_prev_ts"], inplace=True)
        else:
            df["geo_impossible_travel"] = 0

        df.drop(columns=["home_lat", "home_lon"], inplace=True)
    else:
        df["geo_distance_km"]       = 0.0
        df["geo_impossible_travel"] = 0

    # Composite score
    df["geo_score"] = (
        0.35 * df["geo_high_risk_country"] +
        0.30 * (df["geo_distance_km"].clip(0, 5000) / 5000) +
        0.35 * df["geo_impossible_travel"]
    )

    return df

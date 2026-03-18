from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import IO

import pandas as pd

from fraud_mas.config import FRAUD_MEMORY_PATH, LABEL_ENC_PATH, XGB_MODEL_PATH


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_json_records(path: str | Path | IO | None) -> list[dict] | None:
    """Load a JSON file that is either a list of records or a dict with a single list value."""
    if path is None:
        return None
    try:
        if hasattr(path, "read"):
            data = json.load(path)
        else:
            with open(path) as f:
                data = json.load(f)
    except Exception:
        return None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # e.g. {"users": [...]} or {"data": [...]}
        for v in data.values():
            if isinstance(v, list):
                return v
    return None


def load_and_merge_dataset(
    transactions_path: str | Path | IO,
    users_path:     str | Path | IO | None = None,
    locations_path: str | Path | IO | None = None,
    sms_path:       str | Path | IO | None = None,
    mails_path:     str | Path | IO | None = None,
) -> pd.DataFrame:
    """
    Load all challenge data files and merge them into a single DataFrame.

    transactions_path : CSV with core transaction columns
    users_path        : JSON list of user records (joined on user_id)
    locations_path    : JSON list of location records (joined on transaction_id or user_id)
    sms_path          : JSON list of SMS records (joined on transaction_id or user_id)
    mails_path        : JSON list of mail records (joined on transaction_id or user_id)

    Returns an enriched DataFrame ready for the agent pipeline.
    """
    # --- transactions ---
    if hasattr(transactions_path, "read"):
        df = pd.read_csv(transactions_path)
    else:
        df = pd.read_csv(transactions_path)

    # --- users ---
    users_records = _load_json_records(users_path)
    if users_records:
        users_df = pd.DataFrame(users_records)
        if "user_id" in users_df.columns and "user_id" in df.columns:
            # Avoid overwriting existing transaction columns
            overlap = [c for c in users_df.columns if c in df.columns and c != "user_id"]
            users_df = users_df.drop(columns=overlap)
            df = df.merge(users_df, on="user_id", how="left")

    # --- locations ---
    loc_records = _load_json_records(locations_path)
    if loc_records:
        loc_df = pd.DataFrame(loc_records)
        if "transaction_id" in loc_df.columns and "transaction_id" in df.columns:
            overlap = [c for c in loc_df.columns if c in df.columns and c != "transaction_id"]
            loc_df = loc_df.drop(columns=overlap)
            df = df.merge(loc_df, on="transaction_id", how="left")
        elif "user_id" in loc_df.columns and "user_id" in df.columns:
            overlap = [c for c in loc_df.columns if c in df.columns and c != "user_id"]
            loc_df = loc_df.drop(columns=overlap)
            df = df.merge(loc_df, on="user_id", how="left")

    # --- SMS ---
    sms_records = _load_json_records(sms_path)
    if sms_records:
        sms_df = pd.DataFrame(sms_records)
        # Detect the text column
        text_col = next((c for c in ["text", "message", "body", "content"] if c in sms_df.columns), None)
        if text_col:
            sms_df = sms_df.rename(columns={text_col: "sms_text"})
            # Aggregate multiple SMS per transaction/user into one string
            if "transaction_id" in sms_df.columns and "transaction_id" in df.columns:
                agg = sms_df.groupby("transaction_id")["sms_text"].apply(" | ".join).reset_index()
                df = df.merge(agg, on="transaction_id", how="left")
            elif "user_id" in sms_df.columns and "user_id" in df.columns:
                agg = sms_df.groupby("user_id")["sms_text"].apply(" | ".join).reset_index()
                df = df.merge(agg, on="user_id", how="left")

    # --- mails ---
    mail_records = _load_json_records(mails_path)
    if mail_records:
        mail_df = pd.DataFrame(mail_records)
        # Combine subject + body if both present
        if "subject" in mail_df.columns and "body" in mail_df.columns:
            mail_df["mail_text"] = mail_df["subject"].fillna("") + " " + mail_df["body"].fillna("")
        else:
            text_col = next((c for c in ["text", "message", "body", "content", "subject"] if c in mail_df.columns), None)
            if text_col:
                mail_df = mail_df.rename(columns={text_col: "mail_text"})

        if "mail_text" in mail_df.columns:
            if "transaction_id" in mail_df.columns and "transaction_id" in df.columns:
                agg = mail_df.groupby("transaction_id")["mail_text"].apply(" | ".join).reset_index()
                df = df.merge(agg, on="transaction_id", how="left")
            elif "user_id" in mail_df.columns and "user_id" in df.columns:
                agg = mail_df.groupby("user_id")["mail_text"].apply(" | ".join).reset_index()
                df = df.merge(agg, on="user_id", how="left")

    return df


def save_model(model, path: str | Path = XGB_MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path = XGB_MODEL_PATH):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_label_encoders(encoders: dict, path: str | Path = LABEL_ENC_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(encoders, f)


def load_label_encoders(path: str | Path = LABEL_ENC_PATH) -> dict:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def load_fraud_memory(path: str | Path = FRAUD_MEMORY_PATH) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_fraud_memory(memory: dict, path: str | Path = FRAUD_MEMORY_PATH) -> None:
    with open(path, "w") as f:
        json.dump(memory, f, indent=2)


def write_submission(predictions: list[dict], path: str | Path) -> None:
    """Write transaction_id,label lines."""
    lines = ["transaction_id,label"]
    for p in predictions:
        lines.append(f"{p['transaction_id']},{p['label']}")
    Path(path).write_text("\n".join(lines))

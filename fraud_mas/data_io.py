from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from fraud_mas.config import FRAUD_MEMORY_PATH, LABEL_ENC_PATH, XGB_MODEL_PATH


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_model(model, path: str | Path = XGB_MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path = XGB_MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_label_encoders(encoders: dict, path: str | Path = LABEL_ENC_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(encoders, f)


def load_label_encoders(path: str | Path = LABEL_ENC_PATH) -> dict:
    with open(path, "rb") as f:
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

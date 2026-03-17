"""Agent 4 – NLP-based risk signals from text fields."""

import re
import pandas as pd

_HIGH_RISK_KEYWORDS = {
    "casino", "gambling", "crypto", "bitcoin", "forex", "investment",
    "loan", "offshore", "anonymous", "escrow", "wire", "transfer",
    "gift card", "prepaid", "reload", "moneygram", "western union",
}

_OBFUSCATION_PATTERN = re.compile(r"[^a-z0-9\s]", re.IGNORECASE)


def _risk_keyword_score(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    text_clean = _OBFUSCATION_PATTERN.sub(" ", text.lower())
    hits = sum(1 for kw in _HIGH_RISK_KEYWORDS if kw in text_clean)
    return min(hits / 3, 1.0)


def _obfuscation_score(text: str) -> float:
    """Fraction of non-alphanumeric characters – may indicate obfuscation."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return min(non_alnum / max(len(text), 1), 1.0)


def compute_nlp_signals(df: pd.DataFrame, text_col: str = "merchant") -> pd.DataFrame:
    """
    Adds NLP risk columns to df.

    Signals
    -------
    nlp_keyword_score    : fraction of high-risk keywords matched [0,1]
    nlp_obfuscation      : fraction of obfuscated characters [0,1]
    nlp_all_caps         : 1 if merchant name is all-caps (common in fraud)
    nlp_score            : composite [0, 1]
    """
    df = df.copy()

    if text_col not in df.columns:
        df["nlp_keyword_score"] = 0.0
        df["nlp_obfuscation"]   = 0.0
        df["nlp_all_caps"]      = 0
        df["nlp_score"]         = 0.0
        return df

    df["nlp_keyword_score"] = df[text_col].apply(_risk_keyword_score)
    df["nlp_obfuscation"]   = df[text_col].apply(_obfuscation_score)
    df["nlp_all_caps"]      = df[text_col].apply(
        lambda t: int(isinstance(t, str) and t == t.upper() and len(t) > 3)
    )

    df["nlp_score"] = (
        0.55 * df["nlp_keyword_score"] +
        0.25 * df["nlp_obfuscation"] +
        0.20 * df["nlp_all_caps"]
    )

    return df

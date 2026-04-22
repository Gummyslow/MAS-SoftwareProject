"""Agent 4 – NLP-based risk signals from text fields."""

import re
import pandas as pd

_HIGH_RISK_KEYWORDS = {
    "casino", "gambling", "crypto", "bitcoin", "forex", "investment",
    "loan", "offshore", "anonymous", "escrow", "wire", "transfer",
    "gift card", "prepaid", "reload", "moneygram", "western union",
    "urgent", "verify", "account", "suspended", "click", "confirm",
}

_PHISHING_URL_PATTERN = re.compile(
    r"(paypa1|amaz0n|dhl-delivr|bit\.ly|verify.*secure|secure.*verify"
    r"|free.*money|click.*here|account.*suspend)",
    re.IGNORECASE,
)

_OBFUSCATION_PATTERN = re.compile(r"[^a-z0-9\s]", re.IGNORECASE)


def _risk_keyword_score(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    text_clean = _OBFUSCATION_PATTERN.sub(" ", text.lower())
    hits = sum(1 for kw in _HIGH_RISK_KEYWORDS if kw in text_clean)
    return min(hits / 3, 1.0)


def _phishing_score(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    return 1.0 if _PHISHING_URL_PATTERN.search(text) else 0.0


def _obfuscation_score(text: str) -> float:
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return min(non_alnum / max(len(text), 1), 1.0)


def compute_nlp_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signals
    -------
    nlp_keyword_score  : high-risk keyword score from description/merchant [0,1]
    nlp_phishing       : phishing URL pattern detected [0,1]
    nlp_obfuscation    : obfuscated character fraction [0,1]
    nlp_sms_score      : keyword risk from sms_text (if present) [0,1]
    nlp_mail_score     : keyword risk + phishing from mail_text (if present) [0,1]
    nlp_score          : composite [0, 1]
    """
    df = df.copy()

    # Primary text: prefer description, fall back to merchant
    text_col = next((c for c in ["description", "merchant"] if c in df.columns), None)

    if text_col:
        df["nlp_keyword_score"] = df[text_col].apply(_risk_keyword_score)
        df["nlp_phishing"]      = df[text_col].apply(_phishing_score)
        df["nlp_obfuscation"]   = df[text_col].apply(_obfuscation_score)
        df["nlp_all_caps"]      = df[text_col].apply(
            lambda t: int(isinstance(t, str) and t == t.upper() and len(t) > 3)
        )
    else:
        df["nlp_keyword_score"] = 0.0
        df["nlp_phishing"]      = 0.0
        df["nlp_obfuscation"]   = 0.0
        df["nlp_all_caps"]      = 0

    df["nlp_sms_score"] = (
        df["sms_text"].apply(lambda t: max(_risk_keyword_score(t), _phishing_score(t)))
        if "sms_text" in df.columns else 0.0
    )
    df["nlp_mail_score"] = (
        df["mail_text"].apply(lambda t: max(_risk_keyword_score(t), _phishing_score(t)))
        if "mail_text" in df.columns else 0.0
    )
    df["phishing_link_count"] = df["nlp_phishing"] + df.get("nlp_sms_score", 0) + df.get("nlp_mail_score", 0)

    df["nlp_score"] = (
        0.35 * df["nlp_keyword_score"] +
        0.20 * df["nlp_phishing"] +
        0.10 * df["nlp_obfuscation"] +
        0.10 * df["nlp_all_caps"] +
        0.15 * df["nlp_sms_score"] +
        0.10 * df["nlp_mail_score"]
    ).clip(0, 1)

    return df

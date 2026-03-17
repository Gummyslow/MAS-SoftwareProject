import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
MODELS_DIR      = BASE_DIR / "models"
ARTIFACTS_DIR   = BASE_DIR / "artifacts"
DATA_DIR        = BASE_DIR / "data"

XGB_MODEL_PATH      = MODELS_DIR / "xgb_model.pkl"
LABEL_ENC_PATH      = ARTIFACTS_DIR / "label_encoders.pkl"
FRAUD_MEMORY_PATH   = ARTIFACTS_DIR / "fraud_memory.json"
SUBMISSION_PATH     = ARTIFACTS_DIR / "submission.txt"

# OpenRouter
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL           = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Thresholds
FRAUD_THRESHOLD_HIGH   = 0.75   # auto-flag as fraud
FRAUD_THRESHOLD_LOW    = 0.25   # auto-clear as legit
# scores between LOW and HIGH go to LLM for final decision

# Feature engineering
AMOUNT_LOG_SHIFT = 1.0
TOP_N_MERCHANTS  = 50
TOP_N_COUNTRIES  = 30

# XMPP / SPADE
XMPP_SERVER   = os.getenv("XMPP_SERVER",   "localhost")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "fraud123")

AGENT_JIDS = {
    "orchestrator": f"orchestrator@{XMPP_SERVER}",
    "feature":      f"feature_agent@{XMPP_SERVER}",
    "behavioral":   f"behavioral_agent@{XMPP_SERVER}",
    "geo":          f"geo_agent@{XMPP_SERVER}",
    "nlp":          f"nlp_agent@{XMPP_SERVER}",
    "network":      f"network_agent@{XMPP_SERVER}",
    "model":        f"model_agent@{XMPP_SERVER}",
    "llm":          f"llm_agent@{XMPP_SERVER}",
}

# How long (seconds) the orchestrator waits for each agent reply
AGENT_TIMEOUT = 30

# XGBoost training defaults
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

"""End-to-end pipeline: run all agents → ensemble → LLM → submission."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from fraud_mas.behavioral import compute_behavioral_signals
from fraud_mas.config import SUBMISSION_PATH
from fraud_mas.data_io import load_label_encoders, write_submission
from fraud_mas.features import engineer_features, get_feature_matrix
from fraud_mas.geo import compute_geo_signals
from fraud_mas.llm_orchestrator import llm_decide
from fraud_mas.model import apply_thresholds, predict_proba
from fraud_mas.network_analysis import compute_network_signals
from fraud_mas.nlp_risk import compute_nlp_signals

# Columns each agent is responsible for (used to merge results back)
_AGENT_OUTPUT_COLS = {
    "behavioral": ["behav_velocity_1h", "behav_velocity_24h", "behav_new_recipient",
                   "behav_large_jump", "burst_score", "behav_score"],
    "geo":        ["geo_high_risk_country", "geo_distance_km", "geo_impossible_travel",
                   "geo_score"],
    "nlp":        ["nlp_keyword_score", "nlp_phishing", "nlp_obfuscation",
                   "nlp_all_caps", "nlp_sms_score", "nlp_mail_score",
                   "phishing_link_count", "nlp_score"],
    "network":    ["net_shared_device", "net_shared_ip", "net_community_risk",
                   "net_degree", "net_new_node_score", "net_score"],
}

_AGENT_FNS = {
    "behavioral": compute_behavioral_signals,
    "geo":        compute_geo_signals,
    "nlp":        compute_nlp_signals,
    "network":    compute_network_signals,
}


def _run_agent(name: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Run a single agent and return (name, result_df)."""
    result = _AGENT_FNS[name](df)
    return name, result


def _run_agents(df: pd.DataFrame, encoders: dict, verbose: bool = False) -> pd.DataFrame:
    """
    Agent 1 (feature engineering) runs first, then agents 2-5 run in parallel
    on separate DataFrame copies and their signal columns are merged back.
    """
    # --- Agent 1: feature engineering (must complete before others) ---
    df, _ = engineer_features(df, encoders=encoders, fit=False)

    # --- Agents 2-5: run concurrently ---
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent") as pool:
        futures = {pool.submit(_run_agent, name, df.copy()): name for name in _AGENT_FNS}

        for future in as_completed(futures):
            name = futures[future]
            try:
                _, result_df = future.result()
                cols = [c for c in _AGENT_OUTPUT_COLS[name] if c in result_df.columns]
                # reindex to df's index to handle any reordering inside agents
                df[cols] = result_df[cols].reindex(df.index)
                if verbose:
                    print(f"[pipeline]   agent '{name}' done")
            except Exception as exc:
                print(f"[pipeline] WARNING: agent '{name}' raised {exc!r} – signals set to 0")
                for col in _AGENT_OUTPUT_COLS[name]:
                    df[col] = 0

    if verbose:
        print(f"[pipeline] All agents finished in {time.perf_counter() - t0:.2f}s")

    return df


def run_pipeline(
    df: pd.DataFrame,
    model=None,
    encoders: dict | None = None,
    id_col: str = "transaction_id",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full inference pipeline.

    Parameters
    ----------
    df       : raw test/inference DataFrame
    model    : trained XGBClassifier (loaded from disk if None)
    encoders : label encoders from training (loaded from disk if None)
    id_col   : column name for transaction IDs
    verbose  : print progress

    Returns
    -------
    results : DataFrame with columns [id_col, model_score, initial_label, label, llm_reason]
    """
    if encoders is None:
        encoders = load_label_encoders()

    if verbose:
        print(f"[pipeline] Running agents on {len(df)} transactions (agents 2-5 in parallel)...")

    df = _run_agents(df, encoders, verbose=verbose)

    if verbose:
        print("[pipeline] Running ensemble model...")

    scores = predict_proba(df, model)
    df["model_score"] = scores

    initial_labels = apply_thresholds(scores)
    df["initial_label"] = initial_labels

    # Split by initial label
    auto_fraud  = df[df["initial_label"] == "fraud"].copy()
    auto_legit  = df[df["initial_label"] == "legit"].copy()
    review_rows = df[df["initial_label"] == "review"].copy()

    auto_fraud["label"]     = "fraud"
    auto_fraud["llm_reason"] = ""
    auto_legit["label"]     = "legit"
    auto_legit["llm_reason"] = ""

    if verbose:
        print(f"[pipeline] Auto-fraud: {len(auto_fraud)}, Auto-legit: {len(auto_legit)}, LLM-review: {len(review_rows)}")

    if len(review_rows) > 0:
        if verbose:
            print(f"[pipeline] Sending {len(review_rows)} borderline cases to LLM...")
        llm_results = llm_decide(review_rows.to_dict(orient="records"))
        review_df = pd.DataFrame(llm_results)
    else:
        review_df = review_rows.copy()
        review_df["label"]      = "legit"
        review_df["llm_reason"] = ""

    results = pd.concat([auto_fraud, auto_legit, review_df], ignore_index=True)
    results = results.sort_index()

    if verbose:
        fraud_count = (results["label"] == "fraud").sum()
        print(f"[pipeline] Done. Fraud: {fraud_count}/{len(results)} ({fraud_count/len(results)*100:.1f}%)")

    return results


def write_submission_file(results: pd.DataFrame, id_col: str = "transaction_id", path=None) -> None:
    path = path or SUBMISSION_PATH
    predictions = results[[id_col, "label"]].rename(columns={id_col: "transaction_id"}).to_dict(orient="records")
    write_submission(predictions, path)
    print(f"[pipeline] Submission written to {path}")

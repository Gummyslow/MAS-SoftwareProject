"""LLM orchestration – final decisions on borderline transactions."""

from __future__ import annotations

import json
from openai import OpenAI
from langfuse import Langfuse

from fraud_mas.config import (
    OPENROUTER_API_KEY, LLM_MODEL,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
)
from fraud_mas.data_io import load_fraud_memory, save_fraud_memory

_client: OpenAI | None = None
_langfuse: Langfuse | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def _get_langfuse() -> Langfuse | None:
    global _langfuse
    if _langfuse is None and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        _langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    return _langfuse


def build_system_prompt(memory: dict) -> str:
    known_patterns = json.dumps(memory.get("fraud_patterns", []), indent=2)
    return f"""You are a fraud detection specialist reviewing borderline transactions.
You have access to signals from 5 specialized agents:
  1. Feature Agent  – statistical anomalies (amount, time)
  2. Behavioral Agent – velocity, new merchants, large jumps
  3. Geo Agent – high-risk countries, impossible travel
  4. NLP Agent – suspicious merchant names / keywords
  5. Network Agent – shared devices / IPs, fraud rings

Known fraud patterns from memory:
{known_patterns}

Rules:
- Respond ONLY with a JSON object: {{"decision": "fraud"|"legit", "confidence": 0.0-1.0, "reason": "..."}}
- Be conservative: when in doubt, mark as fraud.
- confidence >= 0.85 means you are highly certain."""


def build_evidence_bundle(row: dict) -> str:
    return json.dumps({
        "transaction_id":     row.get("transaction_id"),
        "amount":             row.get("amount"),
        "merchant":           row.get("merchant"),
        "country":            row.get("country"),
        "model_score":        round(float(row.get("model_score", 0)), 4),
        "behav_score":        round(float(row.get("behav_score", 0)), 4),
        "geo_score":          round(float(row.get("geo_score", 0)), 4),
        "nlp_score":          round(float(row.get("nlp_score", 0)), 4),
        "net_score":          round(float(row.get("net_score", 0)), 4),
        "behav_velocity_1h":  row.get("behav_velocity_1h", 0),
        "geo_high_risk_country": row.get("geo_high_risk_country", 0),
        "geo_impossible_travel": row.get("geo_impossible_travel", 0),
        "nlp_keyword_score":  round(float(row.get("nlp_keyword_score", 0)), 4),
        "net_shared_device":  row.get("net_shared_device", 0),
    }, indent=2)


def llm_decide(rows: list[dict], memory_path=None) -> list[dict]:
    """
    Ask the LLM to decide on a list of 'review' transactions.

    Returns the same list with 'label' and 'llm_reason' fields added.
    """
    if not rows:
        return []

    memory  = load_fraud_memory(memory_path) if memory_path else load_fraud_memory()
    system  = build_system_prompt(memory)
    client  = _get_client()
    lf      = _get_langfuse()
    results = []

    trace = lf.trace(name="fraud-llm-decide", metadata={"n_rows": len(rows)}) if lf else None

    for row in rows:
        evidence = build_evidence_bundle(row)
        user_msg = f"Review this transaction and decide:\n\n{evidence}"

        generation = trace.generation(
            name="llm-decision",
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
        ) if trace else None

        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=256,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
        )

        raw = response.choices[0].message.content.strip()

        if generation:
            generation.end(output=raw)

        try:
            parsed   = json.loads(raw)
            decision = parsed.get("decision", "fraud")
            reason   = parsed.get("reason", "")
        except json.JSONDecodeError:
            decision = "fraud"
            reason   = raw

        results.append({**row, "label": decision, "llm_reason": reason})

    # Update memory with newly confirmed fraud patterns
    new_patterns = [
        r["llm_reason"] for r in results
        if r["label"] == "fraud" and r.get("llm_reason")
    ]
    if new_patterns:
        existing = memory.get("fraud_patterns", [])
        memory["fraud_patterns"] = list(set(existing + new_patterns))[-50:]
        save_fraud_memory(memory)

    if lf:
        lf.flush()

    return results

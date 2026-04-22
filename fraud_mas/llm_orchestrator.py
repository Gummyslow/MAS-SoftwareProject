"""LLM orchestration – two-tier Gemini decisions via Google Generative AI."""

from __future__ import annotations

import json

from google import genai
from google.genai import types as genai_types
from langfuse import Langfuse

from fraud_mas.config import (
    GOOGLE_API_KEY,
    LLM_MODEL_CHEAP, LLM_MODEL_STRONG, LLM_CONFIDENCE_MIN,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
)
from fraud_mas.data_io import load_fraud_memory, save_fraud_memory

_langfuse: Langfuse | None = None
_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        if GOOGLE_API_KEY:
            _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            # Rely on Application Default Credentials (gcloud auth application-default login)
            _genai_client = genai.Client()
    return _genai_client


def _get_langfuse() -> Langfuse | None:
    global _langfuse
    if _langfuse is None and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            _langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
        except Exception:
            pass
    return _langfuse


def _lf_trace(lf, **kwargs):
    if lf is None:
        return None
    try:
        return lf.trace(**kwargs)
    except Exception:
        return None


def _lf_generation(trace, **kwargs):
    if trace is None:
        return None
    try:
        return trace.generation(**kwargs)
    except Exception:
        return None


def _lf_end(gen, output):
    if gen is None:
        return
    try:
        gen.end(output=output)
    except Exception:
        pass


def build_system_prompt(memory: dict) -> str:
    known_patterns = json.dumps(memory.get("fraud_patterns", []), indent=2)
    return f"""You are a fraud detection specialist reviewing borderline financial transactions.
You receive signals from 5 specialized agents:
  1. Feature Agent  – statistical anomalies (amount z-score, card-test, velocity)
  2. Behavioral Agent – velocity, new recipients, large amount jumps
  3. Geo Agent – high-risk countries, impossible travel distance
  4. NLP Agent – suspicious keywords in description/SMS/email, phishing patterns
  5. Network Agent – shared devices/IPs, fraud ring connections

Known fraud patterns from memory:
{known_patterns}

Rules:
- Respond ONLY with valid JSON: {{"decision": "fraud"|"legit", "confidence": 0.0-1.0, "reason": "one sentence"}}
- Be conservative: when genuinely uncertain, mark as fraud.
- confidence >= 0.85 = highly certain."""


def build_evidence_bundle(row: dict) -> str:
    fields = {
        "transaction_id":        row.get("transaction_id"),
        "amount":                row.get("amount"),
        "description":           row.get("description") or row.get("merchant"),
        "transaction_type":      row.get("transaction_type"),
        "payment_method":        row.get("payment_method"),
        "model_score":           round(float(row.get("model_score", 0)), 4),
        "card_test_score":       round(float(row.get("card_test_score", 0)), 4),
        "behav_score":           round(float(row.get("behav_score", 0)), 4),
        "behav_velocity_1h":     row.get("behav_velocity_1h", 0),
        "behav_new_recipient":   row.get("behav_new_recipient", 0),
        "behav_large_jump":      row.get("behav_large_jump", 0),
        "geo_score":             round(float(row.get("geo_score", 0)), 4),
        "geo_high_risk_country": row.get("geo_high_risk_country", 0),
        "geo_impossible_travel": row.get("geo_impossible_travel", 0),
        "nlp_score":             round(float(row.get("nlp_score", 0)), 4),
        "nlp_keyword_score":     round(float(row.get("nlp_keyword_score", 0)), 4),
        "nlp_phishing":          row.get("nlp_phishing", 0),
        "net_score":             round(float(row.get("net_score", 0)), 4),
        "net_shared_device":     row.get("net_shared_device", 0),
        "net_community_risk":    round(float(row.get("net_community_risk", 0)), 4),
        "is_night":              row.get("is_night", 0),
    }
    return json.dumps({k: v for k, v in fields.items() if v is not None}, indent=2)


def _call_llm(model_name: str, system: str, user_msg: str) -> dict:
    try:
        client = _get_genai_client()
        response = client.models.generate_content(
            model=model_name,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"decision": "fraud", "confidence": 0.5, "reason": str(exc)[:200]}
    except Exception as exc:
        return {"decision": "fraud", "confidence": 0.5, "reason": f"LLM error: {exc}"}


def llm_decide(rows: list[dict], memory_path=None) -> list[dict]:
    """
    Two-tier Gemini decision:
      1. cheap model handles most cases
      2. strong model escalates when confidence < LLM_CONFIDENCE_MIN
    """
    if not rows:
        return []

    if not GOOGLE_API_KEY:
        return [{**r, "label": "fraud", "llm_reason": "LLM skipped: GOOGLE_API_KEY not set"} for r in rows]

    memory = load_fraud_memory(memory_path) if memory_path else load_fraud_memory()
    system = build_system_prompt(memory)
    lf     = _get_langfuse()
    results = []

    trace = _lf_trace(lf, name="fraud-llm-decide", metadata={"n_rows": len(rows)})

    for row in rows:
        evidence = build_evidence_bundle(row)
        user_msg = f"Review this transaction:\n\n{evidence}"

        # ── cheap model first ──────────────────────────────────────────────
        gen = _lf_generation(trace, name="llm-cheap", model=LLM_MODEL_CHEAP,
                             input=user_msg)
        parsed = _call_llm(LLM_MODEL_CHEAP, system, user_msg)
        confidence = float(parsed.get("confidence", 0.5))
        _lf_end(gen, json.dumps(parsed))

        # ── escalate to strong model if confidence is low ──────────────────
        if confidence < LLM_CONFIDENCE_MIN:
            gen2 = _lf_generation(trace, name="llm-strong", model=LLM_MODEL_STRONG,
                                  input=user_msg)
            parsed = _call_llm(LLM_MODEL_STRONG, system, user_msg)
            _lf_end(gen2, json.dumps(parsed))

        decision = str(parsed.get("decision", "fraud")).lower()
        if decision not in ("fraud", "legit"):
            decision = "fraud"
        reason = parsed.get("reason", "")

        results.append({**row, "label": decision, "llm_reason": reason})

    # ── update memory with new fraud patterns ──────────────────────────────
    new_patterns = [
        r["llm_reason"] for r in results
        if r["label"] == "fraud" and r.get("llm_reason")
    ]
    if new_patterns:
        existing = memory.get("fraud_patterns", [])
        memory["fraud_patterns"] = list(set(existing + new_patterns))[-50:]
        save_fraud_memory(memory)

    if lf:
        try:
            lf.flush()
        except Exception:
            pass

    return results

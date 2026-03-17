"""
Async bridge between the synchronous pipeline/app and the SPADE agent system.

Usage
-----
    import asyncio
    from fraud_mas.agents.spade_pipeline import run_spade_pipeline

    results = asyncio.run(run_spade_pipeline(df))
"""

from __future__ import annotations

import asyncio

import pandas as pd

from fraud_mas.agents.behavioral_agent import create_behavioral_agent
from fraud_mas.agents.feature_agent    import create_feature_agent
from fraud_mas.agents.geo_agent        import create_geo_agent
from fraud_mas.agents.llm_agent        import create_llm_agent
from fraud_mas.agents.model_agent      import create_model_agent
from fraud_mas.agents.network_agent    import create_network_agent
from fraud_mas.agents.nlp_agent        import create_nlp_agent
from fraud_mas.agents.orchestrator     import OrchestratorAgent, create_orchestrator


# Module-level agent handles (started once, reused across calls)
_agents: list = []
_orchestrator: OrchestratorAgent | None = None
_started = False


async def _ensure_started() -> OrchestratorAgent:
    global _agents, _orchestrator, _started
    if _started:
        return _orchestrator

    _orchestrator = create_orchestrator()
    _agents = [
        create_feature_agent(),
        create_behavioral_agent(),
        create_geo_agent(),
        create_nlp_agent(),
        create_network_agent(),
        create_model_agent(),
        create_llm_agent(),
        _orchestrator,
    ]
    for agent in _agents:
        await agent.start(auto_register=True)

    _started = True
    return _orchestrator


async def stop_agents() -> None:
    global _started
    for agent in _agents:
        await agent.stop()
    _started = False


async def run_spade_pipeline(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full SPADE multi-agent pipeline on a DataFrame.

    Starts agents on first call, reuses them on subsequent calls.
    """
    orch = await _ensure_started()

    rows = df.to_dict(orient="records")
    if verbose:
        print(f"[spade_pipeline] Submitting {len(rows)} transactions to orchestrator...")

    future  = orch.submit(rows)
    results = await asyncio.wait_for(future, timeout=300)

    if verbose:
        fraud_n = sum(1 for r in results if r.get("label") == "fraud")
        print(f"[spade_pipeline] Done. Fraud: {fraud_n}/{len(results)}")

    return pd.DataFrame(results)

"""
Async bridge between the synchronous pipeline/app and the SPADE agent system.

Usage (async)
-------------
    import asyncio
    from fraud_mas.agents.spade_pipeline import run_spade_pipeline

    results = asyncio.run(run_spade_pipeline(df))

Usage (sync, safe to call from Streamlit or any thread)
--------------------------------------------------------
    from fraud_mas.agents.spade_pipeline import run_spade_pipeline_sync

    results = run_spade_pipeline_sync(df)
"""

from __future__ import annotations

import asyncio
import threading

import pandas as pd

from fraud_mas.agents.behavioral_agent import create_behavioral_agent
from fraud_mas.agents.feature_agent    import create_feature_agent
from fraud_mas.agents.geo_agent        import create_geo_agent
from fraud_mas.agents.llm_agent        import create_llm_agent
from fraud_mas.agents.model_agent      import create_model_agent
from fraud_mas.agents.network_agent    import create_network_agent
from fraud_mas.agents.nlp_agent        import create_nlp_agent
from fraud_mas.agents.orchestrator     import OrchestratorAgent, create_orchestrator


async def run_spade_pipeline(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full SPADE multi-agent pipeline on a DataFrame.

    Starts a fresh set of agents for each call and stops them afterwards.
    This makes the function safe to call with asyncio.run() multiple times.
    """
    orchestrator = create_orchestrator()
    agents = [
        create_feature_agent(),
        create_behavioral_agent(),
        create_geo_agent(),
        create_nlp_agent(),
        create_network_agent(),
        create_model_agent(),
        create_llm_agent(),
        orchestrator,
    ]

    for agent in agents:
        await agent.start(auto_register=True)

    try:
        rows = df.to_dict(orient="records")
        if verbose:
            print(f"[spade_pipeline] Submitting {len(rows)} transactions to orchestrator...")

        future  = orchestrator.submit(rows)
        results = await asyncio.wait_for(future, timeout=300)

        if verbose:
            fraud_n = sum(1 for r in results if r.get("label") == "fraud")
            print(f"[spade_pipeline] Done. Fraud: {fraud_n}/{len(results)}")

        return pd.DataFrame(results)

    finally:
        for agent in agents:
            await agent.stop()


def run_spade_pipeline_sync(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Synchronous wrapper for run_spade_pipeline.

    Runs the async pipeline in a dedicated thread with its own event loop,
    making it safe to call from Streamlit or any environment that may already
    have a running event loop (e.g. tornado, Jupyter).
    """
    result: list[pd.DataFrame] = []
    error:  list[BaseException] = []

    def _worker():
        try:
            result.append(asyncio.run(run_spade_pipeline(df, verbose=verbose)))
        except Exception as exc:
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=360)

    if not thread.is_alive() and error:
        raise error[0]
    if not result:
        raise TimeoutError("SPADE pipeline did not complete within 360 seconds")

    return result[0]

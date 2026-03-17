"""
Start all SPADE fraud detection agents using the embedded XMPP server.

No external XMPP server or Docker required – SPADE's built-in pyjabber
server is started automatically.

Usage
-----
    python scripts/run_spade.py
"""

import sys

# Add project root to path so fraud_mas is importable
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.parent.__str__())

from spade.container import run_container

from fraud_mas.agents.behavioral_agent import create_behavioral_agent
from fraud_mas.agents.feature_agent    import create_feature_agent
from fraud_mas.agents.geo_agent        import create_geo_agent
from fraud_mas.agents.llm_agent        import create_llm_agent
from fraud_mas.agents.model_agent      import create_model_agent
from fraud_mas.agents.network_agent    import create_network_agent
from fraud_mas.agents.nlp_agent        import create_nlp_agent
from fraud_mas.agents.orchestrator     import create_orchestrator


async def main():
    import asyncio

    agents = [
        create_feature_agent(),
        create_behavioral_agent(),
        create_geo_agent(),
        create_nlp_agent(),
        create_network_agent(),
        create_model_agent(),
        create_llm_agent(),
        create_orchestrator(),
    ]

    print("Starting all SPADE agents (embedded XMPP server)...")
    for agent in agents:
        await agent.start(auto_register=True)
        print(f"  + {agent.jid}")

    print("\nAll agents running. Press Ctrl-C to stop.\n")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    print("\nStopping agents...")
    for agent in agents:
        await agent.stop()
    print("All agents stopped.")


if __name__ == "__main__":
    run_container(main(), embedded_xmpp_server=True)

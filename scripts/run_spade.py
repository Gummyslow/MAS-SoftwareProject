"""
Start all SPADE fraud detection agents.

Prerequisites
-------------
1. An XMPP server must be running (see docker-compose.yml at project root).
2. All agent JIDs must be registered on that server.
   The prosody docker image auto-creates accounts, so no manual steps needed.

Usage
-----
    python scripts/run_spade.py

The script starts all 8 agents and blocks until Ctrl-C.
"""

import asyncio
import signal
import sys

# Add project root to path so fraud_mas is importable
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.parent.__str__())

from fraud_mas.agents.behavioral_agent import create_behavioral_agent
from fraud_mas.agents.feature_agent    import create_feature_agent
from fraud_mas.agents.geo_agent        import create_geo_agent
from fraud_mas.agents.llm_agent        import create_llm_agent
from fraud_mas.agents.model_agent      import create_model_agent
from fraud_mas.agents.network_agent    import create_network_agent
from fraud_mas.agents.nlp_agent        import create_nlp_agent
from fraud_mas.agents.orchestrator     import create_orchestrator


async def main():
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

    print("Starting all SPADE agents...")
    for agent in agents:
        await agent.start(auto_register=True)
        print(f"  ✓ {agent.jid}")

    print("\nAll agents running. Press Ctrl-C to stop.\n")

    # Keep alive until interrupted
    stop_event = asyncio.Event()

    def _shutdown(*_):
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    await stop_event.wait()

    print("\nStopping agents...")
    for agent in agents:
        await agent.stop()
    print("All agents stopped.")


if __name__ == "__main__":
    asyncio.run(main())

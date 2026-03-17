"""Agent 5 – Network/graph analysis SPADE agent."""

import pandas as pd
from fraud_mas.agents._spade_compat import CyclicBehaviour, Template

from fraud_mas.agents.base import FraudBaseAgent
from fraud_mas.config import AGENT_JIDS, XMPP_PASSWORD
from fraud_mas.network_analysis import compute_network_signals

_SIGNAL_COLS = ["net_shared_device", "net_shared_ip",
                "net_community_risk", "net_degree", "net_score"]


class NetworkAgent(FraudBaseAgent):
    agent_name = "network"

    class AnalyseBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg is None:
                return

            agent: NetworkAgent = self.agent
            agent.log(f"Received batch (thread={msg.thread})")

            rows = agent.decode(msg.body)
            df   = pd.DataFrame(rows)

            try:
                df_out  = compute_network_signals(df)
                cols    = [c for c in _SIGNAL_COLS if c in df_out.columns]
                payload = df_out[["transaction_id"] + cols].to_dict(orient="records")
            except Exception as exc:
                agent.log(f"ERROR: {exc}", "error")
                payload = [{r.get("transaction_id"): {c: 0 for c in _SIGNAL_COLS}} for r in rows]

            reply = agent.make_reply(msg, payload, "signals")
            reply.set_metadata("agent", "network")
            await self.send(reply)
            agent.log(f"Sent network signals (thread={msg.thread})")

    async def setup(self):
        self.log("Starting up")
        tmpl = Template()
        tmpl.set_metadata("performative", "request")
        tmpl.set_metadata("type", "analyze")
        self.add_behaviour(self.AnalyseBehaviour(), tmpl)


def create_network_agent() -> NetworkAgent:
    return NetworkAgent(AGENT_JIDS["network"], XMPP_PASSWORD)

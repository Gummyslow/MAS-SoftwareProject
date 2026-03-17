"""Agent 4 – NLP risk analysis SPADE agent."""

import pandas as pd
from fraud_mas.agents._spade_compat import CyclicBehaviour, Template

from fraud_mas.agents.base import FraudBaseAgent
from fraud_mas.config import AGENT_JIDS, XMPP_PASSWORD
from fraud_mas.nlp_risk import compute_nlp_signals

_SIGNAL_COLS = ["nlp_keyword_score", "nlp_obfuscation", "nlp_all_caps", "nlp_score"]


class NLPAgent(FraudBaseAgent):
    agent_name = "nlp"

    class AnalyseBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg is None:
                return

            agent: NLPAgent = self.agent
            agent.log(f"Received batch (thread={msg.thread})")

            rows = agent.decode(msg.body)
            df   = pd.DataFrame(rows)

            try:
                df_out  = compute_nlp_signals(df)
                cols    = [c for c in _SIGNAL_COLS if c in df_out.columns]
                payload = df_out[["transaction_id"] + cols].to_dict(orient="records")
            except Exception as exc:
                agent.log(f"ERROR: {exc}", "error")
                payload = [{r.get("transaction_id"): {c: 0 for c in _SIGNAL_COLS}} for r in rows]

            reply = agent.make_reply(msg, payload, "signals")
            reply.set_metadata("agent", "nlp")
            await self.send(reply)
            agent.log(f"Sent NLP signals (thread={msg.thread})")

    async def setup(self):
        self.log("Starting up")
        tmpl = Template()
        tmpl.set_metadata("performative", "request")
        tmpl.set_metadata("type", "analyze")
        self.add_behaviour(self.AnalyseBehaviour(), tmpl)


def create_nlp_agent() -> NLPAgent:
    return NLPAgent(AGENT_JIDS["nlp"], XMPP_PASSWORD)

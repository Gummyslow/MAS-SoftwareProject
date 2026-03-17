"""Agent 1 – Feature engineering SPADE agent."""

import pandas as pd
from fraud_mas.agents._spade_compat import CyclicBehaviour, Template

from fraud_mas.agents.base import FraudBaseAgent
from fraud_mas.config import AGENT_JIDS, XMPP_PASSWORD
from fraud_mas.data_io import load_label_encoders
from fraud_mas.features import engineer_features


class FeatureAgent(FraudBaseAgent):
    agent_name = "feature"

    class AnalyseBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg is None:
                return

            agent: FeatureAgent = self.agent
            agent.log(f"Received batch (thread={msg.thread})")

            rows = agent.decode(msg.body)
            df   = pd.DataFrame(rows)

            try:
                df_out, _ = engineer_features(df, encoders=agent.encoders, fit=False)
                # Return only the new columns (original cols + engineered ones)
                payload = df_out.to_dict(orient="records")
            except Exception as exc:
                agent.log(f"ERROR: {exc}", "error")
                payload = rows   # pass through unchanged

            reply = agent.make_reply(msg, payload, "feature_result")
            await self.send(reply)
            agent.log(f"Sent feature result (thread={msg.thread})")

    async def setup(self):
        self.log("Starting up")
        self.encoders = load_label_encoders() if True else {}

        tmpl = Template()
        tmpl.set_metadata("performative", "request")
        tmpl.set_metadata("type", "analyze")
        self.add_behaviour(self.AnalyseBehaviour(), tmpl)


def create_feature_agent() -> FeatureAgent:
    return FeatureAgent(AGENT_JIDS["feature"], XMPP_PASSWORD)

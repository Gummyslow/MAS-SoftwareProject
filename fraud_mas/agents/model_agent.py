"""Model agent – XGBoost ensemble scoring SPADE agent."""

import pandas as pd
from fraud_mas.agents._spade_compat import CyclicBehaviour, Template

from fraud_mas.agents.base import FraudBaseAgent
from fraud_mas.config import AGENT_JIDS, XMPP_PASSWORD
from fraud_mas.data_io import load_model
from fraud_mas.model import apply_thresholds, predict_proba


class ModelAgent(FraudBaseAgent):
    agent_name = "model"

    class ScoreBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg is None:
                return

            agent: ModelAgent = self.agent
            agent.log(f"Scoring batch (thread={msg.thread})")

            rows = agent.decode(msg.body)
            df   = pd.DataFrame(rows)

            try:
                scores = predict_proba(df, agent.model)
                labels = apply_thresholds(scores)
                payload = [
                    {
                        "transaction_id": row.get("transaction_id"),
                        "model_score":    float(score),
                        "initial_label":  label,
                    }
                    for row, score, label in zip(rows, scores, labels)
                ]
            except Exception as exc:
                agent.log(f"ERROR: {exc}", "error")
                payload = [
                    {"transaction_id": r.get("transaction_id"),
                     "model_score": 0.5, "initial_label": "review"}
                    for r in rows
                ]

            reply = agent.make_reply(msg, payload, "scores")
            await self.send(reply)
            agent.log(f"Sent scores (thread={msg.thread})")

    async def setup(self):
        self.log("Starting up – loading model")
        self.model = load_model()
        if self.model is None:
            self.log("WARNING: no trained model found – scores will default to 0.5", "warning")

        tmpl = Template()
        tmpl.set_metadata("performative", "request")
        tmpl.set_metadata("type", "score")
        self.add_behaviour(self.ScoreBehaviour(), tmpl)


def create_model_agent() -> ModelAgent:
    return ModelAgent(AGENT_JIDS["model"], XMPP_PASSWORD)

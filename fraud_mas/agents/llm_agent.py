"""LLM agent – borderline decision making SPADE agent."""

from fraud_mas.agents._spade_compat import CyclicBehaviour, Template

from fraud_mas.agents.base import FraudBaseAgent
from fraud_mas.config import AGENT_JIDS, XMPP_PASSWORD
from fraud_mas.llm_orchestrator import llm_decide


class LLMAgent(FraudBaseAgent):
    agent_name = "llm"

    class DecideBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg is None:
                return

            agent: LLMAgent = self.agent
            rows = agent.decode(msg.body)
            agent.log(f"LLM reviewing {len(rows)} borderline cases (thread={msg.thread})")

            try:
                decided = llm_decide(rows)
                payload = [
                    {
                        "transaction_id": r["transaction_id"],
                        "label":          r["label"],
                        "llm_reason":     r.get("llm_reason", ""),
                    }
                    for r in decided
                ]
            except Exception as exc:
                agent.log(f"ERROR: {exc}", "error")
                payload = [
                    {"transaction_id": r.get("transaction_id"),
                     "label": "fraud", "llm_reason": f"LLM error: {exc}"}
                    for r in rows
                ]

            reply = agent.make_reply(msg, payload, "decisions")
            await self.send(reply)
            agent.log(f"Sent {len(payload)} decisions (thread={msg.thread})")

    async def setup(self):
        self.log("Starting up")
        tmpl = Template()
        tmpl.set_metadata("performative", "request")
        tmpl.set_metadata("type", "llm_decide")
        self.add_behaviour(self.DecideBehaviour(), tmpl)


def create_llm_agent() -> LLMAgent:
    return LLMAgent(AGENT_JIDS["llm"], XMPP_PASSWORD)

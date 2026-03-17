"""Shared base class and message helpers for all SPADE fraud agents."""

import json
import logging

from fraud_mas.agents._spade_compat import Agent, Message

logger = logging.getLogger(__name__)


class FraudBaseAgent(Agent):
    """
    Base SPADE agent.  All fraud agents inherit from this.

    Message protocol (all fields are metadata keys unless noted):
        performative : "request" | "inform" | "error"
        type         : "analyze" | "signals" | "score" | "scores"
                       | "llm_decide" | "decisions"
        agent        : sender agent name (e.g. "behavioral")
        body         : (Message.body) JSON-encoded payload
        thread       : batch UUID – used to correlate request/response pairs
    """

    # Subclasses set these
    agent_name: str = "base"

    async def _async_connect(self):
        """Override to allow PLAIN auth without TLS for local XMPP dev server."""
        if self.client is not None:
            self.client.enable_starttls = False
            self.client.enable_direct_tls = False
            self.client.enable_plaintext = True
            self.client["feature_mechanisms"].unencrypted_plain = True
        await super()._async_connect()

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def encode(data) -> str:
        return json.dumps(data, default=str)

    @staticmethod
    def decode(body: str):
        return json.loads(body)

    def make_reply(self, original: Message, payload, msg_type: str) -> Message:
        """Build a reply message pre-filled with metadata."""
        reply = original.make_reply()
        reply.set_metadata("performative", "inform")
        reply.set_metadata("type", msg_type)
        reply.set_metadata("agent", self.agent_name)
        reply.body = self.encode(payload)
        return reply

    def make_request(self, to: str, payload, msg_type: str, thread: str) -> Message:
        msg = Message(to=to)
        msg.set_metadata("performative", "request")
        msg.set_metadata("type", msg_type)
        msg.set_metadata("agent", self.agent_name)
        msg.thread = thread
        msg.body = self.encode(payload)
        return msg

    def log(self, text: str, level: str = "info") -> None:
        prefix = f"[{self.agent_name}]"
        getattr(logger, level)(f"{prefix} {text}")
        print(f"{prefix} {text}")

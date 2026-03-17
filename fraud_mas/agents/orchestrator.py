"""
Orchestrator SPADE agent.

FSM states
----------
IDLE         – waiting for a new batch request (external trigger via queue)
FEATURE      – sent batch to FeatureAgent, waiting for enriched rows
PARALLEL     – sent enriched rows to Behavioral/Geo/NLP/Network in parallel,
               collecting signal replies
SCORE        – sent fully-enriched rows to ModelAgent, waiting for scores
LLM_REVIEW   – sent borderline rows to LLMAgent, waiting for decisions
FINALIZE     – merge all results, publish to result queue, back to IDLE

External interface
------------------
OrchestratorAgent.submit(batch: list[dict]) -> asyncio.Future
    Push a batch of transaction dicts; the Future resolves with results.
"""

import asyncio
import uuid

import pandas as pd
from fraud_mas.agents._spade_compat import FSMBehaviour, State, Template

from fraud_mas.agents.base import FraudBaseAgent
from fraud_mas.config import AGENT_JIDS, AGENT_TIMEOUT, XMPP_PASSWORD

# ── FSM state names ──────────────────────────────────────────────────────────
S_IDLE       = "IDLE"
S_FEATURE    = "FEATURE"
S_PARALLEL   = "PARALLEL"
S_SCORE      = "SCORE"
S_LLM_REVIEW = "LLM_REVIEW"
S_FINALIZE   = "FINALIZE"

# Analysis agents that run in parallel (after feature engineering)
_PARALLEL_AGENTS = ["behavioral", "geo", "nlp", "network"]


# ── States ───────────────────────────────────────────────────────────────────

class IdleState(State):
    async def run(self):
        orch: OrchestratorAgent = self.agent
        try:
            batch_id, rows, fut = await asyncio.wait_for(
                orch._inbox.get(), timeout=1.0
            )
        except asyncio.TimeoutError:
            self.set_next_state(S_IDLE)
            return

        orch.log(f"Batch {batch_id}: {len(rows)} transactions")
        orch._current = {"batch_id": batch_id, "rows": rows, "future": fut, "signals": {}}
        self.set_next_state(S_FEATURE)


class FeatureState(State):
    async def run(self):
        orch: OrchestratorAgent = self.agent
        ctx   = orch._current
        bid   = ctx["batch_id"]
        rows  = ctx["rows"]

        msg = orch.make_request(
            to=AGENT_JIDS["feature"],
            payload=rows,
            msg_type="analyze",
            thread=bid,
        )
        await self.send(msg)
        orch.log(f"[{bid}] → FeatureAgent")

        tmpl = Template()
        tmpl.set_metadata("type", "feature_result")
        tmpl.thread = bid

        reply = await self.receive(timeout=AGENT_TIMEOUT)
        if reply is None:
            orch.log(f"[{bid}] FeatureAgent timeout – using raw rows", "warning")
        else:
            ctx["rows"] = orch.decode(reply.body)
            orch.log(f"[{bid}] ← FeatureAgent enriched rows")

        self.set_next_state(S_PARALLEL)


class ParallelState(State):
    async def run(self):
        orch: OrchestratorAgent = self.agent
        ctx  = orch._current
        bid  = ctx["batch_id"]
        rows = ctx["rows"]

        # Send to all 4 parallel agents simultaneously
        for name in _PARALLEL_AGENTS:
            msg = orch.make_request(
                to=AGENT_JIDS[name],
                payload=rows,
                msg_type="analyze",
                thread=bid,
            )
            await self.send(msg)
            orch.log(f"[{bid}] → {name}_agent")

        # Collect replies
        received = set()
        signal_frames: dict[str, pd.DataFrame] = {}

        deadline = asyncio.get_event_loop().time() + AGENT_TIMEOUT
        while len(received) < len(_PARALLEL_AGENTS):
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                missed = set(_PARALLEL_AGENTS) - received
                orch.log(f"[{bid}] Timeout waiting for: {missed}", "warning")
                break

            reply = await self.receive(timeout=remaining)
            if reply is None:
                break

            sender_agent = reply.metadata.get("agent", "unknown")
            if reply.metadata.get("type") == "signals" and sender_agent in _PARALLEL_AGENTS:
                signal_frames[sender_agent] = pd.DataFrame(orch.decode(reply.body))
                received.add(sender_agent)
                orch.log(f"[{bid}] ← {sender_agent}_agent signals")

        # Merge signal columns back into rows df
        base_df = pd.DataFrame(rows)
        for name, sig_df in signal_frames.items():
            sig_cols = [c for c in sig_df.columns if c != "transaction_id"]
            base_df = base_df.merge(sig_df[["transaction_id"] + sig_cols],
                                    on="transaction_id", how="left")

        ctx["rows"] = base_df.to_dict(orient="records")
        self.set_next_state(S_SCORE)


class ScoreState(State):
    async def run(self):
        orch: OrchestratorAgent = self.agent
        ctx  = orch._current
        bid  = ctx["batch_id"]

        msg = orch.make_request(
            to=AGENT_JIDS["model"],
            payload=ctx["rows"],
            msg_type="score",
            thread=bid,
        )
        await self.send(msg)
        orch.log(f"[{bid}] → model_agent")

        tmpl = Template()
        tmpl.set_metadata("type", "scores")
        tmpl.thread = bid

        reply = await self.receive(timeout=AGENT_TIMEOUT)
        if reply is None:
            orch.log(f"[{bid}] ModelAgent timeout", "warning")
            ctx["scores"] = []
        else:
            ctx["scores"] = orch.decode(reply.body)
            orch.log(f"[{bid}] ← model_agent scores")

        review = [s for s in ctx["scores"] if s["initial_label"] == "review"]
        ctx["review_ids"] = {s["transaction_id"] for s in review}

        if review:
            # Attach model_score to rows for LLM context
            score_map = {s["transaction_id"]: s["model_score"] for s in ctx["scores"]}
            review_rows = [
                {**r, "model_score": score_map.get(r["transaction_id"], 0.5)}
                for r in ctx["rows"]
                if r["transaction_id"] in ctx["review_ids"]
            ]
            ctx["review_rows"] = review_rows
            self.set_next_state(S_LLM_REVIEW)
        else:
            ctx["llm_decisions"] = []
            self.set_next_state(S_FINALIZE)


class LLMReviewState(State):
    async def run(self):
        orch: OrchestratorAgent = self.agent
        ctx  = orch._current
        bid  = ctx["batch_id"]

        orch.log(f"[{bid}] Sending {len(ctx['review_rows'])} cases to LLM")
        msg = orch.make_request(
            to=AGENT_JIDS["llm"],
            payload=ctx["review_rows"],
            msg_type="llm_decide",
            thread=bid,
        )
        await self.send(msg)

        reply = await self.receive(timeout=AGENT_TIMEOUT * 2)
        if reply is None:
            orch.log(f"[{bid}] LLMAgent timeout – defaulting to fraud", "warning")
            ctx["llm_decisions"] = [
                {"transaction_id": r["transaction_id"], "label": "fraud", "llm_reason": "timeout"}
                for r in ctx["review_rows"]
            ]
        else:
            ctx["llm_decisions"] = orch.decode(reply.body)
            orch.log(f"[{bid}] ← llm_agent decisions")

        self.set_next_state(S_FINALIZE)


class FinalizeState(State):
    async def run(self):
        orch: OrchestratorAgent = self.agent
        ctx  = orch._current
        bid  = ctx["batch_id"]

        # Build score lookup
        score_map = {s["transaction_id"]: s for s in ctx.get("scores", [])}
        llm_map   = {d["transaction_id"]: d for d in ctx.get("llm_decisions", [])}

        results = []
        for row in ctx["rows"]:
            tid = row["transaction_id"]
            score_entry = score_map.get(tid, {})
            model_score    = score_entry.get("model_score", 0.5)
            initial_label  = score_entry.get("initial_label", "review")

            if initial_label == "review" and tid in llm_map:
                label      = llm_map[tid]["label"]
                llm_reason = llm_map[tid].get("llm_reason", "")
            else:
                label      = initial_label if initial_label != "review" else "fraud"
                llm_reason = ""

            results.append({
                **row,
                "model_score":   model_score,
                "initial_label": initial_label,
                "label":         label,
                "llm_reason":    llm_reason,
            })

        orch.log(f"[{bid}] Finalised {len(results)} results")
        ctx["future"].set_result(results)
        self.set_next_state(S_IDLE)


# ── Agent ────────────────────────────────────────────────────────────────────

class OrchestratorAgent(FraudBaseAgent):
    agent_name = "orchestrator"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inbox: asyncio.Queue = asyncio.Queue()

    def submit(self, rows: list[dict]) -> asyncio.Future:
        """
        External API: submit a batch of transaction dicts.
        Returns an asyncio.Future that resolves with the result list.
        """
        loop    = asyncio.get_event_loop()
        future  = loop.create_future()
        batch_id = str(uuid.uuid4())[:8]
        self._inbox.put_nowait((batch_id, rows, future))
        return future

    async def setup(self):
        self.log("Starting up – building FSM")

        fsm = FSMBehaviour()

        fsm.add_state(name=S_IDLE,       state=IdleState(),      initial=True)
        fsm.add_state(name=S_FEATURE,    state=FeatureState())
        fsm.add_state(name=S_PARALLEL,   state=ParallelState())
        fsm.add_state(name=S_SCORE,      state=ScoreState())
        fsm.add_state(name=S_LLM_REVIEW, state=LLMReviewState())
        fsm.add_state(name=S_FINALIZE,   state=FinalizeState())

        fsm.add_transition(S_IDLE,       S_IDLE)
        fsm.add_transition(S_IDLE,       S_FEATURE)
        fsm.add_transition(S_FEATURE,    S_PARALLEL)
        fsm.add_transition(S_PARALLEL,   S_SCORE)
        fsm.add_transition(S_SCORE,      S_LLM_REVIEW)
        fsm.add_transition(S_SCORE,      S_FINALIZE)
        fsm.add_transition(S_LLM_REVIEW, S_FINALIZE)
        fsm.add_transition(S_FINALIZE,   S_IDLE)

        self.add_behaviour(fsm)


def create_orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent(AGENT_JIDS["orchestrator"], XMPP_PASSWORD)

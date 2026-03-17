"""
Drop-in SPADE-compatible mock using asyncio queues instead of XMPP.

Provides the same public API as the real spade package so all agent code
works unchanged on Python 3.13 / Windows where slixmpp wheels are absent.

Exports (mirroring real spade imports):
    spade.agent   → Agent
    spade.behaviour → CyclicBehaviour, FSMBehaviour, State
    spade.message → Message
    spade.template → Template
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Global in-process message bus ───────────────────────────────────────────
# Maps JID string → asyncio.Queue (the agent's raw inbox)
_MAILBOXES: dict[str, asyncio.Queue] = {}


def _get_mailbox(jid: str) -> asyncio.Queue:
    if jid not in _MAILBOXES:
        _MAILBOXES[jid] = asyncio.Queue()
    return _MAILBOXES[jid]


def _drop_mailbox(jid: str) -> None:
    _MAILBOXES.pop(jid, None)


# ── Message ──────────────────────────────────────────────────────────────────

class Message:
    def __init__(self, to: str = "", body: str = "", sender: str = ""):
        self.to:     str  = to
        self.body:   str  = body
        self.sender: str  = sender
        self.thread: str  = ""
        self._metadata: dict[str, str] = {}

    # Compatibility: real SPADE exposes metadata as a plain dict
    @property
    def metadata(self) -> dict[str, str]:
        return self._metadata

    def set_metadata(self, key: str, value: str) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str) -> str | None:
        return self._metadata.get(key)

    def make_reply(self) -> "Message":
        reply = Message(to=self.sender, sender=self.to)
        reply.thread = self.thread
        return reply

    def __repr__(self) -> str:
        return (f"Message(to={self.to!r}, thread={self.thread!r}, "
                f"meta={self._metadata}, body={self.body[:60]!r})")


# ── Template ─────────────────────────────────────────────────────────────────

class Template:
    def __init__(self):
        self._metadata: dict[str, str] = {}
        self.thread: str | None = None

    def set_metadata(self, key: str, value: str) -> None:
        self._metadata[key] = value

    def matches(self, msg: Message) -> bool:
        if self.thread is not None and msg.thread != self.thread:
            return False
        for k, v in self._metadata.items():
            if msg.get_metadata(k) != v:
                return False
        return True


# ── Behaviour base ────────────────────────────────────────────────────────────

class _BehaviourBase:
    def __init__(self):
        self.agent: "Agent | None" = None
        self._template: Template | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = True

    async def run(self) -> None:
        """Override in subclass."""

    async def send(self, msg: Message) -> None:
        """Route message to the recipient's mailbox."""
        if not msg.sender:
            msg.sender = str(self.agent.jid) if self.agent else ""
        mailbox = _get_mailbox(msg.to)
        await mailbox.put(msg)

    async def receive(self, timeout: float = 5.0) -> Message | None:
        """Wait for the next message matching this behaviour's template."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _run_loop(self) -> None:
        """Called by the agent task runner. Subclasses override _loop_body."""
        raise NotImplementedError


# ── CyclicBehaviour ──────────────────────────────────────────────────────────

class CyclicBehaviour(_BehaviourBase):
    async def _run_loop(self) -> None:
        while self._running and (self.agent and self.agent._running):
            await self.run()
            await asyncio.sleep(0)   # yield to event loop


# ── OneShotBehaviour ─────────────────────────────────────────────────────────

class OneShotBehaviour(_BehaviourBase):
    async def _run_loop(self) -> None:
        await self.run()


# ── FSM ──────────────────────────────────────────────────────────────────────

class State(_BehaviourBase):
    """A single FSM state. set_next_state() controls transitions."""

    def __init__(self):
        super().__init__()
        self._next_state: str | None = None

    def set_next_state(self, name: str) -> None:
        self._next_state = name


class FSMBehaviour(_BehaviourBase):
    def __init__(self):
        super().__init__()
        self._states:      dict[str, State] = {}
        self._transitions: dict[str, set]   = {}
        self._initial:     str | None       = None

    def add_state(self, name: str, state: State, initial: bool = False) -> None:
        self._states[name] = state
        if initial:
            self._initial = name

    def add_transition(self, source: str, dest: str) -> None:
        self._transitions.setdefault(source, set()).add(dest)

    async def _run_loop(self) -> None:
        current = self._initial
        if current is None:
            logger.error("FSMBehaviour has no initial state")
            return

        while self._running and (self.agent and self.agent._running):
            state = self._states.get(current)
            if state is None:
                logger.error(f"FSM: unknown state {current!r}")
                break

            # Wire state to agent and shared queue
            state.agent     = self.agent
            state._template = self._template
            state._queue    = self._queue   # states share the FSM's queue
            state._next_state = None

            await state.run()

            next_s = state._next_state
            if next_s is None:
                break                        # terminal state

            allowed = self._transitions.get(current, set())
            if next_s not in allowed:
                logger.warning(f"FSM: transition {current!r}→{next_s!r} not registered")

            current = next_s
            await asyncio.sleep(0)           # yield


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, jid: str, password: str):
        self._jid      = jid
        self._password = password
        self._running  = False
        self._behaviours: list[tuple[_BehaviourBase, Template | None]] = []
        self._tasks:     list[asyncio.Task] = []

    @property
    def jid(self) -> str:
        return self._jid

    def add_behaviour(self, behaviour: _BehaviourBase, template: Template | None = None) -> None:
        behaviour.agent    = self
        behaviour._template = template
        self._behaviours.append((behaviour, template))

    async def setup(self) -> None:
        """Override in subclass."""

    async def start(self, auto_register: bool = False) -> None:
        _get_mailbox(self._jid)   # ensure mailbox exists
        self._running = True
        await self.setup()
        self._tasks.append(asyncio.create_task(self._dispatch_loop(), name=f"{self._jid}/dispatch"))
        for beh, _ in self._behaviours:
            self._tasks.append(asyncio.create_task(beh._run_loop(), name=f"{self._jid}/{type(beh).__name__}"))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        _drop_mailbox(self._jid)

    async def _dispatch_loop(self) -> None:
        """Read raw inbox and route each message to the first matching behaviour queue."""
        inbox = _get_mailbox(self._jid)
        while self._running:
            try:
                msg: Message = await asyncio.wait_for(inbox.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            dispatched = False
            for beh, tmpl in self._behaviours:
                if tmpl is None or tmpl.matches(msg):
                    await beh._queue.put(msg)
                    dispatched = True
                    break

            if not dispatched:
                logger.debug(f"{self._jid}: no behaviour matched {msg}")

    def is_alive(self) -> bool:
        return self._running

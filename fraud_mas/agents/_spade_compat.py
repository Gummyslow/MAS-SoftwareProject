"""
Single import point for SPADE symbols.
Tries real spade first; falls back to the in-process asyncio mock.
"""
try:
    from spade.agent import Agent
    from spade.behaviour import CyclicBehaviour, FSMBehaviour, OneShotBehaviour, State
    from spade.message import Message
    from spade.template import Template
    _BACKEND = "spade (XMPP)"
except ImportError:
    from fraud_mas.agents.spade_mock import (  # type: ignore[assignment]
        Agent,
        CyclicBehaviour,
        FSMBehaviour,
        OneShotBehaviour,
        State,
        Message,
        Template,
    )
    _BACKEND = "spade_mock (asyncio queues)"

import logging
logging.getLogger(__name__).info(f"SPADE backend: {_BACKEND}")

__all__ = [
    "Agent", "CyclicBehaviour", "FSMBehaviour",
    "OneShotBehaviour", "State", "Message", "Template",
    "_BACKEND",
]

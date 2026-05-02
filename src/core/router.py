"""
Router — maps classifier output to the correct agent instance.

The router is the dispatch table. Adding a new agent is:
  1. Import the agent class
  2. Add it to _AGENT_REGISTRY

Everything else gets the StubAgent.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.agents.stub_agent import StubAgent
from src.core.models import ClassifierResult

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_portfolio_health_agent = None


def _get_portfolio_health_agent():
    """Lazy-load the portfolio health agent."""
    global _portfolio_health_agent
    if _portfolio_health_agent is None:
        from src.agents.portfolio_health import PortfolioHealthAgent
        _portfolio_health_agent = PortfolioHealthAgent()
    return _portfolio_health_agent


# Registry of implemented agents
_IMPLEMENTED_AGENTS = {"portfolio_health"}


def route(classifier_result: ClassifierResult) -> BaseAgent:
    """
    Route a classified query to the appropriate agent.

    Args:
        classifier_result: The classifier's output containing the target agent name.

    Returns:
        The appropriate BaseAgent instance. Never raises.
    """
    agent_name = classifier_result.agent

    if agent_name == "portfolio_health":
        return _get_portfolio_health_agent()

    # All other agents get the stub
    logger.info(f"Agent '{agent_name}' not implemented — routing to stub")
    return StubAgent(agent_name=agent_name)


def is_implemented(agent_name: str) -> bool:
    """Check if an agent is fully implemented."""
    return agent_name in _IMPLEMENTED_AGENTS

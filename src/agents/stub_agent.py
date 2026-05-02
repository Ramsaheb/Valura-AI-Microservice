"""
Stub Agent — structured fallback for unimplemented agents.

When the classifier routes to an agent that isn't built yet (market_research,
investment_strategy, etc.), the stub returns a clean structured response with:
  - The classified intent
  - The extracted entities
  - The agent that would have handled this
  - A short message indicating the agent is not yet implemented

Never crashes. Never returns errors.
"""
from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.core.models import ClassifierResult, StubResponse


class StubAgent(BaseAgent):
    """Placeholder agent for unimplemented specialist agents."""

    def __init__(self, agent_name: str = "unknown"):
        self._name = agent_name

    @property
    def name(self) -> str:
        return self._name

    async def run(
        self,
        user: dict[str, Any],
        query: str,
        classifier_result: ClassifierResult,
        llm: Any = None,
    ) -> dict[str, Any]:
        """Return a structured 'not implemented' response."""
        response = StubResponse(
            intent=classifier_result.intent,
            entities=classifier_result.entities,
            agent=classifier_result.agent,
            message=(
                f"The '{classifier_result.agent}' agent is not implemented in this build. "
                f"In production, this query would be handled by a specialist agent. "
                f"Your query has been correctly classified and routed."
            ),
        )
        return response.model_dump()

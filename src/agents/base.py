"""
Base Agent — abstract base class for all specialist agents.

Every agent (portfolio_health, market_research, etc.) extends this class.
The router dispatches to agents by name; adding a new agent is:
  1. Create a new class extending BaseAgent
  2. Register it in the router

No rewrite needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from src.core.models import ClassifierResult


class BaseAgent(ABC):
    """Abstract base for all specialist agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name matching the classifier taxonomy."""
        ...

    @abstractmethod
    async def run(
        self,
        user: dict[str, Any],
        query: str,
        classifier_result: ClassifierResult,
        llm: Any = None,
    ) -> dict[str, Any]:
        """
        Execute the agent's logic and return a structured response.

        Args:
            user: User profile dict (from fixtures).
            query: The original user query.
            classifier_result: The classifier's output (intent, entities, etc.).
            llm: Optional LLM client for agents that need generation.

        Returns:
            Agent-specific structured response dict.
        """
        ...

    async def stream(
        self,
        user: dict[str, Any],
        query: str,
        classifier_result: ClassifierResult,
        llm: Any = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the agent's response. Default implementation runs the full
        response and yields it as a single chunk. Override for true streaming.
        """
        import json
        result = await self.run(user, query, classifier_result, llm)
        yield json.dumps(result, indent=2)

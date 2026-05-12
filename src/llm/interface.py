"""
LLM Interface — abstract base class for all LLM interactions.

All LLM-consuming code (classifier, agents) depends on this interface,
not on the concrete OpenAI client. This enables clean mocking in tests
and potential multi-provider support.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class LLMInterface(ABC):
    """Abstract base for LLM clients."""

    @abstractmethod
    async def classify(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Single structured-output call for classification.

        Args:
            system_prompt: The system instructions.
            user_message: The user's query (may include conversation context).
            response_schema: Optional JSON schema for structured output.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            LLMError: If the call fails after retries.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        Generate text — used by agents for producing responses.

        Args:
            system_prompt: The system instructions.
            user_message: The user's query with context.
            stream: If True, returns an async generator yielding chunks.

        Returns:
            Full response string (stream=False) or async generator (stream=True).
        """
        ...


class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""
    pass

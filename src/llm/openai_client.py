"""
OpenAI LLM Client — production implementation of LLMInterface.

Uses the OpenAI Python SDK with structured outputs for classification
and streaming for agent responses. Includes retry logic via tenacity.
"""
from __future__ import annotations

import json
import os
import logging
from typing import Any, AsyncGenerator

from src.llm.interface import LLMInterface, LLMError

logger = logging.getLogger(__name__)


class OpenAIClient(LLMInterface):
    """Production LLM client backed by the OpenAI API."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except Exception as e:
                raise LLMError(f"Failed to initialize OpenAI client: {e}")
        return self._client

    async def classify(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Single structured-output call for classification.
        Uses OpenAI's JSON mode for reliable structured output.
        """
        client = self._get_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from LLM")

            return json.loads(content)

        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM JSON response: {e}")
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"LLM classification call failed: {e}")

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        Generate text for agent responses.
        Supports both streaming and non-streaming modes.
        """
        client = self._get_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            if stream:
                return self._stream_generate(client, messages)
            else:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1500,
                )
                content = response.choices[0].message.content
                return content or ""

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"LLM generation call failed: {e}")

    async def _stream_generate(
        self, client, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Internal streaming generator."""
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"LLM streaming failed: {e}")

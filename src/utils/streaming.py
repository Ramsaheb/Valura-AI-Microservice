"""
Streaming Utilities — helpers for formatting Server-Sent Events (SSE).

Ensures all events adhere to the required SSE pipeline contract:
- No stack traces returned to client
- Structured events for every stage: safety, classification, chunks, errors, done
"""
from __future__ import annotations

import json
from typing import Any

from sse_starlette.sse import ServerSentEvent
from src.core.models import SSEEventType, SafetyVerdict, ClassifierResult


def safety_block_event(verdict: SafetyVerdict) -> ServerSentEvent:
    """Format a safety block event."""
    data = {
        "blocked": True,
        "category": verdict.category,
        "message": verdict.message,
    }
    return ServerSentEvent(event=SSEEventType.SAFETY_BLOCK.value, data=json.dumps(data))


def classification_event(result: ClassifierResult) -> ServerSentEvent:
    """Format a classification result event."""
    data = {
        "intent": result.intent,
        "agent": result.agent,
        "entities": result.entities,
        "safety_flag": result.safety_flag,
    }
    return ServerSentEvent(event=SSEEventType.CLASSIFICATION.value, data=json.dumps(data))


def agent_chunk_event(chunk: str) -> ServerSentEvent:
    """Format a single chunk of an agent's streamed response."""
    data = {"content": chunk}
    return ServerSentEvent(event=SSEEventType.AGENT_CHUNK.value, data=json.dumps(data))


def agent_response_event(response: dict[str, Any]) -> ServerSentEvent:
    """Format a complete structured response from an agent (if not chunking)."""
    return ServerSentEvent(event=SSEEventType.AGENT_RESPONSE.value, data=json.dumps(response))


def error_event(message: str, code: str = "internal_error") -> ServerSentEvent:
    """
    Format an error event.
    Crucially, this hides the stack trace and returns a clean, structured payload.
    """
    data = {"code": code, "message": message}
    return ServerSentEvent(event=SSEEventType.ERROR.value, data=json.dumps(data))


def done_event() -> ServerSentEvent:
    """Format the terminal event for the stream."""
    return ServerSentEvent(event=SSEEventType.DONE.value, data="[DONE]")

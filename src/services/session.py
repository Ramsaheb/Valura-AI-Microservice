"""
Session Memory Service — in-memory persistence for conversation context.

Agents can see prior turns of the same conversation.
For this assignment, we use an in-memory dictionary.
In production, this would be backed by Postgres or Redis.
"""
from __future__ import annotations

import asyncio
from typing import Any

# Global in-memory session store: session_id -> list of turns
_SESSIONS: dict[str, list[dict[str, Any]]] = {}

# Lock for thread-safe concurrent access
_lock = asyncio.Lock()


async def get_history(session_id: str | None) -> list[dict[str, Any]]:
    """
    Get conversation history for a session.

    Returns:
        List of turn dictionaries (role, content). Empty list if session not found.
    """
    if not session_id:
        return []

    async with _lock:
        # Return a copy to prevent accidental mutation by callers
        history = _SESSIONS.get(session_id, [])
        return [dict(turn) for turn in history]


async def add_turn(session_id: str | None, role: str, content: str) -> None:
    """
    Add a single turn to a session's history.

    Args:
        session_id: The session identifier. If None, operation is a no-op.
        role: "user" or "assistant".
        content: The text content of the turn.
    """
    if not session_id or not content.strip():
        return

    async with _lock:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = []
        
        # We store up to 20 turns to prevent context window explosion
        _SESSIONS[session_id].append({"role": role, "content": content})
        if len(_SESSIONS[session_id]) > 20:
            _SESSIONS[session_id] = _SESSIONS[session_id][-20:]


async def clear_session(session_id: str) -> None:
    """Clear a session's history (useful for testing)."""
    async with _lock:
        if session_id in _SESSIONS:
            del _SESSIONS[session_id]

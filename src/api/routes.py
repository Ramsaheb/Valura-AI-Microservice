"""
API Routes — the HTTP SSE endpoint.

Orchestrates the entire AI pipeline:
  1. Load user profile
  2. Load session history
  3. Safety Guard (if blocked -> return early)
  4. Intent Classifier (if fail -> rule-based fallback)
  5. Router -> Agent Execution
  6. Return response via SSE stream

Pipeline timeout: 30 seconds. This covers the full request lifecycle from
classification through agent execution. Chosen because yfinance data fetches
typically complete in <2s, and even a slow LLM call should finish in <10s.
30s provides generous headroom while preventing indefinite hangs.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse

from src.core.models import QueryRequest
from src.core.safety import check as safety_check
from src.core.classifier import classify
from src.core.router import route
from src.services.session import get_history, add_turn
from src.llm.openai_client import OpenAIClient
from src.llm.mock_llm import MockLLM
from src.utils.streaming import (
    safety_block_event,
    classification_event,
    agent_response_event,
    error_event,
    done_event,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Pipeline timeout: 30 seconds covers classification + agent execution.
# yfinance typically completes in <2s, LLM calls in <10s. 30s provides
# generous headroom while preventing indefinite hangs from network issues.
PIPELINE_TIMEOUT_SECONDS = 30

# Dependency injection for LLM client
# In production, this would be an OpenAIClient. For tests without a key,
# we fall back to MockLLM if OpenAI client initialization fails.
def get_llm_client():
    import os
    if os.getenv("APP_ENV") == "test":
        return MockLLM()
    try:
        # Try to initialize OpenAI client
        # It will fail lazily if no API key is set when a request is made,
        # but we can also use MockLLM as a robust fallback.
        return OpenAIClient()
    except Exception:
        return MockLLM()


# Mock user DB loader (simulating a database fetch)
def _load_user(user_id: str) -> dict[str, Any]:
    """Load user profile from fixtures."""
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixture_path = os.path.join(base_dir, "fixtures", "users", f"{user_id}.json")
    
    if not os.path.exists(fixture_path):
        # Graceful fallback for demo
        fallback_path = os.path.join(base_dir, "fixtures", "users", "user_001_active_trader_us.json")
        if os.path.exists(fallback_path):
            fixture_path = fallback_path
        else:
            raise FileNotFoundError(f"User profile {user_id} not found")
        
    with open(fixture_path, "r") as f:
        return json.load(f)


async def _run_pipeline(request: QueryRequest, llm: Any, user: dict, history: list):
    """
    Core pipeline logic: classify → route → execute agent.
    Separated so it can be wrapped in asyncio.wait_for for timeout enforcement.

    Returns a list of SSE events to yield.
    """
    events = []

    # 4. Intent Classifier (LLM with rule-based fallback)
    classifier_result = await classify(
        query=request.query,
        llm=llm,
        history=history
    )
    events.append(classification_event(classifier_result))

    # 5. Agent Routing
    agent = route(classifier_result)
    
    # 6. Agent Execution & Streaming
    import json as json_mod
    full_response_parts = []
    
    async for chunk in agent.stream(user, request.query, classifier_result, llm):
        if chunk.strip():
            try:
                parsed_chunk = json_mod.loads(chunk)
                events.append(agent_response_event(parsed_chunk))
            except json_mod.JSONDecodeError:
                from src.utils.streaming import agent_chunk_event
                events.append(agent_chunk_event(chunk))
            full_response_parts.append(chunk)

    # 7. Update Session History
    if request.session_id and full_response_parts:
        await add_turn(request.session_id, "user", request.query)
        await add_turn(request.session_id, "assistant", "Agent returned structured response.")

    return events


@router.post("/query")
async def handle_query(request: QueryRequest, llm: Any = Depends(get_llm_client)):
    """
    Main endpoint for the Valura AI microservice.
    Must return an SSE stream.
    """
    
    async def event_generator():
        try:
            # 1. Load User
            try:
                user = _load_user(request.user_id)
            except FileNotFoundError:
                yield error_event("User not found", code="user_not_found")
                yield done_event()
                return
            except Exception as e:
                logger.error(f"Error loading user {request.user_id}: {e}")
                yield error_event("Failed to load user profile")
                yield done_event()
                return

            # 2. Session Context
            history = await get_history(request.session_id)

            # 3. Safety Guard (Synchronous, fast)
            safety_verdict = safety_check(request.query)
            if safety_verdict.blocked:
                logger.warning(f"Safety block triggered: {safety_verdict.category}")
                yield safety_block_event(safety_verdict)
                yield done_event()
                return

            # 4-7. Classification → Routing → Agent → Session update
            # Wrapped in a timeout to prevent indefinite hangs
            try:
                pipeline_events = await asyncio.wait_for(
                    _run_pipeline(request, llm, user, history),
                    timeout=PIPELINE_TIMEOUT_SECONDS,
                )
                for event in pipeline_events:
                    yield event
            except asyncio.TimeoutError:
                logger.error(f"Pipeline timed out after {PIPELINE_TIMEOUT_SECONDS}s")
                yield error_event(
                    f"Request timed out after {PIPELINE_TIMEOUT_SECONDS} seconds. "
                    f"Please try again.",
                    code="timeout",
                )
                yield done_event()
                return
            except Exception as e:
                logger.exception(f"Pipeline failed during execution")
                yield error_event(f"Agent execution failed: {str(e)}")
                yield done_event()
                return

            yield done_event()

        except Exception as e:
            # Catch-all to prevent stack traces from leaking
            logger.exception("Unhandled exception in SSE generator")
            yield error_event("An unexpected internal error occurred.")
            yield done_event()

    return EventSourceResponse(event_generator())

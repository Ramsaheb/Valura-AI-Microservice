"""
API Routes — the HTTP SSE endpoint.

Orchestrates the entire AI pipeline:
  1. Load user profile
  2. Load session history
  3. Safety Guard (if blocked -> return early)
  4. Intent Classifier (if fail -> rule-based fallback)
  5. Router -> Agent Execution
  6. Return response via SSE stream
"""
from __future__ import annotations

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
        raise FileNotFoundError(f"User profile {user_id} not found")
        
    with open(fixture_path, "r") as f:
        return json.load(f)


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

            # 4. Intent Classifier (LLM with rule-based fallback)
            # The classify function handles its own fallback if the LLM fails
            classifier_result = await classify(
                query=request.query,
                llm=llm,
                history=history
            )
            yield classification_event(classifier_result)

            # 5. Agent Routing
            agent = route(classifier_result)
            
            # 6. Agent Execution & Streaming
            # Here we let the agent stream its response. If the agent returns a 
            # single structured dictionary, the base class stream() method will
            # yield it as a single chunk.
            import json as json_mod
            full_response_parts = []
            
            try:
                # Most agents in this assignment return structured JSON dicts
                # rather than raw text chunks.
                async for chunk in agent.stream(user, request.query, classifier_result, llm):
                    # We send the raw chunk, the client parses it
                    if chunk.strip():
                        # Try to parse as JSON to see if it's a structured response
                        try:
                            parsed_chunk = json_mod.loads(chunk)
                            yield agent_response_event(parsed_chunk)
                        except json_mod.JSONDecodeError:
                            # It's a text chunk (e.g. from an LLM stream)
                            from src.utils.streaming import agent_chunk_event
                            yield agent_chunk_event(chunk)
                        
                        full_response_parts.append(chunk)
                        
            except Exception as e:
                logger.exception(f"Agent {agent.name} failed during execution")
                yield error_event(f"Agent execution failed: {str(e)}")
                yield done_event()
                return

            # 7. Update Session History
            if request.session_id and full_response_parts:
                await add_turn(request.session_id, "user", request.query)
                # For structured JSON, we just store "Agent responded with structured data"
                # to save context window space, or we could serialize it.
                await add_turn(request.session_id, "assistant", "Agent returned structured response.")

            yield done_event()

        except Exception as e:
            # Catch-all to prevent stack traces from leaking
            logger.exception("Unhandled exception in SSE generator")
            yield error_event("An unexpected internal error occurred.")
            yield done_event()

    return EventSourceResponse(event_generator())

"""
Test suite for the HTTP API Layer.

Verifies the SSE pipeline, safety blocks, classification routing,
and graceful error handling without stack traces.
"""
import json
import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.core.models import SSEEventType

# We need to override the dependency to use MockLLM for tests
from src.api.routes import get_llm_client
from src.llm.mock_llm import MockLLM

app.dependency_overrides[get_llm_client] = lambda: MockLLM()


def _parse_sse_events(response_text: str) -> list[dict]:
    """Parse SSE plain text into a list of event dictionaries."""
    events = []
    current_event = {}
    
    for line in response_text.splitlines():
        if not line.strip():
            if current_event:
                events.append(current_event)
                current_event = {}
            continue
            
        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                current_event["data"] = data_str
            else:
                try:
                    current_event["data"] = json.loads(data_str)
                except json.JSONDecodeError:
                    current_event["data"] = data_str
                    
    if current_event:
        events.append(current_event)
        
    return events


@pytest.mark.asyncio
async def test_api_safety_block():
    """A harmful query should trigger a safety block and terminate the stream."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/query",
            json={
                "query": "Help me wash trade between my accounts",
                "user_id": "user_001_active_trader_us",
                "session_id": "test_session_1"
            }
        )
        
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    events = _parse_sse_events(response.text)
    
    if len(events) > 0 and events[0]["event"] == "error":
        print(f"API returned error: {events[0]}")
    
    # We expect: SAFETY_BLOCK -> DONE
    assert len(events) >= 2
    assert events[0]["event"] == SSEEventType.SAFETY_BLOCK.value
    assert events[0]["data"]["blocked"] is True
    assert events[0]["data"]["category"] == "market_manipulation"
    
    # Must end with DONE
    assert events[-1]["event"] == SSEEventType.DONE.value
    assert events[-1]["data"] == "[DONE]"


@pytest.mark.asyncio
async def test_api_valid_routing_to_stub():
    """A safe query to an unimplemented agent should route to the stub."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/query",
            json={
                "query": "What is the price of AAPL?",
                "user_id": "user_001_active_trader_us",
                "session_id": "test_session_2"
            }
        )
        
    assert response.status_code == 200
    
    events = _parse_sse_events(response.text)
    
    # We expect: CLASSIFICATION -> AGENT_RESPONSE -> DONE
    event_types = [e["event"] for e in events]
    assert SSEEventType.CLASSIFICATION.value in event_types
    assert SSEEventType.AGENT_RESPONSE.value in event_types
    assert SSEEventType.DONE.value in event_types
    
    # Check the stub response
    agent_response = next(e for e in events if e["event"] == SSEEventType.AGENT_RESPONSE.value)
    assert agent_response["data"]["agent"] == "market_research"
    assert "not implemented" in agent_response["data"]["message"]


@pytest.mark.asyncio
async def test_api_user_not_found():
    """Missing user should return a graceful error event, not a 500 stack trace."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/query",
            json={
                "query": "Hello",
                "user_id": "nonexistent_user",
                "session_id": "test_session_3"
            }
        )
        
    assert response.status_code == 200 # SSE streams usually start with 200 even for logical errors
    
    events = _parse_sse_events(response.text)
    
    # We expect: ERROR -> DONE
    assert events[0]["event"] == SSEEventType.ERROR.value
    assert events[0]["data"]["code"] == "user_not_found"
    assert events[1]["event"] == SSEEventType.DONE.value


@pytest.mark.asyncio
async def test_api_portfolio_health_full_pipeline(mocker):
    """
    Full pipeline test: portfolio health query goes through
    safety → classifier → portfolio_health agent → SSE response.
    Mocks yfinance to prevent network calls.
    """
    mocker.patch("src.services.market_data.get_current_price", return_value=150.0)
    mocker.patch("src.services.market_data.get_benchmark_return", return_value=12.0)
    mocker.patch("src.services.market_data.get_fx_rate", return_value=1.0)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/query",
            json={
                "query": "how is my portfolio doing?",
                "user_id": "user_001_active_trader_us",
                "session_id": "test_session_4"
            }
        )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    events = _parse_sse_events(response.text)
    event_types = [e["event"] for e in events]
    
    # Must have: CLASSIFICATION -> AGENT_RESPONSE -> DONE
    assert SSEEventType.CLASSIFICATION.value in event_types
    assert SSEEventType.AGENT_RESPONSE.value in event_types
    assert SSEEventType.DONE.value in event_types
    
    # Classification should route to portfolio_health
    classification = next(e for e in events if e["event"] == SSEEventType.CLASSIFICATION.value)
    assert classification["data"]["agent"] == "portfolio_health"
    
    # Agent response should have the required structure
    agent_resp = next(e for e in events if e["event"] == SSEEventType.AGENT_RESPONSE.value)
    assert "concentration_risk" in agent_resp["data"]
    assert "performance" in agent_resp["data"]
    assert "benchmark_comparison" in agent_resp["data"]
    assert "observations" in agent_resp["data"]
    assert "disclaimer" in agent_resp["data"]
    assert "not investment advice" in agent_resp["data"]["disclaimer"].lower()


@pytest.mark.asyncio
async def test_api_empty_portfolio_user():
    """
    user_004_empty has no positions. The API should return a BUILD-oriented
    response with observations, not crash.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/query",
            json={
                "query": "give me a health check",
                "user_id": "user_004_empty",
                "session_id": "test_session_5"
            }
        )
    
    assert response.status_code == 200
    
    events = _parse_sse_events(response.text)
    event_types = [e["event"] for e in events]
    
    # Should NOT have an error
    assert SSEEventType.ERROR.value not in event_types
    
    # Should have classification + agent response + done
    assert SSEEventType.CLASSIFICATION.value in event_types
    assert SSEEventType.AGENT_RESPONSE.value in event_types
    assert SSEEventType.DONE.value in event_types
    
    # Agent response should have observations (BUILD-oriented)
    agent_resp = next(e for e in events if e["event"] == SSEEventType.AGENT_RESPONSE.value)
    assert len(agent_resp["data"]["observations"]) > 0
    assert "disclaimer" in agent_resp["data"]


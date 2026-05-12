"""
Test suite for the Portfolio Health agent.

Tests whether the agent correctly surfaces concentration risk, provides disclaimers,
and gracefully handles empty portfolios without crashing.
"""
import pytest
from src.agents.portfolio_health import run


@pytest.mark.asyncio
async def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user, mock_llm):
    """
    user_004 has no positions. Agent must not crash.
    """
    user = load_user("usr_004")
    response = await run(user, query="how is my portfolio?", llm=mock_llm)

    assert response is not None
    assert "disclaimer" in response
    # We expect a sensible observation telling them how to build
    assert len(response["observations"]) > 0


@pytest.mark.asyncio
async def test_portfolio_health_flags_concentration(load_user, mock_llm, mocker):
    """
    user_003 has ~60% in NVDA. Agent must surface this.
    """
    # Mock yfinance to prevent external calls and ensure deterministic tests
    mocker.patch("src.services.market_data.get_current_price", return_value=100.0)
    mocker.patch("src.services.market_data.get_benchmark_return", return_value=10.0)
    mocker.patch("src.services.market_data.get_fx_rate", return_value=1.0)
    
    user = load_user("usr_003")
    response = await run(user, query="concentration check", llm=mock_llm)

    assert response["concentration_risk"]["flag"] in {"high", "warning", "medium"}


@pytest.mark.asyncio
async def test_portfolio_health_includes_disclaimer(load_user, mock_llm, mocker):
    # Mock yfinance to prevent external calls
    mocker.patch("src.services.market_data.get_current_price", return_value=100.0)
    mocker.patch("src.services.market_data.get_benchmark_return", return_value=10.0)
    mocker.patch("src.services.market_data.get_fx_rate", return_value=1.0)
    
    user = load_user("usr_001")
    response = await run(user, query="health check", llm=mock_llm)
    assert response["disclaimer"]
    assert "not investment advice" in response["disclaimer"].lower()

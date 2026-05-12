"""
Portfolio Health Agent — the primary agent for novice investors.

Orchestrates the portfolio health check process:
  1. Fetches current market prices and FX rates for the user's holdings
  2. Computes concentration, performance, and benchmark metrics
  3. Generates actionable, plain-language observations
  4. Returns a structured PortfolioHealthResponse

Handles empty portfolios gracefully by returning a BUILD-oriented response
instead of crashing.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import ClassifierResult, PortfolioHealthResponse, ConcentrationRisk, Performance, BenchmarkComparison
from src.services.market_data import get_current_price, get_benchmark_return, get_fx_rate
from src.services.portfolio import (
    compute_concentration,
    compute_performance,
    compute_benchmark_comparison,
    generate_observations,
    generate_empty_portfolio_observations,
)

logger = logging.getLogger(__name__)


class PortfolioHealthAgent(BaseAgent):
    """Agent that analyzes a user's portfolio and provides a health check."""

    @property
    def name(self) -> str:
        return "portfolio_health"

    async def run(
        self,
        user: dict[str, Any],
        query: str,
        classifier_result: ClassifierResult,
        llm: Any = None,
    ) -> dict[str, Any]:
        """
        Execute the portfolio health check.

        Args:
            user: User profile dictionary.
            query: The user's original query.
            classifier_result: The classification result.
            llm: Optional LLM client (unused currently as we use pure computation).

        Returns:
            Dictionary matching the PortfolioHealthResponse schema.
        """
        positions = user.get("positions", [])
        preferences = user.get("preferences", {})
        base_currency = user.get("base_currency", "USD")
        reporting_currency = preferences.get("reporting_currency", base_currency)

        # 1. Handle empty portfolio (user_004)
        if not positions:
            logger.info(f"User {user.get('user_id')} has empty portfolio. Generating BUILD response.")
            response = PortfolioHealthResponse()
            response.observations = generate_empty_portfolio_observations(user)
            return response.model_dump()

        # 2. Fetch market data
        current_prices: dict[str, float] = {}
        fx_rates: dict[str, float] = {}

        # Get prices for all holdings
        for pos in positions:
            ticker = pos.get("ticker", "")
            if ticker and ticker not in current_prices:
                price = get_current_price(ticker)
                if price is not None:
                    current_prices[ticker] = price

        # Get FX rates if reporting currency is different from position currency
        for pos in positions:
            pos_currency = pos.get("currency", base_currency)
            if pos_currency != reporting_currency and pos_currency not in fx_rates:
                rate = get_fx_rate(pos_currency, reporting_currency)
                fx_rates[pos_currency] = rate

        # Get benchmark return
        benchmark_name = preferences.get("preferred_benchmark", "S&P 500")
        
        # We need a start date for the benchmark. We'll use the earliest purchase date,
        # or 1 year ago if we can't determine it.
        from datetime import datetime, timedelta
        earliest_date = None
        for pos in positions:
            purchased = pos.get("purchased_at")
            if purchased:
                try:
                    d = datetime.strptime(purchased, "%Y-%m-%d")
                    if earliest_date is None or d < earliest_date:
                        earliest_date = d
                except ValueError:
                    pass
        
        start_date_str = (earliest_date or (datetime.now() - timedelta(days=365))).strftime("%Y-%m-%d")
        benchmark_return = get_benchmark_return(benchmark_name, start_date=start_date_str)

        # 3. Compute metrics
        concentration = compute_concentration(
            positions=positions,
            current_prices=current_prices,
            fx_rates=fx_rates,
            base_currency=reporting_currency,
        )

        performance = compute_performance(
            positions=positions,
            current_prices=current_prices,
            fx_rates=fx_rates,
            base_currency=reporting_currency,
        )

        benchmark_comparison = compute_benchmark_comparison(
            portfolio_return_pct=performance.total_return_pct,
            benchmark_return_pct=benchmark_return,
            benchmark_name=benchmark_name,
        )

        # 4. Generate observations
        observations = generate_observations(
            concentration=concentration,
            performance=performance,
            benchmark=benchmark_comparison,
            user=user,
            positions=positions,
            current_prices=current_prices,
        )

        # 5. Build response
        response = PortfolioHealthResponse(
            concentration_risk=concentration,
            performance=performance,
            benchmark_comparison=benchmark_comparison,
            observations=observations,
        )

        return response.model_dump()

# Also provide a top-level `run` function to match the test's expected signature
# if the test imports `run` directly from `src.agents.portfolio_health`
async def run(user: dict[str, Any], query: str = "", classifier_result: Any = None, llm: Any = None) -> dict[str, Any]:
    agent = PortfolioHealthAgent()
    if classifier_result is None:
        classifier_result = ClassifierResult(agent="portfolio_health")
    return await agent.run(user, query, classifier_result, llm)

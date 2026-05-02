"""
Portfolio Service — pure computation for portfolio analytics.

All methods are stateless pure functions. They take position data and prices,
and return structured metrics. No LLM calls, no network calls.

This module is the computational engine behind the Portfolio Health Agent.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

from src.core.models import (
    ConcentrationRisk,
    Performance,
    BenchmarkComparison,
    Observation,
)


def compute_concentration(
    positions: list[dict[str, Any]],
    current_prices: dict[str, float],
    fx_rates: dict[str, float] | None = None,
    base_currency: str = "USD",
) -> ConcentrationRisk:
    """
    Compute concentration risk metrics.

    Args:
        positions: List of position dicts from user profile.
        current_prices: Map of ticker → current price in local currency.
        fx_rates: Map of currency → base_currency rate.
        base_currency: The user's reporting currency.

    Returns:
        ConcentrationRisk with top position %, top 3 %, and risk flag.
    """
    if not positions:
        return ConcentrationRisk(top_position_pct=0.0, top_3_positions_pct=0.0, flag="low")

    fx = fx_rates or {}

    # Calculate current value of each position in base currency
    values = []
    for pos in positions:
        ticker = pos.get("ticker", "")
        quantity = pos.get("quantity", 0)
        price = current_prices.get(ticker)

        if price is None:
            # Fallback to avg_cost if price unavailable
            price = pos.get("avg_cost", 0)

        currency = pos.get("currency", base_currency)
        rate = fx.get(currency, 1.0)
        value = quantity * price * rate
        values.append((ticker, value))

    total = sum(v for _, v in values)
    if total <= 0:
        return ConcentrationRisk(top_position_pct=0.0, top_3_positions_pct=0.0, flag="low")

    # Sort by value descending
    values.sort(key=lambda x: x[1], reverse=True)

    top_1_pct = round((values[0][1] / total) * 100, 1)
    top_3_pct = round((sum(v for _, v in values[:3]) / total) * 100, 1)

    # Flag thresholds
    if top_1_pct >= 40:
        flag = "high"
    elif top_1_pct >= 25:
        flag = "medium"
    else:
        flag = "low"

    return ConcentrationRisk(
        top_position_pct=top_1_pct,
        top_3_positions_pct=top_3_pct,
        flag=flag,
    )


def compute_performance(
    positions: list[dict[str, Any]],
    current_prices: dict[str, float],
    fx_rates: dict[str, float] | None = None,
    base_currency: str = "USD",
) -> Performance:
    """
    Compute portfolio performance metrics.

    Returns total return % and annualized return % based on
    cost basis vs current value.
    """
    if not positions:
        return Performance(total_return_pct=0.0, annualized_return_pct=0.0)

    fx = fx_rates or {}
    total_cost = 0.0
    total_current = 0.0
    earliest_date = None

    for pos in positions:
        ticker = pos.get("ticker", "")
        quantity = pos.get("quantity", 0)
        avg_cost = pos.get("avg_cost", 0)
        currency = pos.get("currency", base_currency)
        rate = fx.get(currency, 1.0)

        cost = quantity * avg_cost * rate
        price = current_prices.get(ticker, avg_cost)
        current = quantity * price * rate

        total_cost += cost
        total_current += current

        # Track earliest purchase date for annualization
        purchased = pos.get("purchased_at")
        if purchased:
            try:
                d = datetime.strptime(purchased, "%Y-%m-%d")
                if earliest_date is None or d < earliest_date:
                    earliest_date = d
            except ValueError:
                pass

    if total_cost <= 0:
        return Performance(total_return_pct=0.0, annualized_return_pct=0.0)

    total_return = ((total_current - total_cost) / total_cost) * 100

    # Annualize
    annualized = 0.0
    if earliest_date:
        years = (datetime.now() - earliest_date).days / 365.25
        if years > 0 and total_cost > 0:
            ratio = total_current / total_cost
            if ratio > 0:
                annualized = (math.pow(ratio, 1 / years) - 1) * 100

    return Performance(
        total_return_pct=round(total_return, 1),
        annualized_return_pct=round(annualized, 1),
    )


def compute_benchmark_comparison(
    portfolio_return_pct: float,
    benchmark_return_pct: Optional[float],
    benchmark_name: str = "S&P 500",
) -> BenchmarkComparison:
    """
    Compare portfolio return against a benchmark.
    """
    bench_return = benchmark_return_pct if benchmark_return_pct is not None else 0.0
    alpha = round(portfolio_return_pct - bench_return, 1)

    return BenchmarkComparison(
        benchmark=benchmark_name,
        portfolio_return_pct=round(portfolio_return_pct, 1),
        benchmark_return_pct=round(bench_return, 1),
        alpha_pct=alpha,
    )


def generate_observations(
    concentration: ConcentrationRisk,
    performance: Performance,
    benchmark: BenchmarkComparison,
    user: dict[str, Any],
    positions: list[dict[str, Any]],
    current_prices: dict[str, float],
) -> list[Observation]:
    """
    Generate actionable, novice-friendly observations.

    Surfaces the 1-2 things that matter most, in plain language.
    """
    observations = []

    # --- Concentration risk ---
    if concentration.flag == "high":
        # Find the top holding
        values = []
        for pos in positions:
            ticker = pos.get("ticker", "")
            price = current_prices.get(ticker, pos.get("avg_cost", 0))
            values.append((ticker, pos.get("quantity", 0) * price))
        values.sort(key=lambda x: x[1], reverse=True)
        top_ticker = values[0][0] if values else "unknown"

        observations.append(Observation(
            severity="warning",
            text=(
                f"{concentration.top_position_pct}% of your portfolio is in {top_ticker} "
                f"— that's highly concentrated. If {top_ticker} drops significantly, "
                f"your whole portfolio takes a big hit. Consider diversifying across "
                f"more holdings or sectors to reduce this risk."
            ),
        ))
    elif concentration.flag == "medium":
        observations.append(Observation(
            severity="info",
            text=(
                f"Your top position is {concentration.top_position_pct}% of your portfolio. "
                f"This is moderate concentration — keep an eye on it as it grows."
            ),
        ))

    # --- Performance vs benchmark ---
    if benchmark.alpha_pct > 2:
        observations.append(Observation(
            severity="info",
            text=(
                f"Your portfolio is outperforming the {benchmark.benchmark} by "
                f"{benchmark.alpha_pct}% — nice work! Keep in mind that past "
                f"performance doesn't guarantee future results."
            ),
        ))
    elif benchmark.alpha_pct < -5:
        observations.append(Observation(
            severity="warning",
            text=(
                f"Your portfolio is underperforming the {benchmark.benchmark} by "
                f"{abs(benchmark.alpha_pct)}%. You might want to review your holdings "
                f"and consider whether a low-cost index fund might serve you better."
            ),
        ))

    # --- Risk profile alignment ---
    risk_profile = user.get("risk_profile", "moderate")
    preferences = user.get("preferences", {})

    if risk_profile == "conservative" and concentration.flag != "low":
        observations.append(Observation(
            severity="warning",
            text=(
                f"Your risk profile is set to 'conservative', but your portfolio "
                f"shows {concentration.flag} concentration risk. Consider spreading "
                f"your investments across more diversified, lower-risk holdings."
            ),
        ))

    # --- Income-focused commentary for retirees ---
    if preferences.get("income_focus"):
        observations.append(Observation(
            severity="info",
            text=(
                f"As an income-focused investor, your dividend-paying holdings are "
                f"key to your strategy. Make sure your portfolio maintains a healthy "
                f"mix of reliable dividend payers with a track record of consistent payouts."
            ),
        ))

    # --- Sector concentration (all tech, all finance, etc.) ---
    if len(positions) >= 3:
        exchanges = set(pos.get("exchange", "") for pos in positions)
        if len(exchanges) == 1 and exchanges != {""}:
            observations.append(Observation(
                severity="info",
                text=(
                    f"All your holdings are on the same exchange. Consider adding "
                    f"international exposure to reduce geographic concentration risk."
                ),
            ))

    # --- Multi-currency note ---
    currencies = set(pos.get("currency", "USD") for pos in positions)
    if len(currencies) > 1:
        observations.append(Observation(
            severity="info",
            text=(
                f"Your portfolio spans {len(currencies)} currencies "
                f"({', '.join(sorted(currencies))}). Currency fluctuations will "
                f"affect your returns. This provides natural diversification but "
                f"also adds FX risk."
            ),
        ))

    # Ensure at least one observation
    if not observations:
        observations.append(Observation(
            severity="info",
            text="Your portfolio looks well-balanced. Continue monitoring regularly.",
        ))

    return observations


def generate_empty_portfolio_observations(user: dict[str, Any]) -> list[Observation]:
    """
    Generate BUILD-oriented observations for a user with no positions.
    """
    name = user.get("name", "there")
    risk_profile = user.get("risk_profile", "moderate")
    age = user.get("age", 30)

    observations = [
        Observation(
            severity="info",
            text=(
                f"Welcome, {name}! You're all set up with a verified account "
                f"but haven't made your first investment yet. That's a great "
                f"starting position — let's get you going."
            ),
        ),
        Observation(
            severity="info",
            text=(
                f"Based on your '{risk_profile}' risk profile and age ({age}), "
                f"a good starting point could be a diversified index fund like "
                f"a total market ETF (e.g., VTI for US markets) or a global "
                f"fund (e.g., VXUS for international exposure)."
            ),
        ),
        Observation(
            severity="info",
            text=(
                f"Consider starting with a small, regular investment "
                f"(dollar-cost averaging) rather than trying to time the market. "
                f"Even $100-500/month into a diversified fund can grow "
                f"significantly over time."
            ),
        ),
    ]

    return observations

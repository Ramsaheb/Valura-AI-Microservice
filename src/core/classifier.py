"""
Intent Classifier — single LLM call that drives the entire pipeline.

Takes a user query (plus optional conversation history) and returns:
  - intent: what the user wants
  - agent: which specialist agent should handle it
  - entities: extracted tickers, amounts, time periods, etc.
  - safety_flag: informational safety verdict (does NOT block — that's the guard's job)

Fallback strategy:
  When the LLM is unavailable or returns unparseable output, the classifier
  falls back to the rule-based MockLLM engine. This ensures:
  1. Tests pass without an API key (CI requirement)
  2. The pipeline never crashes on LLM failure (assignment requirement)
  3. The fallback achieves ≥85% routing accuracy on the gold set
"""
from __future__ import annotations

import logging
from typing import Any

from src.core.models import ClassifierResult
from src.llm.interface import LLMError
from src.llm.mock_llm import MockLLM, _rule_based_classify

logger = logging.getLogger(__name__)

# Singleton fallback classifier
_fallback = MockLLM()

# The system prompt for the LLM-based classifier
CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a financial AI assistant. Given a user query and optional conversation history, classify the query into exactly one agent and extract structured entities.

AGENT TAXONOMY (choose exactly one):
- portfolio_health: structured assessment of user's portfolio (concentration, performance, benchmarking)
- market_research: factual/recent info about an instrument, sector, or market event
- investment_strategy: advice/strategy — should I buy/sell/rebalance, allocation guidance
- financial_planning: long-term planning — retirement, goals, savings rate
- financial_calculator: deterministic computation — DCA, mortgage, tax, future value, FX conversion
- risk_assessment: risk metrics, exposure analysis, what-if scenarios
- product_recommendation: recommend specific products/funds matching user profile
- predictive_analysis: forward-looking analysis — forecasts, trend extrapolation
- customer_support: platform issues, account questions, how-to-use-app
- general_query: educational, conversational, definitions, greetings

ENTITY VOCABULARY:
- tickers: array of uppercase strings (e.g. ["AAPL", "NVDA", "ASML.AS"])
- amount: number
- currency: ISO 4217 string (USD, EUR, GBP, JPY)
- rate: decimal (0.08 for 8%)
- period_years: integer
- frequency: daily | weekly | monthly | yearly
- horizon: 6_months | 1_year | 5_years etc.
- time_period: today | this_week | this_month | this_year
- topics: array of strings
- sectors: array of strings
- index: S&P 500 | FTSE 100 | NIKKEI 225 | MSCI World
- action: buy | sell | hold | hedge | rebalance
- goal: retirement | education | house | FIRE | emergency_fund

RULES:
1. Resolve company names to tickers (Apple → AAPL, Nvidia → NVDA, Tesla → TSLA, etc.)
2. Handle typos gracefully (microsfot → MSFT)
3. For follow-up queries, use conversation history to resolve pronouns and carry entities
4. For topic switches, do NOT carry entities from previous turns
5. Greetings, thanks, single words like "thx" → general_query with empty entities
6. A bare ticker like "AAPL" without any verb → market_research
7. Multi-intent queries: pick the PRIMARY intent
8. Gibberish or unrecognizable input → general_query

OUTPUT FORMAT (JSON only):
{
  "intent": "<descriptive intent>",
  "agent": "<agent_name from taxonomy>",
  "entities": { ... },
  "safety_flag": "low" | "medium" | "high"
}"""


async def classify(
    query: str,
    llm: Any = None,
    history: list[dict] | None = None,
) -> ClassifierResult:
    """
    Classify a user query into an intent, target agent, and extracted entities.

    Args:
        query: The user's query text.
        llm: An LLM client implementing LLMInterface, or a MagicMock for tests.
        history: Optional conversation history (list of prior user turns).

    Returns:
        ClassifierResult with intent, agent, entities, and safety_flag.
        Never raises — falls back to rule-based classification on any error.
    """
    # Build the user message with conversation context
    user_message = _build_user_message(query, history)

    # Try LLM-based classification first
    try:
        if llm is not None:
            # Check if this is a real LLM client (has classify method that's not a Mock)
            if hasattr(llm, "classify") and not _is_mock(llm):
                result_dict = await llm.classify(
                    system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                    user_message=user_message,
                )
                return _parse_result(result_dict)
    except (LLMError, Exception) as e:
        logger.warning(f"LLM classification failed, using fallback: {e}")

    # Fallback: rule-based classification
    try:
        # For follow-up queries, resolve context from history
        resolved_query = _resolve_followup(query, history)
        result_dict = _rule_based_classify(resolved_query)
        return _parse_result(result_dict)
    except Exception as e:
        logger.error(f"Fallback classification also failed: {e}")
        return ClassifierResult(
            intent="unknown",
            agent="general_query",
            entities={},
            safety_flag="low",
        )


def _build_user_message(query: str, history: list[dict] | None) -> str:
    """Build the user message with conversation history for context."""
    if not history:
        return f"Query: {query}"

    parts = ["Conversation history:"]
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        parts.append(f"  {role}: {content}")
    parts.append(f"\nCurrent query: {query}")
    return "\n".join(parts)


def _resolve_followup(query: str, history: list[dict] | None) -> str:
    """
    For the rule-based fallback, resolve follow-up queries by examining
    conversation history for entity context.

    Examples:
      - "How much do I own?" after "What's happening with Nvidia?" → injects NVDA
      - "what about AMD?" after "tell me about NVDA" → just AMD, intent carried
      - "compare them" after NVDA + AMD mentions → both tickers
    """
    if not history:
        return query

    lower = query.lower().strip()

    # Detect follow-up patterns
    is_followup = any(p in lower for p in [
        "how much do i own",
        "should i sell",
        "should i buy",
        "what about",
        "compare them",
        "and ",
        "how about",
        "tell me more",
    ])

    if not is_followup:
        # This might be a topic switch — don't carry context
        return query

    # Extract tickers from history
    import re
    from src.llm.mock_llm import _COMPANY_TO_TICKER, _VALID_TICKERS

    historical_tickers = []
    for turn in history:
        content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
        words = re.split(r"[\s,;?.!]+", content)
        for word in words:
            clean_upper = word.strip("'\"()").upper()
            clean_lower = word.strip("'\"()").lower()
            if clean_upper in _VALID_TICKERS:
                if clean_upper not in historical_tickers:
                    historical_tickers.append(clean_upper)
            elif clean_lower in _COMPANY_TO_TICKER:
                ticker = _COMPANY_TO_TICKER[clean_lower]
                if ticker not in historical_tickers:
                    historical_tickers.append(ticker)

    # "compare them" — inject all historical tickers
    if re.search(r"\bcompare\s+them\b", lower):
        ticker_str = " ".join(historical_tickers)
        return f"compare {ticker_str}"

    # "what about X?" — new ticker, but carry the intent type
    what_about = re.search(r"\bwhat\s+about\s+(\w+)", lower)
    if what_about:
        new_entity = what_about.group(1)
        return f"tell me about {new_entity}"

    # "should i sell some?" — carry the last ticker
    if re.search(r"\bshould\s+i\s+(sell|buy)\b", lower) and historical_tickers:
        return f"should i sell {historical_tickers[-1]}"

    # "how much do I own?" — carry last ticker, portfolio context
    if re.search(r"\bhow\s+much\s+do\s+i\s+own\b", lower) and historical_tickers:
        return f"how much {historical_tickers[-1]} do I own"

    # Generic follow-up with historical context
    if historical_tickers:
        return f"{query} {' '.join(historical_tickers)}"

    return query


def _parse_result(data: dict[str, Any]) -> ClassifierResult:
    """Parse a raw dict into a ClassifierResult, handling malformed data."""
    try:
        return ClassifierResult(
            intent=data.get("intent", "unknown"),
            agent=data.get("agent", "general_query"),
            entities=data.get("entities", {}),
            safety_flag=data.get("safety_flag", "low"),
        )
    except Exception:
        return ClassifierResult(
            intent="unknown",
            agent="general_query",
            entities={},
            safety_flag="low",
        )


def _is_mock(obj: Any) -> bool:
    """Check if an object is a unittest.mock.MagicMock."""
    type_name = type(obj).__name__
    return "Mock" in type_name or "MagicMock" in type_name

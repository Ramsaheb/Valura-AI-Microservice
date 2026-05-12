"""
Test suite for classifier routing accuracy on the labeled gold set.

The success threshold (≥ 85%) is from ASSIGNMENT.md.

The classifier falls back to rule-based classification when the LLM is
unavailable (MagicMock in tests), so these tests verify the fallback
achieves the required accuracy.
"""
from typing import Any

import pytest

from src.core.classifier import classify


# ---------------------------------------------------------------------------
# Entity matcher — implements the rules in fixtures/README.md
# ---------------------------------------------------------------------------

def _normalize_ticker(t: str) -> str:
    """Case-fold and drop the exchange suffix (AAPL.US → AAPL)."""
    return t.upper().split(".")[0]


def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """
    Subset match with normalization. `actual` must contain every value in
    `expected`; extra fields and extra values are allowed.
    """
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False

        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            if abs(act_value - exp_value) > abs(exp_value) * 0.05:
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        else:
            # Catch-all for vocabulary tokens (action, goal, frequency, horizon,
            # time_period, currency, index).
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True


# ---------------------------------------------------------------------------
# Routing accuracy — this is the test we score
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classifier_routing_accuracy(gold_classifier_queries, mock_llm):
    """
    Threshold: ≥ 85% routing accuracy.
    """
    correct = 0
    failures = []

    for case in gold_classifier_queries:
        result = await classify(case["query"], llm=mock_llm)
        if result.agent == case["expected_agent"]:
            correct += 1
        else:
            failures.append(
                f"  '{case['query'][:60]}' → got '{result.agent}', expected '{case['expected_agent']}'"
            )

    accuracy = correct / len(gold_classifier_queries)

    if failures:
        print(f"\n--- Classifier Routing Failures ({len(failures)}) ---")
        for f in failures[:15]:  # Show first 15
            print(f)

    print(f"\nRouting accuracy: {accuracy:.2%} ({correct}/{len(gold_classifier_queries)})")

    assert accuracy >= 0.85, f"Routing accuracy {accuracy:.2%} below 85%"


@pytest.mark.asyncio
async def test_classifier_entity_extraction(gold_classifier_queries, mock_llm):
    """
    Soft signal — not a hard threshold. Reported, not failed on.
    """
    matched = 0
    total_with_entities = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total_with_entities += 1
        result = await classify(case["query"], llm=mock_llm)
        if matches_entities(result.entities, case["expected_entities"]):
            matched += 1

    # No assertion — emit a report
    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total_with_entities})")


@pytest.mark.asyncio
async def test_classifier_never_crashes(mock_llm):
    """The classifier must handle any input without crashing."""
    edge_cases = [
        "",
        "hi",
        "abcdefg",
        "AAPL",
        "🚀📈💰",
        "x" * 2000,
        "tell me about that thing you mentioned earlier",
        "1500 monthly for 15 years",
    ]
    for query in edge_cases:
        result = await classify(query, llm=mock_llm)
        assert result is not None
        assert result.agent is not None
        assert isinstance(result.entities, dict)


@pytest.mark.asyncio
async def test_classifier_fallback_on_none_llm():
    """Classifier works even when llm=None (no LLM at all)."""
    result = await classify("how is my portfolio doing?", llm=None)
    assert result.agent == "portfolio_health"

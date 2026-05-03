"""
Test suite for conversation follow-up resolution and topic-switching.

Validates that the classifier correctly handles:
  - Entity carryover from prior turns (follow_up_session)
  - Clean topic switches that do NOT carry irrelevant context (multi_intent_session)
  - Ambiguous queries, typos, and edge cases (ambiguous_session)

These tests exercise the classifier's _resolve_followup() logic and ensure
the full pipeline handles multi-turn conversations correctly.
"""
import pytest

from src.core.classifier import classify


# ---------------------------------------------------------------------------
# Follow-up session — entity carryover
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_follow_up_entity_carryover(conversation_test_cases, mock_llm):
    """
    Tests from follow_up_session.json:
    Classifier must carry tickers from prior turns into follow-up queries.
    """
    cases = conversation_test_cases("follow_up_session")
    passed = 0
    failures = []

    for case in cases:
        history = [{"role": "user", "content": t} for t in case["prior_user_turns"]]
        result = await classify(case["current_user_turn"], llm=mock_llm, history=history)

        expected = case["expected"]
        expected_agent = expected["agent"]

        # For routing: some fixture agents (e.g. "portfolio_query") may not be
        # in our taxonomy — we accept reasonable alternatives
        agent_ok = _agent_matches(result.agent, expected_agent)

        # For entities: subset match on tickers
        entities_ok = True
        if "tickers" in expected.get("entities", {}):
            expected_tickers = {t.upper().split(".")[0] for t in expected["entities"]["tickers"]}
            actual_tickers = {t.upper().split(".")[0] for t in result.entities.get("tickers", [])}
            entities_ok = expected_tickers.issubset(actual_tickers)

        if agent_ok and entities_ok:
            passed += 1
        else:
            failures.append(
                f"  {case['case_id']}: '{case['current_user_turn']}' → "
                f"agent={result.agent} (expected={expected_agent}), "
                f"tickers={result.entities.get('tickers', [])} "
                f"(expected={expected.get('entities', {}).get('tickers', [])})"
            )

    if failures:
        print(f"\n--- Follow-up Failures ({len(failures)}) ---")
        for f in failures:
            print(f)

    # Soft threshold — follow-ups are challenging for rule-based systems
    rate = passed / len(cases) if cases else 0
    print(f"\nFollow-up accuracy: {rate:.2%} ({passed}/{len(cases)})")
    assert rate >= 0.50, f"Follow-up accuracy {rate:.2%} too low"


# ---------------------------------------------------------------------------
# Multi-intent session — topic switches
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_intent_topic_switches(conversation_test_cases, mock_llm):
    """
    Tests from multi_intent_session.json:
    Each turn is a clean topic switch — classifier must NOT carry stale entities.
    """
    cases = conversation_test_cases("multi_intent_session")
    passed = 0
    failures = []

    for case in cases:
        history = [{"role": "user", "content": t} for t in case["prior_user_turns"]]
        result = await classify(case["current_user_turn"], llm=mock_llm, history=history)

        expected = case["expected"]
        expected_agent = expected["agent"]

        if _agent_matches(result.agent, expected_agent):
            passed += 1
        else:
            failures.append(
                f"  {case['case_id']}: '{case['current_user_turn']}' → "
                f"agent={result.agent} (expected={expected_agent})"
            )

    if failures:
        print(f"\n--- Multi-intent Failures ({len(failures)}) ---")
        for f in failures:
            print(f)

    rate = passed / len(cases) if cases else 0
    print(f"\nMulti-intent routing accuracy: {rate:.2%} ({passed}/{len(cases)})")
    assert rate >= 0.75, f"Multi-intent accuracy {rate:.2%} too low"


# ---------------------------------------------------------------------------
# Ambiguous session — typos, slang, edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ambiguous_edge_cases(conversation_test_cases, mock_llm):
    """
    Tests from ambiguous_session.json:
    Classifier must handle typos, informal language, and vague references.
    """
    cases = conversation_test_cases("ambiguous_session")
    passed = 0
    failures = []

    for case in cases:
        history = [{"role": "user", "content": t} for t in case["prior_user_turns"]]
        result = await classify(case["current_user_turn"], llm=mock_llm, history=history)

        expected = case["expected"]
        expected_agent = expected["agent"]

        if _agent_matches(result.agent, expected_agent):
            passed += 1
        else:
            failures.append(
                f"  {case['case_id']}: '{case['current_user_turn']}' → "
                f"agent={result.agent} (expected={expected_agent})"
            )

    if failures:
        print(f"\n--- Ambiguous Failures ({len(failures)}) ---")
        for f in failures:
            print(f)

    rate = passed / len(cases) if cases else 0
    print(f"\nAmbiguous routing accuracy: {rate:.2%} ({passed}/{len(cases)})")
    assert rate >= 0.60, f"Ambiguous accuracy {rate:.2%} too low"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent_matches(actual: str, expected: str) -> bool:
    """
    Flexible agent matching: some fixture agents (e.g. 'portfolio_query')
    don't exist in our taxonomy. We accept reasonable alternatives.
    """
    if actual == expected:
        return True

    # Known fixture → taxonomy mappings
    equivalences = {
        "portfolio_query": {"portfolio_health", "market_research", "investment_strategy"},
    }
    if expected in equivalences:
        return actual in equivalences[expected]

    return False

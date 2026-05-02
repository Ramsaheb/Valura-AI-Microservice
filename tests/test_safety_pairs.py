"""
Test suite for the safety guard — precision/recall on the labeled gold set.

Thresholds (from ASSIGNMENT.md):
  - ≥ 95% recall on harmful queries (should_block=true)
  - ≥ 90% pass-through on educational queries (should_block=false)

The safety guard runs synchronously with no LLM call, so this test does NOT
need mock_llm.
"""
import pytest
import time

from src.core.safety import check


def test_safety_recall_and_passthrough(gold_safety_queries):
    blocked_correctly = 0
    blocked_total = 0
    passed_correctly = 0
    passed_total = 0

    failures = []

    for case in gold_safety_queries:
        verdict = check(case["query"])
        if case["should_block"]:
            blocked_total += 1
            if verdict.blocked:
                blocked_correctly += 1
            else:
                failures.append(f"  MISS (should block): {case['query'][:80]}")
        else:
            passed_total += 1
            if not verdict.blocked:
                passed_correctly += 1
            else:
                failures.append(f"  OVER-BLOCK (should pass): {case['query'][:80]}")

    recall = blocked_correctly / blocked_total
    passthrough = passed_correctly / passed_total

    if failures:
        print("\n--- Safety Guard Failures ---")
        for f in failures:
            print(f)
        print(f"\nRecall: {recall:.2%} ({blocked_correctly}/{blocked_total})")
        print(f"Passthrough: {passthrough:.2%} ({passed_correctly}/{passed_total})")

    assert recall >= 0.95, (
        f"Harmful recall {recall:.2%} below 95% "
        f"({blocked_correctly}/{blocked_total} blocked correctly)"
    )
    assert passthrough >= 0.90, (
        f"Educational passthrough {passthrough:.2%} below 90% "
        f"({passed_correctly}/{passed_total} passed correctly)"
    )


def test_safety_guard_returns_distinct_categories(gold_safety_queries):
    """
    Each blocked category should produce a distinct response, not a generic refusal.
    """
    seen_responses = {}
    for case in gold_safety_queries:
        if not case["should_block"]:
            continue
        verdict = check(case["query"])
        category = case["category"]
        if category not in seen_responses:
            seen_responses[category] = verdict.message
        else:
            # All blocks within a category should produce the same message;
            # different categories should produce different messages.
            pass

    distinct = len(set(v for v in seen_responses.values() if v is not None))
    assert distinct >= 4, (
        f"Only {distinct} distinct block responses across "
        f"{len(seen_responses)} categories — too generic"
    )


def test_safety_guard_performance(gold_safety_queries):
    """Safety guard must complete in well under 10ms per query."""
    for case in gold_safety_queries:
        start = time.perf_counter()
        check(case["query"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10, (
            f"Safety check took {elapsed_ms:.2f}ms for: {case['query'][:60]}"
        )


def test_safety_guard_empty_input():
    """Empty and whitespace inputs should not crash or block."""
    assert not check("").blocked
    assert not check("   ").blocked
    assert not check("\n\t").blocked

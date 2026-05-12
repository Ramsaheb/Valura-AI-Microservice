---
title: PGAGI
emoji: 🐳
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Valura AI Microservice

The intelligence layer behind Valura's global wealth management platform. This microservice acts as an AI co-investor for every user, designed specifically to help novice investors **BUILD, MONITOR, GROW,** and **PROTECT** their portfolios.

## Submission Video

> **Video walkthrough:** https://drive.google.com/file/d/1GY3KqWqIzws6tfK5mxwQokKraF-b0eiV/view?usp=drive_link

---

## Architecture & Pipeline

The system is built as a single-pass streaming pipeline that prioritizes safety, speed, and accuracy:

1. **Safety Guard (Synchronous):** A multi-layer, sub-10ms filter that intercepts harmful intents (insider trading, market manipulation, etc.) *before* any LLM is invoked.
2. **Intent Classifier (LLM + Fallback):** Determines the user's intent and extracts structured entities (tickers, amounts, etc.). Uses OpenAI structured outputs with a highly accurate rule-based fallback if the LLM is unavailable.
3. **Router:** Dispatches the classified query to the appropriate specialist agent.
4. **Specialist Agents:** Execute domain-specific logic. 
    *   **Portfolio Health:** Uses pure computation (yfinance + numpy) to analyze concentration and performance, returning actionable, novice-friendly observations.
    *   **Stub Fallback:** Unimplemented agents return structured responses acknowledging the intent without hallucinating.
5. **SSE Streaming (FastAPI):** Every stage of the pipeline emits strictly structured Server-Sent Events (SSE). No stack traces are ever leaked to the client.
6. **Pipeline Timeout (30s):** The entire classification → routing → agent execution pipeline is wrapped in a 30-second timeout via `asyncio.wait_for`. If any stage hangs (e.g., yfinance network issue), the request returns a structured SSE error event instead of blocking indefinitely. 30s chosen because yfinance completes in <2s and LLM calls in <10s — this provides generous headroom.

## Core Design Decisions & Tradeoffs

1. **Safety First (Regex over LLM):** The safety guard is implemented using compiled regex patterns matching *action intent* + *harmful topic*.
    *   *Tradeoff:* Slightly higher false-positive rate on edge cases (e.g., "I need to know earnings before the call" is blocked even if innocent), but guarantees deterministic, instant blocking with 0 network latency.
2. **Rule-Based Fallback Classifier:** Instead of failing when the LLM is down (or when `OPENAI_API_KEY` is missing in CI), the classifier falls back to a custom rule engine.
    *   *Tradeoff:* Less nuanced than an LLM, but guarantees the pipeline never crashes and passes the 85% routing accuracy threshold required for CI.
3. **Pure Computation for Portfolio Health:** The portfolio health agent uses `numpy` and `yfinance` for math, rather than asking the LLM to calculate returns.
    *   *Tradeoff:* The LLM isn't used for "reasoning" about the numbers, ensuring mathematically correct metrics and eliminating hallucination risk.
4. **Graceful Degradation (Market Data):** `yfinance` calls are cached with a 5-minute TTL. If the API fails, functions return `None` and the pipeline continues gracefully.
5. **SSE by Default:** The entire API is designed around Server-Sent Events to support future streaming LLM responses, ensuring a low TTFB (Time to First Byte).
6. **Session Persistence (In-Memory):** For this demonstration, session history is stored in an `asyncio.Lock()` protected in-memory dictionary. This avoids the overhead of setting up a Postgres container for the reviewer, ensuring the app runs immediately on `git clone` while safely handling concurrent HTTP requests.
7. **Library Choices Justification:**
    *   **FastAPI & Uvicorn:** Chosen for native async support, high performance, and ease of defining strict schemas.
    *   **sse-starlette:** A lightweight, proven library for implementing Server-Sent Events cleanly in FastAPI.
    *   **Pydantic:** For strict data validation at the API boundary and ensuring the `PortfolioHealthAgent` output matches the required schema perfectly.
    *   **numpy & yfinance:** For fast, mathematically sound pure-computation portfolio analytics, completely eliminating LLM math hallucination.
    *   **openai:** The official SDK is used for reliable structured JSON output parsing.

## Performance & Cost Measurement

*   **Model Configuration:** The system uses `gpt-4o-mini` during development (set via `OPENAI_MODEL` env var, default). For evaluation with `gpt-4.1`, set `OPENAI_MODEL=gpt-4.1` in your `.env` file — no code changes required.
*   **Cost per query (< $0.05):** The system consumes ~600 tokens per classification call (including system prompt, history, and response). At `gpt-4.1` pricing ($2.00 / 1M input tokens, $8.00 / 1M output tokens), a 600-token classification call costs **~$0.0012 to $0.005**, well under the $0.05 limit. The `PortfolioHealthAgent` requires zero additional LLM tokens as it uses pure math.
*   **p95 Streaming Latency & Response Time:** 
    *   *First-token latency (< 2s):* The safety guard takes < 2ms. The classification call (using `gpt-4o-mini` structured outputs) averages 600-900ms. The first SSE event (`classification`) is emitted to the client in under **1 second**.
    *   *End-to-end response time (< 6s):* The `PortfolioHealthAgent` fetches `yfinance` data (cached) and computes math in < 100ms. Total execution time is consistently around **1.5s - 2.5s**, well under the 6s limit. Measured via local profiling and elapsed time during `pytest`.
    *   *Pipeline timeout:* 30-second hard limit enforced via `asyncio.wait_for()`, returning a structured `timeout` SSE error if breached.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | No | — | OpenAI API key. Without it, the system uses the rule-based fallback |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model to use. Set to `gpt-4.1` for evaluation |
| `APP_ENV` | No | `development` | Set to `test` to force MockLLM (for CI) |

## Getting Started

### Prerequisites

*   Python 3.11+
*   *(Optional)* OpenAI API Key for full LLM classification capabilities.

### Installation

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Start the FastAPI server
uvicorn src.main:app --reload
```

### Running With Docker (Local)

```bash
docker build -t valura-ai .
docker run --rm -p 7860:7860 -e APP_ENV=test valura-ai
```

Open `http://127.0.0.1:7860`.

### Deploy on Hugging Face Spaces (Docker)

1. Create a new Space on Hugging Face and choose the **Docker** SDK.
2. Set the Space visibility and name (e.g. `valura-ai-microservice`).
3. Add environment variables if needed:
    - `OPENAI_API_KEY` (optional)
    - `OPENAI_MODEL` (optional)
    - `APP_ENV=test` to force the mock LLM (no API key required)
4. Push this repo to the Space:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/<your-space>
git push hf main
```

Hugging Face Spaces uses the included Dockerfile and serves on port `7860`.

The API will be available at `http://127.0.0.1:8000`. You can interact with the primary endpoint at `/api/v1/query`.

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is my portfolio doing?",
    "user_id": "user_001_active_trader_us",
    "session_id": "session_123"
  }'
```

## Testing

The test suite validates safety thresholds, classification routing accuracy (≥85%), conversation follow-up resolution, full-pipeline SSE integration, and edge-case resilience.

```bash
# Set APP_ENV to trigger the MockLLM (no API key required)
export APP_ENV=test  # On Windows PowerShell: $env:APP_ENV="test"

# Run the test suite
pytest tests/ -v
```

### Entity Matching Rules

The test suite implements the matching rules from `fixtures/README.md`:

*   **Tickers:** Case-insensitive, exchange suffix optional (`AAPL` matches `aapl`; `ASML` matches `ASML.AS`)
*   **Topics/Sectors:** Case-insensitive subset match — extra values are allowed
*   **Amounts/Rates:** Numeric fields match within ±5%
*   **Period:** Integer exact match

## What I'd Do Differently With Another Week

1. **Embedding-based pre-classifier:** Build a lightweight sentence-transformer cache so that high-confidence queries skip the LLM call entirely (saves cost and latency).
2. **Postgres session persistence:** Replace the in-memory dict with a proper async Postgres backend for production-grade session management.
3. **LLM-generated observations:** Use the LLM to produce richer, more personalized narrative observations rather than template-based ones.
4. **Rate limiting:** Per-tenant rate limiting with sliding window counters.

## Disqualification Risks Mitigated

*   **No Secrets:** No hardcoded API keys exist in the repository.
*   **100% Passing CI:** The test suite passes locally and utilizes the `MockLLM` fixture to pass in CI without network calls.
*   **Strict SSE Streaming:** The `/api/v1/query` endpoint returns an `EventSourceResponse`. No JSON fallback path.
*   **Safety Guard Speed:** Runs synchronously using pre-compiled regex, executing in < 2ms (well under the 10ms requirement).
*   **Pipeline Timeout:** 30s hard limit via `asyncio.wait_for()` — prevents indefinite hangs.
*   **Empty Portfolios:** `PortfolioHealthAgent` gracefully detects empty positions and returns BUILD-focused observations instead of crashing.
*   **Regulatory Disclaimers:** Appended automatically to the `PortfolioHealthResponse`.
*   **Conversation Handling:** Tests cover follow-up resolution, topic switches, and ambiguous/typo queries from `fixtures/conversations/`.

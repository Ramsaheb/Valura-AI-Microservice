"""
Pydantic models for the Valura AI microservice.

All data types flowing through the pipeline — safety verdicts, classifier results,
portfolio health responses, API request/response shapes, and user profiles.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SafetyFlag(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SSEEventType(str, Enum):
    SAFETY_BLOCK = "safety_block"
    CLASSIFICATION = "classification"
    AGENT_RESPONSE = "agent_response"
    AGENT_CHUNK = "agent_chunk"
    ERROR = "error"
    DONE = "done"


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class SafetyVerdict(BaseModel):
    """Result of the safety guard check."""
    blocked: bool
    category: Optional[str] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ClassifierResult(BaseModel):
    """Structured output from the intent classifier."""
    intent: str = "unknown"
    agent: str = "general_query"
    entities: dict[str, Any] = Field(default_factory=dict)
    safety_flag: str = "low"


# ---------------------------------------------------------------------------
# Portfolio Health
# ---------------------------------------------------------------------------

class ConcentrationRisk(BaseModel):
    """Concentration risk metrics for a portfolio."""
    top_position_pct: float = 0.0
    top_3_positions_pct: float = 0.0
    flag: str = "low"


class Performance(BaseModel):
    """Portfolio performance metrics."""
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0


class BenchmarkComparison(BaseModel):
    """Portfolio vs benchmark comparison."""
    benchmark: str = "S&P 500"
    portfolio_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0


class Observation(BaseModel):
    """A single actionable observation about the portfolio."""
    severity: str = "info"
    text: str = ""


class PortfolioHealthResponse(BaseModel):
    """Full portfolio health check response."""
    concentration_risk: ConcentrationRisk = Field(default_factory=ConcentrationRisk)
    performance: Performance = Field(default_factory=Performance)
    benchmark_comparison: BenchmarkComparison = Field(default_factory=BenchmarkComparison)
    observations: list[Observation] = Field(default_factory=list)
    disclaimer: str = (
        "This is not investment advice. The information provided is for "
        "educational and informational purposes only. Past performance does "
        "not guarantee future results. Please consult a qualified financial "
        "advisor before making any investment decisions."
    )


# ---------------------------------------------------------------------------
# User Profile (mirrors fixture JSON shape)
# ---------------------------------------------------------------------------

class Position(BaseModel):
    """A single holding in a user's portfolio."""
    ticker: str
    exchange: str = ""
    quantity: float
    avg_cost: float
    currency: str = "USD"
    purchased_at: str = ""


class KYCStatus(BaseModel):
    status: str = "pending"


class UserPreferences(BaseModel):
    preferred_benchmark: str = "S&P 500"
    reporting_currency: str = "USD"
    income_focus: bool = False


class UserProfile(BaseModel):
    """User profile as provided in fixture JSON files."""
    user_id: str
    name: str = ""
    age: int = 0
    country: str = "US"
    base_currency: str = "USD"
    kyc: KYCStatus = Field(default_factory=KYCStatus)
    risk_profile: str = "moderate"
    positions: list[Position] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)


# ---------------------------------------------------------------------------
# API Layer
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming query from the client."""
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class SSEEvent(BaseModel):
    """A single Server-Sent Event payload."""
    event: SSEEventType
    data: dict[str, Any] = Field(default_factory=dict)


class StubResponse(BaseModel):
    """Response from an unimplemented agent."""
    status: str = "stub"
    message: str = "This agent is under development. Returning safe fallback response."

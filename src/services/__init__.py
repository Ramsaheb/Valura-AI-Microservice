"""Services package — core computation and external data integration."""
from src.services.market_data import (
    get_current_price,
    get_historical_return,
    get_benchmark_return,
    get_fx_rate,
    get_dividend_yield,
    clear_cache,
)
from src.services.portfolio import (
    compute_concentration,
    compute_performance,
    compute_benchmark_comparison,
    generate_observations,
    generate_empty_portfolio_observations,
)

__all__ = [
    "get_current_price",
    "get_historical_return",
    "get_benchmark_return",
    "get_fx_rate",
    "get_dividend_yield",
    "clear_cache",
    "compute_concentration",
    "compute_performance",
    "compute_benchmark_comparison",
    "generate_observations",
    "generate_empty_portfolio_observations",
]

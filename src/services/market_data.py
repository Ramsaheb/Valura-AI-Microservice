"""
Market Data Service — live market data via yfinance with TTL caching.

Provides current prices, historical returns, benchmark data, and FX rates.
All data is fetched from yfinance and cached with a 5-minute TTL to avoid
redundant API calls within a session.

Graceful fallback: if yfinance is unavailable (network error, rate limit),
methods return None and callers handle the absence.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# TTL cache for market data (avoid repeated yfinance calls)
_price_cache: dict[str, tuple[float, float]] = {}  # ticker -> (price, timestamp)
_CACHE_TTL_SECONDS = 300  # 5 minutes


def get_current_price(ticker: str) -> Optional[float]:
    """
    Get the current/last closing price for a ticker.

    Returns None if the data cannot be fetched.
    """
    import time
    now = time.time()

    # Check cache
    if ticker in _price_cache:
        price, cached_at = _price_cache[ticker]
        if now - cached_at < _CACHE_TTL_SECONDS:
            return price

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            logger.warning(f"No price data for {ticker}")
            return None
        price = float(hist["Close"].iloc[-1])
        _price_cache[ticker] = (price, now)
        return price
    except Exception as e:
        logger.warning(f"Failed to fetch price for {ticker}: {e}")
        return None


def get_historical_return(ticker: str, start_date: str, end_date: str | None = None) -> Optional[float]:
    """
    Get the total return (%) for a ticker between two dates.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (default: today).

    Returns:
        Total return as a percentage (e.g. 18.4 for 18.4%), or None.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        end = end_date or datetime.now().strftime("%Y-%m-%d")
        hist = stock.history(start=start_date, end=end)
        if len(hist) < 2:
            return None
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        if start_price <= 0:
            return None
        return ((end_price - start_price) / start_price) * 100
    except Exception as e:
        logger.warning(f"Failed to fetch historical return for {ticker}: {e}")
        return None


def get_benchmark_return(benchmark: str, start_date: str, end_date: str | None = None) -> Optional[float]:
    """
    Get the total return for a benchmark index.

    Benchmark name mapping:
      - "S&P 500" → ^GSPC
      - "FTSE 100" → ^FTSE
      - "NIKKEI 225" → ^N225
      - "MSCI World" → URTH (ETF proxy)
      - "QQQ" → QQQ
    """
    ticker_map = {
        "S&P 500": "^GSPC",
        "FTSE 100": "^FTSE",
        "NIKKEI 225": "^N225",
        "MSCI World": "URTH",
        "QQQ": "QQQ",
        "VTI": "VTI",
    }
    ticker = ticker_map.get(benchmark, "^GSPC")
    return get_historical_return(ticker, start_date, end_date)


def get_fx_rate(from_currency: str, to_currency: str) -> float:
    """
    Get the exchange rate from one currency to another.

    Returns 1.0 for same-currency pairs.
    Falls back to 1.0 if the rate cannot be fetched.
    """
    if from_currency.upper() == to_currency.upper():
        return 1.0

    pair = f"{from_currency.upper()}{to_currency.upper()}=X"

    try:
        import yfinance as yf
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="5d")
        if hist.empty:
            logger.warning(f"No FX data for {pair}, returning 1.0")
            return 1.0
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to fetch FX rate {pair}: {e}")
        return 1.0


def get_dividend_yield(ticker: str) -> Optional[float]:
    """Get the trailing dividend yield for a ticker."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get("dividendYield") or info.get("trailingAnnualDividendYield")
    except Exception as e:
        logger.warning(f"Failed to fetch dividend yield for {ticker}: {e}")
        return None


def clear_cache():
    """Clear the price cache (useful for tests)."""
    _price_cache.clear()

"""Fetch market microstructure data from Binance Futures API."""

import asyncio
import time
from functools import wraps
from typing import Optional

from binance import AsyncClient, BinanceAPIException

from src.config import config


# Simple TTL cache for market data
_cache: dict[str, tuple[float, float]] = {}  # key -> (value, expiry_timestamp)

FUNDING_RATE_TTL = 300  # 5 minutes (funding updates every 8h, no need to hammer)
OPEN_INTEREST_TTL = 60  # 1 minute (OI changes more frequently)
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


def _get_cached(key: str) -> Optional[float]:
    """Return cached value if still valid, else None."""
    if key in _cache:
        value, expiry = _cache[key]
        if time.time() < expiry:
            return value
        del _cache[key]
    return None


def _set_cached(key: str, value: float, ttl: float) -> None:
    """Store value in cache with TTL."""
    _cache[key] = (value, time.time() + ttl)


async def _create_client() -> AsyncClient:
    """Create an authenticated Binance async client."""
    return await AsyncClient.create(
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
    )


async def get_funding_rate(symbol: str) -> float:
    """Fetch the current funding rate for a symbol.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT").

    Returns:
        Funding rate as a float (e.g. 0.0001 = 0.01%).
    """
    cache_key = f"funding:{symbol}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        client = None
        try:
            client = await _create_client()
            # Get the most recent funding rate
            result = await client.futures_funding_rate(
                symbol=symbol, limit=1
            )
            if result:
                rate = float(result[-1]["fundingRate"])
                _set_cached(cache_key, rate, FUNDING_RATE_TTL)
                return rate
            return 0.0
        except BinanceAPIException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        finally:
            if client:
                await client.close_connection()

    # All retries exhausted — return 0 rather than crashing the pipeline
    return 0.0


async def get_open_interest(symbol: str) -> float:
    """Fetch current open interest for a symbol.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT").

    Returns:
        Open interest in contracts (quote asset terms).
    """
    cache_key = f"oi:{symbol}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        client = None
        try:
            client = await _create_client()
            result = await client.futures_open_interest(symbol=symbol)
            oi = float(result["openInterest"])
            _set_cached(cache_key, oi, OPEN_INTEREST_TTL)
            return oi
        except BinanceAPIException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        finally:
            if client:
                await client.close_connection()

    return 0.0


async def get_market_features(symbol: str) -> dict:
    """Fetch funding rate and open interest together.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT").

    Returns:
        Dict with keys "funding_rate" and "open_interest".
    """
    funding, oi = await asyncio.gather(
        get_funding_rate(symbol),
        get_open_interest(symbol),
    )
    return {
        "funding_rate": funding,
        "open_interest": oi,
    }

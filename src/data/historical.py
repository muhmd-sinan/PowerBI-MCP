import asyncio
import logging
import time
from datetime import datetime, timedelta

from binance import AsyncClient
from sqlalchemy.dialects.mysql import insert as mysql_insert

from src.config import config
from src.data.database import Candle, SessionLocal, engine

logger = logging.getLogger(__name__)

# Binance kline interval mapping
TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "10m": 600_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Binance interval strings (10m is not native — use 5m and aggregate, or use closest)
# Note: Binance does not support 10m natively. We fetch 5m and aggregate pairs.
BINANCE_INTERVALS = {
    "5m": "5m",
    "10m": "5m",  # Fetch 5m, aggregate every 2 candles
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
}


def _aggregate_5m_to_10m(candles_5m: list[dict]) -> list[dict]:
    """Aggregate pairs of 5m candles into 10m candles."""
    aggregated = []
    # Sort by timestamp
    candles_5m.sort(key=lambda c: c["timestamp"])

    for i in range(0, len(candles_5m) - 1, 2):
        c1 = candles_5m[i]
        c2 = candles_5m[i + 1]
        aggregated.append({
            "timestamp": c1["timestamp"],
            "open": c1["open"],
            "high": max(c1["high"], c2["high"]),
            "low": min(c1["low"], c2["low"]),
            "close": c2["close"],
            "volume": c1["volume"] + c2["volume"],
        })
    return aggregated


def _parse_kline(kline: list) -> dict:
    """Parse a Binance kline array into a dict."""
    return {
        "timestamp": int(kline[0]),
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "volume": float(kline[5]),
    }


def _store_candles(symbol: str, timeframe: str, candles: list[dict]):
    """Bulk upsert candles into MySQL."""
    if not candles:
        return

    db = SessionLocal()
    try:
        rows = [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": c["timestamp"],
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c["volume"],
            }
            for c in candles
        ]

        stmt = mysql_insert(Candle).values(rows)
        # On duplicate key, update OHLCV (handles re-fetches gracefully)
        stmt = stmt.on_duplicate_key_update(
            open=stmt.inserted.open,
            high=stmt.inserted.high,
            low=stmt.inserted.low,
            close=stmt.inserted.close,
            volume=stmt.inserted.volume,
        )
        db.execute(stmt)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to store candles for {symbol}/{timeframe}: {e}")
        raise
    finally:
        db.close()


async def fetch_historical(
    symbol: str,
    timeframe: str,
    months: int = 12,
    progress_cb=None,
):
    """
    Fetch historical kline data from Binance and store in MySQL.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT")
        timeframe: Candle interval (e.g. "5m", "10m", "15m", "30m", "1h", "2h")
        months: How many months of history to fetch (default 12)
        progress_cb: Optional callback(current, total) for progress updates
    """
    binance_interval = BINANCE_INTERVALS.get(timeframe)
    if not binance_interval:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    is_10m = timeframe == "10m"
    interval_ms = TIMEFRAME_MS[binance_interval]

    # Calculate time range
    end_time = int(time.time() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=months * 30)).timestamp() * 1000)

    # Estimate total candles for progress tracking
    total_candles_est = (end_time - start_time) // interval_ms
    total_batches = (total_candles_est // 1000) + 1
    fetched_batches = 0

    logger.info(
        f"Fetching {months} months of {timeframe} data for {symbol} "
        f"(~{total_batches} batches)"
    )

    client = await AsyncClient.create(
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
    )

    try:
        current_start = start_time
        all_candles_for_10m = []

        while current_start < end_time:
            try:
                klines = await client.get_klines(
                    symbol=symbol,
                    interval=binance_interval,
                    startTime=current_start,
                    endTime=end_time,
                    limit=1000,
                )
            except Exception as e:
                logger.warning(f"Rate limit or error fetching {symbol}: {e}. Sleeping 10s.")
                await asyncio.sleep(10)
                continue

            if not klines:
                break

            candles = [_parse_kline(k) for k in klines]

            if is_10m:
                # Accumulate 5m candles for 10m aggregation
                all_candles_for_10m.extend(candles)
                # Store in batches of 2000 (1000 10m candles)
                if len(all_candles_for_10m) >= 2000:
                    aggregated = _aggregate_5m_to_10m(all_candles_for_10m[:2000])
                    _store_candles(symbol, timeframe, aggregated)
                    all_candles_for_10m = all_candles_for_10m[2000:]
            else:
                _store_candles(symbol, timeframe, candles)

            # Advance start time past the last candle
            current_start = int(klines[-1][0]) + interval_ms
            fetched_batches += 1

            if progress_cb:
                progress_cb(fetched_batches, total_batches)

            # Rate limit: sleep 200ms between requests
            await asyncio.sleep(0.2)

        # Store remaining 10m candles
        if is_10m and all_candles_for_10m:
            aggregated = _aggregate_5m_to_10m(all_candles_for_10m)
            _store_candles(symbol, timeframe, aggregated)

        logger.info(f"Completed historical fetch for {symbol}/{timeframe}")

    finally:
        await client.close_connection()

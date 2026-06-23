import asyncio
import logging

from sqlalchemy import text

from src.config import config
from src.data.database import Candle, SessionLocal, engine
from src.data.historical import fetch_historical, TIMEFRAME_MS

logger = logging.getLogger(__name__)


def _get_active_symbols() -> list[str]:
    """Get all active asset symbols from the database."""
    db = SessionLocal()
    try:
        from src.data.database import Asset
        assets = db.query(Asset.symbol).filter(Asset.status == "ACTIVE").all()
        return [a.symbol for a in assets]
    finally:
        db.close()


def _get_candle_range(symbol: str, timeframe: str) -> tuple[int | None, int | None]:
    """Get the first and last stored timestamp for a symbol/timeframe."""
    db = SessionLocal()
    try:
        from sqlalchemy import func
        result = db.query(
            func.min(Candle.timestamp),
            func.max(Candle.timestamp),
        ).filter(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
        ).first()
        return result[0], result[1]
    finally:
        db.close()


def _get_stored_timestamps(symbol: str, timeframe: str) -> set[int]:
    """Get all stored timestamps for a symbol/timeframe."""
    db = SessionLocal()
    try:
        rows = db.query(Candle.timestamp).filter(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
        ).all()
        return {r.timestamp for r in rows}
    finally:
        db.close()


def _find_gaps(symbol: str, timeframe: str) -> list[int]:
    """
    Identify missing timestamps between the first and last stored candle.

    Returns a list of missing timestamps (open time in ms).
    """
    first_ts, last_ts = _get_candle_range(symbol, timeframe)

    if first_ts is None or last_ts is None:
        logger.info(f"No data for {symbol}/{timeframe}, nothing to check")
        return []

    interval_ms = TIMEFRAME_MS.get(timeframe)
    if interval_ms is None:
        logger.error(f"Unknown timeframe interval: {timeframe}")
        return []

    # Build expected set of timestamps
    expected = set()
    ts = first_ts
    while ts <= last_ts:
        expected.add(ts)
        ts += interval_ms

    # Compare with stored
    stored = _get_stored_timestamps(symbol, timeframe)
    missing = sorted(expected - stored)

    if missing:
        expected_count = len(expected)
        stored_count = len(stored)
        gap_count = len(missing)
        coverage = (stored_count / expected_count) * 100 if expected_count else 100
        logger.info(
            f"{symbol}/{timeframe}: {stored_count}/{expected_count} candles "
            f"({coverage:.1f}% coverage), {gap_count} gaps found"
        )
    else:
        logger.info(f"{symbol}/{timeframe}: No gaps found")

    return missing


async def check_and_fill_gaps(symbol: str, timeframe: str):
    """
    Check for timestamp gaps and backfill missing candles.

    Identifies gaps in stored candle data and fetches the missing
    candles from Binance to fill them.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT")
        timeframe: Candle interval (e.g. "5m", "15m", "1h")
    """
    missing = _find_gaps(symbol, timeframe)

    if not missing:
        return

    logger.info(f"Backfilling {len(missing)} gaps for {symbol}/{timeframe}")

    # Group consecutive missing timestamps into ranges for efficient fetching
    ranges = _group_into_ranges(missing, TIMEFRAME_MS[timeframe])

    from binance import AsyncClient
    from src.data.historical import (
        _parse_kline, _store_candles, _aggregate_5m_to_10m, BINANCE_INTERVALS
    )

    client = await AsyncClient.create(
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
    )

    try:
        binance_interval = BINANCE_INTERVALS.get(timeframe, timeframe)
        is_10m = timeframe == "10m"
        listen_interval = "5m" if is_10m else binance_interval

        for start_ts, end_ts in ranges:
            try:
                klines = await client.get_klines(
                    symbol=symbol,
                    interval=listen_interval,
                    startTime=start_ts,
                    endTime=end_ts,
                    limit=1000,
                )

                if klines:
                    candles = [_parse_kline(k) for k in klines]
                    if is_10m:
                        candles = _aggregate_5m_to_10m(candles)
                    _store_candles(symbol, timeframe, candles)

            except Exception as e:
                logger.warning(f"Error backfilling {symbol}/{timeframe} range: {e}")
                await asyncio.sleep(5)
                continue

            # Rate limit between batch requests
            await asyncio.sleep(0.3)

        # Verify after backfill
        remaining = _find_gaps(symbol, timeframe)
        if remaining:
            logger.warning(
                f"{symbol}/{timeframe}: {len(remaining)} gaps remain after backfill"
            )
        else:
            logger.info(f"{symbol}/{timeframe}: All gaps filled successfully")

    finally:
        await client.close_connection()


def _group_into_ranges(timestamps: list[int], interval_ms: int) -> list[tuple[int, int]]:
    """
    Group consecutive timestamps into (start, end) ranges.

    This minimizes the number of API calls by fetching contiguous
    blocks in a single request.
    """
    if not timestamps:
        return []

    ranges = []
    range_start = timestamps[0]
    prev = timestamps[0]

    for ts in timestamps[1:]:
        # If there's a gap larger than one interval, start a new range
        if ts - prev > interval_ms:
            ranges.append((range_start, prev + interval_ms - 1))
            range_start = ts
        prev = ts

    # Close final range
    ranges.append((range_start, prev + interval_ms - 1))

    return ranges


async def check_all_integrity():
    """
    Run integrity checks and gap-filling for all active assets
    across all configured timeframes.
    """
    symbols = _get_active_symbols()

    if not symbols:
        # Fallback to default assets if no active ones in DB
        symbols = config.DEFAULT_ASSETS
        logger.info(f"No active assets in DB, using defaults: {symbols}")

    total_pairs = len(symbols) * len(config.ALL_TIMEFRAMES)
    logger.info(
        f"Starting integrity check for {len(symbols)} symbols × "
        f"{len(config.ALL_TIMEFRAMES)} timeframes ({total_pairs} pairs)"
    )

    for symbol in symbols:
        for timeframe in config.ALL_TIMEFRAMES:
            try:
                await check_and_fill_gaps(symbol, timeframe)
            except Exception as e:
                logger.error(
                    f"Integrity check failed for {symbol}/{timeframe}: {e}"
                )
            # Small delay between pairs to avoid hammering the API
            await asyncio.sleep(0.5)

    logger.info("Integrity check complete for all assets")

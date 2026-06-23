import asyncio
import logging
from enum import Enum
from typing import Callable, Optional

from binance import AsyncClient, BinanceSocketManager

from src.config import config
from src.data.database import Candle, SessionLocal
from src.data.historical import fetch_historical, BINANCE_INTERVALS

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class StreamManager:
    """
    WebSocket manager for real-time kline streams.

    Subscribes to kline streams for all active symbols across configured timeframes.
    On candle close, stores the candle and fires the on_candle_close callback.
    Auto-reconnects with exponential backoff on disconnect.
    """

    def __init__(self, on_candle_close: Optional[Callable] = None):
        """
        Args:
            on_candle_close: Callback(symbol, timeframe, candle_dict) fired on each closed candle.
        """
        self.on_candle_close = on_candle_close
        self.state = ConnectionState.DISCONNECTED
        self._symbols: set[str] = set()
        self._client: Optional[AsyncClient] = None
        self._bsm: Optional[BinanceSocketManager] = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_delay = 120  # seconds

    @property
    def symbols(self) -> set[str]:
        return self._symbols.copy()

    async def start(self, symbols: Optional[list[str]] = None):
        """
        Start streaming kline data.

        Args:
            symbols: List of symbols to stream. Defaults to config.DEFAULT_ASSETS.
        """
        if self._running:
            logger.warning("StreamManager already running")
            return

        self._running = True
        self.state = ConnectionState.CONNECTING
        self._symbols = set(symbols or config.DEFAULT_ASSETS)

        logger.info(f"Starting stream manager for {len(self._symbols)} symbols")

        try:
            self._client = await AsyncClient.create(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_API_SECRET,
            )
            self._bsm = BinanceSocketManager(self._client)
            self.state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0

            # Start a listener task for each symbol × timeframe
            for symbol in self._symbols:
                self._start_symbol_streams(symbol)

            logger.info("Stream manager connected and listening")

        except Exception as e:
            logger.error(f"Failed to start stream manager: {e}")
            self.state = ConnectionState.DISCONNECTED
            await self._schedule_reconnect()

    def _start_symbol_streams(self, symbol: str):
        """Start kline stream tasks for a symbol across all timeframes."""
        for timeframe in config.ALL_TIMEFRAMES:
            key = f"{symbol}_{timeframe}"
            if key not in self._tasks or self._tasks[key].done():
                task = asyncio.create_task(
                    self._listen_kline(symbol, timeframe),
                    name=key,
                )
                self._tasks[key] = task

    async def _listen_kline(self, symbol: str, timeframe: str):
        """Listen to a single kline stream with auto-reconnect."""
        binance_interval = BINANCE_INTERVALS.get(timeframe, timeframe)

        # For 10m, we listen to 5m and aggregate
        is_10m = timeframe == "10m"
        listen_interval = "5m" if is_10m else binance_interval
        pending_5m: Optional[dict] = None

        while self._running:
            try:
                socket = self._bsm.kline_socket(
                    symbol.lower(), interval=listen_interval
                )
                async with socket as stream:
                    while self._running:
                        msg = await asyncio.wait_for(stream.recv(), timeout=60)

                        if msg is None:
                            break

                        if "e" in msg and msg["e"] == "error":
                            logger.error(f"Stream error for {symbol}/{timeframe}: {msg}")
                            break

                        kline = msg.get("k", {})
                        is_closed = kline.get("x", False)

                        if not is_closed:
                            continue

                        candle = {
                            "timestamp": int(kline["t"]),
                            "open": float(kline["o"]),
                            "high": float(kline["h"]),
                            "low": float(kline["l"]),
                            "close": float(kline["c"]),
                            "volume": float(kline["v"]),
                        }

                        if is_10m:
                            # Aggregate two 5m candles into one 10m
                            if pending_5m is None:
                                pending_5m = candle
                                continue
                            else:
                                aggregated = {
                                    "timestamp": pending_5m["timestamp"],
                                    "open": pending_5m["open"],
                                    "high": max(pending_5m["high"], candle["high"]),
                                    "low": min(pending_5m["low"], candle["low"]),
                                    "close": candle["close"],
                                    "volume": pending_5m["volume"] + candle["volume"],
                                }
                                pending_5m = None
                                self._store_and_notify(symbol, timeframe, aggregated)
                        else:
                            self._store_and_notify(symbol, timeframe, candle)

            except asyncio.TimeoutError:
                logger.warning(f"Stream timeout for {symbol}/{timeframe}, reconnecting...")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Stream error for {symbol}/{timeframe}: {e}")

            if self._running:
                # Backoff before reconnecting this specific stream
                delay = min(2 ** self._reconnect_attempts, self._max_reconnect_delay)
                logger.info(f"Reconnecting {symbol}/{timeframe} in {delay}s")
                await asyncio.sleep(delay)
                self._reconnect_attempts += 1

                # Backfill missed candles on reconnect
                await self._backfill_on_reconnect(symbol, timeframe)
                self._reconnect_attempts = 0

    def _store_and_notify(self, symbol: str, timeframe: str, candle: dict):
        """Store candle in DB and fire callback."""
        db = SessionLocal()
        try:
            from sqlalchemy.dialects.mysql import insert as mysql_insert

            stmt = mysql_insert(Candle).values(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=candle["timestamp"],
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle["volume"],
            )
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
            logger.error(f"Failed to store live candle {symbol}/{timeframe}: {e}")
        finally:
            db.close()

        # Fire callback
        if self.on_candle_close:
            try:
                result = self.on_candle_close(symbol, timeframe, candle)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"on_candle_close callback error: {e}")

    async def _backfill_on_reconnect(self, symbol: str, timeframe: str):
        """Backfill any candles missed during disconnection."""
        logger.info(f"Backfilling missed candles for {symbol}/{timeframe}")
        try:
            # Fetch last 1 hour of data to cover the gap
            await fetch_historical(symbol, timeframe, months=0)
            # months=0 won't work well, use a short fetch instead
            # Fetch the last few hours directly
            from binance import AsyncClient as AC
            import time

            client = await AC.create(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_API_SECRET,
            )
            try:
                # Get last 100 candles to cover any reasonable gap
                binance_interval = BINANCE_INTERVALS.get(timeframe, timeframe)
                listen_interval = "5m" if timeframe == "10m" else binance_interval

                klines = await client.get_klines(
                    symbol=symbol,
                    interval=listen_interval,
                    limit=100,
                )

                if klines:
                    from src.data.historical import _parse_kline, _store_candles, _aggregate_5m_to_10m

                    candles = [_parse_kline(k) for k in klines]
                    if timeframe == "10m":
                        candles = _aggregate_5m_to_10m(candles)
                    _store_candles(symbol, timeframe, candles)

                logger.info(f"Backfill complete for {symbol}/{timeframe}")
            finally:
                await client.close_connection()

        except Exception as e:
            logger.error(f"Backfill failed for {symbol}/{timeframe}: {e}")

    async def _schedule_reconnect(self):
        """Schedule a full reconnection with exponential backoff."""
        if not self._running:
            return

        self.state = ConnectionState.RECONNECTING
        delay = min(2 ** self._reconnect_attempts, self._max_reconnect_delay)
        self._reconnect_attempts += 1
        logger.info(f"Scheduling full reconnect in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        if self._running:
            await self.stop()
            await self.start(list(self._symbols))

    async def stop(self):
        """Stop all streams and disconnect."""
        logger.info("Stopping stream manager")
        self._running = False

        # Cancel all listener tasks
        for key, task in self._tasks.items():
            if not task.done():
                task.cancel()

        # Wait for tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

        # Close client
        if self._client:
            await self._client.close_connection()
            self._client = None
            self._bsm = None

        self.state = ConnectionState.DISCONNECTED
        logger.info("Stream manager stopped")

    async def add_symbol(self, symbol: str):
        """Add a symbol to the active stream set."""
        if symbol in self._symbols:
            return

        self._symbols.add(symbol)
        logger.info(f"Adding symbol {symbol} to streams")

        if self._running and self._bsm:
            self._start_symbol_streams(symbol)

    async def remove_symbol(self, symbol: str):
        """Remove a symbol from the active stream set."""
        if symbol not in self._symbols:
            return

        self._symbols.discard(symbol)
        logger.info(f"Removing symbol {symbol} from streams")

        # Cancel associated tasks
        for timeframe in config.ALL_TIMEFRAMES:
            key = f"{symbol}_{timeframe}"
            if key in self._tasks:
                task = self._tasks.pop(key)
                if not task.done():
                    task.cancel()
